"""CPU-only: choose one saved answer per epoch x truncation x correctness cell."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from html import escape
from pathlib import Path

from recipe.common import (digest, file_hash, fresh_dir, iter_jsonl, now, read_json,
                           snapshot_code, stable_seed, write_json, write_jsonl)
from recipe.paths import artifact_path

CATEGORIES = (
    ("truncated_correct", "被上限截断 · 判对", True, True),
    ("truncated_incorrect", "被上限截断 · 判错", True, False),
    ("not_truncated_correct", "未被截断 · 判对", False, True),
    ("not_truncated_incorrect", "未被截断 · 判错", False, False),
)
DATASETS = {"math500": "MATH500", "gpqa": "GPQA Diamond"}
SELECTION_RULE = ("Deterministic greedy cover: repeatedly choose the question covering the most remaining "
                  "epoch/category cells; fill all such cells from that question. Seed breaks ties. "
                  "This favors cross-epoch reuse, not a random/representative sample or an exact minimum cover.")


def inside(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Input path escapes this job: {path}")
    return path


def grading_directories(job, plan):
    """Prefer one full-panel result, else one committed grade result per group."""
    def find(directory, complete_name, state_name):
        complete, state = directory / complete_name, job / "grouped_state" / state_name
        if complete.is_file():
            target = inside(read_json(complete)["grades"], job)
        elif state.is_file():
            target = inside(read_json(state)["output"], job)
        else:
            target = directory / "grades"
            if not (target / "grading_manifest.json").is_file():
                return None
        if not (target / "grading_manifest.json").is_file():
            raise ValueError(f"Saved grading completion points to missing results: {target}")
        return target

    full = find(job, "grouped_complete.json", "full-grading.json")
    if full:
        return [(full, plan["panel_ids"])], "full_panel"
    found = []
    for group in plan["groups"]:
        directory = inside(job / "groups" / group["name"], job)
        target = find(directory, "complete.json", group["name"] + "-grading.json")
        if target:
            found.append((target, group["selected_ids"]))
    if not found:
        raise ValueError("No completed grading results yet. Wait for one group's grading to finish; "
                         "this script does not regenerate or regrade answers.")
    return found, "completed_group_grades"


def load_evidence(job, epochs):
    plan_path = job / "grouped_pipeline.json"
    plan = read_json(plan_path)
    run = Path(plan["run"])
    panel_path = run / "data" / "panel.jsonl"
    if file_hash(run / "run.json") != plan["run_sha256"] or file_hash(panel_path) != plan["panel_sha256"]:
        raise ValueError("Run/panel changed since this evaluation job was created")
    if not set(epochs) <= set(plan["generation_epochs"]):
        raise ValueError("Requested epochs are not included in this job's generation protocol")
    panel_rows = list(iter_jsonl(panel_path))
    panel = {r["sample_id"]: r for r in panel_rows}
    if len(panel) != len(panel_rows) or list(panel) != plan["panel_ids"]:
        raise ValueError("Duplicate or mismatched panel IDs")
    directories, mode = grading_directories(job, plan)
    evidence, records, signatures = [], {}, set()
    for directory, expected_ids in directories:
        manifest_path = directory / "grading_manifest.json"
        manifest_hash = file_hash(manifest_path)
        manifest = read_json(manifest_path)
        seen_epochs = set()
        for source in manifest["sources"]:
            identity = source["identity"]
            epoch = identity["epoch"]
            if epoch not in epochs:
                continue
            if epoch in seen_epochs:
                raise ValueError(f"Duplicate epoch in grading manifest: {manifest_path}")
            seen_epochs.add(epoch)
            if (identity["run_sha256"] != plan["run_sha256"]
                    or identity["panel_sha256"] != plan["panel_sha256"]
                    or identity["selected_ids"] != expected_ids
                    or identity["settings"] != plan["settings"]
                    or identity["code_sha256"] != plan["code_sha256"]
                    or identity.get("backend", "transformers") != plan.get("generation_backend", "transformers")):
                raise ValueError(f"Grading input belongs to a different evaluation: {directory}")
            signatures.add(digest({
                "generation": {k: identity.get(k) for k in
                               ("settings", "generation_config", "backend_config", "seed_policy")},
                "grader_version": manifest["grader_version"], "grader_code": manifest.get("code_sha256"),
            }))
            raw_path = inside(source["responses"], job)
            if raw_path.parent.name != f"epoch-{epoch:03d}":
                raise ValueError("Response path/epoch mismatch")
            if read_json(raw_path.parent / "identity.json") != identity or file_hash(raw_path) != source["sha256"]:
                raise ValueError(f"Original responses changed since grading: {raw_path}")
            graded_path = directory / f"epoch-{epoch:03d}" / "graded.jsonl"
            before_hash = file_hash(graded_path)
            raw_rows = {}
            for raw in iter_jsonl(raw_path):
                if raw["sample_id"] in raw_rows:
                    raise ValueError(f"Duplicate original response: {raw_path}")
                raw_rows[raw["sample_id"]] = raw
            if file_hash(raw_path) != source["sha256"]:
                raise ValueError(f"Original responses changed while reading: {raw_path}")
            actual = set()
            limit = identity["settings"]["max_new_tokens"]
            for row in iter_jsonl(graded_path):
                sid = row["sample_id"]
                key = (epoch, sid)
                if sid not in expected_ids or key in records or sid not in raw_rows:
                    raise ValueError(f"Duplicate/unexpected graded sample: {key}")
                question, raw = panel[sid], raw_rows[sid]
                if any(row.get(k) != v for k, v in raw.items()):
                    raise ValueError(f"Grading modified the original generated answer: {key}")
                if (row["epoch"] != epoch or row["task"] not in DATASETS
                        or any(row[k] != question[k] for k in ("question_id", "task", "messages", "gold_answer"))
                        or row["grader_version"] != manifest["grader_version"]):
                    raise ValueError(f"Mismatched graded question/protocol: {key}")
                flags = ("length_truncated", "reached_length_limit", "ended_with_eos", "normal_chat_end")
                if any(type(row[k]) is not bool for k in flags):
                    raise ValueError(f"Invalid stop flags: {key}")
                n = row["generated_tokens"]
                if (type(n) is not int or n < 1 or n > limit or n != len(row["generated_token_ids"])
                        or row["reached_length_limit"] != (n == limit)
                        or row["length_truncated"] != (n == limit and not row["ended_with_eos"])
                        or (row["normal_chat_end"] and not row["ended_with_eos"])):
                    raise ValueError(f"Inconsistent length/stop metadata: {key}")
                if row["correct"] is not None and type(row["correct"]) is not bool:
                    raise ValueError(f"Invalid correctness value: {key}")
                if row["correct"] is True and (not row["extractable"] or row["extracted_answer"] is None):
                    raise ValueError(f"Correct answer has no extracted answer: {key}")
                records[key] = {"row": row, "question": question, "limit": limit,
                                "source": {"graded_file": str(graded_path), "responses_file": str(raw_path),
                                           "grading_manifest": str(manifest_path)}}
                actual.add(sid)
            if actual != set(raw_rows) or not actual <= set(expected_ids) or file_hash(graded_path) != before_hash:
                raise ValueError(f"Incomplete or concurrently changed grading file: {graded_path}")
            evidence.append({"graded_file": str(graded_path), "graded_sha256": before_hash,
                             "responses_file": str(raw_path), "responses_sha256": source["sha256"],
                             "grading_manifest": str(manifest_path), "manifest_sha256": manifest_hash})
        if file_hash(manifest_path) != manifest_hash:
            raise ValueError(f"Grading manifest changed while reading: {manifest_path}")
    if len(signatures) != 1 or not records:
        raise ValueError("No requested graded answers, or mixed generation/grading protocols")
    return plan, panel, records, evidence, mode


def select_examples(records, epochs, seed):
    candidates = {(e, c): set() for e in epochs for c, _, _, _ in CATEGORIES}
    for (epoch, sid), item in records.items():
        row = item["row"]
        if row.get("grading_error") is not None or type(row["correct"]) is not bool:
            continue  # Unknown grader outcomes are not incorrect model answers.
        category = next(c for c, _, truncated, correct in CATEGORIES
                        if row["length_truncated"] == truncated and row["correct"] == correct)
        candidates[epoch, category].add(sid)
    remaining = {cell for cell, ids in candidates.items() if ids}
    choices = {}
    while remaining:
        coverage = {}
        for cell in remaining:
            for sid in candidates[cell]:
                coverage.setdefault(sid, set()).add(cell)
        best = min(coverage, key=lambda sid: (-len(coverage[sid]), stable_seed(seed, sid), sid))
        for cell in coverage[best]:
            choices[cell] = best
        remaining -= coverage[best]
    return candidates, choices


def text(value):
    return escape(str(value), quote=True)


def render_report(slots, coverage, manifest):
    labels = {c: label for c, label, _, _ in CATEGORIES}
    lookup = {(s["epoch"], s["category"]): s for s in slots}
    html = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width,initial-scale=1'><title>生成回答案例报告</title>",
            "<style>body{max-width:1120px;margin:32px auto;padding:0 20px;font:16px/1.65 system-ui;color:#182536;background:#f7f9fc}"
            "table{border-collapse:collapse;width:100%;margin:12px 0}th,td{padding:10px;border:1px solid #ccd5df;text-align:left;vertical-align:top}"
            "article{background:white;border:1px solid #ccd5df;border-radius:10px;padding:22px;margin:24px 0;scroll-margin-top:16px}"
            "article:target{outline:3px solid #397cb7}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.7 ui-monospace,monospace}"
            "a{color:#16598e}summary,button{cursor:pointer}button{padding:8px 14px;margin:8px 8px 8px 0}"
            ".note{background:#eaf2fa;padding:14px;border-radius:6px}.missing{color:#7c4c13}small{color:#536274}</style>",
            "<body><h1>生成回答案例报告</h1>",
            f"<p>共 {len(slots)} 个位置，选中 {manifest['selected_count']} 条回答，涉及 "
            f"{manifest['distinct_questions']} 道题。生成时间（UTC）：{text(manifest['created_at'])}</p>",
            "<p class='note'>分类按 length_truncated：达到生成上限且没有结束符才算被截断。"
            "恰好在上限生成结束符属于未被截断。判分异常不作为判错。候选来自已保存的判分，未完成组不参与本次选择。"
            "每轮四类在 MATH500 与 GPQA 合并候选中各取一条，不是每个数据集各取一条。</p>",
            "<p>优先让同一道题覆盖多个 epoch 的位置，再补选其他题；这是一组便于检查的案例，不是随机样本或总体正确率估计。"
            "同一题可随 epoch 转入另一类别。空位置明确保留，不以其他类别代替。</p>",
            "<h2>已保存判分的覆盖情况</h2><table><tr><th>Epoch</th><th>已保存 / 面板题数</th><th>判分异常或未知</th><th>覆盖完整</th></tr>"]
    for c in coverage:
        html.append(f"<tr><td>{c['epoch']}</td><td>{c['observed']}/{c['expected']}</td>"
                    f"<td>{c['unknown']}</td><td>{'是' if c['complete'] else '否：当前结果快照'}</td></tr>")
    html.append("</table><h2>案例导航</h2><table><tr><th>Epoch</th>" +
                "".join(f"<th>{text(label)}</th>" for _, label, _, _ in CATEGORIES) + "</tr>")
    for epoch in manifest["epochs"]:
        html.append(f"<tr><th>{epoch}</th>")
        for category, _, _, _ in CATEGORIES:
            s = lookup[epoch, category]
            label = f"{s['dataset']} · {s['sample_id'][:14]} · {s['generated_tokens']} tokens" if s["status"] == "selected" else "无可选记录"
            html.append(f"<td><a href='#{s['slot_id']}'>{text(label)}</a><br><small>候选 {s['candidate_count']} 条</small></td>")
        html.append("</tr>")
    html.append("</table><button onclick=\"document.querySelectorAll('details.answer').forEach(x=>x.open=true)\">展开全部完整回答</button>"
                "<button onclick=\"document.querySelectorAll('details.answer').forEach(x=>x.open=false)\">收起全部完整回答</button>")
    uses = Counter(s["sample_id"] for s in slots if s["status"] == "selected")
    for slot in slots:
        html.append(f"<article id='{slot['slot_id']}'><h2>Epoch {slot['epoch']} · {text(labels[slot['category']])}</h2>")
        if slot["status"] != "selected":
            html.append(f"<p class='missing'>{text(slot['reason'])}</p></article>")
            continue
        r, q = slot["graded_response"], slot["question_record"]
        metadata = [("数据集", slot["dataset"]), ("Epoch", slot["epoch"]), ("生成 token 数", slot["generated_tokens"]),
                    ("生成上限", slot["max_new_tokens"]), ("sample_id", slot["sample_id"]),
                    ("停止原因", r["stop_reason"]), ("达到上限 / 被截断", f"{r['reached_length_limit']} / {r['length_truncated']}"),
                    ("判分结果", "正确" if r["correct"] else "错误"), ("提取答案", r["extracted_answer"]),
                    ("抽取方式", r["extraction_method"]), ("本题被选入的位置数", uses[slot["sample_id"]])]
        html.append("<table>" + "".join(f"<tr><th>{text(k)}</th><td>{text(v)}</td></tr>" for k, v in metadata) + "</table>")
        html.append(f"<h3>题目</h3><pre>{text(q['question'])}</pre>")
        if q.get("choices"):
            html.append("<pre>" + text("\n".join(f"{a}. {b}" for a, b in zip("ABCD", q["choices"]))) + "</pre>")
        gold = str(q["gold_answer"])
        if q["task"] == "gpqa" and q.get("choices") and gold in "ABCD" and len(gold) == 1:
            gold += ". " + q["choices"]["ABCD".index(gold)]
        html.append(f"<h3>正确答案</h3><pre>{text(gold)}</pre>")
        prompt = "\n\n".join(m["role"] + ":\n" + m["content"] for m in r["messages"])
        html.append(f"<details><summary>实际提示词</summary><pre>{text(prompt)}</pre></details>")
        html.append(f"<p><small>来源：{text(slot['source']['graded_file'])}<br>完整字段与 token IDs：selected_examples.jsonl</small></p>")
        # The full raw response is the LAST content in each example, without shortening or stripping tokens.
        html.append(f"<details class='answer'><summary>完整回答（{r['generated_tokens']} tokens；点击展开）</summary>"
                    f"<pre>{text(r['raw_response'])}</pre></details></article>")
    html.append("</body></html>")
    return "\n".join(html)


def build_report(job, out, epochs, seed=42):
    plan, panel, records, evidence, mode = load_evidence(job, epochs)
    candidates, choices = select_examples(records, epochs, seed)
    coverage = []
    for epoch in epochs:
        rows = [x["row"] for (e, _), x in records.items() if e == epoch]
        unknown = sum(r.get("grading_error") is not None or type(r["correct"]) is not bool for r in rows)
        coverage.append({"epoch": epoch, "observed": len(rows), "expected": len(panel), "unknown": unknown,
                         "complete": len(rows) == len(panel) and unknown == 0})
    coverage_map = {r["epoch"]: r for r in coverage}
    slots = []
    for epoch in epochs:
        for category, _, _, _ in CATEGORIES:
            cell = (epoch, category)
            slot = {"slot_id": f"e{epoch}-{category}", "epoch": epoch, "category": category,
                    "candidate_count": len(candidates[cell])}
            if cell not in choices:
                complete = coverage_map[epoch]["complete"]
                slot.update(status="category_empty" if complete else "no_candidate_in_available_results",
                            reason="该 epoch 的完整判分中没有此类样本。" if complete else
                            "当前可用判分中没有此类样本；部分结果尚未完成或判分未知，不能认定整个面板没有此类。")
            else:
                sid = choices[cell]
                item = records[epoch, sid]
                slot.update(status="selected", sample_id=sid, dataset=DATASETS[item["row"]["task"]],
                            generated_tokens=item["row"]["generated_tokens"], max_new_tokens=item["limit"],
                            graded_response=item["row"], question_record=item["question"], source=item["source"])
            slots.append(slot)
    manifest = {"created_at": now(), "job": str(job), "job_manifest_sha256": file_hash(job / "grouped_pipeline.json"),
                "panel_sha256": plan["panel_sha256"], "epochs": epochs, "seed": seed, "selection_rule": SELECTION_RULE,
                "length_field": "length_truncated", "correctness_source": "saved grade only; errors/None excluded",
                "backend": plan.get("generation_backend", "transformers"), "input_mode": mode, "inputs": evidence,
                "coverage": coverage, "slot_count": len(slots), "selected_count": len(choices),
                "distinct_questions": len(set(choices.values())), "missing_slots": len(slots) - len(choices)}
    out = fresh_dir(out)  # Existing reports are never overwritten.
    manifest["report_code_sha256"] = snapshot_code(out / "code")
    write_jsonl(out / "selected_examples.jsonl", slots)
    columns = ["slot_id", "epoch", "category", "status", "candidate_count", "sample_id", "dataset", "generated_tokens", "max_new_tokens", "reason"]
    with (out / "selection.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(slots)
    (out / "report.html").write_text(render_report(slots, coverage, manifest), encoding="utf-8")
    manifest["output_hashes"] = {n: file_hash(out / n) for n in ("report.html", "selected_examples.jsonl", "selection.csv")}
    write_json(out / "manifest.json", manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", required=True, help="Grouped evaluation job, e.g. runs/random-lora-s42/postprocess/grouped-e4plus-vllm-v1")
    p.add_argument("--out", required=True, help="NEW report directory on the data volume")
    p.add_argument("--epochs", default="4,8,16,32,64")
    p.add_argument("--seed", type=int, default=42, help="Tie-break seed for reproducible cross-epoch selection")
    args = p.parse_args()
    try:
        epochs = sorted(set(int(x) for x in args.epochs.split(",")))
    except ValueError:
        p.error("--epochs must be comma-separated integers")
    if not epochs or min(epochs) < 0:
        p.error("Expected nonnegative epochs")
    result = build_report(artifact_path(args.job), artifact_path(args.out), epochs, args.seed)
    print(f"Saved {result['selected_count']}/{result['slot_count']} examples; "
          f"{result['distinct_questions']} distinct questions; {result['missing_slots']} empty slots.")
    print(f"Open {artifact_path(args.out) / 'report.html'}")


if __name__ == "__main__":
    main()
