"""SERVER: let the unadapted Base solve the frozen random/high/low training questions."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import csv
from html import escape
import importlib.metadata
import io
import os
from pathlib import Path
import re
import shutil
import sys
import uuid

from recipe.common import (ROOT, append_jsonl, digest, environment, file_hash, iter_jsonl, now,
                           read_json, recover_jsonl, snapshot_code, stable_seed, write_json, write_jsonl)
from recipe.encoding import normalize_question, question_id
from recipe.grading import GRADER_VERSION, grade_one
from recipe.paths import artifact_path, configure_runtime
from recipe.pipeline import Pipeline
from recipe.resume import verified_code, verify_run_inputs
from recipe.vllm_generation import completed_rows, engine_options, sampling_options, validate_config
from generate import stop_details
from generate_vllm import configure_vllm_runtime, execute, runtime_identity
from grade import summarize

GROUP_SIZES = {"random": 400, "high": 100, "low": 100}
FORMAT = "math-cot-training-base-v1"


def build_panel(groups, sizes=GROUP_SIZES):
    """Deduplicate identical question + prompt, retaining every original membership/text."""
    panel, by_id, golds = [], {}, {}
    if set(groups) != set(sizes):
        raise ValueError("Expected random, high and low training groups")
    for group, size in sizes.items():
        rows = groups[group]
        if len(rows) != size or len({r["question_id"] for r in rows}) != size:
            raise ValueError(f"{group}: expected {size} distinct training questions")
        for index, row in enumerate(rows):
            qid, messages = row["question_id"], row["messages"]
            if (qid != question_id(row["question"])
                    or row["sample_id"] != digest([qid, row["response"]])):
                raise ValueError(f"Training text/identity mismatch: {group}/{index}")
            if ([m["role"] for m in messages] not in (["user"], ["system", "user"])
                    or normalize_question(row["question"]) not in normalize_question(messages[-1]["content"])):
                raise ValueError("Expected the original question-only training prompt")
            for message in messages:
                if any(t in message["content"] for t in ("<|im_start|>", "<|im_end|>", "<|endoftext|>")):
                    raise ValueError("Reserved chat delimiter in question prompt")
            if qid in golds and golds[qid] != row["answer"]:
                raise ValueError(f"Conflicting gold answers for shared question: {qid}")
            golds[qid] = row["answer"]
            sid = "math-cot-base-" + digest([qid, messages])
            if sid not in by_id:
                item = {"sample_id": sid, "question_id": qid, "task": "math", "dataset": "jasonrqh/Math-CoT-20k",
                        "question": row["question"], "gold_answer": row["answer"], "messages": messages,
                        "groups": [], "references": []}
                panel.append(item)
                by_id[sid] = item
            item = by_id[sid]
            item["groups"].append(group)
            item["references"].append({"group": group, "training_index": index, "sample": row})
    return panel


def verify_setup(out):
    setup = out / "setup"
    if read_json(setup / "complete.json")["plan_sha256"] != file_hash(setup / "plan.json"):
        raise ValueError("Frozen experiment plan changed")
    plan = read_json(setup / "plan.json")
    if plan["format_version"] != FORMAT or plan["model_mode"] != "base_only_no_adapter":
        raise ValueError("Not a training-question Base experiment")
    verified_code(setup / "code", plan["code_sha256"])
    for name, checksum in plan["files"].items():
        path = (setup / name).resolve()
        if not path.is_relative_to(setup.resolve()) or file_hash(path) != checksum:
            raise ValueError(f"Frozen experiment input changed: {name}")
    return plan


def prepare(out, sources, config=None, engine_config=None):
    if (out / "setup").exists():
        plan = verify_setup(out)
        if plan["source_paths"] != {k: str(v) for k, v in sources.items()}:
            raise ValueError("Resume source paths changed; choose a new --out")
        for path, name in ((config, "generation.json"), (engine_config, "engine.json")):
            if path and read_json(path) != read_json(out / "setup" / name):
                raise ValueError("Resume settings changed; choose a new --out")
        return plan
    if any(p.name != "pipeline.lock" and not p.name.startswith(".setup-partial-") for p in out.iterdir()):
        raise ValueError("Nonempty output has no recognized setup; choose a new --out")
    metadata = {k: verify_run_inputs(v) for k, v in sources.items()}
    base = metadata["random"]
    if (base["model"]["id"] != "Qwen/Qwen3-4B-Base"
            or not re.fullmatch(r"[0-9a-f]{40}", base["model"]["revision"])):
        raise ValueError("Expected the training run's pinned Qwen/Qwen3-4B-Base revision")
    for group, meta in metadata.items():
        if meta["model"] != base["model"] or meta["tokenizer_files"] != base["tokenizer_files"]:
            raise ValueError(f"{group}: Base model/tokenizer differs from the random run")
    groups = {k: list(iter_jsonl(v / "data/train.jsonl")) for k, v in sources.items()}
    panel = build_panel(groups)
    settings = read_json(config or ROOT / "configs/generation.json")
    sampling_options(settings, 151643, 151645)  # Validate schema without loading a tokenizer/model.
    engine = validate_config(read_json(engine_config or ROOT / "configs/vllm.json"))
    stage = out / (".setup-partial-" + uuid.uuid4().hex)
    stage.mkdir()
    write_json(stage / "generation.json", settings)
    write_json(stage / "engine.json", engine)
    write_json(stage / "runs.json", metadata)
    for group, source in sources.items():
        target = stage / "training" / group
        target.mkdir(parents=True)
        shutil.copy2(source / "data/train.jsonl", target / "train.jsonl")
        shutil.copy2(source / "run.json", target / "run.json")
        if (source / "data/manifest.json").is_file():
            shutil.copy2(source / "data/manifest.json", target / "data_manifest.json")
    shutil.copytree(sources["random"] / "tokenizer", stage / "tokenizer")
    write_jsonl(stage / "panel.jsonl", panel)
    files = {p.relative_to(stage).as_posix(): file_hash(p) for p in stage.rglob("*") if p.is_file()}
    code_sha = snapshot_code(stage / "code")
    counts = Counter(len(q["groups"]) for q in panel)
    plan = {"format_version": FORMAT, "created_at": now(), "model_mode": "base_only_no_adapter",
            "model": base["model"], "source_paths": {k: str(v) for k, v in sources.items()},
            "source_run_sha256": {k: file_hash(v / "run.json") for k, v in sources.items()},
            "group_sizes": GROUP_SIZES, "group_memberships": sum(GROUP_SIZES.values()),
            "unique_prompts": len(panel), "unique_questions": len({q["question_id"] for q in panel}),
            "shared_prompts": sum(n for memberships, n in counts.items() if memberships > 1),
            "deduplication": "one generation per identical normalized question + exact original messages",
            "seed_policy": "stable_seed(generation seed, question+prompt identity); shared by all memberships",
            "prompt_policy": "original messages only; reference response and gold answer never appended",
            "files": files, "code_sha256": code_sha}
    write_json(stage / "plan.json", plan)
    write_json(stage / "complete.json", {"plan_sha256": file_hash(stage / "plan.json")})
    stage.rename(out / "setup")
    return plan


def generation_inputs(out):
    """Return the existing vLLM runner's protocol, with LoRA explicitly disabled."""
    plan = verify_setup(out)
    from transformers import AutoConfig, AutoTokenizer
    setup = out / "setup"
    metadata = read_json(setup / "runs.json")["random"]
    settings = read_json(setup / "generation.json")
    tokenizer = AutoTokenizer.from_pretrained(setup / "tokenizer", local_files_only=True)
    eos_id, end_id = tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")
    if (type(eos_id) is not int or type(end_id) is not int
            or tokenizer.decode([end_id], skip_special_tokens=False) != "<|im_end|>"):
        raise ValueError("Missing Qwen EOS/chat-end tokens")
    panel = list(iter_jsonl(setup / "panel.jsonl"))
    prompts = {q["sample_id"]: tokenizer.apply_chat_template(q["messages"], tokenize=True,
                                                           add_generation_prompt=True) for q in panel}
    model_config = AutoConfig.from_pretrained(plan["model"]["id"], revision=plan["model"]["revision"],
                                             trust_remote_code=False)
    if model_config.model_type != "qwen3":
        raise ValueError("Expected dense Qwen3")
    options = engine_options(metadata, read_json(setup / "engine.json"), prompts,
                             model_config.max_position_embeddings, settings, base_only=True)
    runtime = {**runtime_identity(), "adapter_compute_dtype": None, "model_mode": "base_only_no_adapter"}
    from recipe.evaluation import bind_output
    destination = out / "generation/epoch-000"
    bind_output(destination, {"plan_sha256": file_hash(setup / "plan.json"), "epoch": 0,
                 "model": plan["model"], "model_mode": plan["model_mode"], "settings": settings,
                 "selected_ids": [q["sample_id"] for q in panel], "eos_id": eos_id, "end_id": end_id,
                 "prompt_ids_sha256": digest(prompts), "backend": "vllm", "runtime": runtime,
                 "engine_options": options, "model_config_sha256": digest(model_config.to_dict())})
    done = completed_rows(destination, panel, prompts, 0, settings, eos_id, end_id)
    if not (destination / "environment.json").exists():
        write_json(destination / "environment.json", environment())
    return {"options": options, "checkpoints": [{"epoch": 0, "out": destination, "checkpoint": None,
                                               "pending": len(panel) - len(done)}],
            "panel": panel, "prompts": prompts, "settings": settings, "tokenizer": tokenizer,
            "eos_id": eos_id, "end_id": end_id, "base_only": True, "progress_label": "Math-CoT training questions"}


def saved_responses(out, allow_partial=False):
    """CPU-only integrity checks; partial progress never counts missing answers as wrong."""
    plan = verify_setup(out)
    panel = {q["sample_id"]: q for q in iter_jsonl(out / "setup/panel.jsonl")}
    folder = out / "generation/epoch-000"
    identity = read_json(folder / "identity.json")
    if (identity["plan_sha256"] != file_hash(out / "setup/plan.json")
            or identity["model_mode"] != "base_only_no_adapter" or identity["epoch"] != 0
            or identity["settings"] != read_json(out / "setup/generation.json")
            or identity["selected_ids"] != list(panel)):
        raise ValueError("Generation identity differs from the frozen setup")
    log, marker = folder / "responses.jsonl", folder / "complete.json"
    if marker.exists() and read_json(marker) != {"samples": len(panel), "responses_sha256": file_hash(log)}:
        raise ValueError("Completed generation was modified")
    rows = recover_jsonl(log)
    seen = set()
    for row in rows:
        sid = row["sample_id"]
        if sid not in panel or sid in seen:
            raise ValueError("Duplicate/unexpected saved response")
        q = panel[sid]
        ids = row["generated_token_ids"]
        if (row["epoch"] != 0 or row["backend"] != "vllm"
                or any(row[k] != q[k] for k in ("question_id", "task", "gold_answer", "messages"))
                or row["seed"] != stable_seed(identity["settings"]["seed"], sid)
                or not ids or any(type(t) is not int or t < 0 for t in ids)
                or row["generated_tokens"] != len(ids) or len(ids) > identity["settings"]["max_new_tokens"]
                or row["prompt_tokens"] != len(row["prompt_token_ids"])):
            raise ValueError(f"Invalid saved response: {sid}")
        detail = stop_details(ids, identity["eos_id"], identity["end_id"], identity["settings"]["max_new_tokens"])
        if detail["stop_reason"] == "other" or any(row[k] != v for k, v in detail.items()):
            raise ValueError(f"Invalid stop metadata: {sid}")
        seen.add(sid)
    if len(rows) != len(panel) and not allow_partial:
        raise ValueError("Generation incomplete; resume it or use --stage grade --allow-partial")
    return plan, panel, rows


def grade_saved(out, allow_partial=False, stop_check=lambda: None):
    import math_verify  # Fail explicitly if unavailable; do not silently change judging.
    plan, panel, responses = saved_responses(out, allow_partial)
    directory = out / "grades"
    from recipe.evaluation import bind_output
    bind_output(directory, {"plan_sha256": file_hash(out / "setup/plan.json"),
                "generation_identity_sha256": file_hash(out / "generation/epoch-000/identity.json"),
                "grader_version": GRADER_VERSION, "math_verify_version": importlib.metadata.version("math-verify")})
    graded_path = directory / "graded.jsonl"
    rows = recover_jsonl(graded_path)
    response_by_id = {r["sample_id"]: r for r in responses}
    seen = set()
    for row in rows:
        sid = row["sample_id"]
        if sid in seen or sid not in response_by_id or row["response_sha256"] != digest(response_by_id[sid]):
            raise ValueError("Saved grading has duplicate IDs or changed source responses")
        seen.add(sid)
    marker = directory / "complete.json"
    if marker.exists():
        if read_json(marker) != {"graded_sha256": file_hash(graded_path),
                                 "responses_sha256": file_hash(out / "generation/epoch-000/responses.jsonl")}:
            raise ValueError("Completed grading changed")
    for response in responses:
        stop_check()
        sid = response["sample_id"]
        if sid in seen:
            continue
        q = panel[sid]
        row = {**q, **response, **grade_one(response["raw_response"], response["gold_answer"], "math"),
               "response_sha256": digest(response), "grader_version": GRADER_VERSION}
        append_jsonl(graded_path, row)
        rows.append(row)
        seen.add(sid)
        print(f"GRADE {len(rows)}/{len(panel)} correct={row['correct']} id={sid}", flush=True)
    if len(rows) == len(panel):
        write_json(marker, {"graded_sha256": file_hash(graded_path),
                            "responses_sha256": file_hash(out / "generation/epoch-000/responses.jsonl")})
    report(out, plan, panel, rows)


def report(out, plan, panel, graded):
    directory = out / "report"
    directory.mkdir(exist_ok=True)
    order = {sid: i for i, sid in enumerate(panel)}
    graded = sorted(graded, key=lambda r: order[r["sample_id"]])
    summaries = []
    for group, expected in plan["group_sizes"].items():
        members = [r for r in graded if group in r["groups"]]
        summary = {"group": group, "expected": expected, "completed": len(members),
                   "missing": expected - len(members), "complete": len(members) == expected,
                   **(summarize(members) if members else {"accuracy": None})}
        summaries.append(summary)
        write_jsonl(directory / f"{group}.jsonl", members)
    write_json(directory / "summary.json", {"model": plan["model"], "model_mode": plan["model_mode"],
               "unique_prompts": len(panel), "completed_unique_prompts": len(graded),
               "group_sizes": plan["group_sizes"], "groups": summaries,
               "denominator": "completed questions in each group; missing are excluded and explicitly counted",
               "caution": "one stochastic sample per prompt, not pass@k; selected training questions, not held-out generalization"})
    keys = ["group", "expected", "completed", "missing", "accuracy", "mean_generated_tokens", "eos_end_rate",
            "normal_chat_end_rate", "length_truncated_rate", "extractable_rate", "grading_errors"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(summaries)
    atomic_text(directory / "summary.csv", buffer.getvalue())
    body = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Base 解训练题</title>',
            '<style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:20px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f5f5;padding:14px}details{margin:12px 0}</style>',
            '<h1>Qwen3-4B-Base 解 Math-CoT-20k 训练题</h1>',
            f'<p>模型 revision：{escape(plan["model"]["revision"])}；BF16，不加载 LoRA。仅输入原题及原提示词。</p>',
            f'<p>三组共 {plan["group_memberships"]} 个成员；合并相同题目及提示词后 {len(panel)} 次生成，目前完成 {len(graded)} 次。</p>',
            '<p>每题一次采样；这衡量所选训练题的初始解题表现，不是泛化评测。未完成题不计为错误，部分报告分母为已完成题。判分异常仍计入分母。</p>',
            '<p>生成参数：<code>' + escape(str(read_json(out / "setup/generation.json"))) + '</code>；vLLM 并发 8。</p>',
            '<p>沿用现有数学判分：提取最后完整 boxed（或答案标记），用 math-verify 比较。尚未闭合的 think 不提取；截断但可提取的回答仍可能判对。</p>',
            '<p><a href="summary.csv">汇总 CSV</a> · <a href="summary.json">汇总 JSON</a> · <a href="../grades/graded.jsonl">全部逐题数据</a></p>',
            '<table><tr><th>组</th><th>已完成/计划</th><th>正确率</th><th>平均 token</th><th>EOS 结束率</th><th>聊天结束率</th><th>截断率</th><th>提取率</th><th>判分异常</th></tr>']
    for s in summaries:
        pct = lambda k: '-' if s.get(k) is None else f'{s[k]:.2%}'
        mean = '-' if not s["completed"] else f'{s["mean_generated_tokens"]:.1f}'
        cells = [s["group"], f'{s["completed"]}/{s["expected"]}', pct("accuracy"), mean, pct("eos_end_rate"),
                 pct("normal_chat_end_rate"), pct("length_truncated_rate"), pct("extractable_rate"), s.get("grading_errors", 0)]
        body.append('<tr>' + ''.join('<td>' + escape(str(x)) + '</td>' for x in cells) + '</tr>')
    body.append('</table>')
    for group in plan["group_sizes"]:
        body.append(f'<h2>{group}</h2><p><a href="{group}.jsonl">该组完整逐题 JSONL</a></p>')
        for row in graded:
            if group not in row["groups"]:
                continue
            reference = next(r["sample"] for r in row["references"] if r["group"] == group)
            body.append(f'<details><summary>row {reference["row_index"]} · correct={row["correct"]} · {row["generated_tokens"]} tokens · {escape(row["stop_reason"])}</summary>')
            fields = [("数据集", row["dataset"]), ("来源", reference["source"]), ("原训练 sample_id", reference["sample_id"]),
                      ("所属组", ", ".join(row["groups"])), ("题目", row["question"]), ("正确答案", row["gold_answer"]),
                      ("提取答案", row["extracted_answer"]), ("判分异常", row["grading_error"])]
            body.extend(f'<h3>{label}</h3><pre>{escape(str(value))}</pre>' for label, value in fields)
            body.append('<details><summary>数据集参考解答（未输入模型）</summary><pre>' + escape(reference["response"]) + '</pre></details>')
            body.append('<h3>Base 完整回答</h3><pre>' + escape(row["raw_response"]) + '</pre></details>')
    atomic_text(directory / "report.html", ''.join(body) + '</html>')
    print(f"REPORT: {directory / 'report.html'}", flush=True)


def atomic_text(path, value):
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temp.open("w", encoding="utf-8", newline="") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-run", default="runs/random-lora-s42")
    parser.add_argument("--experiment", default="runs/likelihood-100x16-s42", help="Contains high/ and low/ training runs")
    parser.add_argument("--out", default="runs/math-cot-training-base-s42")
    parser.add_argument("--config", help="Generation JSON; default configs/generation.json, frozen on first start")
    parser.add_argument("--engine-config", help="vLLM JSON; default configs/vllm.json, frozen on first start")
    parser.add_argument("--prepare-only", action="store_true", help="Freeze/verify the exact training texts; no model/GPU")
    parser.add_argument("--stage", choices=["all", "generate", "grade"], default="all")
    parser.add_argument("--allow-partial", action="store_true", help="Only for --stage grade: summarize completed answers")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.allow_partial and args.stage != "grade":
        parser.error("--allow-partial requires --stage grade")
    configure_runtime()
    if args.worker:
        out = Path(args.worker).resolve()
        with Pipeline(out, inherited=True) as worker:
            if args.stage == "generate":
                configure_vllm_runtime()
                raise SystemExit(asyncio.run(execute(generation_inputs(out))))
            grade_saved(out, args.allow_partial, worker.check_stop)
        return
    out = artifact_path(args.out)
    experiment = artifact_path(args.experiment)
    sources = {"random": artifact_path(args.random_run), "high": experiment / "high", "low": experiment / "low"}
    with Pipeline(out) as worker:
        plan = prepare(out, sources, args.config, args.engine_config)
        print(f"TRAINING BASE: random=400, high=100, low=100; unique prompts={plan['unique_prompts']}; output={out}", flush=True)
        if args.prepare_only:
            print("Inputs frozen. Repeat without --prepare-only to solve the questions.", flush=True)
            return
        code = out / "setup/code"
        if args.stage in ("all", "grade"):
            # Dependency preflight before spending GPU time. Actual grading is a separate process.
            import math_verify
        for stage in (["generate", "grade"] if args.stage == "all" else [args.stage]):
            write_json(out / "progress.json", {"stage": stage, "updated_at": now()})
            worker.invoke([sys.executable, "-B", "-u", code / "solve_training_base.py", "--worker", out,
                           "--stage", stage, *(["--allow-partial"] if args.allow_partial else [])], code)
        write_json(out / "progress.json", {"stage": "complete" if (out / "grades/complete.json").exists()
                                           else "generation_complete" if args.stage == "generate" else "partial_report", "updated_at": now()})


if __name__ == "__main__":
    main()
