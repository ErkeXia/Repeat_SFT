"""CPU-only reports of correct-to-incorrect validation answers, using saved grades."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from html import escape
import json
from pathlib import Path

from recipe.common import ROOT, file_hash, fresh_dir, now, write_json, write_jsonl
from recipe.paths import artifact_path
from recipe.regressions import TASKS, Evidence, analyze_series, discover, load_sources

CLUES = {
    "new_truncation": "后来开始被长度上限截断", "both_truncated": "前后均被截断",
    "lost_answer_extraction": "后来无法提取答案/代码", "lost_eos": "后来不再正常生成结束符",
    "unclosed_think": "回答含未闭合的 think 标签", "length_at_least_doubled": "输出长度至少翻倍",
    "length_at_least_halved": "输出长度降至一半以下", "increased_token_repetition": "重复 token 片段明显增加",
    "more_repeated_long_lines": "重复长文本行增加", "more_answer_markers": "最终答案标记增加",
    "extractable_but_incorrect": "仍能提取答案，但判分为错", "code_wrong_answer": "代码测试答案错误",
    "code_runtime_error": "代码运行异常", "code_timeout": "单个代码测试超时",
    "code_problem_timeout": "整题代码测试超时", "code_compile_error": "代码语法错误",
    "code_no_code": "无法提取代码", "code_output_limit": "代码输出超限",
    "code_invalid_function_output": "函数返回格式无效",
}
STATES = {"correct": "正确", "incorrect": "错误", "grading_unknown": "判分异常/未知", "missing": "无可用判分"}
STATUSES = {"complete": "完整", "partial": "部分完成", "no_completed_grades": "尚无完成判分",
            "not_requested_in_this_series": "本评测未安排此节点"}
NOTE = ("这里只描述已保存的单次采样结果，答对→答错不等于已证明过拟合或永久遗忘。"
        "缺失节点、未完成判分和判分异常不计为答错；退化区间连接该题相邻的有效观测节点，不能确定中间哪一轮开始退化。"
        "不同评测任务目录、生成参数或判分协议分别分析，不拼接成一条曲线。")
STYLE = """body{font:16px/1.65 system-ui,sans-serif;max-width:1450px;margin:28px auto;padding:0 20px;color:#243247;background:#f6f8fb}
h1,h2,h3{line-height:1.3}a{color:#175a94}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #ccd4df;padding:8px;text-align:left;vertical-align:top}
th{background:#edf2f8}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 ui-monospace,monospace;margin:0;padding:14px;background:#fff}
article,.note{background:#fff;border:1px solid #d2dbe6;border-radius:8px;padding:18px;margin:20px 0}details{margin:12px 0}summary{cursor:pointer;font-weight:600}
.scroll{overflow:auto}.answers{display:grid;grid-template-columns:1fr 1fr;gap:16px}.answers>div{min-width:0}select,input,button{padding:8px;margin:4px;font:inherit}
.muted{color:#637084}.wrong{color:#9a2424}.right{color:#21623a}.metrics{display:flex;flex-wrap:wrap;gap:16px}.metrics b{font-size:24px}small{color:#526176}
@media(max-width:850px){.answers{grid-template-columns:1fr}}"""


def esc(value):
    return escape(str(value), quote=True)


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


def page(title, body):
    return (f'<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title>'
            f'<style>{STYLE}</style><body>{body}</body></html>')


def table(headers, rows):
    return ('<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{esc(k)}</th>' for k in headers) +
            '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{esc(v)}</td>' for v in row) +
            '</tr>' for row in rows) + '</tbody></table></div>')


def save_csv(path, rows, fallback_fields):
    fields = list(dict.fromkeys(k for r in rows for k in r)) or fallback_fields
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def render_question(item):
    q = item["question"]
    answers = {r["epoch"]: r for r in item["answers"]}
    title = f'{TASKS[q["task"]]} · {q["sample_id"]}'
    body = f'<p><a href="../report.html">← 退化报告</a> · <a href="{item["file_id"]}.json">完整记录与 token IDs</a></p><h1>{esc(title)}</h1>'
    body += f'<p>评测：{esc(item["name"])} · 系列 {esc(item["series_id"])}</p><p class="note">{esc(NOTE)}</p>'
    body += f'<h2>题目</h2><pre>{esc(q.get("question", ""))}</pre>'
    if q.get("choices"):
        body += '<pre>' + esc('\n'.join(f'{chr(65+i)}. {x}' for i, x in enumerate(q['choices']))) + '</pre>'
    gold = q.get("gold_answer")
    if q["task"] == "livecodebench":
        gold = "以公开和隐藏代码测试判分；此数据没有单一的参考答案字符串或参考解答代码。"
    body += f'<h2>参考答案</h2><pre>{esc(gold)}</pre>'
    if q.get("original"):
        body += f'<details><summary>数据集中保存的原题/参考解答</summary><pre>{esc(json_text(q["original"]))}</pre></details>'
    body += f'<details><summary>实际提示词与题目元数据</summary><pre>{esc(json_text(q))}</pre></details>'
    body += '<h2>已观察到的正确性轨迹</h2>' + table(["Epoch", "结果"],
                 [(r["epoch"], STATES[r["state"]]) for r in item["timeline"]])
    for i, event in enumerate(item["events"], 1):
        before, after = event["before_epoch"], event["after_epoch"]
        a, b = answers[before], answers[after]
        body += f'<article><h2>退化区间 {i}：epoch {before} 答对 → epoch {after} 答错</h2>'
        recovery = event["recovered_at_epoch"]
        body += f'<p>之后再次观测答对：{esc(recovery if recovery is not None else "截至现有结果尚未观察到")}'
        body += f'；本次连续观测答错节点：{esc(event["wrong_observed_epochs_until_recovery"])}。'
        body += f'区间内无有效判分的已列节点：{esc(event["unknown_epochs_between"])}。</p>'
        body += '<p>排查线索（非原因判定）：' + esc('；'.join(CLUES.get(k, k) for k in event["clues"]) or "未检测到预设输出行为变化，请检查完整回答") + '</p>'
        metrics = [("生成 token", "generated_tokens"), ("停止原因", "stop_reason"), ("截断", "length_truncated"),
                   ("正常 EOS", "ended_with_eos"), ("答案可提取", "extractable"), ("判分状态", "grading_status"),
                   ("重复 16-token 窗口比例", "repeated_16gram_fraction"), ("重复长行次数", "repeated_long_line_occurrences"),
                   ("最终答案标记数", "answer_marker_count")]
        body += table(["指标", f"epoch {before}", f"epoch {after}"],
                      [(label, a["behavior"].get(key), b["behavior"].get(key)) for label, key in metrics])
        body += '<div class="answers">'
        for answer in (a, b):
            r = answer["graded_response"]
            body += f'<div><h3>Epoch {r["epoch"]} · {"正确" if r["correct"] else "错误"}</h3><p>提取答案/代码：</p><pre>{esc(r.get("extracted_answer"))}</pre>'
            if r.get("code_execution") is not None:
                body += f'<details open><summary>代码测试结果、实际输出及 stderr</summary><pre>{esc(json_text(r["code_execution"]))}</pre></details>'
            if answer["behavior"]["most_repeated_line_excerpt"]:
                body += '<details><summary>重复长行示例（节选）</summary><pre>' + esc(answer["behavior"]["most_repeated_line_excerpt"]) + '</pre></details>'
            body += f'<details><summary>来源文件与诊断字段</summary><pre>{esc(json_text({"sources": answer["sources"], "behavior": answer["behavior"]}))}</pre></details>'
            # Complete original answers are last, side by side, never shortened or interpreted as HTML.
            body += f'<details open class="answer"><summary>完整回答 · {r.get("generated_tokens")} tokens</summary><pre>{esc(r["raw_response"])}</pre></details></div>'
        body += '</div></article>'
    body += '<h2>本题全部已保存回答</h2>'
    for answer in item["answers"]:
        r = answer["graded_response"]
        body += f'<details class="answer"><summary>Epoch {r["epoch"]} · correct={esc(r["correct"])} · {r.get("generated_tokens")} tokens</summary><pre>{esc(r["raw_response"])}</pre></details>'
    return page(title, body)


def render_index(results, manifest):
    events, coverage, questions = results["events"], results["coverage"], results["questions"]
    body = '<h1>验证集逐题退化报告</h1><p class="note">' + esc(NOTE) + '</p>'
    body += f'<p>生成时间（UTC）：{esc(manifest["created_at"])}。这是已完成判分的只读快照；后续完成的评测需另建报告目录。</p>'
    body += f'<div class="metrics"><span><b>{len(questions)}</b> 个系列内的退化题</span><span><b>{len(events)}</b> 次答对→答错</span><span><b>{sum(e["no_observed_recovery"] for e in events)}</b> 次尚未观测到恢复</span></div>'
    body += '<p><a href="coverage.csv">覆盖情况</a> · <a href="regressions.csv">退化事件 CSV</a> · <a href="paired_changes.csv">所有节点两两配对比较</a> · <a href="manifest.json">配置与来源</a></p>'
    body += '<details open><summary>各评测集实际覆盖情况</summary><p>正确率只针对有效判分；部分完成节点的题目组成可能不同，不能直接比较总体正确率。两两配对 CSV 仅使用双方都有有效判分的同题。</p>'
    body += table(["评测 / 系列", "数据集", "Epoch", "有效 / 总题数", "答对", "答错", "缺失", "判分未知", "有效题正确率", "状态"],
                  [(r["name"] + " / " + r["series_id"][:6], TASKS[r["task"]], r["epoch"],
                    f'{r["valid_grades"]}/{r["expected_questions"]}', r["correct"], r["incorrect"], r["missing"], r["grading_unknown"],
                    f'{r["accuracy_on_valid"]:.1%}' if r["accuracy_on_valid"] is not None else "—", STATUSES[r["status"]]) for r in coverage]) + '</details>'
    counts = Counter(tag for e in events for tag in e["clues"])
    body += '<details><summary>排查线索统计与定义</summary><p>各类可重叠；重复统计是文本特征，不能自动判断推理错误。重复窗口比例 = 1 − 不同的连续 16-token 窗口数 / 总窗口数；增幅至少 0.2 且后值至少 0.5 才标记明显增加。重复长行按空白归一化后至少 40 字符计数。答案标记多次出现也不必然代表循环。</p>'
    body += table(["线索", "退化事件数"], [(CLUES.get(k, k), n) for k, n in counts.most_common()]) + '</details>'
    body += '<h2>找出具体退化题</h2><label>数据集 <select id="task"><option value="">全部</option>' + ''.join(f'<option value="{k}">{v}</option>' for k, v in TASKS.items()) + '</select></label>'
    body += '<label>恢复 <select id="recovery"><option value="">全部</option><option value="no">尚未观测到恢复</option><option value="yes">后来再次答对</option></select></label><input id="search" placeholder="搜索题目、epoch 或线索"><span id="count"></span>'
    body += '<div class="scroll"><table id="events"><thead><tr><th>数据集 / 评测</th><th>区间</th><th>恢复节点</th><th>排查线索</th><th>题目与完整回答</th></tr></thead><tbody>'
    questions_by_path = {f'questions/{q["file_id"]}.html': q for q in questions}
    for e in sorted(events, key=lambda r: (not r["no_observed_recovery"], r["task"], r["after_epoch"], r["sample_id"])):
        q = questions_by_path[e["question_report"]]["question"]
        body += f'<tr data-task="{e["task"]}" data-recovery="{"no" if e["no_observed_recovery"] else "yes"}"><td>{TASKS[e["task"]]}<br>{esc(e["name"])} / {e["series_id"][:6]}</td>'
        body += f'<td>{e["before_epoch"]} → {e["after_epoch"]}</td><td>{esc(e["recovered_at_epoch"] if e["recovered_at_epoch"] is not None else "未观察到")}</td>'
        body += f'<td>{esc("；".join(CLUES.get(k, k) for k in e["clues"]))}</td><td><a href="{e["question_report"]}">{esc(q.get("question", e["sample_id"])[:180])}</a></td></tr>'
    body += '</tbody></table></div>'
    if not events:
        body += '<p>当前可比较的有效观测中未发现答对→答错；这不代表未评测节点没有退化。</p>'
    if manifest["notes"]:
        body += '<details><summary>未纳入的结果与读取说明</summary><pre>' + esc('\n'.join(manifest['notes'])) + '</pre></details>'
    body += """<script>
const task=document.getElementById('task'), recovery=document.getElementById('recovery'), search=document.getElementById('search');
function filter(){let n=0;for(const r of document.querySelectorAll('#events tbody tr')){
 const show=(!task.value||r.dataset.task===task.value)&&(!recovery.value||r.dataset.recovery===recovery.value)&&r.textContent.toLowerCase().includes(search.value.toLowerCase());
 r.hidden=!show;if(show)n++;}document.getElementById('count').textContent='显示 '+n+' 条事件';}
for(const el of [task,recovery,search])el.addEventListener('input',filter);filter();
</script>"""
    return page("验证集退化报告", body)


def build_report(run, out, jobs=(), grades=(), epochs=None):
    run, out = Path(run).resolve(), Path(out).resolve()
    jobs, grades = discover(run, jobs, grades)
    if not jobs and not grades:
        raise ValueError("No recognized grading jobs. Use --job JOB or --grades GRADING_DIRECTORY explicitly.")
    # A separate fresh output cannot overwrite input evidence or a live evaluation.
    if out == run or run.is_relative_to(out) or any(out.is_relative_to(p) or p.is_relative_to(out) for p in jobs + grades):
        raise ValueError("Choose a separate NEW report directory outside evaluation jobs and grades")
    if out.exists():
        raise FileExistsError(f"Report already exists; choose a new --out to retain previous snapshots: {out}")
    evidence = Evidence()
    series = load_sources(run, jobs, grades, evidence)
    observed = {e for s in series for e, _ in s["records"]}
    epochs = sorted(set(epochs) if epochs is not None else observed | {e for s in series for e in s["planned_epochs"]})
    if not epochs or any(type(e) is not int or e < 0 for e in epochs):
        raise ValueError("At least one nonnegative epoch is required")
    results = {k: [] for k in ("coverage", "pairs", "events", "questions")}
    for s in series:
        part = analyze_series(s, epochs)
        for key in results:
            results[key].extend(part[key])
    evidence.verify()
    manifest = {"format": "observed-regressions-v1", "created_at": now(), "run": str(run), "epochs": epochs,
                "interpretation": NOTE, "notes": evidence.notes,
                "series": [{k: s[k] for k in ("series_id", "name", "origin", "protocol", "planned_epochs")} for s in series],
                "source_sha256": evidence.files,
                "analysis_code_sha256": {p: file_hash(ROOT / p) for p in ("analyze_regressions.py", "recipe/regressions.py", "report_examples.py")},
                "regression_events": len(results["events"]), "regressed_questions_per_series": len(results["questions"]),
                "excluded": "Uncommitted transfer epochs, unknown grading results, ungraded generation; no inference or regrading performed"}
    fresh_dir(out)
    (out / "questions").mkdir()
    save_csv(out / "coverage.csv", results["coverage"], ["series_id", "task", "epoch"])
    save_csv(out / "paired_changes.csv", results["pairs"], ["series_id", "task", "before_epoch", "after_epoch", "matched", "lost", "gained", "unknown"])
    save_csv(out / "regressions.csv", results["events"], ["series_id", "task", "sample_id", "before_epoch", "after_epoch", "clues"])
    write_jsonl(out / "regressions.jsonl", results["events"])
    for item in results["questions"]:
        write_json(out / "questions" / (item["file_id"] + ".json"), item)
        (out / "questions" / (item["file_id"] + ".html")).write_text(render_question(item), encoding="utf-8")
    (out / "report.html").write_text(render_index(results, manifest), encoding="utf-8")
    write_json(out / "manifest.json", manifest)
    print(f"Saved {len(results['events'])} observed regressions from {len(results['questions'])} series/questions: {out / 'report.html'}")
    return results, manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="runs/random-lora-s42")
    p.add_argument("--out", required=True, help="NEW report directory; prior snapshots are never overwritten")
    p.add_argument("--job", action="append", default=[], help="Optional grouped/transfer job; repeat for multiple jobs")
    p.add_argument("--grades", action="append", default=[], help="Optional standalone grade.py output; repeat if needed")
    p.add_argument("--epochs", help="Optional display/filter nodes; default: all actual/planned nodes across selected jobs")
    args = p.parse_args()
    epochs = [int(x) for x in args.epochs.split(",")] if args.epochs else None
    build_report(artifact_path(args.run), artifact_path(args.out), [artifact_path(p) for p in args.job],
                 [artifact_path(p) for p in args.grades], epochs)


if __name__ == "__main__":
    main()
