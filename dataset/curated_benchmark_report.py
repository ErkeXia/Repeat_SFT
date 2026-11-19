"""Four-task reports from complete epoch results; no GPU or model imports."""
from __future__ import annotations

from collections import Counter
import csv
from html import escape
import io
import json
import os
from pathlib import Path
import uuid

from grade import summarize
from recipe.analysis import correctness_milestones
from recipe.common import file_hash, iter_jsonl, read_json, write_json, write_jsonl
from recipe.transfer_eval import load_panel

TASKS = {"math500": "MATH500", "gpqa": "GPQA", "bbh": "BBH", "livecodebench": "LiveCodeBench"}
COUNTS = {"math500": 128, "gpqa": 64, "bbh": 100, "livecodebench": 100}
PANELS = ("math-gpqa", "bbh-livecodebench")
STYLE = '''<style>body{font:16px/1.6 system-ui;max-width:1300px;margin:24px auto;padding:16px;color:#18202a}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:9px;border:1px solid #dbe0e5;text-align:right}
th{background:#edf2f7}td:first-child,th:first-child{text-align:left}.scroll{overflow-x:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f6f7;padding:14px}
details{border-top:1px solid #ccc;padding:12px 0}a{color:#1659ad}.pending{color:#707781}</style>'''


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def csv_file(path, rows):
    if not rows:
        return
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def collect_epoch(directory, epoch):
    """Require both complete panels, exact IDs/counts and their committed grade hashes."""
    result, sources = [], {}
    for name in PANELS:
        child = directory / name
        complete = read_json(child / "complete.json")
        if complete["epochs"] != [epoch]:
            raise ValueError("Child evaluated a different epoch")
        panel, _ = load_panel(child / "setup/panel")
        expected = {r["sample_id"]: r for r in panel}
        file = child / "grades" / f"epoch-{epoch:03d}" / "graded.jsonl"
        rows = list(iter_jsonl(file))
        if (read_json(file.parent / "complete.json") != {"samples": len(rows), "graded_sha256": file_hash(file)}
                or len(rows) != len(expected) or {r["sample_id"] for r in rows} != set(expected)):
            raise ValueError("Epoch report requires complete, unchanged, nonduplicate grades")
        sources[file.relative_to(directory).as_posix()] = file_hash(file)
        for row in rows:
            question = expected[row["sample_id"]]
            if (row["epoch"] != epoch or row["task"] != question["task"]
                    or row["question_id"] != question["question_id"]):
                raise ValueError("Graded epoch/task/question differs from frozen panel")
            result.append({**row, "question": question["question"], "gold_answer": question["gold_answer"],
                           "messages": question["messages"], "choices": question.get("choices"),
                           "reference_path": f"{name}/setup/panel/{question['reference_file']}"})
    if Counter(r["task"] for r in result) != COUNTS or len({r["sample_id"] for r in result}) != 392:
        raise ValueError("Expected 392 unique questions covering all four tasks")
    return result, sources


def table(rows, epochs, task=None):
    by_key = {(r["epoch"], r["task"]): r for r in rows}
    headers = ["Epoch" if task else "Benchmark", "正确 / 题数", "正确率", "平均 tokens",
               "EOS 结束率", "Chat end 率", "截断率", "提取率", "判分异常"]
    body = '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr></thead><tbody>'
    keys = [(e, task) for e in sorted(epochs)] if task else [(epochs[0], t) for t in TASKS]
    for epoch, name in keys:
        label = str(epoch) if task else TASKS[name]
        row = by_key.get((epoch, name))
        if row is None:
            body += f'<tr class="pending"><td>{label}</td><td colspan="8">未完成，未计入统计</td></tr>'
            continue
        values = [label, f"{row['correct']} / {row['n']}", f"{row['accuracy']:.2%}",
                  f"{row['mean_generated_tokens']:.1f}", f"{row['eos_end_rate']:.2%}",
                  f"{row['normal_chat_end_rate']:.2%}", f"{row['length_truncated_rate']:.2%}",
                  f"{row['extractable_rate']:.2%}", str(row['grading_errors'])]
        body += '<tr>' + ''.join(f'<td>{escape(v)}</td>' for v in values) + '</tr>'
    return body + '</tbody></table></div>'


def header(title, plan):
    settings = plan["generation"]
    return ('<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{escape(title)}</title>{STYLE}</head><body><h1>{escape(title)}</h1>'
            f'<p>固定 392 题：MATH500 128、GPQA 64、BBH 100、LiveCodeBench 100。每题一次采样，'
            f'并发 8，上限 {settings["max_new_tokens"]} 新 token，temperature={settings["temperature"]}，'
            f'top-p={settings["top_p"]}，top-k={settings["top_k"]}。</p>'
            '<p>EOS 结束包括 chat_end/base_eos；Chat end 只统计聊天结束符。截断指因长度限制停止，提取成功不等于正确。'
            '判分异常保留并计入总题数。LiveCodeBench 使用本项目的 Python 测试执行协议。</p>')


def epoch_report(job, epoch, plan):
    directory = job / "epochs" / f"epoch-{epoch:03d}"
    rows, sources = collect_epoch(directory, epoch)
    out = directory / "report"
    out.mkdir(exist_ok=True)
    summaries = [{"epoch": epoch, "task": task, **summarize([r for r in rows if r["task"] == task])} for task in TASKS]
    write_json(out / "summary.json", {"epoch": epoch, "complete": True, "samples": len(rows), "tasks": summaries})
    csv_file(out / "curves.csv", summaries)
    write_jsonl(out / "samples.jsonl", rows)
    body = header(f"Epoch {epoch} · 四个 benchmark", plan) + table(summaries, [epoch])
    body += '<p><a href="curves.csv">指标 CSV</a> · <a href="samples.jsonl">逐题原文、token、提取和判分 JSONL</a></p><h2>逐题详情</h2>'
    for task in TASKS:
        body += f'<h3>{TASKS[task]}</h3>'
        for row in rows:
            if row["task"] != task:
                continue
            esc = lambda v: escape(str(v))
            body += (f'<details id="{esc(row["sample_id"])}"><summary>{esc(row["sample_id"])} · '
                     f'correct={esc(row["correct"])} · {row["generated_tokens"]} tokens · {esc(row["stop_reason"])}</summary>'
                     f'<p>来源：{esc(row["source"])}；判分状态：{esc(row["grading_status"])}；'
                     f'抽取：{esc(row["extraction_method"])}</p><h4>题目</h4><pre>{esc(row["question"])}</pre>')
            if row.get("choices"):
                body += '<pre>' + esc('\n'.join(f'{chr(65+i)}. {v}' for i, v in enumerate(row['choices']))) + '</pre>'
            gold = '公开及隐藏测试，无参考解答代码' if task == 'livecodebench' else row['gold_answer']
            body += f'<p>参考答案：{esc(gold)}</p><h4>提取结果</h4><pre>{esc(row.get("extracted_answer"))}</pre>'
            if row.get('code_execution') or row.get('grading_error'):
                body += '<h4>判分详情</h4><pre>' + esc(json.dumps(row.get('code_execution') or row['grading_error'], ensure_ascii=False)) + '</pre>'
            body += '<h4>完整回答</h4><pre>' + esc(row['raw_response']) + '</pre></details>'
    atomic_text(out / "report.html", body + '</body></html>\n')
    write_json(out / "manifest.json", {"epoch": epoch, "samples": len(rows), "graded_sources": sources,
               "files": {name: file_hash(out / name) for name in ('report.html', 'summary.json', 'curves.csv', 'samples.jsonl')}})


def overview(job, plan, completed):
    """Rebuild derived overview after each complete epoch. Never count partial epochs."""
    out = job / "report"
    out.mkdir(exist_ok=True)
    curves, observations = [], {}
    for epoch in sorted(completed):
        directory = job / 'epochs' / f'epoch-{epoch:03d}' / 'report'
        curves.extend(read_json(directory / 'summary.json')['tasks'])
        observations[epoch] = {r['sample_id']: r for r in iter_jsonl(directory / 'samples.jsonl')}
    summary = {"requested_epochs": plan['epochs'], "completed_epochs": sorted(completed),
               "complete": set(completed) == set(plan['epochs']), "questions_per_epoch": 392,
               "responses_completed": 392 * len(completed), "curves": curves}
    write_json(out / 'summary.json', summary)
    csv_file(out / 'curves.csv', curves)
    body = header('四个 benchmark · 各 epoch 总报告', plan)
    body += f'<p>完整节点 {len(completed)}/{len(plan["epochs"])}；执行顺序：{escape(str(plan["epochs"]))}。表格按训练 epoch 升序展示。</p>'
    body += '<p>' + ('<a href="curves.csv">指标 CSV</a> · ' if curves else '尚无完整节点的指标 · ')
    body += '<a href="summary.json">汇总 JSON</a></p>'
    for task, label in TASKS.items():
        body += f'<h2>{label}</h2>' + table(curves, plan['epochs'], task)
    body += '<h2>各节点完整报告</h2><ul>'
    for epoch in sorted(completed):
        body += f'<li><a href="../epochs/epoch-{epoch:03d}/report/report.html">Epoch {epoch}：四任务指标及全部题目、回答</a></li>'
    body += '</ul>'
    if observations:
        epochs = sorted(observations)
        ids = set(observations[epochs[0]])
        if any(set(observations[e]) != ids for e in epochs):
            raise ValueError('Epoch reports contain different evaluation questions')
        milestones = []
        for sid in sorted(ids):
            answers = [observations[e][sid] for e in epochs]
            milestones.append({'sample_id': sid, 'task': answers[0]['task'],
                               **correctness_milestones(epochs, answers, 2)})
        csv_file(out / 'question_milestones.csv', milestones)
        body += '<p><a href="question_milestones.csv">逐题答对/退化节点 CSV</a>：按训练轮次排序，仅基于已完成节点；不是 GPU 执行时的先后顺序。</p>'
    atomic_text(out / 'report.html', body + '</body></html>\n')
    return summary
