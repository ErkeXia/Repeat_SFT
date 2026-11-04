"""Portable per-question transfer curves and full answers from saved grades."""
import argparse
from collections import defaultdict
import csv
import html
import os
from pathlib import Path

from grade import summarize
from recipe.analysis import correctness_milestones
from recipe.common import file_hash, iter_jsonl, read_json, write_json, write_jsonl
from recipe.paths import artifact_path
from recipe.transfer_eval import load_panel


def csv_write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def report(panel_dir, grades, out, selected_ids=None):
    panel_dir, grades, out = Path(panel_dir), Path(grades), Path(out)
    if any(out.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(out.resolve()) for p in (panel_dir, grades)):
        raise ValueError("Report output must be separate from source grades and panel")
    panel, _ = load_panel(panel_dir)
    if selected_ids is not None:
        panel = [p for p in panel if p["sample_id"] in selected_ids]
        if len(panel) != len(selected_ids):
            raise ValueError("Report selection differs from the frozen panel")
    wanted = {p["sample_id"] for p in panel}
    observations, sources = {}, {}
    for file in sorted(grades.glob("epoch-*/graded.jsonl")):
        rows = list(iter_jsonl(file))
        if (read_json(file.parent / "complete.json") != {"samples": len(rows), "graded_sha256": file_hash(file)}
                or len(rows) != len(wanted) or {r["sample_id"] for r in rows} != wanted):
            raise ValueError("Report requires a complete, unchanged graded panel at every requested epoch")
        epoch = int(file.parent.name.split("-")[1])
        if any(r["epoch"] != epoch for r in rows):
            raise ValueError("Graded epoch mismatch")
        observations[epoch] = {r["sample_id"]: r for r in rows}
        sources[str(epoch)] = file_hash(file)
    if not observations:
        raise ValueError("No grades available")
    epochs = sorted(observations)
    out.mkdir(parents=True, exist_ok=True)
    (out / "questions").mkdir(exist_ok=True)
    summaries = []
    for epoch, by_id in sorted(observations.items()):
        groups = defaultdict(list)
        for row in by_id.values():
            groups[(row["task"], "all")].append(row)
            if row["task"] == "bbh":
                groups[(row["task"], row["subtask"])].append(row)
        for (task, subset), rows in groups.items():
            summaries.append({"epoch": epoch, "task": task, "subset": subset, **summarize(rows)})
    csv_write(out / "curves.csv", summaries)
    write_jsonl(out / "summary.jsonl", summaries)
    milestones, links = [], []
    style = '<style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f6f7;padding:15px}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}details{margin:16px 0}</style>'
    esc = lambda x: html.escape(str(x))
    for question in panel:
        sid = question["sample_id"]
        rows = [observations[e][sid] for e in epochs]
        event = {"sample_id": sid, "task": question["task"], **correctness_milestones(epochs, rows, 2)}
        milestones.append(event)
        write_json(out / "questions" / f"{sid}.json", {"question": question, "milestones": event, "answers": rows})
        reference = os.path.relpath(panel_dir / question["reference_file"], out / "questions").replace(os.sep, "/")
        body = f'<h1>{esc(question["task"])} · {esc(sid)}</h1><pre>{esc(question["question"])}</pre>'
        body += f'<p>参考答案：{esc(question["gold_answer"] if question["task"] == "bbh" else "公开和隐藏代码测试；无参考解答代码")}</p>'
        body += f'<p><a href="{esc(reference)}">完整原题与测试数据</a> · <a href="{esc(sid)}.json">回答和全部 token / 判分 JSON</a></p>'
        for row in rows:
            body += (f'<details><summary>epoch {row["epoch"]} · correct={row["correct"]} · '
                     f'{row["generated_tokens"]} tokens · {esc(row["stop_reason"])}</summary>'
                     f'<p>判分：{esc(row["grading_status"])}；提取：{esc(row["extraction_method"])}</p>'
                     f'<pre>{esc(row["raw_response"])}</pre></details>')
        (out / "questions" / f"{sid}.html").write_text('<!doctype html><meta charset="utf-8">' + style + body, encoding="utf-8")
        links.append(f'<li><a href="questions/{esc(sid)}.html">{esc(question["task"])} · {esc(question["question"][:100])}</a></li>')
    csv_write(out / "question_milestones.csv", milestones)
    body = '<h1>BBH / LiveCodeBench 迁移评测</h1><p>同一固定面板、每题一次采样。LiveCodeBench 为项目 Python 执行协议，不能当作官方完整排行榜分数。首次答对仅指这些观测节点。</p>'
    body += '<p><a href="curves.csv">正确率和输出行为 CSV</a> · <a href="question_milestones.csv">逐题首次答对节点</a></p>'
    body += '<table><tr><th>Epoch</th><th>任务</th><th>题数</th><th>正确率</th><th>平均长度</th><th>正常 EOS</th><th>截断率</th><th>答案/代码提取率</th></tr>'
    for row in summaries:
        if row["subset"] == "all":
            body += '<tr>' + ''.join(f'<td>{esc(v)}</td>' for v in (row["epoch"], row["task"], row["n"], f'{row["accuracy"]:.1%}',
                  round(row["mean_generated_tokens"]), f'{row["eos_end_rate"]:.1%}', f'{row["length_truncated_rate"]:.1%}', f'{row["extractable_rate"]:.1%}')) + '</tr>'
    body += '</table><h2>逐题全文</h2><ul>' + ''.join(links) + '</ul>'
    (out / "report.html").write_text('<!doctype html><meta charset="utf-8">' + style + body, encoding="utf-8")
    write_json(out / "manifest.json", {"graded_sources": sources, "panel_sha256": file_hash(panel_dir / "panel.jsonl"),
               "epochs": epochs, "samples": len(panel), "full_reference_location": str(panel_dir.resolve())})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel-dir", required=True)
    p.add_argument("--grades", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    report(artifact_path(args.panel_dir), artifact_path(args.grades), artifact_path(args.out))


if __name__ == "__main__":
    main()
