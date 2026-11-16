"""CPU-only: rebuild ranked summaries and a portable HTML report from saved scores."""
from __future__ import annotations

import argparse
import csv
import html
import io
import math
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recipe.common import digest, file_hash, iter_jsonl, now, read_json, write_json, write_jsonl
from recipe.paths import artifact_path
from recipe.pipeline import Pipeline
from dataset.likelihood_utils import DATA_ROWS, DATA_SHA256, iter_rows, load_scores


def write_text(path, text):
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def aggregate(rows):
    n = len(rows)
    if not n:
        raise ValueError("Cannot summarize zero scores")
    largest = max(x["sum_logprob"] for x in rows)
    # The arithmetic mean of raw sequence probabilities, represented in log10
    # space to avoid silently underflowing every long response to zero.
    log_mean_sequence = largest + math.log(math.fsum(math.exp(x["sum_logprob"] - largest) for x in rows)) - math.log(n)
    return {"count": n,
            "mean_normalized_likelihood": math.fsum(x["normalized_likelihood"] for x in rows) / n,
            "mean_logprob": math.fsum(x["mean_logprob"] for x in rows) / n,
            "mean_nll": math.fsum(x["mean_nll"] for x in rows) / n,
            "token_weighted_nll": -math.fsum(x["sum_logprob"] for x in rows) / sum(x["n_tokens"] for x in rows),
            "mean_sequence_likelihood_log10": log_mean_sequence / math.log(10)}


def render_sample(sample):
    esc = lambda value: html.escape(str(value), quote=True)
    return (f'<details id="row-{sample["row_index"]}"><summary>排名 {sample["rank"]} · '
            f'row {sample["row_index"]} · {esc(sample["source"])} · '
            f'似然 {sample["normalized_likelihood"]:.6f} · {sample["n_tokens"]} tokens</summary>'
            f'<p>question_id: <code>{esc(sample["question_id"])}</code><br>'
            f'sample_id: <code>{esc(sample["sample_id"])}</code><br>'
            f'平均 NLL: {sample["mean_nll"]:.6f}；整段 log-likelihood: {sample["sum_logprob"]:.6f}；'
            f'结束符 log-probability: {sample["end_token_logprob"]:.6f}</p>'
            f'<h4>题目</h4><pre>{esc(sample["question"])}</pre>'
            f'<h4>参考答案</h4><pre>{esc(sample["answer"])}</pre>'
            '<h4>实际提示词（不参与评分）</h4>' +
            ''.join(f'<p>{esc(m["role"])}</p><pre>{esc(m["content"])}</pre>' for m in sample["messages"]) +
            f'<h4>数据集原始完整解答</h4><pre>{esc(sample["response"])}</pre></details>')


def build_report(out, allow_partial=False):
    out = Path(out)
    job, prepared = read_json(out / "job.json"), read_json(out / "prepared.json")
    if (job["identity"] != digest(job["spec"]) or prepared["job_identity"] != job["identity"]
            or prepared["index_sha256"] != file_hash(out / "index.jsonl")):
        raise ValueError("Job/prepared manifest mismatch")
    if file_hash(out / "source.parquet") != DATA_SHA256:
        raise ValueError("Frozen source Parquet was modified")
    index = list(iter_jsonl(out / "index.jsonl"))
    if [x["row_index"] for x in index] != list(range(job["spec"]["selected_rows"])):
        raise ValueError("Prepared index is incomplete")
    scores = load_scores(out, job, index)
    if len(scores) != len(index) and not allow_partial:
        raise ValueError(f"Only {len(scores)}/{len(index)} scored; resume scoring or explicitly use --allow-partial")
    if not scores:
        raise ValueError("No committed scores to report")
    ranked = [{"rank": i + 1, **row} for i, row in enumerate(sorted(
        scores.values(), key=lambda x: (-x["mean_logprob"], x["row_index"])))]
    top, bottom = ranked[:100], ranked[-100:]
    summary = {"job_identity": job["identity"], "timestamp": now(), "model": job["spec"]["model"],
               "dataset": job["spec"]["dataset"], "selected_rows": len(index), "scored_rows": len(scores),
               "complete_selected": len(scores) == len(index), "complete_corpus": len(scores) == DATA_ROWS,
               "ranking": job["spec"]["ranking"],
               "metric": "exp(mean assistant-token logprob), including im_end; arithmetic mean across rows",
               "all": aggregate(ranked), "top100": aggregate(top), "bottom100": aggregate(bottom),
               "highest": ranked[0], "lowest": ranked[-1]}
    selected = {x["row_index"]: x for x in top + bottom}
    samples = {}
    for row in iter_rows(out / "source.parquet", job["spec"]["selected_rows"]):
        if row["row_index"] in selected:
            score = selected[row["row_index"]]
            if row["sample_id"] != score["sample_id"]:
                raise ValueError("Score/source sample identity mismatch")
            samples[row["row_index"]] = {**row, **score}
    report = out / "report"
    report.mkdir(exist_ok=True)
    write_jsonl(report / "scores.jsonl", ranked)
    csv_file = io.StringIO(newline="")
    writer = csv.DictWriter(csv_file, fieldnames=list(ranked[0]))
    writer.writeheader()
    writer.writerows(ranked)
    write_text(report / "scores.csv", "\ufeff" + csv_file.getvalue())
    write_jsonl(report / "top100.jsonl", [samples[x["row_index"]] for x in top])
    write_jsonl(report / "bottom100.jsonl", [samples[x["row_index"]] for x in bottom])
    write_json(report / "missing_rows.json", [x["row_index"] for x in index if x["row_index"] not in scores])
    labels = [("全部已完成样本", ranked), (f"最高 {len(top)} 条", top), (f"最低 {len(bottom)} 条", bottom),
              ("最高一条", [ranked[0]]), ("最低一条", [ranked[-1]])]
    table = "".join(f'<tr><td>{label}</td><td>{len(rows)}</td>'
                    f'<td>{aggregate(rows)["mean_normalized_likelihood"]:.8f}</td>'
                    f'<td>{aggregate(rows)["mean_nll"]:.6f}</td></tr>' for label, rows in labels)
    title = "Math-CoT-20k · Base 解答似然"
    status = "全量完成" if summary["complete_corpus"] else "部分数据 / smoke：不能当作全量排名"
    esc = html.escape
    doc = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>body{{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.65 system-ui;color:#182a38;background:#fafbfd}}
table{{border-collapse:collapse;width:100%;background:white}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #ccd5dd}}
details{{margin:14px 0;padding:14px;border:1px solid #ccd5dd;border-radius:8px;background:white}}summary{{cursor:pointer;font-weight:600}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.65 ui-monospace,monospace;background:#f3f6f9;padding:16px}}
code{{overflow-wrap:anywhere}}a{{color:#126086}}</style>
<h1>{title}</h1><p><strong>{status}</strong>：已评分 {len(scores):,} / 本次选中 {len(index):,} / 全集 {DATA_ROWS:,} 条。</p>
<p>模型：{esc(job['spec']['model']['id'])}，revision <code>{esc(job['spec']['model']['revision'])}</code>。BF16 Base，无 LoRA。</p>
<p>固定原解答前文，计算每个 token 的条件概率；范围为完整推理、最终答案及 &lt;|im_end|&gt;。
用户提示词和结束符后的换行不计分。这里衡量对已有解答的拟合，不是自由解题正确率。</p>
<p><b>主指标 = exp(Σ log p(token | 前文) / 解答 token 数)</b>，即 token 条件概率的几何平均；越高越容易模仿。
各组平均数再对样本等权取算术平均。按平均 log-probability 降序排列，同分按原始行号；“前后 100”不是文件原始顺序。
整段解答概率容易数值下溢，scores 文件额外保留 sum_logprob 和 sequence_log10_likelihood。
summary.json 也保留原始整段概率均值的 log10 表示。</p>
<table><tr><th>范围</th><th>数量</th><th>长度归一化似然平均值</th><th>平均 NLL</th></tr>{table}</table>
<p><a href="scores.csv">全部分数 CSV</a> · <a href="scores.jsonl">全部分数 JSONL</a> ·
<a href="summary.json">统计 JSON</a> · <a href="top100.jsonl">最高样本完整文本</a> ·
<a href="bottom100.jsonl">最低样本完整文本</a></p>
<p>发布者没有提供独立题号。row_index 从 0 开始，须与固定数据 revision 一起使用；question_id、sample_id 是本项目生成的哈希标识。
解答来自 Qwen3-32B thinking，题目抽自 OpenR1-Math-220k；
<a href="https://arxiv.org/html/2604.06628v1#A2.SS1">数据构造说明</a>。</p>
<p>以下解答按原文完整保留，展开即可阅读；不运行其中的 HTML，不依赖外部数学渲染服务。
少于 200 条的部分报告中，最高和最低组会有重叠。</p>
<h2>最高 {len(top)} 条</h2>{''.join(render_sample(samples[x['row_index']]) for x in top)}
<h2>最低 {len(bottom)} 条</h2>{''.join(render_sample(samples[x['row_index']]).replace('id="row-', 'id="bottom-row-', 1) for x in bottom)}
</html>'''
    write_text(report / "report.html", doc)
    # Last commit: a report can be regenerated without touching any GPU results.
    write_json(report / "summary.json", summary)
    print(f"Report saved: {report / 'report.html'} ({len(scores)}/{DATA_ROWS}; {status})", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="dataset/math-cot-20k/likelihood/base-v1", help="Existing scoring output directory")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    out = artifact_path(args.out)
    if not (out / "job.json").exists():
        raise FileNotFoundError(f"Not a likelihood job: {out}")
    with Pipeline(out):
        build_report(out, allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()
