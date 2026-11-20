"""Compare committed Base/checkpoint scores of the same frozen manual answers."""
import csv
from html import escape
from pathlib import Path

from recipe.common import file_hash, iter_jsonl, read_json, write_json, write_jsonl
from dataset.likelihood_utils import load_scores


def build_comparison(out, plan):
    rows = list(iter_jsonl(out / "setup/samples.jsonl"))
    index = list(iter_jsonl(out / "setup/index.jsonl"))
    report = out / "report"
    report.mkdir(exist_ok=True)
    summaries, samples, by_epoch = [], [], {}
    for epoch in [0, *plan["epochs"]]:
        directory = out / "setup/base" if epoch == 0 else out / "epochs" / f"epoch-{epoch:03d}"
        label = "Base" if epoch == 0 else f"Epoch {epoch}"
        if not (directory / "complete.json").exists():
            summaries.append({"epoch": epoch, "model": label, "status": "pending"})
            continue
        job = read_json(directory / "job.json")
        scores = load_scores(directory, job, index)
        marker = read_json(directory / "complete.json")
        if (len(scores) != len(rows) or marker["samples"] != len(rows) or marker["job_identity"] != job["identity"]
                or marker["summary_sha256"] != file_hash(directory / "summary.json")
                or marker["records"] != {p.name: file_hash(p) for p in sorted((directory / "records").glob("*.json"))}):
            raise ValueError(f"Incomplete/changed committed likelihood results: {directory}")
        by_epoch[epoch] = scores
        summary = read_json(directory / "summary.json")
        summaries.append({"epoch": epoch, "model": label, "status": "complete", **summary["all"],
                          "historical_groups": summary["historical_groups"]})
        for row in rows:
            score = scores[row["row_index"]]
            base = by_epoch[0][row["row_index"]]
            samples.append({**row, **score, "epoch": epoch, "model": label,
                            "delta_nll_vs_base": score["mean_nll"] - base["mean_nll"],
                            "likelihood_ratio_vs_base": score["normalized_likelihood"] / base["normalized_likelihood"]
                            if base["normalized_likelihood"] else None,
                            "token_file": (directory / score["token_file"]).relative_to(out).as_posix()})
    write_json(report / "summary.json", {"models": summaries, "epochs": [0, *plan["epochs"]],
               "note": "Teacher-forced likelihood of fixed training answers; not free-generation accuracy. "
                       "Base scores reused; vLLM adapters evaluated in BF16. High/low are historical labels."})
    write_jsonl(report / "samples.jsonl", samples)
    fields = ["epoch", "model", "status", "samples", "mean_sample_nll", "mean_normalized_likelihood",
              "token_weighted_nll", "token_weighted_normalized_likelihood", "mean_supervised_tokens"]
    with (report / "curves.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    fields = ["epoch", "row_index", "sample_id", "n_tokens", "mean_nll", "mean_logprob",
              "normalized_likelihood", "delta_nll_vs_base", "likelihood_ratio_vs_base", "end_token_logprob", "token_file"]
    with (report / "samples.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(samples)

    body = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>Manual answer likelihood comparison</title>',
            '<style>body{max-width:1150px;margin:32px auto;padding:16px;font:16px/1.6 sans-serif}'
            'table{border-collapse:collapse;width:100%;margin:16px 0}th,td{border:1px solid #ccc;padding:7px;text-align:right}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere}details{border-top:1px solid #ccc;padding:12px 0}</style>',
            '<h1>10 条人工回答：Base 与训练检查点的似然度</h1>',
            '<p>给定固定回答前文，统计助手推理、最终答案和结束符的概率。NLL 越低、归一化似然越高，表示越容易拟合这条回答；'
            '这不是自由解题正确率。归一化似然 = exp(平均 token logprob)，不是整条回答概率。Base 复用原结果；'
            '检查点使用 BF16 LoRA。High/Low 是修改前的历史分组。</p>',
            '<h2>整体对比</h2><table><tr><th>模型</th><th>状态</th><th>逐样本平均 NLL</th>'
            '<th>逐样本平均似然</th><th>Token 加权 NLL</th><th>Token 加权似然</th></tr>']
    for s in summaries:
        values = ''.join(f'<td>{s[k]:.6f}</td>' for k in ('mean_sample_nll', 'mean_normalized_likelihood',
                        'token_weighted_nll', 'token_weighted_normalized_likelihood')) if s['status'] == 'complete' else '<td colspan="4">尚未完成</td>'
        body.append(f'<tr><td>{escape(s["model"])}</td><td>{escape(s["status"])}</td>{values}</tr>')
    body.append('</table><h2>逐题对比</h2>')
    for row in rows:
        body += [f'<details><summary>row {row["row_index"]} · {escape(row["curation"]["historical_group"])} · '
                 f'{by_epoch[0][row["row_index"]]["n_tokens"]} supervised tokens</summary>',
                 '<table><tr><th>模型</th><th>NLL</th><th>归一化似然</th><th>NLL 相对 Base 的变化</th><th>结束符 logprob</th></tr>']
        for epoch in [0, *plan["epochs"]]:
            score = by_epoch.get(epoch, {}).get(row['row_index'])
            label = 'Base' if epoch == 0 else f'Epoch {epoch}'
            values = (f'<td>{score["mean_nll"]:.6f}</td><td>{score["normalized_likelihood"]:.6f}</td>'
                      f'<td>{score["mean_nll"] - by_epoch[0][row["row_index"]]["mean_nll"]:+.6f}</td>'
                      f'<td>{score["end_token_logprob"]:.6f}</td>') if score else '<td colspan="4">尚未完成</td>'
            body.append(f'<tr><td>{label}</td>{values}</tr>')
        body += ['</table><h3>题目</h3><pre>' + escape(row['question']) + '</pre>',
                 '<h3>数据集短答案</h3><pre>' + escape(str(row['answer'])) + '</pre>',
                 '<h3>完整人工回答（所有模型均对这份文本评分）</h3><pre>' + escape(row['response']) + '</pre></details>']
    (report / "report.html").write_text('\n'.join(body) + '</html>\n', encoding='utf-8')
    return len(by_epoch) == 1 + len(plan["epochs"])
