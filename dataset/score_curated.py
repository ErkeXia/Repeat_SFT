"""SERVER worker: unadapted BF16 Base likelihood of the ten frozen edited answers."""
from __future__ import annotations

import argparse
import asyncio
import csv
from html import escape
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recipe.common import (append_jsonl, digest, environment, file_hash, iter_jsonl,
                           now, read_json, write_json, write_jsonl)
from recipe.encoding import encode_example
from recipe.paths import configure_runtime
from recipe.pipeline import Pipeline
from dataset.curated_training import verify_bundle
from dataset.likelihood_utils import describe, extract_logprobs, load_scores, save_score


def initialize(out, plan):
    versions = {name: importlib.metadata.version(name)
                for name in ("vllm", "torch", "transformers", "numpy")}
    if versions["vllm"] != "0.11.0":
        raise ValueError("Use the existing separate vllm==0.11.0 environment")
    spec = {"protocol": "curated-ten-base-likelihood-v1", "bundle_identity": plan["identity"],
            "model": plan["model"], "scoring": plan["scoring"], "versions": versions,
            "semantics": "teacher forcing; assistant reasoning + final answer + im_end; no adapter",
            "ranking": "mean_logprob descending; normalized likelihood = exp(mean_logprob)"}
    job = {"identity": digest(spec), "spec": spec}
    if (out / "job.json").exists():
        if read_json(out / "job.json") != job:
            raise ValueError("Scoring environment or inputs changed; restore them or use a new experiment")
    else:
        if any((out / "records").glob("*.json")):
            raise ValueError("Committed scores found without job.json; restore the manifest")
        write_json(out / "job.json", job)
        write_json(out / "environment.json", environment())
    return job


async def score_pending(engine, sampling, out, job, rows, tokenizer, index, done, concurrency=8,
                        *, lora_request=None, label="BASE SCORE", request_prefix="curated"):
    expected = {r["row_index"]: r for r in index}
    pending = iter(r for r in rows if r["row_index"] not in done)
    finished = len(done)

    async def worker():
        nonlocal finished
        for row in pending:
            encoded = encode_example(tokenizer, row, 10240)
            if encoded is None or describe(row, encoded) != expected[row["row_index"]]:
                raise ValueError("Scoring and training tokenization differ from the frozen index")
            request_id = f"{request_prefix}-{row['row_index']}"
            final = None
            try:
                options = {"lora_request": lora_request} if lora_request is not None else {}
                async for result in engine.generate({"prompt_token_ids": encoded["input_ids"]}, sampling, request_id, **options):
                    if result.finished:
                        final = result
                if final is None:
                    raise ValueError("Missing completed vLLM prompt logprobs")
                score = save_score(out, job["identity"], expected[row["row_index"]], encoded,
                                   extract_logprobs(final, encoded))
            except BaseException:
                try:
                    await asyncio.wait_for(engine.abort(request_id), timeout=5)
                except Exception:
                    pass
                raise
            finished += 1
            append_jsonl(out / "progress.jsonl", {"timestamp": now(), "completed": finished,
                         "row_index": row["row_index"], "mean_nll": score["mean_nll"]})
            print(f"{label} {finished}/{len(rows)} row={row['row_index']} "
                  f"tokens={score['n_tokens']} nll={score['mean_nll']:.6f} "
                  f"likelihood={score['normalized_likelihood']:.6f}", flush=True)

    tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def execute(out, job, plan, rows, tokenizer, index, done):
    import torch
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 support required")
    task, loop = asyncio.current_task(), asyncio.get_running_loop()
    stopped, engine = [], None

    def stop(signum):
        if not stopped:
            stopped.append(signum)
            print("Stopping Base scoring; committed samples retained. Repeat the launcher to resume.", flush=True)
            task.cancel()

    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop, signum)
    try:
        settings = plan["scoring"]
        options = {"model": plan["model"]["id"], "revision": plan["model"]["revision"],
                   "tokenizer": str(out.parent / "bundle/prepared/train/tokenizer"),
                   "trust_remote_code": False, "dtype": "bfloat16", "tensor_parallel_size": 1,
                   "enable_lora": False, "enable_prefix_caching": False,
                   "enable_chunked_prefill": True, "enforce_eager": True,
                   "max_model_len": settings["max_model_len"], "max_num_seqs": settings["concurrency"],
                   "max_num_batched_tokens": settings["max_num_batched_tokens"],
                   "gpu_memory_utilization": settings["gpu_memory_utilization"],
                   "generation_config": "vllm", "seed": plan["config"]["seed"]}
        append_jsonl(out / "engine_startups.jsonl", {"timestamp": now(), "options": options,
                     "gpu": torch.cuda.get_device_name(0), "already_completed": len(done)})
        engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**options))
        # Exactly the same scoring protocol as the previous full-corpus vLLM job.
        # One discarded output token completes the request; only prompt logprobs are used.
        sampling = SamplingParams(prompt_logprobs=0, max_tokens=1, temperature=1.0, top_p=1.0,
                                  top_k=-1, min_p=0.0, repetition_penalty=1.0, ignore_eos=True,
                                  detokenize=False, output_kind=RequestOutputKind.FINAL_ONLY)
        await score_pending(engine, sampling, out, job, rows, tokenizer, index, done, settings["concurrency"])
    except asyncio.CancelledError:
        if stopped:
            return 128 + stopped[0]
        raise
    finally:
        if engine is not None:
            engine.shutdown()
        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(signum)
    return 0


def build_report(out, job, rows, index, *, model_label="Base"):
    scores = load_scores(out, job, index)
    if len(scores) != len(rows):
        raise ValueError("Scoring must finish all ten samples before publishing the model report")
    combined = [{**row, **scores[row["row_index"]]} for row in rows]
    ranked = sorted(combined, key=lambda r: (-r["mean_logprob"], r["row_index"]))
    write_jsonl(out / "samples.jsonl", ranked)  # Full questions, prompts and edited answers.

    def aggregate(items):
        total_tokens = sum(r["n_tokens"] for r in items)
        mean_logprob = math.fsum(r["sum_logprob"] for r in items) / total_tokens
        return {"samples": len(items), "supervised_tokens": total_tokens,
                "mean_sample_nll": math.fsum(r["mean_nll"] for r in items) / len(items),
                "mean_normalized_likelihood": math.fsum(r["normalized_likelihood"] for r in items) / len(items),
                "token_weighted_nll": -mean_logprob, "token_weighted_normalized_likelihood": math.exp(mean_logprob),
                "mean_supervised_tokens": total_tokens / len(items)}

    summary = {"job_identity": job["identity"], "all": aggregate(combined),
               "historical_groups": {g: aggregate([r for r in combined if r["curation"]["historical_group"] == g])
                                     for g in ("high", "low")},
               "highest_likelihood_row": ranked[0]["row_index"], "lowest_likelihood_row": ranked[-1]["row_index"],
               "note": "High/low describes the ORIGINAL answer's group, not the edited answer's new rank."}
    write_json(out / "summary.json", summary)
    fields = ["row_index", "historical_group", "sample_id", "n_tokens", "mean_nll", "mean_logprob",
              "normalized_likelihood", "sum_logprob", "sequence_log10_likelihood", "end_token_logprob"]
    with (out / "scores.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in ranked:
            writer.writerow({key: row["curation"][key] if key == "historical_group" else row[key] for key in fields})
    body = [f'<!doctype html><html lang="zh"><meta charset="utf-8"><title>Curated {escape(model_label)} likelihood</title>',
            '<style>body{max-width:1050px;margin:32px auto;padding:16px;font:16px/1.6 sans-serif}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere}details{border-top:1px solid #ccc;padding:16px 0}</style>',
            f'<h1>10 条改写回答的 {escape(model_label)} 似然度</h1>',
            '<p>仅统计助手推理、最终答案及结束符；归一化似然 = exp(平均 token logprob)。'
            'High/Low 表示改写前的历史分组。本报告按改写后的平均似然排序。</p>',
            '<pre>' + escape(json.dumps(summary, ensure_ascii=False, indent=2)) + '</pre>']
    for row in ranked:
        body += [f'<details><summary>row {row["row_index"]} · {escape(row["curation"]["historical_group"])} · '
                 f'{row["n_tokens"]} tokens · NLL {row["mean_nll"]:.6f} · '
                 f'likelihood {row["normalized_likelihood"]:.6f}</summary>',
                 '<h2>题目</h2><pre>' + escape(row["question"]) + '</pre>',
                 '<h2>数据集短答案</h2><pre>' + escape(str(row["answer"])) + '</pre>',
                 '<h2>完整改写回答</h2><pre>' + escape(row["response"]) + '</pre></details>']
    (out / "report.html").write_text('\n'.join(body) + '</html>\n', encoding="utf-8")
    write_json(out / "complete.json", {"job_identity": job["identity"], "samples": len(rows),
               "summary_sha256": file_hash(out / "summary.json"),
               "records": {p.name: file_hash(p) for p in sorted((out / "records").glob("*.json"))}})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--inherited-lock", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if os.name != "posix":
        raise RuntimeError("Run GPU scoring on the Linux VM")
    configure_runtime()
    experiment = Path(args.experiment).resolve()
    with Pipeline(experiment, inherited=args.inherited_lock) as pipeline:
        plan = verify_bundle(experiment)
        out = experiment / "base_likelihood"
        rows = list(iter_jsonl(experiment / "bundle/prepared/train/train.jsonl"))
        index = list(iter_jsonl(experiment / "bundle/index.jsonl"))
        job = initialize(out, plan)
        done = load_scores(out, job, index)
        print(f"Retained Base scores: {len(done)}/{len(rows)}", flush=True)
        pipeline.check_stop()
        if len(done) < len(rows):
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(experiment / "bundle/prepared/train/tokenizer", local_files_only=True)
            code = asyncio.run(execute(out, job, plan, rows, tokenizer, index, done))
            if code:
                raise SystemExit(code)
        pipeline.check_stop()
        build_report(out, job, rows, index)
        print(f"Base likelihood complete: {out / 'report.html'}", flush=True)


if __name__ == "__main__":
    main()
