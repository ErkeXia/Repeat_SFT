"""SERVER ONLY: score EVERY Math-CoT-20k solution under the unadapted BF16 Base."""
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import os
from pathlib import Path
import re
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recipe.common import (append_jsonl, digest, environment, file_hash, iter_jsonl,
                           now, read_json, write_json, write_jsonl)
from recipe.encoding import inspect_example
from recipe.paths import artifact_path, configure_runtime
from recipe.pipeline import Pipeline
from dataset.likelihood_utils import (DATASET, DATA_FILE, DATA_REVISION, DATA_ROWS, DATA_SHA256,
                                      PROTOCOL, copy_atomic, describe, encode_full, extract_logprobs,
                                      iter_rows, load_scores, save_score)

CODE_FILES = ["dataset/__init__.py", "dataset/score_likelihood.py", "dataset/likelihood_utils.py",
              "recipe/common.py", "recipe/data.py", "recipe/encoding.py", "recipe/paths.py",
              "recipe/pipeline.py"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="dataset/math-cot-20k/likelihood/base-v1")
    p.add_argument("--input", default="dataset/math-cot-20k/raw/Math-CoT-20k.parquet",
                   help="Relative artifact paths resolve under /data/repeat_sft")
    p.add_argument("--download-data", action="store_true", help="Download the pinned Parquet if missing, on VM only")
    p.add_argument("--base-run", help="Use the run's pinned Base revision and frozen tokenizer; NO adapter loaded")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    p.add_argument("--revision", default="main", help="Resolved to a commit once; --base-run takes precedence")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-num-batched-tokens", type=int, default=1024)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--limit", type=int, help="Smoke only: first N original rows in a SEPARATE output directory")
    p.add_argument("--prepare-only", action="store_true", help="Validate/tokenize all selected rows; no model weights loaded")
    args = p.parse_args()
    if (args.concurrency < 1 or args.max_num_batched_tokens < args.concurrency
            or not 0 < args.gpu_memory_utilization < 1
            or args.limit is not None and not 1 <= args.limit <= DATA_ROWS):
        p.error("Invalid concurrency, token budget, memory fraction or limit")
    return args


def file_map(directory):
    return {p.relative_to(directory).as_posix(): file_hash(p)
            for p in sorted(Path(directory).rglob("*")) if p.is_file()}


def initialize(args, out):
    """A committed manifest fixes the data, Base, tokenizer, software and scoring code."""
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoConfig, AutoTokenizer
    import pyarrow.parquet as pq

    request = {"input": str(artifact_path(args.input)),
               "base_run": str(artifact_path(args.base_run)) if args.base_run else None,
               "model": args.model, "revision": args.revision, "limit": args.limit,
               "concurrency": args.concurrency, "max_num_batched_tokens": args.max_num_batched_tokens,
               "gpu_memory_utilization": args.gpu_memory_utilization}
    versions = {name: importlib.metadata.version(name)
                for name in ("vllm", "torch", "transformers", "numpy", "pyarrow")}
    if versions["vllm"] != "0.11.0":
        raise ValueError("Use the separate vllm==0.11.0 environment; see dataset/LIKELIHOOD.md")
    code = {name: file_hash(ROOT / name) for name in CODE_FILES}
    manifest = out / "job.json"
    if manifest.exists():
        job = read_json(manifest)
        if job["identity"] != digest(job["spec"]):
            raise ValueError("Job manifest was modified")
        spec = job["spec"]
        if (spec["request"] != request or spec["versions"] != versions or spec["code"] != code):
            raise ValueError("Resume arguments/software/scoring code differ; use the original command/environment/code, "
                             "or a new --out. Existing scores were retained.")
        if file_map(out / "tokenizer") != spec["tokenizer_files"]:
            raise ValueError("Frozen tokenizer changed")
        if file_hash(out / "source.parquet") != DATA_SHA256:
            raise ValueError("Frozen dataset changed")
        if file_hash(out / "base_config.json") != spec["base_config_sha256"]:
            raise ValueError("Frozen model configuration changed")
        return job, AutoTokenizer.from_pretrained(out / "tokenizer", local_files_only=True)

    if any((out / "records").glob("*.json")):
        raise ValueError("Committed scores exist without job.json; restore the manifest")
    source = artifact_path(args.input)
    if not source.exists():
        if not args.download_data:
            raise FileNotFoundError(f"{source}: upload the local Parquet, or use --download-data on the VM")
        cached = hf_hub_download(DATASET, DATA_FILE, repo_type="dataset", revision=DATA_REVISION)
        copy_atomic(cached, source)
    if file_hash(source) != DATA_SHA256 or pq.ParquetFile(source).metadata.num_rows != DATA_ROWS:
        raise ValueError("Expected the pinned, unmodified 20,480-row Math-CoT-20k Parquet")
    copy_atomic(source, out / "source.parquet")
    if args.base_run:
        run = artifact_path(args.base_run)
        metadata = read_json(run / "run.json")
        model = metadata["model"]
        actual = file_map(run / "tokenizer")
        if actual != metadata["tokenizer_files"]:
            raise ValueError("Training run tokenizer hashes do not match run.json")
        tokenizer = AutoTokenizer.from_pretrained(run / "tokenizer", local_files_only=True)
        base_run_hash = file_hash(run / "run.json")
    else:
        model = {"id": args.model, "revision": HfApi().model_info(args.model, revision=args.revision).sha}
        tokenizer = AutoTokenizer.from_pretrained(model["id"], revision=model["revision"], trust_remote_code=False)
        base_run_hash = None
    if (model["id"] != "Qwen/Qwen3-4B-Base"
            or not re.fullmatch(r"[0-9a-f]{40}", model.get("revision", ""))):
        raise ValueError("This experiment requires pinned Qwen/Qwen3-4B-Base, without a LoRA adapter")
    config = AutoConfig.from_pretrained(model["id"], revision=model["revision"], trust_remote_code=False)
    if config.model_type != "qwen3":
        raise ValueError("Expected dense Qwen3")
    tokenizer.save_pretrained(out / "tokenizer")
    write_json(out / "base_config.json", config.to_dict())
    for name in CODE_FILES + ["dataset/report_likelihood.py", "dataset/LIKELIHOOD.md"]:
        copy_atomic(ROOT / name, out / "code" / name)
    write_json(out / "environment.json", environment())
    spec = {"protocol": PROTOCOL, "request": request, "versions": versions, "code": code,
            "dataset": {"id": DATASET, "revision": DATA_REVISION, "sha256": DATA_SHA256, "rows": DATA_ROWS},
            "model": model, "base_run_sha256": base_run_hash, "adapter": None,
            "dtype": "bfloat16", "context_length": int(config.max_position_embeddings),
            "selected_rows": args.limit or DATA_ROWS, "tokenizer_files": file_map(out / "tokenizer"),
            "base_config_sha256": file_hash(out / "base_config.json"),
            "semantics": "teacher forcing; assistant reasoning + final answer + im_end; prompt/trailing newline excluded",
            "ranking": "mean_logprob descending (equivalent to exp(mean_logprob)); row_index ascending for ties"}
    job = {"identity": digest(spec), "created": now(), "spec": spec}
    write_json(manifest, job)
    return job, tokenizer


def prepare_index(out, job, tokenizer, check_stop=lambda: None):
    marker = out / "prepared.json"
    if marker.exists():
        saved = read_json(marker)
        if saved["job_identity"] != job["identity"] or saved["index_sha256"] != file_hash(out / "index.jsonl"):
            raise ValueError("Prepared index belongs to another job or changed")
        index = list(iter_jsonl(out / "index.jsonl"))
    else:
        index, inspections = [], []
        for row in iter_rows(out / "source.parquet", job["spec"]["selected_rows"]):
            check_stop()
            encoded = encode_full(tokenizer, row, job["spec"]["context_length"])
            index.append(describe(row, encoded))
            if len(inspections) < 3:
                inspections.append(inspect_example(tokenizer, row, encoded))
            if len(index) % 100 == 0:
                print(f"PREPARE {len(index)}/{job['spec']['selected_rows']}", flush=True)
        write_jsonl(out / "index.jsonl", index)
        write_jsonl(out / "inspection.jsonl", inspections)
        write_json(marker, {"job_identity": job["identity"], "index_sha256": file_hash(out / "index.jsonl"),
                            "rows": len(index), "max_total_tokens": max(x["total_tokens"] for x in index)})
    if [x["row_index"] for x in index] != list(range(job["spec"]["selected_rows"])):
        raise ValueError("Incomplete/incorrect index: all selected original rows must be present")
    print(f"Prepared {len(index)} rows; max input={max(x['total_tokens'] for x in index)} tokens; no filtering/truncation",
          flush=True)
    return index


async def score_pending(engine, sampling, out, job, tokenizer, index, done):
    # Lazily read the Parquet. At most concurrency long prompts are in flight.
    pending = (row for row in iter_rows(out / "source.parquet", job["spec"]["selected_rows"])
               if row["row_index"] not in done)
    began = time.monotonic()
    finished = 0

    async def worker():
        nonlocal finished
        for row in pending:
            encoded = encode_full(tokenizer, row, job["spec"]["context_length"])
            desc = describe(row, encoded)
            if desc != index[row["row_index"]]:
                raise ValueError("Tokenization differs from frozen preflight index")
            rid = f"row-{row['row_index']:05d}"
            final = None
            try:
                async for result in engine.generate({"prompt_token_ids": encoded["input_ids"]}, sampling, rid):
                    if result.finished:
                        final = result
                if final is None:
                    raise ValueError(f"vLLM did not complete {rid}")
                result = save_score(out, job["identity"], desc, encoded, extract_logprobs(final, encoded))
            except BaseException:
                try:
                    await asyncio.wait_for(engine.abort(rid), timeout=5)
                except Exception:
                    pass  # Preserve the original failure; shutdown handles a broken engine.
                raise
            finished += 1
            count = len(done) + finished
            elapsed = time.monotonic() - began
            eta = (len(index) - count) * elapsed / finished
            append_jsonl(out / "progress.jsonl", {"timestamp": now(), "completed": count,
                         "row_index": row["row_index"], "mean_nll": result["mean_nll"], "elapsed_seconds": elapsed})
            print(f"SCORE {count}/{len(index)} row={row['row_index']} tokens={desc['n_tokens']} "
                  f"nll={result['mean_nll']:.6f} likelihood={result['normalized_likelihood']:.6f} "
                  f"eta={eta/3600:.2f}h", flush=True)

    workers = [asyncio.create_task(worker()) for _ in range(job["spec"]["request"]["concurrency"])]
    try:
        await asyncio.gather(*workers)
    finally:
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)


async def execute(out, job, tokenizer, index, done):
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This scorer requires a CUDA GPU with BF16 support")
    spec, task, loop = job["spec"], asyncio.current_task(), asyncio.get_running_loop()
    engine, stopped = None, []

    def stop(signum):
        if not stopped:
            stopped.append(signum)
            print("Stopping likelihood scoring: committed rows retained; repeat the command to resume.", flush=True)
            task.cancel()

    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop, signum)
    try:
        options = {"model": spec["model"]["id"], "revision": spec["model"]["revision"],
                   "tokenizer": str(out / "tokenizer"), "trust_remote_code": False,
                   "dtype": "bfloat16", "tensor_parallel_size": 1, "enable_lora": False,
                   "max_model_len": spec["context_length"], "enable_prefix_caching": False,
                   "enable_chunked_prefill": True, "enforce_eager": True,
                   "max_num_seqs": spec["request"]["concurrency"],
                   "max_num_batched_tokens": spec["request"]["max_num_batched_tokens"],
                   "gpu_memory_utilization": spec["request"]["gpu_memory_utilization"],
                   "generation_config": "vllm", "seed": 42}
        append_jsonl(out / "engine_startups.jsonl", {"timestamp": now(), "options": options,
                     "gpu": torch.cuda.get_device_name(0), "already_completed": len(done)})
        engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**options))
        # vLLM returns the actual prompt token even with top-k logprobs=0.
        # The extra generated token merely finishes the request; it is NEVER scored.
        sampling = SamplingParams(prompt_logprobs=0, max_tokens=1, temperature=1.0, top_p=1.0,
                                  top_k=-1, min_p=0.0, repetition_penalty=1.0, ignore_eos=True,
                                  detokenize=False, output_kind=RequestOutputKind.FINAL_ONLY)
        await score_pending(engine, sampling, out, job, tokenizer, index, done)
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


def main():
    args = parse_args()
    if os.name != "posix":
        raise RuntimeError("Run GPU likelihood scoring on the Linux VM; local unit tests need no model")
    configure_runtime()  # Before importing HF/vLLM: downloads and temp files go to /data.
    out = artifact_path(args.out)
    with Pipeline(out) as pipeline:
        job, tokenizer = initialize(args, out)
        pipeline.check_stop()
        index = prepare_index(out, job, tokenizer, pipeline.check_stop)
        if args.prepare_only:
            print(f"Preflight ready: {out / 'inspection.jsonl'}; rerun without --prepare-only to score")
            return
        done = load_scores(out, job, index)
        print(f"Retained {len(done)}/{len(index)} committed scores", flush=True)
        pipeline.check_stop()
        if len(done) < len(index):
            code = asyncio.run(execute(out, job, tokenizer, index, done))
            if code:
                raise SystemExit(code)
        from dataset.report_likelihood import build_report
        build_report(out, allow_partial=False)
        print(f"Finished: {out / 'report' / 'report.html'}", flush=True)


if __name__ == "__main__":
    main()
