"""Bounded concurrent generation and durable records; importable without vLLM."""
from __future__ import annotations

import asyncio
import math
import time

from generate import stop_details
from .adapters import validate_finetuning
from .common import append_jsonl, file_hash, now, read_json, recover_jsonl, stable_seed, write_json

VLLM_VERSION = "0.11.0"


def validate_config(config):
    keys = {"max_num_seqs", "max_num_batched_tokens", "gpu_memory_utilization", "max_model_len",
            "max_lora_rank", "dtype", "lora_dtype", "enforce_eager"}
    if set(config) != keys:
        raise ValueError("Unexpected/missing vLLM engine configuration")
    if type(config["max_num_seqs"]) is not int or config["max_num_seqs"] != 8:
        raise ValueError("This vLLM protocol requires concurrency max_num_seqs=8")
    for key in ("max_num_batched_tokens", "max_lora_rank"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["max_lora_rank"] not in (8, 16, 32, 64, 128, 256, 320, 512):
        raise ValueError("Unsupported vLLM max_lora_rank")
    if (not math.isfinite(config["gpu_memory_utilization"])
            or not 0 < config["gpu_memory_utilization"] < 1):
        raise ValueError("gpu_memory_utilization must be between 0 and 1")
    limit = config["max_model_len"]
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("max_model_len must be null or a positive integer")
    if config["dtype"] != "bfloat16" or config["lora_dtype"] != "bfloat16":
        raise ValueError("The vLLM evaluation protocol uses explicit BF16 Base and BF16 adapters")
    if type(config["enforce_eager"]) is not bool:
        raise ValueError("enforce_eager must be boolean")
    return config


def engine_options(metadata, config, prompts, model_context, settings):
    validate_config(config)
    ft = validate_finetuning(metadata["config"].get("finetuning"))
    if ft["base_quantization"] != "none":
        raise ValueError("vLLM backend currently supports BF16 LoRA only; use Transformers for NF4 runs")
    if ft["rank"] > config["max_lora_rank"]:
        raise ValueError("Adapter rank exceeds configured max_lora_rank")
    budget = max(map(len, prompts.values())) + settings["max_new_tokens"]
    limit = config["max_model_len"] or budget
    if limit < budget or limit > model_context:
        raise ValueError(f"Need prompt + new tokens = {budget}; max_model_len={limit}, Base context={model_context}. "
                         "Do not truncate prompts or silently reduce the generation budget.")
    return {"model": metadata["model"]["id"], "revision": metadata["model"]["revision"],
            "dtype": config["dtype"], "lora_dtype": config["lora_dtype"], "quantization": None,
            "enable_lora": True, "max_lora_rank": config["max_lora_rank"], "max_loras": 1,
            "max_cpu_loras": 8, "max_num_seqs": 8, "max_num_batched_tokens": config["max_num_batched_tokens"],
            "max_model_len": limit, "gpu_memory_utilization": config["gpu_memory_utilization"],
            "enforce_eager": config["enforce_eager"], "tensor_parallel_size": 1,
            "enable_prefix_caching": False, "enable_chunked_prefill": True,
            "skip_tokenizer_init": True, "trust_remote_code": False, "generation_config": "vllm",
            "seed": settings["seed"], "disable_log_stats": False}


def sampling_options(settings, eos_id, end_id):
    keys = {"seed", "do_sample", "temperature", "top_p", "top_k", "max_new_tokens", "repetition_penalty"}
    if set(settings) != keys or settings["max_new_tokens"] < 1:
        raise ValueError("Unexpected generation settings or nonpositive token budget")
    if not settings["do_sample"]:
        raise ValueError("This evaluation protocol requires sampling, not greedy decoding")
    return {"n": 1, "temperature": settings["temperature"], "top_p": settings["top_p"],
            "top_k": settings["top_k"], "max_tokens": settings["max_new_tokens"],
            "repetition_penalty": settings["repetition_penalty"], "presence_penalty": 0.0,
            "frequency_penalty": 0.0, "min_p": 0.0, "min_tokens": 0, "ignore_eos": False,
            "stop_token_ids": list(dict.fromkeys([end_id, eos_id])), "stop": [],
            "detokenize": False, "skip_special_tokens": False, "truncate_prompt_tokens": None}


def completed_rows(out, panel, prompts, epoch, settings, eos_id, end_id):
    marker = out / "complete.json"
    if marker.exists():
        saved = read_json(marker)
        log = out / "responses.jsonl"
        if not log.is_file() or saved != {"samples": len(panel), "responses_sha256": file_hash(log)}:
            raise ValueError(f"Completed vLLM responses changed: {out}")
    rows = recover_jsonl(out / "responses.jsonl")
    expected = {r["sample_id"]: r for r in panel}
    done = set()
    for row in rows:
        sid = row["sample_id"]
        if sid not in expected or sid in done:
            raise ValueError("Duplicate/unexpected vLLM response; no automatic deduplication")
        question = expected[sid]
        if (row["epoch"] != epoch or row.get("backend") != "vllm"
                or any(row[k] != question[k] for k in ("question_id", "task", "messages", "gold_answer"))
                or row["seed"] != stable_seed(settings["seed"], sid)
                or row["prompt_token_ids"] != prompts[sid] or row["prompt_tokens"] != len(prompts[sid])):
            raise ValueError(f"Saved response identity/prompt changed: {sid}")
        ids = row["generated_token_ids"]
        if (not ids or any(type(t) is not int or t < 0 for t in ids)
                or row["generated_tokens"] != len(ids) or len(ids) > settings["max_new_tokens"]):
            raise ValueError(f"Saved response has invalid token IDs/count: {sid}")
        detail = stop_details(ids, eos_id, end_id, settings["max_new_tokens"])
        if detail["stop_reason"] == "other" or any(row[k] != value for k, value in detail.items()):
            raise ValueError(f"Saved response has invalid stop metadata: {sid}")
        done.add(sid)
    if marker.exists() and len(done) != len(panel):
        raise ValueError(f"Completed vLLM responses changed: {out}")
    return done


def response_record(output, request_id, row, prompt, epoch, seed, tokenizer, eos_id, end_id, limit, seconds):
    if (output.request_id != request_id or not output.finished or len(output.outputs) != 1
            or list(output.prompt_token_ids) != prompt):
        raise ValueError("vLLM returned an incomplete/mismatched request or changed prompt token IDs")
    completion = output.outputs[0]
    ids = list(completion.token_ids)
    if not ids or len(ids) > limit or any(type(t) is not int or t < 0 for t in ids):
        raise ValueError("vLLM returned invalid/over-budget generated token IDs")
    detail = stop_details(ids, eos_id, end_id, limit)
    if completion.finish_reason not in ("stop", "length") or detail["stop_reason"] == "other":
        raise ValueError("vLLM request did not finish at a declared EOS or the token limit")
    if completion.stop_reason is not None and completion.stop_reason not in (eos_id, end_id):
        raise ValueError("vLLM stopped for an unexpected stop token/string")
    return {"sample_id": row["sample_id"], "question_id": row["question_id"], "task": row["task"],
            "epoch": epoch, "seed": seed, "messages": row["messages"], "gold_answer": row["gold_answer"],
            "prompt_token_ids": prompt, "generated_token_ids": ids, "prompt_tokens": len(prompt),
            "generated_tokens": len(ids), "raw_response": tokenizer.decode(ids, skip_special_tokens=False),
            "response_text": tokenizer.decode(ids, skip_special_tokens=True), "generation_seconds": seconds,
            "backend": "vllm", "engine_finish_reason": completion.finish_reason,
            "engine_stop_reason": completion.stop_reason, **detail}


async def generate_epoch(engine, lora, panel, prompts, out, epoch, settings, tokenizer, eos_id, end_id,
                         sampling_factory, final_output_kind, progress_label=""):
    """At most eight requests in flight. Every completed answer is fsynced immediately."""
    done = completed_rows(out, panel, prompts, epoch, settings, eos_id, end_id)
    pending = [r for r in panel if r["sample_id"] not in done]
    queue = asyncio.Queue()
    for row in pending:
        queue.put_nowait(row)
    sampling = sampling_options(settings, eos_id, end_id)
    started = time.monotonic()
    before, produced_tokens = len(done), 0
    status = "interrupted"

    async def worker():
        nonlocal produced_tokens
        while not queue.empty():
            row = queue.get_nowait()
            sid = row["sample_id"]
            request_id = f"epoch-{epoch:03d}:{sid}"
            seed = stable_seed(settings["seed"], sid)
            params = sampling_factory(**sampling, seed=seed, output_kind=final_output_kind)
            start = time.monotonic()
            saved = False
            try:
                async for result in engine.generate({"prompt_token_ids": prompts[sid]}, params,
                                                    request_id, lora_request=lora):
                    if not result.finished:
                        continue
                    if saved:
                        raise ValueError(f"Duplicate final output: {request_id}")
                    record = response_record(result, request_id, row, prompts[sid], epoch, seed, tokenizer,
                                             eos_id, end_id, settings["max_new_tokens"], time.monotonic() - start)
                    append_jsonl(out / "responses.jsonl", record)
                    done.add(sid)
                    produced_tokens += record["generated_tokens"]
                    saved = True
                    write_json(out / "progress.json", {"epoch": epoch, "completed": len(done), "total": len(panel),
                                                        "last_sample_id": sid, "backend": "vllm", "concurrency": 8})
                    print(f"{progress_label} epoch={epoch} {len(done)}/{len(panel)} "
                          f"tokens={record['generated_tokens']} stop={record['stop_reason']} backend=vllm concurrency=8",
                          flush=True)
                if not saved:
                    raise RuntimeError(f"Request ended without a complete answer: {request_id}")
            except BaseException as exc:
                try:
                    await asyncio.wait_for(engine.abort(request_id), timeout=5)
                except Exception:
                    pass  # The engine may already be dead; retain the original error.
                if isinstance(exc, Exception):
                    append_jsonl(out / "errors.jsonl", {"epoch": epoch, "sample_id": sid, "seed": seed,
                                                       "error": str(exc), "timestamp": now()})
                raise

    tasks = [asyncio.create_task(worker()) for _ in range(min(8, len(pending)))]
    try:
        await asyncio.gather(*tasks)
        if len(done) != len(panel):
            raise RuntimeError("Incomplete panel; no completion marker written")
        write_json(out / "complete.json", {"samples": len(panel), "responses_sha256": file_hash(out / "responses.jsonl")})
        status = "complete"
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.monotonic() - started
        append_jsonl(out / "generation_attempts.jsonl", {"timestamp": now(), "status": status,
                     "previously_completed": before, "newly_completed": len(done) - before,
                     "new_generated_tokens": produced_tokens, "elapsed_seconds": elapsed,
                     "generated_tokens_per_second": produced_tokens / elapsed if elapsed else None,
                     "note": "Per-epoch wall time excludes engine startup; per-request times overlap."})
