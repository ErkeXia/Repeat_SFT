"""SERVER ONLY: BF16 Base + LoRA, eight concurrent requests, per-answer resume."""
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import os
import signal
import time

from recipe.adapters import adapter_spec
from recipe.common import (append_jsonl, checkpoint_path, digest, environment, file_hash, iter_jsonl, now, read_json,
                           snapshot_code, write_json)
from recipe.evaluation import bind_output
from recipe.grouped import generation_epochs, select_panel
from recipe.paths import artifact_path, configure_runtime, storage_root
from recipe.resume import verify_run_inputs
from recipe.replay import evaluation_epochs
from recipe.vllm_generation import (VLLM_VERSION, completed_rows, engine_options, generate_epoch,
                                   sampling_options, validate_config)


def configure_vllm_runtime():
    configure_runtime()
    root = storage_root()
    for key, suffix in (("VLLM_CACHE_ROOT", "vllm"), ("VLLM_CONFIG_ROOT", "vllm-config"),
                        ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"), ("TRITON_CACHE_DIR", "triton"),
                        ("TORCH_HOME", "torch")):
        os.environ.setdefault(key, str(root / ".cache" / suffix))
    for key, value in (("VLLM_USE_V1", "1"), ("VLLM_ENABLE_V1_MULTIPROCESSING", "1"),
                       ("VLLM_WORKER_MULTIPROC_METHOD", "spawn")):
        if os.environ.get(key, value) != value:
            raise ValueError(f"This pinned async vLLM protocol requires {key}={value}")
        os.environ[key] = value


def runtime_identity():
    import torch
    if os.name != "posix" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Run vLLM on the Linux VM with a BF16-capable NVIDIA GPU")
    versions = {name: importlib.metadata.version(name) for name in ("vllm", "torch", "transformers")}
    if (versions["vllm"] != VLLM_VERSION or versions["transformers"] != "4.57.1"
            or versions["torch"].split("+")[0] != "2.8.0"):
        raise RuntimeError("Use the separate environment pinned in requirements-vllm.txt; do not change an active job's backend")
    return {"versions": versions, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "attention_backend_override": os.environ.get("VLLM_ATTENTION_BACKEND"),
            "engine": "V1 AsyncLLM", "adapter_compute_dtype": "bfloat16",
            "reproducibility": "Per-question seed; completed answers immutable. No cross-backend or batch-invariant bitwise guarantee."}


def prepare(args):
    from transformers import AutoConfig, AutoTokenizer
    run = artifact_path(args.run)
    metadata = verify_run_inputs(run)
    settings = read_json(args.config)
    config = validate_config(read_json(args.engine_config))
    tokenizer = AutoTokenizer.from_pretrained(run / "tokenizer", local_files_only=True)
    eos_id = tokenizer.eos_token_id
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if (type(eos_id) is not int or type(end_id) is not int
            or tokenizer.decode([end_id], skip_special_tokens=False) != "<|im_end|>"):
        raise ValueError("Expected Qwen EOS/chat-end tokens are missing")
    sampling = sampling_options(settings, eos_id, end_id)
    full_panel = list(iter_jsonl(run / "data" / "panel.jsonl"))
    panel = full_panel
    if args.sample_ids:
        if args.limit_per_task is not None:
            raise ValueError("--sample-ids and --limit-per-task cannot be combined")
        panel = select_panel(panel, read_json(artifact_path(args.sample_ids)))
    if args.limit_per_task is not None:
        if args.limit_per_task < 1:
            raise ValueError("--limit-per-task must be positive")
        panel = [r for task in sorted({r["task"] for r in panel})
                 for r in [x for x in panel if x["task"] == task][:args.limit_per_task]]
    # Use ALL 192 frozen prompts to choose a common context budget across groups.
    # Passing token IDs avoids a second/different chat template inside vLLM.
    prompts = {r["sample_id"]: tokenizer.apply_chat_template(r["messages"], tokenize=True,
                                                           add_generation_prompt=True) for r in full_panel}
    model_config = AutoConfig.from_pretrained(metadata["model"]["id"], revision=metadata["model"]["revision"],
                                              trust_remote_code=False)
    if model_config.model_type != "qwen3":
        raise ValueError("This backend is validated for dense Qwen3 LoRA")
    options = engine_options(metadata, config, prompts, model_config.max_position_embeddings, settings)
    runtime = runtime_identity()
    backend_config = {"engine_options": options, "runtime": runtime,
                      "base_config_sha256": digest(model_config.to_dict())}
    base = artifact_path(args.out) if args.out else run / "evaluations" / "generation-vllm"
    checkpoints = []
    for epoch in generation_epochs(args.epochs):
        if epoch not in evaluation_epochs(metadata):
            raise ValueError(f"Epoch {epoch} is not a saved training checkpoint")
        ckpt = checkpoint_path(run, epoch)
        marker = read_json(ckpt / "complete.json")
        if (marker["config_sha256"] != metadata["config_sha256"]
                or marker["step"] != epoch * metadata["config"]["expected_train_size"]):
            raise ValueError(f"Checkpoint belongs to another training run: {ckpt}")
        adapter_spec(ckpt, finetuning=metadata["config"]["finetuning"], base_model=metadata["model"])
        out = base / f"epoch-{epoch:03d}"
        bind_output(out, {"run_sha256": file_hash(run / "run.json"), "epoch": epoch, "checkpoint": marker,
                         "settings": settings, "generation_config": {**sampling, "output_kind": "FINAL_ONLY"},
                         "backend": "vllm", "backend_config": backend_config,
                         "panel_sha256": file_hash(run / "data" / "panel.jsonl"),
                         "selected_ids": [r["sample_id"] for r in panel],
                         "seed_policy": "seed + question identity, same at every checkpoint"})
        if not (out / "code").exists():
            snapshot_code(out / "code")
            write_json(out / "environment.json", environment())
        done = completed_rows(out, panel, prompts, epoch, settings, eos_id, end_id)
        checkpoints.append({"epoch": epoch, "checkpoint": ckpt, "out": out, "pending": len(panel) - len(done)})
    return {"options": options, "checkpoints": checkpoints, "panel": panel, "prompts": prompts,
            "settings": settings, "tokenizer": tokenizer, "eos_id": eos_id, "end_id": end_id,
            "progress_label": args.progress_label}


async def run_checkpoints(prepared, engine_factory, lora_factory, sampling_factory, final_output_kind):
    engine = None
    startup_seconds = None
    try:
        for item in prepared["checkpoints"]:
            epoch, out = item["epoch"], item["out"]
            if item["pending"] == 0:
                write_json(out / "complete.json", {"samples": len(prepared["panel"]),
                                                    "responses_sha256": file_hash(out / "responses.jsonl")})
                print(f"SKIP epoch={epoch}: all saved answers retained", flush=True)
                continue
            if engine is None:
                start = time.monotonic()
                engine = engine_factory(prepared["options"])
                startup_seconds = time.monotonic() - start
                print(f"vLLM Base loaded once for this group; startup={startup_seconds:.1f}s; concurrency=8", flush=True)
            append_jsonl(out / "engine_startups.jsonl", {"timestamp": now(), "startup_seconds": startup_seconds,
                                                        "note": "Shared Base instance across this group's checkpoints"})
            lora = lora_factory(f"epoch-{epoch:03d}", epoch + 1, str(item["checkpoint"]))
            await generate_epoch(engine, lora, prepared["panel"], prepared["prompts"], out, epoch,
                                 prepared["settings"], prepared["tokenizer"], prepared["eos_id"], prepared["end_id"],
                                 sampling_factory, final_output_kind, prepared["progress_label"])
    finally:
        if engine is not None:
            engine.shutdown()


async def execute(prepared):
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    stopped = []

    def stop(signum):
        if not stopped:
            stopped.append(signum)
            print("Stopping vLLM; completed answers are saved; unfinished requests will restart on resume.", flush=True)
            task.cancel()

    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stop, signum)
    try:
        await run_checkpoints(prepared, lambda options: AsyncLLM.from_engine_args(AsyncEngineArgs(**options)),
                              LoRARequest, SamplingParams, RequestOutputKind.FINAL_ONLY)
    except asyncio.CancelledError:
        if stopped:
            return 128 + stopped[0]
        raise
    finally:
        for signum in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(signum)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--epochs", help="Default 64,32,16,8,4; always descending")
    p.add_argument("--config", default="configs/generation.json")
    p.add_argument("--engine-config", default="configs/vllm.json")
    p.add_argument("--out", help="Default RUN/evaluations/generation-vllm; never the Transformers output directory")
    p.add_argument("--sample-ids")
    p.add_argument("--limit-per-task", type=int, help="Optional smoke-only subset; not a concurrency benchmark")
    p.add_argument("--progress-label", default="")
    args = p.parse_args()
    configure_vllm_runtime()
    from grouped_evaluation import job_lock
    run = artifact_path(args.run)
    base = artifact_path(args.out) if args.out else run / "evaluations" / "generation-vllm"
    # Protect direct invocation as well as invocation through the grouped controller.
    with job_lock(base):
        prepared = prepare(args)
        raise SystemExit(asyncio.run(execute(prepared)))


if __name__ == "__main__":
    main()
