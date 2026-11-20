"""SERVER: score the frozen manual answers with LoRA checkpoints and compare to Base."""
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recipe.adapters import adapter_spec
from recipe.common import (append_jsonl, digest, environment, file_hash, iter_jsonl, now,
                           read_json, snapshot_code, write_json)
from recipe.paths import artifact_path, configure_runtime
from recipe.pipeline import Pipeline
from recipe.replay import IDENTITY_KEYS, parse_epochs
from recipe.resume import verified_code, verify_run_inputs
from dataset.curated_training import verify_bundle
from dataset.curated_likelihood_report import build_comparison
from dataset.likelihood_training import tree_hashes
from dataset.likelihood_utils import load_scores
from dataset.score_curated import build_report, score_pending

EXTRA_CODE = ("__init__.py", "score_curated_checkpoints.py", "curated_likelihood_report.py",
              "score_curated.py", "curated_training.py", "likelihood_utils.py", "likelihood_training.py")
PROTOCOL = "curated-checkpoint-likelihood-v1"


def versions():
    result = {name: importlib.metadata.version(name) for name in ("vllm", "torch", "transformers", "numpy")}
    if result["vllm"] != "0.11.0":
        raise ValueError("Use the existing separate vllm==0.11.0 environment")
    return result


def prepare(experiment, out, epochs):
    if (out / "setup").exists():
        plan = verify(out)
        if plan["experiment"] != str(experiment) or plan["epochs"] != epochs:
            raise ValueError("Scoring inputs differ; repeat the original command or choose a new --job-name")
        return plan
    if any(p.name != "pipeline.lock" and not p.name.startswith(".setup-partial-") for p in out.iterdir()):
        raise ValueError("Output is not this scoring job; choose a new --job-name")
    original = verify_bundle(experiment)
    run = experiment / "train"
    meta = verify_run_inputs(run)
    prepared = experiment / "bundle/prepared/train"
    if (meta["model"] != original["model"] or meta["config"] != original["config"]
            or meta["max_length"] != original["max_length"]
            or meta["config_sha256"] != digest({k: meta[k] for k in IDENTITY_KEYS})
            or file_hash(run / "data/train.jsonl") != file_hash(prepared / "train.jsonl")
            or tree_hashes(run / "tokenizer") != tree_hashes(prepared / "tokenizer")):
        raise ValueError("Training run does not match the frozen curated experiment")
    ft = meta["config"]["finetuning"]
    if ft["base_quantization"] != "none" or ft["rank"] not in (8, 16, 32, 64, 128, 256, 320, 512):
        raise ValueError("Expected BF16 LoRA with a supported vLLM rank")
    if not set(epochs) <= set(meta["config"]["checkpoint_epochs"]):
        raise ValueError("Requested checkpoint epochs were not scheduled in this run")
    checkpoints = {}
    for epoch in epochs:
        path = run / "checkpoints" / f"epoch-{epoch:03d}"
        marker = read_json(path / "complete.json")
        if marker["config_sha256"] != meta["config_sha256"] or marker["step"] != epoch * meta["config"]["expected_train_size"]:
            raise ValueError(f"Checkpoint identity/step mismatch: {path}")
        adapter_spec(path, finetuning=ft, base_model=meta["model"])
        checkpoints[str(epoch)] = {"path": str(path), "marker_sha256": file_hash(path / "complete.json")}
    rows = list(iter_jsonl(prepared / "train.jsonl"))
    index = list(iter_jsonl(experiment / "bundle/index.jsonl"))
    if len(rows) != 10 or len(index) != 10 or len({r["sample_id"] for r in rows}) != 10:
        raise ValueError("Expected the ten distinct frozen manual answers")
    base = experiment / "base_likelihood"
    job, complete = read_json(base / "job.json"), read_json(base / "complete.json")
    if (job["identity"] != digest(job["spec"]) or job["spec"]["protocol"] != "curated-ten-base-likelihood-v1"
            or job["spec"]["bundle_identity"] != original["identity"] or job["spec"]["model"] != original["model"]
            or job["spec"]["scoring"] != original["scoring"] or complete["job_identity"] != job["identity"]
            or complete["samples"] != 10 or len(load_scores(base, job, index)) != 10
            or complete["summary_sha256"] != file_hash(base / "summary.json")
            or complete["records"] != {p.name: file_hash(p) for p in sorted((base / "records").glob("*.json"))}):
        raise ValueError("Original Base scores are incomplete or do not match these ten answers")
    current_versions = versions()
    if current_versions != job["spec"]["versions"]:
        raise ValueError(f"Use the same scoring package versions as Base: {job['spec']['versions']}; got {current_versions}")
    stage = out / (".setup-partial-" + uuid.uuid4().hex)
    stage.mkdir()
    shutil.copy2(prepared / "train.jsonl", stage / "samples.jsonl")
    shutil.copy2(experiment / "bundle/index.jsonl", stage / "index.jsonl")
    shutil.copytree(prepared / "tokenizer", stage / "tokenizer")
    shutil.copytree(base, stage / "base")
    if tree_hashes(stage / "base") != tree_hashes(base):
        raise ValueError("Base results changed during copying")
    code_hash = snapshot_code(stage / "code")
    (stage / "code/dataset").mkdir()
    for name in EXTRA_CODE:
        shutil.copy2(ROOT / "dataset" / name, stage / "code/dataset" / name)
    plan = {"protocol": PROTOCOL, "created_at": now(), "experiment": str(experiment), "epochs": epochs,
            "bundle_identity": original["identity"], "run_sha256": file_hash(run / "run.json"),
            "model": meta["model"], "finetuning": ft, "checkpoints": checkpoints,
            "scoring": original["scoring"], "seed": meta["config"]["seed"], "versions": current_versions,
            "lora_dtype": "bfloat16", "base_job_identity": job["identity"], "code_sha256": code_hash,
            "semantics": "Teacher forcing on frozen assistant reasoning, final answer and im_end; "
                         "raw prompt logprobs, no truncation; existing Base scores reused",
            "files": tree_hashes(stage)}
    plan["identity"] = digest(plan)
    write_json(stage / "plan.json", plan)
    stage.rename(out / "setup")
    return verify(out)


def verify(out):
    setup = out / "setup"
    plan = read_json(setup / "plan.json")
    if plan["protocol"] != PROTOCOL or plan["identity"] != digest({k: v for k, v in plan.items() if k != "identity"}):
        raise ValueError("Frozen scoring plan changed")
    for name, expected in plan["files"].items():
        path = (setup / name).resolve()
        if not path.is_relative_to(setup.resolve()) or file_hash(path) != expected:
            raise ValueError(f"Frozen scoring input/code changed: {name}")
    verified_code(setup / "code", plan["code_sha256"])
    run = Path(plan["experiment"]) / "train"
    if file_hash(run / "run.json") != plan["run_sha256"]:
        raise ValueError("Source training metadata changed")
    for checkpoint in plan["checkpoints"].values():
        path = Path(checkpoint["path"])
        if file_hash(path / "complete.json") != checkpoint["marker_sha256"]:
            raise ValueError("Checkpoint completion marker changed")
        adapter_spec(path, finetuning=plan["finetuning"], base_model=plan["model"])
    return plan


def epoch_job(out, plan, epoch):
    path = out / "epochs" / f"epoch-{epoch:03d}"
    spec = {"protocol": PROTOCOL, "plan_identity": plan["identity"], "epoch": epoch,
            "checkpoint": plan["checkpoints"][str(epoch)], "model": plan["model"],
            "scoring": {**plan["scoring"], "adapter": "lora", "lora_dtype": plan["lora_dtype"]},
            "versions": plan["versions"], "semantics": plan["semantics"]}
    job = {"identity": digest(spec), "spec": spec}
    if (path / "job.json").exists():
        if read_json(path / "job.json") != job:
            raise ValueError("Checkpoint score identity changed")
    else:
        if any((path / "records").glob("*.json")):
            raise ValueError("Scores found without their job identity")
        write_json(path / "job.json", job)
    return path, job


def engine_options(plan, out):
    settings = plan["scoring"]
    return {"model": plan["model"]["id"], "revision": plan["model"]["revision"],
            "tokenizer": str(out / "setup/tokenizer"), "trust_remote_code": False,
            "dtype": "bfloat16", "tensor_parallel_size": 1, "enable_lora": True,
            "lora_dtype": plan["lora_dtype"], "max_lora_rank": plan["finetuning"]["rank"],
            "max_loras": 1, "max_cpu_loras": max(1, len(plan["epochs"])),
            "enable_prefix_caching": False, "enable_chunked_prefill": True, "enforce_eager": True,
            "max_model_len": settings["max_model_len"], "max_num_seqs": settings["concurrency"],
            "max_num_batched_tokens": settings["max_num_batched_tokens"],
            "gpu_memory_utilization": settings["gpu_memory_utilization"],
            "generation_config": "vllm", "seed": plan["seed"]}


async def score_models(engine, sampling, lora_class, out, plan, tokenizer):
    rows = list(iter_jsonl(out / "setup/samples.jsonl"))
    index = list(iter_jsonl(out / "setup/index.jsonl"))
    for epoch in plan["epochs"]:
        path, job = epoch_job(out, plan, epoch)
        done = load_scores(path, job, index)
        print(f"EPOCH {epoch}: retained {len(done)}/10 manual answer scores", flush=True)
        if len(done) < len(rows):
            lora = lora_class(f"curated-epoch-{epoch}", epoch, plan["checkpoints"][str(epoch)]["path"])
            await score_pending(engine, sampling, path, job, rows, tokenizer, index, done,
                                plan["scoring"]["concurrency"], lora_request=lora,
                                label=f"EPOCH {epoch} SCORE", request_prefix=f"curated-epoch-{epoch}")
        if not (path / "complete.json").exists():
            build_report(path, job, rows, index, model_label=f"Epoch {epoch}")
        build_comparison(out, plan)


async def execute(out, plan):
    import torch
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM
    if versions() != plan["versions"]:
        raise ValueError("Scoring versions changed; restore the frozen environment")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 support required")
    tokenizer = AutoTokenizer.from_pretrained(out / "setup/tokenizer", local_files_only=True)
    task, loop, stopped, engine = asyncio.current_task(), asyncio.get_running_loop(), [], None

    def stop(signum):
        if not stopped:
            stopped.append(signum)
            print("Stopping likelihood scoring; committed answers retained; repeat the command to resume", flush=True)
            task.cancel()

    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop, signum)
    try:
        options = engine_options(plan, out)
        append_jsonl(out / "engine_startups.jsonl", {"timestamp": now(), "options": options,
                                                   "gpu": torch.cuda.get_device_name(0), "environment": environment()})
        engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**options))
        sampling = SamplingParams(prompt_logprobs=0, max_tokens=1, temperature=1.0, top_p=1.0,
                                  top_k=-1, min_p=0.0, repetition_penalty=1.0, ignore_eos=True,
                                  detokenize=False, output_kind=RequestOutputKind.FINAL_ONLY)
        await score_models(engine, sampling, LoRARequest, out, plan, tokenizer)
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


def finish(out, plan):
    if not build_comparison(out, plan):
        return False
    write_json(out / "complete.json", {"plan_identity": plan["identity"], "epochs": plan["epochs"],
               "checkpoint_samples": 10 * len(plan["epochs"]), "reused_base_samples": 10,
               "report_files": tree_hashes(out / "report")})
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="runs/curated-10x128-s42")
    parser.add_argument("--epochs", default="64,128")
    parser.add_argument("--job-name", default="e64-e128-v1")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="Refresh reports from saved scores; no GPU")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.job_name):
        parser.error("--job-name must be a single directory name")
    if os.name != "posix":
        raise RuntimeError("Run this likelihood experiment on the Linux VM")
    configure_runtime()
    experiment = artifact_path(args.experiment)
    out = Path(args.worker).resolve() if args.worker else experiment / "checkpoint_likelihood" / args.job_name
    with Pipeline(out, inherited=bool(args.worker)) as pipeline:
        if not args.worker:
            plan = prepare(experiment, out, parse_epochs(args.epochs))
            print(f"Checkpoints: {plan['epochs']}; 10 frozen answers each; reused Base scores; output={out}", flush=True)
            if args.prepare_only:
                return
            code = out / "setup/code"
            command = [sys.executable, "-B", "-u", code / "dataset/score_curated_checkpoints.py", "--worker", out]
            if args.report_only:
                command += ["--report-only"]
            pipeline.invoke(command, code)
            return
        plan = verify(out)
        rows = list(iter_jsonl(out / "setup/samples.jsonl"))
        index = list(iter_jsonl(out / "setup/index.jsonl"))
        pending = False
        for epoch in plan["epochs"]:
            path, job = epoch_job(out, plan, epoch)
            if len(load_scores(path, job, index)) == len(rows):
                if not (path / "complete.json").exists():
                    build_report(path, job, rows, index, model_label=f"Epoch {epoch}")
            else:
                pending = True
        finish(out, plan)
        if pending and not args.report_only:
            pipeline.check_stop()
            status = asyncio.run(execute(out, plan))
            if status:
                raise SystemExit(status)
        pipeline.check_stop()
        finish(out, plan)
        print(f"Likelihood comparison: {out / 'report/report.html'}", flush=True)


if __name__ == "__main__":
    main()
