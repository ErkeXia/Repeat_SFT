"""SERVER ONLY: one epoch / 400 examples for four fixed batch/attention combinations."""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid

from recipe.common import (ROOT, append_jsonl, digest, environment, epoch_order, file_hash,
                           fresh_dir, iter_jsonl, lr_for_update, now, read_json,
                           seed_everything, snapshot_code, validate_prepared, write_json, write_jsonl)
from recipe.paths import artifact_path, configure_runtime
from recipe.resume import verified_code

CASES = (
    {"name": "b1-sdpa", "batch_size": 1, "attention": "sdpa"},
    {"name": "b4-sdpa", "batch_size": 4, "attention": "sdpa"},
    {"name": "b1-flash", "batch_size": 1, "attention": "flash_attention_2"},
    {"name": "b4-flash", "batch_size": 4, "attention": "flash_attention_2"},
)
TRAIN_SIZE = 400
GIB = 1024**3


def batch_indices(order, batch_size):
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


def validate_rows(rows):
    if len(rows) != TRAIN_SIZE:
        raise ValueError(f"Benchmark requires exactly {TRAIN_SIZE} prepared training examples")
    for key in ("sample_id", "question_id"):
        if len({r[key] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate {key} in benchmark data")


def validate_config(config, warmup_samples):
    if config["expected_train_size"] != TRAIN_SIZE:
        raise ValueError("expected_train_size must be 400")
    if not isinstance(warmup_samples, int) or not 0 <= warmup_samples < TRAIN_SIZE or warmup_samples % 4:
        raise ValueError("--warmup-samples must be a multiple of 4 in [0, 396]")
    for key in ("learning_rate", "adam_epsilon", "max_grad_norm"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if config["logit_chunk_size"] < 1 or config["warmup_steps"] < 0:
        raise ValueError("Invalid logit chunk size or LR warmup")


def prepare_run(args):
    if not args.data:
        raise ValueError("--data is required for a new benchmark")
    data = artifact_path(args.data)
    prepared = validate_prepared(data)
    if prepared["model"]["id"] != "Qwen/Qwen3-4B-Base" or not prepared["model"]["revision"]:
        raise ValueError("Expected a pinned Qwen/Qwen3-4B-Base in prepared data")
    rows = list(iter_jsonl(data / "train.jsonl"))
    validate_rows(rows)
    config = read_json(args.config or ROOT / "configs" / "baseline.json")
    if args.seed is not None:
        config["seed"] = args.seed
    warmup = 16 if args.warmup_samples is None else args.warmup_samples
    validate_config(config, warmup)
    # Retain the full source configuration, but explicitly freeze the one-epoch protocol.
    protocol = dict(config, epochs=1, checkpoint_epochs=[1], attention_implementation="per_case")
    out = fresh_dir(artifact_path(args.out))
    (out / "data").mkdir()
    for name in ("train.jsonl", "manifest.json", "chat_template.jinja"):
        shutil.copy2(data / name, out / "data" / name)
    shutil.copytree(data / "tokenizer", out / "tokenizer")
    write_json(out / "base_config.json", config)
    order = epoch_order(TRAIN_SIZE, config["seed"], 0)
    write_json(out / "order.json", {"indices": order, "sample_ids": [rows[i]["sample_id"] for i in order]})
    files = {p.relative_to(out).as_posix(): file_hash(p)
             for folder in (out / "data", out / "tokenizer") for p in folder.rglob("*") if p.is_file()}
    files.update({name: file_hash(out / name) for name in ("base_config.json", "order.json")})
    manifest = {"format_version": 1, "kind": "training_speed_benchmark", "created_at": now(),
                "model": prepared["model"], "max_length": prepared["arguments"]["max_length"],
                "data_source": str(data), "config": protocol, "cases": list(CASES), "files": files,
                "warmup_samples": warmup, "save_weights": bool(args.save_weights),
                "kernel_probe": args.kernel_probe is not False,
                "semantics": {"dtype": "bfloat16", "full_parameter": True,
                              "gradient_checkpointing": True, "gradient_accumulation_steps": 1,
                              "optimizer": "bitsandbytes.AdamW8bit", "padding": "right, no packing or length sorting",
                              "loss": "mean over all assistant target tokens in each batch, including im_end",
                              "warmup": "LR warmup counts updates; timing warmup counts examples and does not skip training",
                              "resume": "skip completed cases; retry incomplete cases from Base in a new attempt"}}
    manifest["code_sha256"] = snapshot_code(out / "code")
    manifest["identity"] = digest(manifest)
    write_json(out / "benchmark.json", manifest)
    return out, manifest


def verify_run(out):
    manifest = read_json(out / "benchmark.json")
    if manifest["identity"] != digest({k: v for k, v in manifest.items() if k != "identity"}):
        raise ValueError("Benchmark manifest changed")
    if manifest["cases"] != list(CASES):
        raise ValueError("Benchmark case definitions changed")
    for name, expected in manifest["files"].items():
        path = (out / name).resolve()
        if not path.is_relative_to(out.resolve()) or file_hash(path) != expected:
            raise ValueError(f"Frozen benchmark input changed: {name}")
    code = verified_code(out / "code", manifest["code_sha256"])
    return manifest, code


def summarize_steps(records, warmup_samples):
    def aggregate(items):
        seconds = sum(r["update_seconds"] for r in items)
        examples = sum(r["batch_size"] for r in items)
        tokens = sum(r["input_tokens"] for r in items)
        padded = sum(r["padded_tokens"] for r in items)
        supervised = sum(r["supervised_tokens"] for r in items)
        return {"updates": len(items), "examples": examples, "update_seconds": seconds,
                "input_tokens": tokens, "padded_tokens": padded, "supervised_tokens": supervised,
                "examples_per_second": examples / seconds if seconds else None,
                "nonpadding_tokens_per_second": tokens / seconds if seconds else None,
                "supervised_tokens_per_second": supervised / seconds if seconds else None,
                "padding_fraction": 1 - tokens / padded if padded else None}
    return {"all_updates": aggregate(records),
            "after_timing_warmup": aggregate([r for r in records if r["sample_start"] >= warmup_samples])}


def train_epoch(model, optimizer, scheduler, tokenizer, rows, encoded, order, case, config, attempt,
                warmup_samples, *, device="cuda"):
    """The timed region includes collation, transfer, forward/backward and optimizer update.

    CPU injection exists only for tiny offline tests. The CLI always requires CUDA/BF16.
    Synchronous per-update measurements exclude log I/O; epoch wall time includes it.
    """
    import torch
    from recipe.encoding import pad_batch
    from recipe.modeling import training_loss

    on_cuda = torch.device(device).type == "cuda"
    sync = torch.cuda.synchronize if on_cuda else lambda: None
    sync()
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
    records = []
    batches = batch_indices(order, case["batch_size"])
    epoch_start = time.perf_counter()
    for step, indices in enumerate(batches, 1):
        examples = [encoded[i] for i in indices]
        sync()
        started = time.perf_counter()
        batch = {k: v.to(device) for k, v in pad_batch(examples, tokenizer.pad_token_id).items()}
        optimizer.zero_grad(set_to_none=True)
        lr = optimizer.param_groups[0]["lr"]
        expected = lr_for_update(step, config["learning_rate"], config["warmup_steps"])
        if abs(lr - expected) > 1e-14:
            raise ValueError("Scheduler mismatch")
        loss = training_loss(model, batch, config["logit_chunk_size"])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Nonfinite loss at update {step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"], error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        sync()
        seconds = time.perf_counter() - started
        record = {"step": step, "epoch": 1, "sample_start": (step - 1) * case["batch_size"],
                  "samples_seen": min(step * case["batch_size"], len(rows)), "batch_size": len(indices),
                  "sample_ids": [rows[i]["sample_id"] for i in indices],
                  "question_ids": [rows[i]["question_id"] for i in indices],
                  "sequence_lengths": [len(e["input_ids"]) for e in examples],
                  "input_tokens": sum(len(e["input_ids"]) for e in examples),
                  "padded_tokens": batch["input_ids"].numel(),
                  "supervised_tokens": sum(e["supervised_tokens"] for e in examples),
                  "loss": loss.item(), "learning_rate": lr, "grad_norm_before_clip": norm.item(),
                  "update_seconds": seconds, "timestamp": now()}
        del loss, batch, norm
        records.append(record)
        append_jsonl(attempt / "steps.jsonl", record)
        elapsed = time.perf_counter() - epoch_start
        seen = record["samples_seen"]
        progress = {"status": "running", "step": step, "total_steps": len(batches),
                    "samples_seen": seen, "total_samples": len(rows), "elapsed_seconds": elapsed,
                    "eta_seconds": elapsed / seen * (len(rows) - seen)}
        write_json(attempt / "progress.json", progress)
        filled = int(20 * seen / len(rows))
        print(f"{case['name']} [{'#' * filled}{'-' * (20 - filled)}] {seen}/{len(rows)} "
              f"step={step}/{len(batches)} loss={record['loss']:.6f} "
              f"update={seconds:.2f}s elapsed={elapsed:.1f}s ETA={progress['eta_seconds']:.1f}s", flush=True)
    sync()
    metrics = summarize_steps(records, warmup_samples)
    metrics.update(epoch_wall_seconds=time.perf_counter() - epoch_start,
                   peak_allocated_gib=torch.cuda.max_memory_allocated() / GIB if on_cuda else None,
                   peak_reserved_gib=torch.cuda.max_memory_reserved() / GIB if on_cuda else None)
    write_json(attempt / "progress.json", dict(progress, status="training_complete", eta_seconds=0))
    return metrics


def attention_evidence(names):
    matches = sorted({n for n in names if any(k in n.lower() for k in ("attention", "flash", "fmha"))})
    backends = []
    for marker, label in (("_scaled_dot_product_flash_attention", "pytorch_flash"),
                          ("_scaled_dot_product_efficient_attention", "pytorch_efficient"),
                          ("_scaled_dot_product_cudnn_attention", "pytorch_cudnn"),
                          ("_scaled_dot_product_attention_math", "pytorch_math"),
                          ("flash_attn", "external_flash_attn")):
        if any(marker in n.lower() for n in matches):
            backends.append(label)
    return {"inferred_backends": backends or ["unclassified; inspect operator_names"], "operator_names": matches}


def probe_attention(model, optimizer, tokenizer, examples, config):
    import torch
    from recipe.encoding import pad_batch
    from recipe.modeling import training_loss
    from torch.profiler import ProfilerActivity, profile

    # This extra forward/backward is AFTER the epoch, with no optimizer update.
    # Its profiler overhead and memory are excluded from all training measurements.
    optimizer.zero_grad(set_to_none=True)
    batch = {k: v.to("cuda") for k, v in pad_batch(examples, tokenizer.pad_token_id).items()}
    torch.cuda.synchronize()
    started = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        loss = training_loss(model, batch, config["logit_chunk_size"])
        loss.backward()
        torch.cuda.synchronize()
    result = attention_evidence([item.key for item in prof.key_averages()])
    result.update(status="complete", seconds=time.perf_counter() - started,
                  sequence_lengths=[len(e["input_ids"]) for e in examples],
                  scope="First batch shape, probed after training; other shapes may use other kernels")
    optimizer.zero_grad(set_to_none=True)
    return result


def worker(out, case_name, attempt):
    manifest, _ = verify_run(out)
    case = next(c for c in CASES if c["name"] == case_name)
    result = {"case": case_name, "identity": manifest["identity"], "attempt": str(attempt),
              "batch_size": case["batch_size"], "attention": case["attention"], "status": "running"}
    write_json(attempt / "result.json", result)
    started = time.perf_counter()
    try:
        config = manifest["config"]
        # HF 4.57.1 reads this flag for the external FA2 backward implementation.
        os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1" if config["deterministic"] else "0"
        write_json(attempt / "environment.json", {**environment(), "environment_variables": {
            k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "HF_HOME", "TMPDIR",
                                           "FLASH_ATTENTION_DETERMINISTIC", "PYTORCH_CUDA_ALLOC_CONF")}})
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Benchmark requires a CUDA GPU with BF16 support; do not run locally")
        import bitsandbytes as bnb
        from transformers import AutoTokenizer
        from recipe.encoding import encode_example, inspect_example
        from recipe.modeling import load_model
        if case["attention"] == "flash_attention_2":
            import inspect
            from flash_attn import flash_attn_varlen_func
            if config["deterministic"] and "deterministic" not in inspect.signature(flash_attn_varlen_func).parameters:
                raise RuntimeError("Installed flash-attn does not support deterministic backward")
        seed_everything(config["seed"], config["deterministic"])
        rows = list(iter_jsonl(out / "data" / "train.jsonl"))
        validate_rows(rows)
        order = read_json(out / "order.json")["indices"]
        if order != epoch_order(TRAIN_SIZE, config["seed"], 0):
            raise ValueError("Frozen sample order mismatch")
        tokenizer = AutoTokenizer.from_pretrained(out / "tokenizer", local_files_only=True)
        encoded = [encode_example(tokenizer, row, manifest["max_length"]) for row in rows]
        if any(e is None for e in encoded):
            raise ValueError("Over-length example; benchmark never truncates or drops examples")
        write_jsonl(attempt / "inspection.jsonl", [inspect_example(tokenizer, rows[i], encoded[i]) for i in order[:3]])
        write_json(attempt / "config.json", {**config, "batch_size": case["batch_size"],
                   "attention_implementation": case["attention"], "total_updates": TRAIN_SIZE // case["batch_size"],
                   "timing_warmup_samples": manifest["warmup_samples"]})
        write_json(attempt / "hardware.json", {
            "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
            "total_memory_gib": torch.cuda.get_device_properties(0).total_memory / GIB,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "sdpa_flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "sdpa_mem_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "sdpa_math_enabled": torch.backends.cuda.math_sdp_enabled()})
        torch.cuda.synchronize()
        loading = time.perf_counter()
        model = load_model(manifest["model"]["id"], revision=manifest["model"]["revision"],
                           attention=case["attention"], training=True)
        torch.cuda.synchronize()
        result["model_load_seconds"] = time.perf_counter() - loading
        actual = model.config._attn_implementation
        if actual != case["attention"]:
            raise RuntimeError(f"Unexpected attention implementation: {actual}")
        result["loaded_attention"] = actual
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=config["learning_rate"],
                                      betas=tuple(config["betas"]), eps=config["adam_epsilon"],
                                      weight_decay=config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda completed: lr_for_update(completed + 1, 1.0, config["warmup_steps"]))
        result["setup_seconds"] = time.perf_counter() - started
        result["metrics"] = train_epoch(model, optimizer, scheduler, tokenizer, rows, encoded, order,
                                        case, config, attempt, manifest["warmup_samples"])
        write_json(attempt / "training_result.json", dict(result, status="training_complete"))
        result["save_weights_seconds"] = 0.0
        if manifest["save_weights"]:
            from recipe.checkpoints import save_checkpoint
            saving = time.perf_counter()
            save_checkpoint(attempt / "weights", model, tokenizer, TRAIN_SIZE // case["batch_size"],
                            digest([manifest["identity"], case_name]))
            result["save_weights_seconds"] = time.perf_counter() - saving
            result["weights"] = str(attempt / "weights")
        if manifest["kernel_probe"]:
            try:
                result["attention_probe"] = probe_attention(model, optimizer, tokenizer,
                    [encoded[i] for i in order[:case["batch_size"]]], config)
            except Exception as exc:
                result["attention_probe"] = {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
                print(f"Kernel probe failed (training timings preserved): {exc}", flush=True)
            write_json(attempt / "attention_probe.json", result["attention_probe"])
        else:
            result["attention_probe"] = {"status": "disabled"}
        result["status"] = "complete"
    except BaseException as exc:
        result.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        print(result["traceback"], file=sys.stderr, flush=True)
    result.update(finished_at=now(), worker_wall_seconds=time.perf_counter() - started)
    write_json(attempt / "result.json", result)
    return 0 if result["status"] == "complete" else 1


def case_result(out, case_name, identity):
    results = []
    for path in sorted((out / case_name).glob("attempt-*/result.json")):
        result = read_json(path)
        if result["identity"] != identity or result["case"] != case_name:
            raise ValueError(f"Benchmark result identity mismatch: {path}")
        results.append(result)
    return next((r for r in results if r["status"] == "complete"), results[-1] if results else None)


def write_summary(out, manifest):
    results = [case_result(out, c["name"], manifest["identity"]) or {**c, "case": c["name"], "status": "pending"}
               for c in CASES]
    baseline = results[0].get("metrics", {}).get("epoch_wall_seconds") if results[0]["status"] == "complete" else None
    table = []
    for r in results:
        metrics = r.get("metrics", {})
        wall = metrics.get("epoch_wall_seconds")
        steady = metrics.get("after_timing_warmup", {})
        probe = r.get("attention_probe", {})
        table.append({"case": r["case"], "status": r["status"], "batch_size": r["batch_size"],
                      "attention": r["attention"], "updates": metrics.get("all_updates", {}).get("updates"),
                      "epoch_wall_seconds": wall,
                      "speedup_vs_b1_sdpa": baseline / wall if baseline and wall and r["status"] == "complete" else None,
                      "nonpadding_tokens_per_second": steady.get("nonpadding_tokens_per_second"),
                      "padding_fraction": metrics.get("all_updates", {}).get("padding_fraction"),
                      "peak_allocated_gib": metrics.get("peak_allocated_gib"),
                      "peak_reserved_gib": metrics.get("peak_reserved_gib"),
                      "model_load_seconds": r.get("model_load_seconds"),
                      "save_weights_seconds": r.get("save_weights_seconds"),
                      "probe_status": probe.get("status"),
                      "observed_backends": ";".join(probe.get("inferred_backends", [])),
                      "attempt": r.get("attempt"), "error": r.get("error")})
    write_json(out / "summary.json", {"updated_at": now(), "identity": manifest["identity"],
               "timing_warmup_samples": manifest["warmup_samples"], "table": table, "results": results,
               "timing_scope": "epoch_wall_seconds includes all 400 examples and log I/O; excludes loading, saving and profiling. "
                               "Tokens/s excludes the same initial warmup examples and log I/O for every case.",
               "memory_scope": "PyTorch allocator peaks during training, including optimizer state; excludes profiler. "
                               "Reserved includes allocator caching; neither metric is whole-device nvidia-smi memory."})
    temp = out / ("summary.csv.tmp-" + uuid.uuid4().hex)
    with temp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, out / "summary.csv")
    return table


def invoke_worker(command, attempt):
    with (attempt / "console.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            process.stdout.close()


def run_cases(out, manifest, code):
    interrupted = False
    write_summary(out, manifest)
    for case in CASES:
        existing = case_result(out, case["name"], manifest["identity"])
        if existing and existing["status"] == "complete":
            print(f"Skipping completed {case['name']}", flush=True)
            continue
        attempt = fresh_dir(out / case["name"] / ("attempt-" + now().replace(":", "-") + "-" + uuid.uuid4().hex[:8]))
        command = [sys.executable, "-B", "-u", str(code / "benchmark_train.py"), "--out", str(out),
                   "--worker", case["name"], "--attempt", str(attempt)]
        write_json(attempt / "command.json", command)
        print(f"Starting {case['name']} from Base; output: {attempt}", flush=True)
        try:
            returncode = invoke_worker(command, attempt)
        except KeyboardInterrupt:
            returncode, interrupted = 130, True
        path = attempt / "result.json"
        result = read_json(path) if path.exists() else {
            "case": case["name"], "batch_size": case["batch_size"], "attention": case["attention"],
            "identity": manifest["identity"], "attempt": str(attempt), "status": "running"}
        if result["status"] == "running":
            result.update(status="interrupted" if interrupted else "failed",
                          error=f"Worker exited without a final result; see console.log (exit {returncode})")
        result["returncode"] = returncode
        write_json(path, result)
        write_summary(out, manifest)
        if interrupted:
            break
    table = write_summary(out, manifest)
    print("\ncase           status       epoch(s)    speedup    tokens/s     peak GiB", flush=True)
    for r in table:
        def fmt(key):
            return f"{r[key]:.2f}" if r.get(key) is not None else "-"
        print(f"{r['case']:<14} {r['status']:<12} {fmt('epoch_wall_seconds'):>9} "
              f"{fmt('speedup_vs_b1_sdpa'):>10} {fmt('nonpadding_tokens_per_second'):>11} "
              f"{fmt('peak_allocated_gib'):>12}", flush=True)
    print(f"Results: {out / 'summary.csv'}", flush=True)
    return 130 if interrupted else 0 if all(r["status"] == "complete" for r in table) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Prepared data; required for a new benchmark")
    parser.add_argument("--out", required=True, help="New output directory under /data/repeat_sft, or an absolute path")
    parser.add_argument("--config", help="Training settings; defaults to configs/baseline.json")
    parser.add_argument("--seed", type=int, help="Training order/RNG seed; defaults to the config seed")
    parser.add_argument("--warmup-samples", type=int, help="Initial examples excluded ONLY from steady tokens/s (default 16)")
    parser.add_argument("--save-weights", action="store_true", default=None, help="Also save final weights (no optimizer); time separately")
    parser.add_argument("--no-kernel-probe", action="store_false", dest="kernel_probe", default=None,
                        help="Disable the separate post-epoch attention profiler probe")
    parser.add_argument("--resume", action="store_true", help="Skip completed cases; rerun incomplete cases from Base, retaining old attempts")
    parser.add_argument("--worker", choices=[c["name"] for c in CASES], help=argparse.SUPPRESS)
    parser.add_argument("--attempt", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Use one GPU/process; do not launch with torchrun")
    configure_runtime()
    out = artifact_path(args.out)
    if args.worker:
        attempt = Path(args.attempt).resolve()
        if not attempt.is_relative_to(out / args.worker) or not attempt.is_dir():
            parser.error("Worker attempt must be an existing directory within its case")
        return worker(out, args.worker, attempt)
    if args.resume:
        if any(getattr(args, key) is not None for key in ("data", "config", "seed", "warmup_samples", "save_weights", "kernel_probe")):
            parser.error("--resume uses frozen settings; pass only --out and --resume")
        manifest, code = verify_run(out)
    else:
        out, manifest = prepare_run(args)
        code = out / "code"
    return run_cases(out, manifest, code)


if __name__ == "__main__":
    sys.exit(main())
