"""SERVER ONLY: real Base smoke run, forced process death, resume, and evaluation."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import subprocess
import sys
import uuid

from recipe.common import (ROOT, append_jsonl, file_hash, fresh_dir, iter_jsonl, now, read_json,
                           recover_jsonl, snapshot_code, write_json, write_jsonl)
from recipe.checkpoints import latest_recovery
from recipe.paths import artifact_path, configure_runtime
from recipe.resume import original_code, verified_code, verify_run_inputs
from recipe.storage import GIB, require_free_space


def canonical_logs(run):
    records = {}
    for attempt in iter_jsonl(Path(run) / "attempts.jsonl"):
        # A new attempt supersedes all later uncommitted events from the old one.
        records = {k: v for k, v in records.items() if k <= attempt["start_step"]}
        from recipe.common import recover_jsonl
        for row in recover_jsonl(Path(run) / "logs" / (attempt["attempt"] + ".jsonl")):
            records[row["step"]] = row
    return records


def compare_weights(a, b):
    import torch
    from safetensors import safe_open

    def weight_map(path):
        result = {}
        for shard in Path(path).glob("*.safetensors"):
            with safe_open(shard, framework="pt", device="cpu") as f:
                for key in f.keys():
                    result[key] = shard
        return result

    left, right = weight_map(a), weight_map(b)
    if not left or set(left) != set(right):
        raise AssertionError("Saved model parameter keys differ")
    max_error, unequal = 0.0, []
    # Compare one parameter at a time, avoiding two entire 4B models in CPU RAM.
    for key in left:
        with safe_open(left[key], framework="pt", device="cpu") as f, safe_open(right[key], framework="pt", device="cpu") as g:
            x, y = f.get_tensor(key), g.get_tensor(key)
            if not torch.equal(x, y):
                error = (x.float() - y.float()).abs().max().item()
                max_error = max(max_error, error)
                unequal.append(key)
    return {"parameters_compared": len(left), "max_absolute_error": max_error,
            "unequal_parameter_names": unequal, "bitwise_equal": not unequal}


def frozen_smoke(out, data):
    """Validate frozen inputs and use the same source for both trajectories.

    Legacy smoke runs have no top-level source snapshot. Recover their original
    source from the training attempts rather than changing the recorded hashes.
    """
    out, data = Path(out), Path(data)
    config = read_json(out / "smoke_train.json")
    if (config["epochs"] != 4 or config["expected_train_size"] != 8
            or config["checkpoint_epochs"] != [0, 4] or config["recovery_every_steps"] != 8):
        raise ValueError("Unexpected smoke protocol; expected 8 examples and 32 updates")
    subset_hash = file_hash(out / "subset.jsonl")
    prepared_hash = file_hash(data / "manifest.json")
    records = []
    for name in ("continuous", "resumed"):
        run = out / name
        if run.exists():
            if not (run / "run.json").exists():
                raise ValueError(f"Incomplete initialization at {run}; original files retained. Use a new smoke output.")
            metadata = verify_run_inputs(run)
            if (metadata["config"] != config or metadata["prepared_manifest_sha256"] != prepared_hash
                    or file_hash(run / "data" / "train.jsonl") != subset_hash):
                raise ValueError(f"Frozen smoke inputs differ: {run}")
            records.append((run, metadata))
    if records and any(r[1]["config_sha256"] != records[0][1]["config_sha256"]
                       or r[1]["code_sha256"] != records[0][1]["code_sha256"] for r in records):
        raise ValueError("The two smoke trajectories used different code or experiment identities")
    manifest_path = out / "smoke_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        for name, expected in manifest["files"].items():
            if file_hash(out / name) != expected:
                raise ValueError(f"Frozen smoke input changed: {name}")
        if manifest["prepared_manifest_sha256"] != prepared_hash:
            raise ValueError("Prepared dataset differs from the original smoke run")
        if records and records[0][1]["code_sha256"] != manifest["code_sha256"]:
            raise ValueError("Smoke and training code identities differ")
        return verified_code(out / "code", manifest["code_sha256"])
    if not records:
        raise ValueError("Legacy smoke has no recoverable training metadata or source snapshot")
    return original_code(records[0][0])


def crash_evidence(out):
    """Recover the intentional interruption evidence, including legacy runs."""
    out = Path(out)
    run = out / "resumed"
    if not (run / "attempts.jsonl").exists():
        return None
    commands = read_json(out / "commands.json")
    planned = any("--simulate-crash-at" in c and "--run" in c
                  and c[c.index("--simulate-crash-at") + 1] == "13"
                  and Path(c[c.index("--run") + 1]).name == "resumed" for c in commands)
    if not planned:
        return None
    attempts = list(iter_jsonl(run / "attempts.jsonl"))
    for i, attempt in enumerate(attempts):
        rows = recover_jsonl(run / "logs" / (attempt["attempt"] + ".jsonl"))
        if attempt["start_step"] < 13 and [r["step"] for r in rows] == list(range(attempt["start_step"] + 1, 14)):
            if i + 1 < len(attempts) and attempts[i + 1]["start_step"] != 8:
                raise ValueError("The intentional interruption was not first resumed from step 8")
            return {"crash_attempt": attempt["attempt"], "crash_step": 13,
                    "restored_step": 8, "continuation_started": i + 1 < len(attempts)}
    return None


def continuation(run):
    run = Path(run)
    checkpoint = latest_recovery(run)
    marker = read_json(checkpoint / "complete.json")
    identity = read_json(run / "run.json")["config_sha256"]
    if marker["config_sha256"] != identity or not 0 <= marker["step"] <= 32:
        raise ValueError(f"Invalid smoke recovery point: {checkpoint}")
    return checkpoint, marker["step"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--resume", action="store_true", help="Continue an existing smoke run using its frozen inputs and original code")
    p.add_argument("--config", help="Fresh run only; defaults to configs/baseline.json (BF16 LoRA)")
    p.add_argument("--min-free-gib", type=float, default=10, help="Output-space precheck, excluding Base download; default 10 GiB")
    p.add_argument("--generation-max-new-tokens", type=int,
                   help="Smoke only; default 512, formal generation.json uses 8192")
    args = p.parse_args()
    if not math.isfinite(args.min_free_gib) or args.min_free_gib < 0:
        p.error("--min-free-gib must be finite and nonnegative")
    if args.generation_max_new_tokens is not None and args.generation_max_new_tokens < 1:
        p.error("--generation-max-new-tokens must be positive")
    configure_runtime()
    data, out = artifact_path(args.data), artifact_path(args.out)
    storage = require_free_space(out, args.min_free_gib * GIB)
    print(f"Smoke directory: {out}; free space {storage['free_bytes'] / GIB:.1f} GiB", flush=True)
    if args.resume:
        if args.config:
            p.error("--resume uses the frozen smoke config; do not pass --config")
        if not out.is_dir():
            raise FileNotFoundError(f"Smoke output missing: {out}. Migrate the existing output to the volume first.")
        code_root = frozen_smoke(out, data)
        generation = read_json(out / "smoke_generation.json")
        if args.generation_max_new_tokens is not None and args.generation_max_new_tokens != generation["max_new_tokens"]:
            raise ValueError("Resume must use the frozen generation settings")
        commands = read_json(out / "commands.json") if (out / "commands.json").exists() else []
        print(f"Resuming with original verified source: {code_root}", flush=True)
    else:
        out = fresh_dir(out)
        ordered = sorted(iter_jsonl(data / "train.jsonl"), key=lambda r: (r["n_tokens"], r["sample_id"]))
        if len(ordered) < 8:
            raise ValueError("Smoke test requires at least eight prepared train examples")
        # A short-only smoke can pass while the longest real solution OOMs.
        rows = ordered[:7] + ordered[-1:]
        write_jsonl(out / "subset.jsonl", rows)
        config = read_json(args.config or ROOT / "configs" / "baseline.json")
        from recipe.adapters import validate_finetuning
        validate_finetuning(config.get("finetuning"))
        config.update(epochs=4, expected_train_size=8, checkpoint_epochs=[0, 4], recovery_every_steps=8, keep_recovery=1)
        write_json(out / "smoke_train.json", config)
        generation = read_json(ROOT / "configs" / "generation.json")
        generation["max_new_tokens"] = args.generation_max_new_tokens or 512
        write_json(out / "smoke_generation.json", generation)
        code_hash = snapshot_code(out / "code")
        write_json(out / "smoke_manifest.json", {"code_sha256": code_hash,
                   "prepared_manifest_sha256": file_hash(data / "manifest.json"),
                   "files": {n: file_hash(out / n) for n in ("subset.jsonl", "smoke_train.json", "smoke_generation.json")}})
        code_root = out / "code"
        commands = []

    def invoke(script, *flags, expected=0):
        command = [sys.executable, "-B", str(code_root / script), *map(str, flags)]
        commands.append(command)
        write_json(out / "commands.json", commands)
        print("RUN", command, flush=True)
        result = subprocess.run(command, cwd=code_root)
        append_jsonl(out / "command_results.jsonl", {"command": command, "exit_code": result.returncode,
                                                     "expected_exit_code": expected, "timestamp": now()})
        if result.returncode != expected:
            raise RuntimeError(f"{script}: expected exit {expected}, got {result.returncode}")

    def train_stage(name, force_crash=False):
        run = out / name
        extra = ["--simulate-crash-at", 13] if force_crash else []
        expected = 86 if force_crash else 0
        if run.exists():
            source, step = continuation(run)
            if force_crash and step >= 13:
                raise ValueError("No verified forced interruption, but the resumed trajectory already passed step 13")
            if step == 32:
                if not (run / "checkpoints" / "epoch-004" / "complete.json").exists():
                    raise ValueError("Final smoke checkpoint missing")
                print(f"Skipping completed {name} training", flush=True)
                return
            invoke("train.py", "--run", run, "--resume", source, *extra, expected=expected)
        else:
            invoke("train.py", "--data", data, "--subset", out / "subset.jsonl",
                   "--config", out / "smoke_train.json", "--run", run, *extra, expected=expected)

    train_stage("continuous")
    evidence = crash_evidence(out)
    if evidence is None:
        train_stage("resumed", force_crash=True)
        evidence = crash_evidence(out)
        if evidence is None:
            raise AssertionError("Intentional interruption evidence is missing")
    if not evidence["continuation_started"]:
        _, restored = continuation(out / "resumed")
        if restored != 8:
            raise AssertionError(f"Expected recovery at step 8 after intentional crash, got {restored}")
    restored_step = evidence["restored_step"]
    write_json(out / "interruption.json", evidence)
    train_stage("resumed")
    left, right = canonical_logs(out / "continuous"), canonical_logs(out / "resumed")
    if set(left) != set(range(1, 33)) or set(right) != set(left):
        raise AssertionError("Missing/repeated update numbers")
    mismatches = []
    for step in left:
        for key in ("sample_id", "learning_rate", "loss", "grad_norm_before_clip"):
            if left[step][key] != right[step][key]:
                mismatches.append({"step": step, "field": key, "continuous": left[step][key], "resumed": right[step][key]})
    weights = compare_weights(out / "continuous" / "checkpoints" / "epoch-004",
                              out / "resumed" / "checkpoints" / "epoch-004")
    write_json(out / "resume_comparison.json", {"restored_step": restored_step, "log_mismatches": mismatches, **weights})
    if mismatches or not weights["bitwise_equal"]:
        raise AssertionError("Resume did not reproduce the continuous run. Inspect resume_comparison.json before long training.")
    invoke("evaluate_loss.py", "--run", out / "resumed", "--epochs", "0,4", "--limit-per-split", 8)
    invoke("generate.py", "--run", out / "resumed", "--epochs", "0,4", "--config", out / "smoke_generation.json", "--limit-per-task", 2)
    grades = out / "grades"
    if grades.exists():
        grades = out / ("grades-retry-" + uuid.uuid4().hex[:8])
    invoke("grade.py", "--responses", out / "resumed" / "evaluations" / "generation", "--out", grades)
    summaries = list(iter_jsonl(grades / "summary.jsonl"))
    if any(r["grading_errors"] for r in summaries):
        raise AssertionError("Smoke grader reported errors")
    write_json(out / "SUCCESS.json", {"updates_per_run": 32, "forced_crash_step": 13,
                                      "restored_step": restored_step, "weights_bitwise_equal": True,
                                      "subset_sha256": file_hash(out / "subset.jsonl"),
                                      "grading_output": str(grades),
                                      "inspection": str(out / "resumed" / "data" / "inspection.jsonl"),
                                      "note": "Generation cap is smoke-only; inspect input/labels and loss before formal training."})
    print(f"Smoke test passed. Inspect {out}/SUCCESS.json and data/inspection.jsonl before the baseline.")


if __name__ == "__main__":
    main()
