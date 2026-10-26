"""SERVER ONLY: real Base smoke run, forced process death, resume, and evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from recipe.common import ROOT, file_hash, fresh_dir, iter_jsonl, read_json, write_json, write_jsonl


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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--generation-max-new-tokens", type=int, default=512,
                   help="Smoke only; formal generation.json remains 30000")
    args = p.parse_args()
    data = Path(args.data).resolve()
    out = fresh_dir(args.out)
    # Use genuine, short, already verified examples; never fabricate training data.
    rows = sorted(iter_jsonl(data / "train.jsonl"), key=lambda r: (r["n_tokens"], r["sample_id"]))[:8]
    if len(rows) != 8:
        raise ValueError("Smoke test requires at least eight prepared train examples")
    write_jsonl(out / "subset.jsonl", rows)
    config = read_json(ROOT / "configs" / "baseline.json")
    config.update(epochs=4, expected_train_size=8, checkpoint_epochs=[0, 4],
                  recovery_every_steps=8, keep_recovery=1)
    write_json(out / "smoke_train.json", config)
    generation = read_json(ROOT / "configs" / "generation.json")
    generation["max_new_tokens"] = args.generation_max_new_tokens
    write_json(out / "smoke_generation.json", generation)
    commands = []

    def invoke(script, *flags, expected=0):
        command = [sys.executable, "-B", str(ROOT / script), *map(str, flags)]
        commands.append(command)
        write_json(out / "commands.json", commands)
        print("RUN", command, flush=True)
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != expected:
            raise RuntimeError(f"{script}: expected exit {expected}, got {result.returncode}")

    common = ["--data", data, "--subset", out / "subset.jsonl", "--config", out / "smoke_train.json"]
    invoke("train.py", *common, "--run", out / "continuous")
    invoke("train.py", *common, "--run", out / "resumed", "--simulate-crash-at", 13, expected=86)
    # At crash step 13, latest durable state is step 8. Steps 9--13 must replay.
    from recipe.checkpoints import latest_recovery
    restored_step = read_json(latest_recovery(out / "resumed") / "complete.json")["step"]
    if restored_step != 8:
        raise AssertionError(f"Expected recovery at step 8, got {restored_step}")
    invoke("train.py", "--run", out / "resumed", "--resume")
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
    invoke("grade.py", "--responses", out / "resumed" / "evaluations" / "generation", "--out", out / "grades")
    summaries = list(iter_jsonl(out / "grades" / "summary.jsonl"))
    if any(r["grading_errors"] for r in summaries):
        raise AssertionError("Smoke grader reported errors")
    write_json(out / "SUCCESS.json", {"updates_per_run": 32, "forced_crash_step": 13,
                                      "restored_step": restored_step, "weights_bitwise_equal": True,
                                      "subset_sha256": file_hash(out / "subset.jsonl"),
                                      "inspection": str(out / "resumed" / "data" / "inspection.jsonl"),
                                      "note": "Generation cap is smoke-only; inspect input/labels and loss before formal training."})
    print(f"Smoke test passed. Inspect {out}/SUCCESS.json and data/inspection.jsonl before the baseline.")


if __name__ == "__main__":
    main()
