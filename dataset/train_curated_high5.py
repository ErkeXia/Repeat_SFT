"""SERVER: fresh Base LoRA on the five frozen manual high-group answers, 64 epochs."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import os
from pathlib import Path
import re
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recipe.adapters import validate_finetuning
from recipe.common import (append_jsonl, digest, file_hash, iter_jsonl, now, read_json,
                           recover_jsonl, snapshot_code, validate_prepared, write_json, write_jsonl)
from recipe.data import OverlapIndex
from recipe.paths import artifact_path, configure_runtime
from recipe.pipeline import Pipeline
from recipe.replay import validate_checkpoint
from dataset.likelihood_training import tree_hashes, verify_bundle
from dataset.train_likelihood_extremes import check_run, command_for

SIZE = 5
EPOCHS = [0, 16, 32, 48, 64]
PROTOCOL = "curated-historical-high-five-64-v1"
EXTRA_CODE = ("__init__.py", "train_curated_high5.py", "train_likelihood_extremes.py",
              "likelihood_training.py", "likelihood_utils.py", "CURATED_HIGH5.md")


def training_config(original, seed):
    config = deepcopy(original)
    validate_finetuning(config["finetuning"])
    if config["finetuning"]["base_quantization"] != "none":
        raise ValueError("This experiment requires the same BF16 Base and LoRA protocol")
    config.update(seed=seed, expected_train_size=SIZE, epochs=64,
                  checkpoint_epochs=EPOCHS.copy(), recovery_every_steps=SIZE, keep_recovery=2)
    return config


def selected_rows(rows):
    if (len(rows) != 10 or len({r["sample_id"] for r in rows}) != 10
            or Counter(r["curation"]["historical_group"] for r in rows) != {"high": 5, "low": 5}):
        raise ValueError("Source must contain the original ten curated answers, five high and five low")
    selected = [r for r in rows if r["curation"]["historical_group"] == "high"]
    if len({r["question_id"] for r in selected}) != SIZE or len({r["row_index"] for r in selected}) != SIZE:
        raise ValueError("Duplicate high-group questions/row indices")
    return selected


def prepare(out, request, check_stop=lambda: None):
    source = Path(request["source_experiment"])
    if out.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(out.resolve()):
        raise ValueError("Use a separate experiment directory, not inside the source experiment")
    if any(p.name != "pipeline.lock" and not p.name.startswith(".initializing-") for p in out.iterdir()):
        raise ValueError("Output is not an empty high-five experiment; choose a new --out")
    parent = verify_bundle(source)
    if parent["protocol"] != "curated-ten-base-likelihood-lora-128-v1":
        raise ValueError("Select the frozen ten-answer manual experiment as --source-experiment")
    data = source / "bundle/prepared/train"
    original = validate_prepared(data)
    if (parent["max_length"] != 10240 or original["arguments"]["max_length"] != 10240
            or original["model"] != parent["model"] or parent["model"]["id"] != "Qwen/Qwen3-4B-Base"
            or not re.fullmatch(r"[0-9a-f]{40}", parent["model"]["revision"])):
        raise ValueError("Expected the original pinned Qwen Base and 10,240-token protocol")
    rows = selected_rows(list(iter_jsonl(data / "train.jsonl")))
    config = training_config(parent["config"], request["seed"])
    index_by_id = {r["sample_id"]: r for r in iter_jsonl(source / "bundle/index.jsonl")}
    inspection_by_id = {r["sample_id"]: r for r in iter_jsonl(data / "inspection.jsonl")}
    references = [r for name in ("holdout.jsonl", "panel.jsonl", "overlap_references.jsonl") for r in iter_jsonl(data / name)]
    overlap = OverlapIndex(references)
    index, inspections = [], []
    for row in rows:
        check_stop()
        description = index_by_id[row["sample_id"]]
        if (description["question_id"] != row["question_id"] or description["row_index"] != row["row_index"]
                or not 0 < description["n_tokens"] <= description["total_tokens"] <= 10240):
            raise ValueError("Frozen answer token identity/length mismatch")
        if overlap.match(row):
            raise ValueError(f"Training/evaluation overlap: {row['sample_id']}")
        index.append(description)
        inspections.append(inspection_by_id[row["sample_id"]])
    # Copy the exact frozen texts/tokenizer; train.py encodes and checks them again.
    stage = out / (".initializing-" + uuid.uuid4().hex)
    stage.mkdir()
    prepared = stage / "prepared/train"
    prepared.mkdir(parents=True)
    for name in ("holdout.jsonl", "panel.jsonl", "overlap_references.jsonl", "chat_template.jinja"):
        shutil.copy2(data / name, prepared / name)
        if file_hash(prepared / name) != original["files"][name]:
            raise ValueError(f"Source evaluation input changed during copying: {name}")
    shutil.copytree(data / "tokenizer", prepared / "tokenizer")
    if {"tokenizer/" + k: v for k, v in tree_hashes(prepared / "tokenizer").items()} != {
            k: v for k, v in original["files"].items() if k.startswith("tokenizer/")}:
        raise ValueError("Source tokenizer changed during copying")
    write_jsonl(prepared / "train.jsonl", rows)
    write_jsonl(prepared / "inspection.jsonl", inspections)
    write_jsonl(stage / "index.jsonl", index)
    selection = {"historical_group": "high", "samples": SIZE,
                 "rule": "The five manually edited answers whose original training group was high; no reranking",
                 "source_experiment": str(source), "source_bundle_identity": parent["identity"],
                 "source_train_sha256": original["files"]["train.jsonl"],
                 "selected": [{k: row[k] for k in ("row_index", "sample_id", "question_id", "curation")} for row in rows]}
    write_json(stage / "selection.json", selection)
    write_json(stage / "source_plan.json", parent)
    write_json(stage / "config.json", config)
    write_json(prepared / "manifest.json", {"format_version": 1,
               "arguments": {**original["arguments"], "train_size": SIZE},
               "model": parent["model"], "sources": original.get("sources", {}),
               "parent_manifest_sha256": file_hash(data / "manifest.json"), "selection": selection,
               "curation": {"edited": True, "historical_groups": {"high": 5, "low": 0}},
               "stats": {**original.get("stats", {}), "train": SIZE}, "files": tree_hashes(prepared)})
    snapshot_code(stage / "code")
    (stage / "code/dataset").mkdir()
    for name in EXTRA_CODE:
        shutil.copy2(ROOT / "dataset" / name, stage / "code/dataset" / name)
    plan = {"protocol": PROTOCOL, "created_at": now(), "request": request, "model": parent["model"],
            "config": config, "max_length": 10240, "updates_per_group": SIZE * config["epochs"],
            "selection": selection, "initialization": "fresh pinned Base and newly initialized LoRA; no prior adapter",
            "continuation": "Use dataset/continue_training.py --source-run OUT/train after the final epoch",
            "files": tree_hashes(stage)}
    plan["identity"] = digest(plan)
    write_json(stage / "plan.json", plan)
    check_stop()
    stage.rename(out / "bundle")
    return verify_bundle(out)


def training_complete(out, plan):
    run = out / "train"
    if not (run / "run.json").exists():
        return False
    metadata = check_run(out, "train", plan)
    final = run / "checkpoints" / f"epoch-{plan['config']['epochs']:03d}"
    if not (final / "complete.json").exists():
        return False
    for epoch in plan["config"]["checkpoint_epochs"]:
        validate_checkpoint(run / "checkpoints" / f"epoch-{epoch:03d}", metadata, epoch * SIZE)
    return True


def execute(out, plan, pipeline, stop_after=None):
    recover_jsonl(out / "commands.jsonl")
    pipeline.check_stop()
    if not training_complete(out, plan):
        command = command_for(out, "train", plan)
        if stop_after is not None:
            command += ["--stop-after", str(stop_after)]
        append_jsonl(out / "commands.jsonl", {"timestamp": now(), "job_identity": plan["identity"],
                     "group": "train", "command": list(map(str, command))})
        write_json(out / "progress.json", {"status": "running", "job_identity": plan["identity"], "total_steps": 320})
        pipeline.invoke(command, out / "bundle/code")
        if not training_complete(out, plan):
            write_json(out / "progress.json", {"status": "paused", "job_identity": plan["identity"]})
            print("Paused; repeat the same command without --stop-after to resume", flush=True)
            return False
    result = {"job_identity": plan["identity"], "updates": 320, "epochs": EPOCHS,
              "checkpoint_markers": {str(e): file_hash(out / "train/checkpoints" / f"epoch-{e:03d}" / "complete.json") for e in EPOCHS}}
    if (out / "complete.json").exists() and read_json(out / "complete.json") != result:
        raise ValueError("Previously completed high-five checkpoint markers changed")
    write_json(out / "complete.json", result)
    write_json(out / "progress.json", {"status": "complete", "job_identity": plan["identity"], "total_steps": 320})
    print(f"Complete: {out / 'train'}; use continue_training.py to extend beyond epoch 64", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-experiment", default="runs/curated-10x128-s42",
                        help="Existing ten-answer manual experiment, without /train")
    parser.add_argument("--out", default="runs/curated-high5-64-s42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare-only", action="store_true", help="Freeze five answers/config/code without loading a model")
    parser.add_argument("--stop-after", type=int, help="Pause at an absolute update, e.g. 7; range 1..320")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stop_after is not None and not 1 <= args.stop_after <= 320:
        parser.error("--stop-after must be between 1 and 320")
    if os.name != "posix":
        raise RuntimeError("Run this training experiment on the Linux VM")
    configure_runtime()
    out = Path(args.worker).resolve() if args.worker else artifact_path(args.out)
    with Pipeline(out, inherited=bool(args.worker)) as pipeline:
        try:
            if args.worker:
                execute(out, verify_bundle(out), pipeline, args.stop_after)
                return
            request = {"source_experiment": str(artifact_path(args.source_experiment)), "seed": args.seed}
            if (out / "bundle").exists():
                plan = verify_bundle(out)
                if plan["protocol"] != PROTOCOL or plan["request"] != request:
                    raise ValueError("Different experiment/seed; repeat the original command or choose a new --out")
            else:
                plan = prepare(out, request, pipeline.check_stop)
            config = plan["config"]
            print(f"5 manual high-group answers x 64 epochs = 320 updates; checkpoints={EPOCHS}\n"
                  f"Rows: {[r['row_index'] for r in plan['selection']['selected']]}\n"
                  f"Fresh Base LoRA; lr={config['learning_rate']}; warmup={config['warmup_steps']} updates "
                  f"({config['warmup_steps'] / SIZE:g} epochs); run={out / 'train'}", flush=True)
            pipeline.check_stop()
            if args.prepare_only:
                print(f"Frozen five-answer supervision: {out / 'bundle/prepared/train/inspection.jsonl'}", flush=True)
                return
            code = out / "bundle/code"
            command = [sys.executable, "-B", "-u", code / "dataset/train_curated_high5.py", "--worker", out]
            if args.stop_after is not None:
                command += ["--stop-after", str(args.stop_after)]
            pipeline.invoke(command, code)
        except BaseException as exc:
            if not (out / "complete.json").exists():
                write_json(out / "progress.json", {"status": "paused" if isinstance(exc, (SystemExit, KeyboardInterrupt)) else "failed",
                                                   "error": str(exc), "timestamp": now()})
            raise


if __name__ == "__main__":
    main()
