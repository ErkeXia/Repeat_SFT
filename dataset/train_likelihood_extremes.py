"""One command: train the highest/lowest-likelihood eligible 100 examples, 16 epochs each."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recipe.adapters import adapter_spec
from recipe.checkpoints import latest_recovery
from recipe.common import append_jsonl, file_hash, now, read_json, recover_jsonl, write_json
from recipe.paths import artifact_path, configure_runtime
from recipe.pipeline import Pipeline
from recipe.storage import GIB
from dataset.likelihood_training import (EPOCHS, GROUPS, initialize_bundle, plan_experiment,
                                         tree_hashes, verify_bundle)


def check_run(out, group, plan):
    """Bind a resumable run to the selected text, seed, Base and frozen trainer."""
    run = out / group
    metadata = read_json(run / "run.json")
    bundle = out / "bundle"
    prepared = bundle / "prepared" / group
    if (metadata["config"] != plan["config"] or metadata["model"] != plan["model"]
            or metadata["prepared_manifest_sha256"] != file_hash(prepared / "manifest.json")
            or metadata["max_length"] != plan["max_length"]
            or metadata["code_sha256"] != read_json(bundle / "code" / "manifest.json")["code_sha256"]):
        raise ValueError(f"Existing {group} run does not belong to this experiment")
    for name, expected in metadata["data_files"].items():
        if file_hash(run / "data" / name) != expected:
            raise ValueError(f"Run input changed: {group}/{name}")
    if file_hash(run / "data" / "train.jsonl") != file_hash(prepared / "train.jsonl"):
        raise ValueError("Selected training samples changed")
    if tree_hashes(run / "tokenizer") != metadata["tokenizer_files"]:
        raise ValueError("Run tokenizer changed")
    return metadata


def is_complete(out, group, plan):
    run = out / group
    if not (run / "run.json").exists():
        return False
    metadata = check_run(out, group, plan)
    final = run / "checkpoints" / "epoch-016"
    if not (final / "complete.json").exists():
        return False
    # A successful run must retain EVERY requested milestone, not just epoch 16.
    for epoch in EPOCHS:
        path = run / "checkpoints" / f"epoch-{epoch:03d}"
        marker = read_json(path / "complete.json")
        if (marker["step"] != epoch * plan["config"]["expected_train_size"]
                or marker["config_sha256"] != metadata["config_sha256"]
                or not marker["has_training_state"] or not (path / "training_state.pt").is_file()):
            raise ValueError(f"Missing/wrong complete training checkpoint: {path}")
        adapter_spec(path, finetuning=plan["config"]["finetuning"], base_model=plan["model"])
    return True


def archive_uninitialized(out, group, plan):
    """If killed before epoch-0 recovery existed, preserve that attempt and restart Base."""
    run = out / group
    owned = any(r.get("group") == group and r.get("job_identity") == plan["identity"]
                for r in recover_jsonl(out / "commands.jsonl"))
    if not owned or run.is_symlink() or run.resolve().parent != out.resolve():
        raise ValueError(f"Unrecognized existing directory, will not move or overwrite: {run}")
    destination = out / "incomplete_initializations" / (group + "-" + uuid.uuid4().hex)
    destination.parent.mkdir(exist_ok=True)
    if destination.resolve().parent.parent != out.resolve():
        raise ValueError("Unsafe initialization archive path")
    run.rename(destination)
    append_jsonl(out / "archives.jsonl", {"timestamp": now(), "group": group, "path": str(destination),
                                          "reason": "interrupted before first complete recovery"})
    print(f"Preserved incomplete initialization: {destination}; restarting this group from Base", flush=True)


def command_for(out, group, plan):
    run, code = out / group, out / "bundle" / "code"
    command = [sys.executable, "-B", "-u", str(code / "train.py"), "--run", str(run)]
    if run.exists():
        if (run / "run.json").exists():
            metadata = check_run(out, group, plan)
            try:
                recovery = latest_recovery(run)
            except FileNotFoundError:
                recovery = None
            if recovery is not None:
                marker = read_json(recovery / "complete.json")
                if (marker["config_sha256"] != metadata["config_sha256"]
                        or not 0 <= marker["step"] < plan["updates_per_group"]):
                    raise ValueError("Recovery checkpoint belongs to a different or already completed run")
                return command + ["--resume", str(recovery)]
        archive_uninitialized(out, group, plan)
    # No adapter path or recovery from the other group: each fresh run starts Base.
    return command + ["--data", str(out / "bundle" / "prepared" / group),
                      "--config", str(out / "bundle" / "config.json")]


def execute(out, plan, pipeline):
    recover_jsonl(out / "commands.jsonl")  # Retain complete entries after a torn last write.
    finished = []
    for group in GROUPS:
        pipeline.check_stop()
        if not is_complete(out, group, plan):
            command = command_for(out, group, plan)
            append_jsonl(out / "commands.jsonl", {"timestamp": now(), "job_identity": plan["identity"],
                                                  "group": group, "command": command})
            write_json(out / "progress.json", {"status": "running", "active_group": group,
                                                "completed_groups": finished, "job_identity": plan["identity"]})
            print(f"TRAIN {group}: 100 examples x 16 epochs; output={out / group}", flush=True)
            pipeline.invoke(command, out / "bundle" / "code")
            if not is_complete(out, group, plan):
                # SIGTERM sent directly to train.py exits successfully after saving.
                # Respect that pause instead of starting the second experiment.
                write_json(out / "progress.json", {"status": "paused", "active_group": group,
                                                    "completed_groups": finished, "job_identity": plan["identity"]})
                print(f"Paused in {group}; repeat the launcher command to resume", flush=True)
                return False
        else:
            print(f"SKIP {group}: all requested checkpoints are complete", flush=True)
        finished.append(group)
    write_json(out / "complete.json", {"timestamp": now(), "job_identity": plan["identity"],
                "groups": finished, "epochs": EPOCHS,
                "checkpoint_markers": {f"{g}/{e}": file_hash(out / g / "checkpoints" / f"epoch-{e:03d}" / "complete.json")
                                       for g in GROUPS for e in EPOCHS}})
    write_json(out / "progress.json", {"status": "complete", "completed_groups": finished,
                                       "job_identity": plan["identity"]})
    print(f"Both LoRA experiments finished: {out}", flush=True)
    return True


def show_plan(plan):
    storage = plan["storage"]
    print(f"Selection: highest/lowest 100 from {plan['eligible_pool_size']} original eligible samples", flush=True)
    for group in GROUPS:
        stats = plan["groups"][group]
        print(f"  {group}: mean initial NLL={stats['mean_initial_nll']:.6f}; "
              f"mean assistant tokens={stats['mean_supervised_tokens']:.1f}; "
              f"max full length={stats['max_total_tokens']}", flush=True)
    print(f"Each run: 1,600 updates; checkpoints={EPOCHS}\n"
          f"Adapter parameters: {storage['lora_parameters']:,}; "
          f"FP32 adapter weights={storage['adapter_weight_bytes']/1024**2:.1f} MiB/checkpoint\n"
          f"Estimated retained checkpoint storage: {storage['typical_retained_bytes']/GIB:.2f} GiB\n"
          f"Conservative initial free-space check: {(storage['required_bytes']+storage['reserve_bytes'])/GIB:.2f} GiB",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", default="dataset/math-cot-20k/likelihood/base-v1")
    parser.add_argument("--data", default="artifacts/data-s42")
    parser.add_argument("--config", default=str(ROOT / "configs" / "baseline.json"))
    parser.add_argument("--out", default="runs/likelihood-100x16-s42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plan-only", action="store_true", help="Read-only selection/space estimate, no weights loaded")
    parser.add_argument("--prepare-only", action="store_true", help="Freeze both subsets/config/code without training")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = Path(args.worker).resolve() if args.worker else artifact_path(args.out)
    if args.worker:
        with Pipeline(out, inherited=True) as pipeline:
            try:
                execute(out, verify_bundle(out), pipeline)
            except BaseException as exc:
                progress = read_json(out / "progress.json") if (out / "progress.json").exists() else {}
                progress.update(status="paused" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                                stopped=now(), error=str(exc))
                write_json(out / "progress.json", progress)
                raise
        return
    request = {"scores": str(artifact_path(args.scores)), "data": str(artifact_path(args.data)),
               "config": str(Path(args.config).resolve()), "seed": args.seed}
    if args.plan_only:
        if (out / "bundle").exists():
            plan = verify_bundle(out)
        else:
            plan, _, _, _ = plan_experiment(Path(request["scores"]), Path(request["data"]),
                                             Path(request["config"]), args.seed)
        show_plan(plan)
        return
    configure_runtime()
    with Pipeline(out) as pipeline:
        if (out / "bundle").exists():
            plan = verify_bundle(out)
            if request != plan["request"]:
                raise ValueError("Different selection inputs/seed/config path; repeat the original command or choose a new --out")
        else:
            plan, groups, inspections, prepared = plan_experiment(
                Path(request["scores"]), Path(request["data"]), Path(request["config"]), args.seed)
            pipeline.check_stop()
            plan = initialize_bundle(out, plan, groups, inspections, prepared)
        show_plan(plan)
        pipeline.check_stop()
        if args.prepare_only:
            print(f"Frozen selections ready in {out / 'bundle' / 'prepared'}", flush=True)
            return
        # Using the frozen launcher and trainer also makes later git pulls harmless to this job.
        code = out / "bundle" / "code"
        pipeline.invoke([sys.executable, "-B", "-u", code / "dataset" / "train_likelihood_extremes.py",
                         "--worker", out], code)


if __name__ == "__main__":
    main()
