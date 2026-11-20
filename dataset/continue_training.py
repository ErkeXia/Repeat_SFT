"""SERVER: extend a completed LoRA run; repeat the same command to resume."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recipe.checkpoints import latest_recovery
from recipe.common import append_jsonl, environment, file_hash, now, read_json, recover_jsonl, write_json
from recipe.extension import BOOTSTRAP, initialize, plan_extension, verify_extension
from recipe.paths import artifact_path, configure_runtime
from recipe.pipeline import Pipeline
from recipe.replay import validate_checkpoint
from recipe.storage import GIB


def check_request(plan, source, args):
    if (str(source) != plan["source_run"] or args.additional_epochs != plan["additional_epochs"]
            or args.checkpoint_every != plan["checkpoint_every"]):
        raise ValueError("Continuation inputs differ; repeat the original command or choose a new --out")


def execute(out, pipeline, stop_after=None):
    plan, meta = verify_extension(out)
    control = out / "continuation"
    selected = latest_recovery(out)
    step = validate_checkpoint(selected, meta)["step"]
    if not plan["start_step"] <= step <= plan["total_steps"]:
        raise ValueError("Recovery step is outside the continuation budget")
    if stop_after is not None and not step < stop_after <= plan["total_steps"]:
        raise ValueError("--stop-after must exceed the restored global step and not exceed the final step")
    # Every crossed milestone must already exist; do not hide missing results.
    for epoch in plan["target_epochs"]:
        if epoch * plan["epoch_size"] <= step:
            validate_checkpoint(out / "checkpoints" / f"epoch-{epoch:03d}", meta, epoch * plan["epoch_size"])
    if step < plan["total_steps"]:
        recover_jsonl(control / "commands.jsonl")
        append_jsonl(control / "environments.jsonl", environment())
        code = out / BOOTSTRAP
        command = [sys.executable, "-B", "-u", code / "train.py", "--run", out, "--resume", selected]
        if stop_after is not None:
            command += ["--stop-after", str(stop_after)]
        append_jsonl(control / "commands.jsonl", {"timestamp": now(), "start_step": step,
                                                  "command": list(map(str, command))})
        write_json(control / "progress.json", {"status": "running", "step": step, "total_steps": plan["total_steps"]})
        print(f"CONTINUE step={step}/{plan['total_steps']}, epoch={step / plan['epoch_size']:g}; "
              "restoring optimizer, scheduler, RNG and shuffle cursor", flush=True)
        pipeline.invoke(command, code)
        step = validate_checkpoint(latest_recovery(out), meta)["step"]
    complete = step == plan["total_steps"]
    if complete:
        markers = {}
        for epoch in plan["target_epochs"]:
            path = out / "checkpoints" / f"epoch-{epoch:03d}"
            validate_checkpoint(path, meta, epoch * plan["epoch_size"])
            markers[str(epoch)] = file_hash(path / "complete.json")
        result = {"plan_sha256": file_hash(control / "plan.json"), "step": step,
                  "epochs": plan["target_epochs"], "checkpoint_markers": markers}
        if (control / "complete.json").exists():
            if read_json(control / "complete.json") != result:
                raise ValueError("Completed continuation artifacts changed")
        else:
            write_json(control / "complete.json", result)
    write_json(control / "progress.json", {"status": "complete" if complete else "paused",
                                           "step": step, "total_steps": plan["total_steps"]})
    print(f"{'Complete' if complete else 'Paused; repeat the same command without --stop-after'}: {out}", flush=True)
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", help="Completed source run (the directory containing run.json)")
    parser.add_argument("--out", help="New, separate training run directory")
    parser.add_argument("--additional-epochs", type=int, default=128)
    parser.add_argument("--checkpoint-every", type=int, default=16)
    parser.add_argument("--plan-only", action="store_true", help="Read-only plan and storage estimate; no torch/model load")
    parser.add_argument("--prepare-only", action="store_true", help="Copy/validate full state on CPU; do not train")
    parser.add_argument("--stop-after", type=int, help="Pause at this global update, e.g. 1290 after starting at 1280")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.worker and (not args.source_run or not args.out):
        parser.error("--source-run and --out are required")
    out = Path(args.worker).resolve() if args.worker else artifact_path(args.out)
    if not args.worker:
        source = artifact_path(args.source_run)
        if out.exists():
            plan, _ = verify_extension(out)
            check_request(plan, source, args)
        else:
            plan = plan_extension(source, out, args.additional_epochs, args.checkpoint_every)
        if args.stop_after is not None and not plan["start_step"] < args.stop_after <= plan["total_steps"]:
            parser.error("--stop-after must be a global update after the anchor and within the extended budget")
        print(f"Epoch {plan['start_epoch']} -> {plan['end_epoch']}; {plan['epoch_size']} samples; "
              f"{plan['additional_updates']} new updates; save {plan['target_epochs']}\n"
              f"Output: {out}\nEstimated free-space budget: "
              f"{(plan['storage']['required_bytes'] + plan['storage']['reserve_bytes']) / GIB:.2f} GiB\n"
              f"{plan['storage']['note']}", flush=True)
        if args.plan_only:
            return
    if os.name != "posix":
        raise RuntimeError("Run continuation on the Linux VM; --plan-only and offline tests need no GPU")
    configure_runtime()
    # Outside the run, so initialization can publish the whole run atomically.
    lock_dir = out.parent / ("." + out.name + ".continuation-lock")
    with Pipeline(lock_dir, inherited=bool(args.worker)) as pipeline:
        try:
            if args.worker:
                execute(out, pipeline, args.stop_after)
                return
            if not out.exists():
                initialize(plan)
            frozen_plan, _ = verify_extension(out)
            check_request(frozen_plan, source, args)
            if args.prepare_only:
                print("Full continuation state ready; repeat without --prepare-only to train", flush=True)
                return
            frozen = out / "continuation/code"
            command = [sys.executable, "-B", "-u", frozen / "dataset/continue_training.py", "--worker", out]
            if args.stop_after is not None:
                command += ["--stop-after", str(args.stop_after)]
            pipeline.invoke(command, frozen)
        except BaseException as exc:
            path = out / "continuation/progress.json"
            if path.exists() and not (out / "continuation/complete.json").exists():
                progress = read_json(path)
                progress.update(status="paused" if isinstance(exc, (SystemExit, KeyboardInterrupt)) else "failed",
                                error=str(exc), timestamp=now())
                write_json(path, progress)
            raise


if __name__ == "__main__":
    main()
