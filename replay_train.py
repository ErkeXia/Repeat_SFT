"""SERVER: recover missing epochs into a separate run using the original trainer."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import subprocess
import sys

from recipe.common import append_jsonl, environment, file_hash, now, read_json, write_json
from recipe.paths import artifact_path, configure_runtime
from recipe.replay import (DEFAULT_EPOCHS, continuation, initialize, parse_epochs, plan_replay,
                           publish_target, validate_checkpoint, verify_replay)
from recipe.storage import GIB

LOCK_ENV = "REPEAT_SFT_REPLAY_LOCK_FD"
_lock_fd = None
_child = None
_stop_signal = None


@contextmanager
def replay_lock(out):
    global _lock_fd
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / f".{out.name}.replay.lock").open("a+b") as lock:
        try:
            if os.name == "posix":
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                import msvcrt
                if lock.seek(0, 2) == 0:
                    lock.write(b"\0")
                    lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError(f"Replay already running: {out}") from exc
        previous = _lock_fd
        _lock_fd = lock.fileno() if os.name == "posix" else None
        try:
            yield
        finally:
            _lock_fd = previous
            # Close, do not explicitly unlock on POSIX: an orphaned child must
            # retain the lock after SIGKILL of its launcher.
            if os.name != "posix":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def stop(signum, _frame):
    global _stop_signal
    if _stop_signal is not None:
        return
    _stop_signal = signum
    print("Stop requested; waiting for the current update and recovery save. Resume with --out SAME_PATH --resume.", flush=True)
    if _child is not None and _child.poll() is None:
        _child.terminate()
    # Never wait inside a Python signal handler: the interrupted Popen.wait()
    # may already hold its non-reentrant waitpid lock. Let that wait finish.


def check_stop():
    if _stop_signal is not None:
        raise SystemExit(128 + _stop_signal)


def invoke(command, cwd):
    global _child
    check_stop()
    options = {}
    if os.name == "posix" and _lock_fd is not None:
        options = {"pass_fds": (_lock_fd,), "env": {**os.environ, LOCK_ENV: str(_lock_fd)}}
    _child = subprocess.Popen([str(x) for x in command], cwd=cwd, **options)
    try:
        if _stop_signal is not None:
            _child.terminate()
        code = _child.wait()
    finally:
        _child = None
    check_stop()
    if code:
        raise RuntimeError(f"Training stage exited {code}; resume the replay to continue")


def show_plan(plan):
    for segment in plan["segments"]:
        print(f"Original epoch {segment['anchor_epoch']} -> save {segment['target_epochs']}", flush=True)
    storage = plan["storage"]
    print(f"Total: {plan['training_updates']:,} updates; output={plan['out']}\n"
          f"New target checkpoints estimate: {storage['target_checkpoint_bytes_estimate'] / GIB:.2f} GiB\n"
          f"Free space budget including anchors/recovery/temp/reserve: "
          f"{(storage['required_bytes'] + storage['reserve_bytes']) / GIB:.2f} GiB\n"
          f"{storage['note']}", flush=True)


def execute(out):
    plan, metadata = verify_replay(out)
    code = out / "attempts" / "replay-bootstrap" / "code"
    completed = []
    markers = {}
    append_jsonl(out / "replay_environments.jsonl", environment())
    for segment in plan["segments"]:
        for epoch in segment["target_epochs"]:
            check_stop()
            destination = out / "checkpoints" / f"epoch-{epoch:03d}"
            target_step = epoch * plan["epoch_size"]
            if destination.exists():
                validate_checkpoint(destination, metadata, target_step)
                print(f"SKIP epoch={epoch}: checkpoint already saved", flush=True)
            else:
                step, selected = continuation(out, plan, metadata, segment, epoch)
                if step < target_step:
                    write_json(out / "replay_progress.json", {"status": "running", "target_epoch": epoch,
                               "restored_step": step, "completed_epochs": completed,
                               "target_epochs": plan["target_epochs"], "training_updates": plan["training_updates"]})
                    command = [sys.executable, "-B", "-u", code / "train.py", "--run", out,
                               "--resume", selected, "--stop-after", str(target_step)]
                    append_jsonl(out / "replay_commands.jsonl", {"timestamp": now(), "target_epoch": epoch,
                                 "anchor_epoch": segment["anchor_epoch"], "start_step": step,
                                 "command": [str(x) for x in command]})
                    print(f"REPLAY epoch={epoch}: restore step {step}, pause at {target_step}; "
                          f"completed {len(completed)}/{len(plan['target_epochs'])}", flush=True)
                    invoke(command, code)
                    step, selected = continuation(out, plan, metadata, segment, epoch)
                if step != target_step:
                    # A signal sent directly to train.py can return success after
                    # an earlier graceful pause. Never auto-restart over that pause.
                    write_json(out / "replay_progress.json", {"status": "paused", "target_epoch": epoch,
                               "saved_step": step, "completed_epochs": completed})
                    print(f"Paused at step {step}; resume to reach epoch {epoch}", flush=True)
                    return False
                publish_target(out, metadata, epoch, selected, step)
                print(f"SAVED epoch={epoch}: {destination}", flush=True)
            completed.append(epoch)
            markers[str(epoch)] = file_hash(destination / "complete.json")
            write_json(out / "replay_progress.json", {"status": "running", "completed_epochs": completed,
                       "target_epochs": plan["target_epochs"], "training_updates": plan["training_updates"]})
    write_json(out / "replay_complete.json", {"completed_at": now(), "epochs": completed,
               "checkpoint_markers": markers, "plan_sha256": file_hash(out / "replay_plan.json"),
               "trajectory_comparison_performed": False})
    write_json(out / "replay_progress.json", {"status": "complete", "completed_epochs": completed,
               "target_epochs": plan["target_epochs"], "training_updates": plan["training_updates"]})
    print(f"Replay complete: {len(completed)} checkpoints saved in {out / 'checkpoints'}", flush=True)
    return True


def main():
    global _lock_fd
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", help="Original LoRA run; required for a new replay")
    parser.add_argument("--out", help="Separate replay run directory")
    parser.add_argument("--epochs", help="Default: " + ",".join(map(str, DEFAULT_EPOCHS)))
    parser.add_argument("--resume", action="store_true", help="Keep completed epochs and resume remaining work")
    parser.add_argument("--plan-only", action="store_true", help="Read-only plan/storage estimate; no GPU imports")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
    if args.worker:
        if os.name == "posix":
            _lock_fd = int(os.environ[LOCK_ENV])
            os.fstat(_lock_fd)
        out = Path(args.worker).resolve()
        try:
            execute(out)
        except BaseException as exc:
            path = out / "replay_progress.json"
            if path.exists():
                progress = read_json(path)
                progress.update(status="paused" if isinstance(exc, (SystemExit, KeyboardInterrupt)) else "failed",
                                stopped_at=now(), error=str(exc))
                write_json(path, progress)
            raise
        return
    if not args.out or (not args.resume and not args.source_run):
        parser.error("New replay needs --source-run and --out; resume needs --out --resume")
    out = artifact_path(args.out)
    if args.resume:
        plan, _ = verify_replay(out)
        if args.source_run and str(artifact_path(args.source_run)) != plan["source_run"]:
            parser.error("Source run differs from the saved replay")
        if args.epochs and parse_epochs(args.epochs) != plan["target_epochs"]:
            parser.error("Epochs differ from the saved replay; use another output directory")
    else:
        if out.exists():
            raise FileExistsError(f"Output exists; use --resume: {out}")
        plan = plan_replay(artifact_path(args.source_run), out,
                           parse_epochs(args.epochs) if args.epochs else DEFAULT_EPOCHS)
    show_plan(plan)
    if args.plan_only:
        return
    configure_runtime()
    with replay_lock(out):
        if not args.resume:
            initialize(plan)
        frozen = out / "replay_code"
        invoke([sys.executable, "-B", "-u", frozen / "replay_train.py", "--worker", out], frozen)


if __name__ == "__main__":
    main()
