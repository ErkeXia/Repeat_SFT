"""Atomic checkpoint directories and exact continuation state."""
from __future__ import annotations

import os
import errno
from pathlib import Path
import re
import shutil
import uuid

from .common import capture_rng, now, read_json, restore_rng, write_json
from .storage import checkpoint_estimate, require_free_space


def save_checkpoint(path, model, tokenizer, step, config_hash, optimizer=None, scheduler=None, cursor=None):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        completed = read_json(path / "complete.json")
        if completed["step"] != step or completed["config_sha256"] != config_hash:
            raise ValueError(f"Conflicting checkpoint: {path}")
        return path
    estimate = checkpoint_estimate(model, optimizer)
    require_free_space(path.parent, estimate["required_bytes"], estimate["reserve_bytes"])
    rng = capture_rng()
    temp = path.parent / (path.name + ".partial-" + uuid.uuid4().hex)
    created_temp = False
    try:
        temp.mkdir()
        created_temp = True
        model.save_pretrained(temp, safe_serialization=True, max_shard_size="4GB")
        tokenizer.save_pretrained(temp)
        if optimizer is not None:
            torch.save({"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "rng": rng, "cursor": cursor, "config_sha256": config_hash}, temp / "training_state.pt")
        write_json(temp / "complete.json", {"step": step, "config_sha256": config_hash,
                                           "has_training_state": optimizer is not None,
                                           "storage_estimate": estimate, "timestamp": now()})
        for item in temp.iterdir():
            if item.is_file():
                with item.open("rb+") as f:
                    os.fsync(f.fileno())
        os.replace(temp, path)
        if os.name != "nt":
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    except BaseException as exc:
        # Clean only the unpublished directory created by this call. Completed
        # checkpoints and partials left by other attempts remain untouched.
        if created_temp and temp.exists():
            if temp.is_symlink() or temp.resolve().parent != path.parent.resolve():
                raise RuntimeError(f"Unexpected temporary checkpoint path: {temp}") from exc
            try:
                shutil.rmtree(temp)
            except OSError as cleanup_error:
                print(f"Could not remove failed temporary checkpoint {temp}: {cleanup_error}", flush=True)
        if getattr(exc, "errno", None) == errno.ENOSPC or "no space left on device" in str(exc).lower():
            raise OSError(errno.ENOSPC, f"Checkpoint write ran out of space at {path}. Previous complete "
                          "checkpoints were preserved. Move the run to a larger filesystem before resuming.") from exc
        raise
    finally:
        restore_rng(rng)
    return path


def latest_recovery(run):
    run = Path(run)
    candidates = []
    for parent in (run / "recovery", run / "checkpoints"):
        if parent.exists():
            for path in parent.iterdir():
                if (re.fullmatch(r"(?:epoch|step)-\d+", path.name)
                        and (path / "complete.json").exists() and (path / "training_state.pt").exists()):
                    candidates.append((read_json(path / "complete.json")["step"], path))
    if not candidates:
        raise FileNotFoundError("No complete recovery checkpoint")
    return max(candidates, key=lambda x: x[0])[1]


def prune_recovery(run, keep):
    parent = (Path(run) / "recovery").resolve()
    if not parent.exists():
        return
    if keep < 1:
        raise ValueError("At least one rolling recovery checkpoint must be kept")
    paths = sorted([p for p in parent.iterdir()
                    if re.fullmatch(r"step-\d+", p.name) and (p / "complete.json").exists()],
                   key=lambda p: read_json(p / "complete.json")["step"])
    for path in paths[:-keep]:
        # Delete only validated completed recovery directories, never milestones.
        if path.resolve().parent != parent or path.is_symlink():
            raise ValueError("Unsafe recovery path")
        shutil.rmtree(path)
