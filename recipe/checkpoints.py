"""Atomic checkpoint directories and exact continuation state."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import uuid

from .common import capture_rng, now, read_json, restore_rng, write_json


def save_checkpoint(path, model, tokenizer, step, config_hash, optimizer=None, scheduler=None, cursor=None):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        completed = read_json(path / "complete.json")
        if completed["step"] != step or completed["config_sha256"] != config_hash:
            raise ValueError(f"Conflicting checkpoint: {path}")
        return path
    rng = capture_rng()
    temp = path.parent / (path.name + ".partial-" + uuid.uuid4().hex)
    temp.mkdir()
    model.save_pretrained(temp, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(temp)
    if optimizer is not None:
        torch.save({"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "rng": rng, "cursor": cursor, "config_sha256": config_hash}, temp / "training_state.pt")
    write_json(temp / "complete.json", {"step": step, "config_sha256": config_hash,
                                       "has_training_state": optimizer is not None, "timestamp": now()})
    # Sync all weight/state files before publishing the directory.
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
    restore_rng(rng)
    return path


def latest_recovery(run):
    run = Path(run)
    candidates = []
    for parent in (run / "recovery", run / "checkpoints"):
        if parent.exists():
            for path in parent.iterdir():
                if (path / "complete.json").exists() and (path / "training_state.pt").exists():
                    candidates.append((read_json(path / "complete.json")["step"], path))
    if not candidates:
        raise FileNotFoundError("No complete recovery checkpoint")
    return max(candidates, key=lambda x: x[0])[1]


def prune_recovery(run, keep):
    parent = (Path(run) / "recovery").resolve()
    if not parent.exists():
        return
    paths = sorted([p for p in parent.iterdir() if (p / "complete.json").exists()],
                   key=lambda p: read_json(p / "complete.json")["step"])
    for path in paths[:-keep]:
        # Delete only validated completed recovery directories, never milestones.
        if path.resolve().parent != parent or path.is_symlink():
            raise ValueError("Unsafe recovery path")
        shutil.rmtree(path)
