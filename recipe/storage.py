"""Estimate checkpoint storage and check the destination filesystem."""
from __future__ import annotations

import errno
import os
from pathlib import Path
import shutil

GIB = 1024**3


def require_free_space(path, required_bytes, reserve_bytes=0):
    destination = Path(path).resolve()
    existing = destination
    while not existing.exists():
        existing = existing.parent
    required = int(required_bytes + reserve_bytes)
    free = shutil.disk_usage(existing).free
    if free < required:
        raise OSError(errno.ENOSPC,
                      f"Insufficient disk space at {destination}: need approximately {required / GIB:.2f} GiB "
                      f"free, found {free / GIB:.2f} GiB. A new checkpoint must fit before old recovery files are pruned.")
    if hasattr(os, "statvfs"):
        stat = os.statvfs(existing)
        if stat.f_files > 0 and stat.f_favail == 0:
            raise OSError(errno.ENOSPC, f"No free inodes on the filesystem containing {destination}")
    return {"destination": str(destination), "free_bytes": free, "required_bytes": required}


def tensor_storage_bytes(value):
    import torch
    storages, containers = set(), set()

    def visit(obj):
        if isinstance(obj, torch.Tensor):
            storage = obj.untyped_storage()
            key = (str(obj.device), storage.data_ptr(), storage.nbytes())
            if key in storages:
                return 0
            storages.add(key)
            return storage.nbytes()
        if isinstance(obj, (dict, list, tuple)):
            if id(obj) in containers:
                return 0
            containers.add(id(obj))
            return sum(visit(x) for x in (obj.values() if isinstance(obj, dict) else obj))
        return 0

    return visit(value)


def checkpoint_estimate(model, optimizer=None):
    from .adapters import checkpoint_state_dict
    weights = tensor_storage_bytes(checkpoint_state_dict(model))
    state = tensor_storage_bytes(optimizer.state_dict()) if optimizer is not None else 0
    return {"weight_bytes": weights, "optimizer_bytes": state,
            "required_bytes": int((weights + state) * 1.05) + 64 * 1024**2,
            "reserve_bytes": 512 * 1024**2}
