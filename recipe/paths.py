"""Resolve experiment artifacts onto the VM data volume before heavy imports."""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ROOT = "/data/repeat_sft"


def storage_root():
    root = Path(os.environ.get("REPEAT_SFT_ROOT", DEFAULT_ROOT)).expanduser()
    if not root.is_absolute():
        raise ValueError("REPEAT_SFT_ROOT must be an absolute directory")
    if os.name == "nt" and "REPEAT_SFT_ROOT" not in os.environ:
        raise RuntimeError("Run experiments on the server; set REPEAT_SFT_ROOT explicitly for local tests")
    # A missing mount must never turn /data into an unnoticed system-disk folder.
    if os.name != "nt" and root.is_relative_to(Path("/data")) and not Path("/data").is_mount():
        raise RuntimeError("/data is not mounted. Mount the data volume before starting experiments.")
    return root.resolve()


def artifact_path(value):
    """Relative run/data/output paths are relative to REPEAT_SFT_ROOT, not cwd.

    Absolute paths remain explicit overrides, including legacy inputs. Relative
    paths may not escape the storage root through '..' or a symlink.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        # Also check the mount for explicitly supplied /data paths.
        if os.name != "nt" and path.is_relative_to(Path("/data")) and not Path("/data").is_mount():
            raise RuntimeError("/data is not mounted")
        return path.resolve()
    root = storage_root()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Relative artifact path escapes the storage root: {value}")
    return resolved


def configure_runtime():
    """Default new HF caches and temporary files to the volume.

    Explicit environment settings are respected. Move/link old cache directories
    using the documented migration before resuming an existing run.
    """
    root = storage_root()
    defaults = {"HF_HOME": root / ".cache" / "huggingface", "TMPDIR": root / ".cache" / "tmp"}
    for key, default in defaults.items():
        os.environ.setdefault(key, str(default))
        Path(os.environ[key]).expanduser().mkdir(parents=True, exist_ok=True)
    return root
