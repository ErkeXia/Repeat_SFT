"""Small, dependency-free provenance and append-only artifact utilities."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(value).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def recover_jsonl(path):
    """Keep complete lines; preserve a torn trailing write before repairing it."""
    path = Path(path)
    if not path.exists():
        return []
    rows, end = [], 0
    with path.open("rb") as f:
        while line := f.readline():
            if not line.endswith(b"\n"):
                break
            rows.append(json.loads(line))  # Corruption inside the file is fatal.
            end = f.tell()
    if end != path.stat().st_size:
        shutil.copy2(path, path.with_name(path.name + ".torn-" + uuid.uuid4().hex))
        with path.open("r+b") as f:
            f.truncate(end)
    return rows


def fresh_dir(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def stable_seed(seed, identity):
    return int(digest([seed, identity])[:8], 16)


def epoch_order(size, seed, epoch):
    order = list(range(size))
    random.Random(stable_seed(seed, ["epoch", epoch])).shuffle(order)
    return order


def lr_for_update(step, base_lr, warmup):
    """step is one-based: first update lr/100, update 100 reaches lr."""
    if step < 1:
        raise ValueError("Update numbers start at 1")
    return base_lr * min(1.0, step / warmup) if warmup else base_lr


def environment():
    packages = {}
    for name in ("torch", "transformers", "datasets", "huggingface-hub", "bitsandbytes",
                 "accelerate", "peft", "numpy", "safetensors", "math-verify"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"timestamp": now(), "python": sys.version, "platform": platform.platform(),
            "packages": packages, "command": sys.argv,
            "installed_packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}


def source_files():
    paths = list(ROOT.glob("*.py")) + list(ROOT.glob("*.md")) + [ROOT / "requirements.txt"]
    for folder in ("recipe", "configs", "tests"):
        paths.extend(p for p in (ROOT / folder).rglob("*") if p.suffix in (".py", ".json", ".md"))
    return sorted(set(p for p in paths if p.is_file()))


def code_fingerprint():
    return digest({p.relative_to(ROOT).as_posix(): file_hash(p) for p in source_files()})


def snapshot_code(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for src in source_files():
        if src.is_file():
            rel = src.relative_to(ROOT)
            target = destination / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            manifest[rel.as_posix()] = file_hash(src)
    git = {}
    for key, args in (("commit", ["rev-parse", "HEAD"]), ("status", ["status", "--porcelain"]),
                      ("diff", ["diff", "HEAD", "--"])):
        result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                                text=True, encoding="utf-8", errors="replace") if shutil.which("git") else None
        git[key] = result.stdout if result and result.returncode == 0 else None
    write_json(destination / "manifest.json", {"files": manifest, "code_sha256": digest(manifest), "git": git})
    return digest(manifest)


def seed_everything(seed, deterministic=True):
    import numpy as np
    import torch
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def capture_rng():
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def validate_prepared(path):
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    for name, expected in manifest["files"].items():
        if file_hash(path / name) != expected:
            raise ValueError(f"Prepared data changed: {name}")
    return manifest


def checkpoint_path(run, epoch):
    path = Path(run) / "checkpoints" / f"epoch-{epoch:03d}"
    if not (path / "complete.json").exists():
        raise FileNotFoundError(f"Incomplete/missing checkpoint: {path}")
    return path
