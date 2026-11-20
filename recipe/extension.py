"""Extend a completed LoRA run in a separate directory using its frozen trainer."""
from copy import deepcopy
from pathlib import Path
import shutil
import uuid

from .common import (ROOT, digest, epoch_order, file_hash, iter_jsonl, lr_for_update,
                     now, read_json, snapshot_code, write_json)
from .replay import IDENTITY_KEYS, tree_bytes, tree_hashes, validate_checkpoint
from .resume import original_code, verified_code, verify_run_inputs
from .storage import GIB, require_free_space

BOOTSTRAP = "attempts/continuation-bootstrap/code"
FORMAT = "full-state-training-extension-v1"


def plan_extension(source, out, additional_epochs=128, checkpoint_every=16):
    source, out = Path(source).resolve(), Path(out).resolve()
    if out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError("Continuation output must be separate from the source run")
    if additional_epochs < 1 or checkpoint_every < 1:
        raise ValueError("Additional epochs and checkpoint interval must be positive")
    meta = verify_run_inputs(source)
    if meta.get("replay"):
        raise ValueError("Continue the completed training run, not a missing-epoch replay")
    if digest({k: meta[k] for k in IDENTITY_KEYS}) != meta["config_sha256"]:
        raise ValueError("Source run config/data fingerprint mismatch")
    start, size = meta["config"]["epochs"], meta["config"]["expected_train_size"]
    if start < 1 or size < 1 or len(list(iter_jsonl(source / "data/train.jsonl"))) != size:
        raise ValueError("Source training size/budget is invalid")
    anchor = source / "checkpoints" / f"epoch-{start:03d}"
    validate_checkpoint(anchor, meta, start * size)
    code = original_code(source)
    end = start + additional_epochs
    targets = sorted(set(range(start + checkpoint_every, end + 1, checkpoint_every)) | {end})
    config = deepcopy(meta["config"])
    config.update(epochs=end, checkpoint_epochs=targets)
    largest = tree_bytes(anchor)
    return {"format": FORMAT, "created_at": now(), "source_run": str(source), "out": str(out),
            "source_run_sha256": file_hash(source / "run.json"),
            "source_checkpoint": str(anchor), "source_checkpoint_files": tree_hashes(anchor),
            "training_code": str(code), "training_code_sha256": meta["code_sha256"],
            "start_epoch": start, "end_epoch": end, "epoch_size": size,
            "additional_epochs": additional_epochs, "checkpoint_every": checkpoint_every,
            "target_epochs": targets, "start_step": start * size, "total_steps": end * size,
            "additional_updates": additional_epochs * size, "config": config,
            "storage": {"target_checkpoint_bytes_estimate": largest * len(targets),
                        "required_bytes": int(largest * (len(targets) + config["keep_recovery"] + 2) * 1.25)
                        + tree_bytes(source / "data") + tree_bytes(source / "tokenizer") + tree_bytes(code),
                        "reserve_bytes": GIB,
                        "note": "Includes copied anchor, milestones, rolling recovery and temporary save. "
                                "Excludes a missing Base cache, new environments and evaluation outputs."}}


def validate_state(state, meta, step):
    required = {"step", "optimizer", "scheduler", "rng", "cursor", "config_sha256"}
    if not required <= state.keys() or state["step"] != step or state["config_sha256"] != meta["config_sha256"]:
        raise ValueError("Incomplete or incompatible full recovery state")
    cfg = meta["config"]
    size = cfg["expected_train_size"]
    cursor = {"epoch_index": step // size, "position": step % size,
              "order": epoch_order(size, cfg["seed"], step // size)}
    if state["cursor"] != cursor:
        raise ValueError("Source shuffle/data cursor mismatch")
    if state["scheduler"].get("last_epoch") != step:
        raise ValueError("Source scheduler step differs from the global update count")
    expected_lr = lr_for_update(step + 1, cfg["learning_rate"], cfg["warmup_steps"])
    groups = state["optimizer"].get("param_groups", [])
    if not groups or any(abs(g["lr"] - expected_lr) > 1e-14 for g in groups):
        raise ValueError("Source learning rate differs from the original warmup/constant schedule")
    if not {"python", "numpy", "torch", "cuda"} <= state["rng"].keys():
        raise ValueError("Source RNG state is incomplete")


def initialize(plan):
    """Copy first, change only budget identities in the COPY, then atomically publish."""
    import torch
    out, source = Path(plan["out"]), Path(plan["source_run"])
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}")
    if file_hash(source / "run.json") != plan["source_run_sha256"]:
        raise ValueError("Source run changed after planning")
    require_free_space(out, plan["storage"]["required_bytes"], plan["storage"]["reserve_bytes"])
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = out.with_name(out.name + ".continuation-partial-" + uuid.uuid4().hex)
    stage.mkdir()
    try:
        meta = deepcopy(verify_run_inputs(source))
        write_json(stage / "continuation/source_run.json", meta)
        for folder, field in (("data", "data_files"), ("tokenizer", "tokenizer_files")):
            for name in meta[field]:
                if Path(name).is_absolute() or ".." in Path(name).parts:
                    raise ValueError("Frozen input paths must be relative")
                target = stage / folder / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / folder / name, target)
        shutil.copytree(plan["training_code"], stage / BOOTSTRAP)
        verified_code(stage / BOOTSTRAP, plan["training_code_sha256"])
        anchor = stage / "checkpoints" / f"epoch-{plan['start_epoch']:03d}"
        shutil.copytree(plan["source_checkpoint"], anchor)
        if tree_hashes(anchor) != plan["source_checkpoint_files"]:
            raise ValueError("Source checkpoint changed during copy")
        # Only trusted self-produced training state; no external model pickle.
        state = torch.load(anchor / "training_state.pt", map_location="cpu", weights_only=False)
        validate_state(state, meta, plan["start_step"])
        marker = read_json(anchor / "complete.json")
        write_json(stage / "continuation/source_checkpoint.json", marker)
        old_identity = meta["config_sha256"]
        meta["config"] = deepcopy(plan["config"])
        meta["config_sha256"] = digest({k: meta[k] for k in IDENTITY_KEYS})
        meta["continuation"] = {"format": FORMAT, "source_run": str(source),
                                "source_run_sha256": plan["source_run_sha256"],
                                "source_config_sha256": old_identity, "start_epoch": plan["start_epoch"],
                                "start_step": plan["start_step"], "target_epochs": plan["target_epochs"],
                                "changed_config_fields": ["epochs", "checkpoint_epochs"],
                                "policy": "Full optimizer/scheduler/RNG/cursor continuation; global epochs/steps preserved"}
        # A new budget has a new identity. Optimizer tensors, scheduler, RNG,
        # cursor, adapter files and step are carried over without resetting.
        state["config_sha256"] = meta["config_sha256"]
        torch.save(state, anchor / "training_state.pt")
        del state
        marker["config_sha256"] = meta["config_sha256"]
        write_json(anchor / "complete.json", marker)
        write_json(stage / "run.json", meta)
        verify_run_inputs(stage)
        validate_checkpoint(anchor, meta, plan["start_step"])
        frozen = deepcopy(plan)
        frozen["run_sha256"] = file_hash(stage / "run.json")
        frozen["anchor_files"] = tree_hashes(anchor)
        launcher = stage / "continuation/code"
        frozen["launcher_code_sha256"] = snapshot_code(launcher)
        (launcher / "dataset").mkdir()
        for name in ("__init__.py", "continue_training.py"):
            shutil.copy2(ROOT / "dataset" / name, launcher / "dataset" / name)
        frozen["launcher_extra_files"] = tree_hashes(launcher / "dataset")
        frozen["provenance_files"] = {name: file_hash(stage / "continuation" / name)
                                      for name in ("source_run.json", "source_checkpoint.json")}
        write_json(stage / "continuation/plan.json", frozen)
        write_json(stage / "continuation/manifest.json", {"plan_sha256": file_hash(stage / "continuation/plan.json")})
        write_json(stage / "continuation/progress.json", {"status": "prepared", "step": plan["start_step"],
                                                         "total_steps": plan["total_steps"]})
        if out.exists():
            raise FileExistsError(f"Refusing to replace existing output: {out}")
        stage.rename(out)
    except BaseException:
        if stage.exists():
            if stage.is_symlink() or stage.resolve().parent != out.parent.resolve():
                raise RuntimeError(f"Unsafe staging path: {stage}")
            shutil.rmtree(stage)
        raise
    return frozen


def verify_extension(out):
    out = Path(out).resolve()
    control = out / "continuation"
    if file_hash(control / "plan.json") != read_json(control / "manifest.json")["plan_sha256"]:
        raise ValueError("Frozen continuation plan changed")
    plan = read_json(control / "plan.json")
    if plan["format"] != FORMAT or plan["out"] != str(out):
        raise ValueError("Continuation format/location mismatch")
    if file_hash(out / "run.json") != plan["run_sha256"]:
        raise ValueError("Continuation run configuration changed")
    meta = verify_run_inputs(out)
    verified_code(out / BOOTSTRAP, plan["training_code_sha256"])
    verified_code(control / "code", plan["launcher_code_sha256"])
    if tree_hashes(control / "code/dataset") != plan["launcher_extra_files"]:
        raise ValueError("Frozen continuation launcher changed")
    for name, expected in plan["provenance_files"].items():
        if file_hash(control / name) != expected:
            raise ValueError("Continuation provenance changed")
    anchor = out / "checkpoints" / f"epoch-{plan['start_epoch']:03d}"
    if tree_hashes(anchor) != plan["anchor_files"]:
        raise ValueError("Frozen continuation anchor changed")
    return plan, meta
