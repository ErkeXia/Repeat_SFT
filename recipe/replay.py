"""Plan and publish missing epochs without changing an original training run."""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import re
import shutil
import uuid

from .adapters import adapter_spec, validate_finetuning
from .common import (digest, file_hash, iter_jsonl, now, read_json, snapshot_code,
                     write_json)
from .resume import original_code, verified_code, verify_run_inputs
from .storage import GIB, require_free_space

DEFAULT_EPOCHS = [6, 7, 9, 10, 12, 14, 20, 22, 26, 30]
IDENTITY_KEYS = ("config", "model", "data_files", "tokenizer_files", "semantics")


def parse_epochs(value):
    try:
        values = [int(part.strip()) for part in value.split(",")]
    except (ValueError, AttributeError) as exc:
        raise ValueError("--epochs must be comma-separated positive integers") from exc
    if not values or min(values) < 1 or len(set(values)) != len(values):
        raise ValueError("--epochs must be nonempty, unique and positive")
    return sorted(values)


def evaluation_epochs(metadata):
    """Replay retains the original training config; its evaluation nodes differ."""
    replay = metadata.get("replay")
    return list(replay["target_epochs"] if replay else metadata["config"]["checkpoint_epochs"])


def tree_files(path):
    path = Path(path).resolve()
    files = {}
    for item in sorted(path.rglob("*")):
        if item.is_symlink() or not item.resolve().is_relative_to(path):
            raise ValueError(f"Symlinks are not allowed inside copied artifacts: {item}")
        if item.is_file():
            files[item.relative_to(path).as_posix()] = item
    return files


def tree_hashes(path):
    return {name: file_hash(file) for name, file in tree_files(path).items()}


def tree_bytes(path):
    return sum(file.stat().st_size for file in tree_files(path).values())


def validate_checkpoint(path, metadata, step=None):
    path = Path(path)
    marker = read_json(path / "complete.json")
    if (marker.get("config_sha256") != metadata["config_sha256"]
            or not marker.get("has_training_state")
            or not (path / "training_state.pt").is_file()
            or (step is not None and marker["step"] != step)):
        raise ValueError(f"Missing or incompatible full training state: {path}")
    adapter_spec(path, finetuning=metadata["config"]["finetuning"], base_model=metadata["model"])
    return marker


def plan_replay(source, out, targets):
    source, out = Path(source).resolve(), Path(out).resolve()
    if out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError("Replay output must be separate from the source run, not inside it or its parent")
    metadata = verify_run_inputs(source)
    validate_finetuning(metadata["config"].get("finetuning"))
    if metadata.get("replay"):
        raise ValueError("Select the original run, not another replay branch")
    if digest({key: metadata[key] for key in IDENTITY_KEYS}) != metadata["config_sha256"]:
        raise ValueError("Original run config/data fingerprint mismatch")
    targets = parse_epochs(",".join(map(str, targets)))
    if targets[-1] > metadata["config"]["epochs"]:
        raise ValueError("Replay targets must lie within the original training budget")
    size = metadata["config"]["expected_train_size"]
    if len(list(iter_jsonl(source / "data" / "train.jsonl"))) != size:
        raise ValueError("Original training set size differs from its configuration")
    code = original_code(source)
    available = {}
    for path in sorted((source / "checkpoints").glob("epoch-*")):
        if not re.fullmatch(r"epoch-\d+", path.name) or not (path / "complete.json").exists():
            continue
        epoch = int(path.name.split("-")[1])
        marker = read_json(path / "complete.json")
        if epoch in targets:
            raise ValueError(f"Epoch {epoch} is already saved in the source run; evaluate it there")
        if marker.get("has_training_state") and (path / "training_state.pt").is_file():
            available[epoch] = path
    segments = []
    for target in targets:
        predecessors = [epoch for epoch in available if epoch < target]
        if not predecessors:
            raise ValueError(f"No earlier checkpoint with full training state for epoch {target}")
        anchor = max(predecessors)
        if not segments or segments[-1]["anchor_epoch"] != anchor:
            segments.append({"anchor_epoch": anchor, "target_epochs": []})
        segments[-1]["target_epochs"].append(target)
    anchors = {}
    for segment in segments:
        epoch = segment["anchor_epoch"]
        path = available[epoch]
        validate_checkpoint(path, metadata, epoch * size)
        anchors[str(epoch)] = {"source": str(path), "files": tree_hashes(path), "bytes": tree_bytes(path)}
    largest = max(item["bytes"] for item in anchors.values())
    # Full-state adapter checkpoints are small. Include kept recoveries, one
    # in-flight atomic save, repeated source snapshots/logs, and a safety reserve.
    copies = len(targets) + len(anchors) + metadata["config"]["keep_recovery"] + 1
    inputs = tree_bytes(source / "data") + tree_bytes(source / "tokenizer") + tree_bytes(code)
    estimate = {"largest_source_checkpoint_bytes": largest,
                "target_checkpoint_bytes_estimate": largest * len(targets),
                "peak_checkpoint_copies_budget": copies,
                "required_bytes": int(copies * largest * 1.25) + inputs + GIB,
                "reserve_bytes": GIB,
                "note": "Estimate from actual full checkpoints; includes anchors/recovery/temp/logs. "
                        "Excludes a missing Base cache, new environments and later evaluation outputs."}
    return {"version": 1, "created_at": now(), "source_run": str(source), "out": str(out),
            "source_run_sha256": file_hash(source / "run.json"), "training_code": str(code),
            "training_code_sha256": metadata["code_sha256"], "target_epochs": targets,
            "epoch_size": size, "segments": segments, "anchors": anchors,
            "training_updates": sum((s["target_epochs"][-1] - s["anchor_epoch"]) * size for s in segments),
            "storage": estimate}


def initialize(plan):
    """Publish an independent run atomically. No original state is edited/retagged."""
    out, source = Path(plan["out"]), Path(plan["source_run"])
    if out.exists():
        raise FileExistsError(f"Output exists; use --resume: {out}")
    if file_hash(source / "run.json") != plan["source_run_sha256"]:
        raise ValueError("Source run changed after planning")
    estimate = plan["storage"]
    require_free_space(out, estimate["required_bytes"], estimate["reserve_bytes"])
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.with_name(out.name + ".partial-" + uuid.uuid4().hex)
    staging.mkdir()
    try:
        metadata = deepcopy(verify_run_inputs(source))
        metadata["replay"] = {"format_version": 1, "source_run": str(source),
                              "source_run_sha256": plan["source_run_sha256"],
                              "target_epochs": plan["target_epochs"], "segments": plan["segments"],
                              "policy": "Independent continuations from original full-state anchors; no trajectory comparison"}
        for name in ("data", "tokenizer"):
            # Only frozen inputs, not evaluations, logs, caches or other runs.
            (staging / name).mkdir()
            field = "data_files" if name == "data" else "tokenizer_files"
            for filename in metadata[field]:
                if Path(filename).is_absolute() or ".." in Path(filename).parts:
                    raise ValueError(f"Frozen input must have a relative path: {filename}")
                target = staging / name / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / name / filename, target)
        training_code = staging / "attempts" / "replay-bootstrap" / "code"
        shutil.copytree(plan["training_code"], training_code)
        verified_code(training_code, plan["training_code_sha256"])
        for epoch, anchor in plan["anchors"].items():
            target = staging / "anchors" / f"epoch-{int(epoch):03d}"
            shutil.copytree(anchor["source"], target)
            if tree_hashes(target) != anchor["files"]:
                raise ValueError(f"Anchor changed during copy: {anchor['source']}")
        write_json(staging / "run.json", metadata)
        verify_run_inputs(staging)
        frozen_plan = deepcopy(plan)
        frozen_plan["run_sha256"] = file_hash(staging / "run.json")
        frozen_plan["launcher_code_sha256"] = snapshot_code(staging / "replay_code")
        write_json(staging / "replay_plan.json", frozen_plan)
        write_json(staging / "replay_manifest.json", {"plan_sha256": file_hash(staging / "replay_plan.json")})
        write_json(staging / "replay_progress.json", {"status": "prepared", "target_epochs": plan["target_epochs"],
                                                     "completed_epochs": [], "training_updates": plan["training_updates"]})
        if out.exists():
            raise FileExistsError(f"Refusing to replace an existing output: {out}")
        staging.rename(out)
    except BaseException:
        # Delete only this call's unpublished staging directory, never a run.
        if staging.exists():
            if staging.is_symlink() or staging.resolve().parent != out.parent.resolve():
                raise RuntimeError(f"Unsafe staging path: {staging}")
            shutil.rmtree(staging)
        raise
    return frozen_plan


def verify_replay(out):
    out = Path(out).resolve()
    if file_hash(out / "replay_plan.json") != read_json(out / "replay_manifest.json")["plan_sha256"]:
        raise ValueError("Frozen replay plan changed")
    plan = read_json(out / "replay_plan.json")
    if plan["version"] != 1 or plan["out"] != str(out):
        raise ValueError("Replay plan version/location mismatch")
    if file_hash(out / "run.json") != plan["run_sha256"]:
        raise ValueError("Frozen replay run metadata changed")
    metadata = verify_run_inputs(out)
    verified_code(out / "replay_code", plan["launcher_code_sha256"])
    verified_code(out / "attempts" / "replay-bootstrap" / "code", plan["training_code_sha256"])
    for epoch, anchor in plan["anchors"].items():
        path = out / "anchors" / f"epoch-{int(epoch):03d}"
        if tree_hashes(path) != anchor["files"]:
            raise ValueError(f"Frozen anchor changed: {path}")
    return plan, metadata


def continuation(out, plan, metadata, segment, target):
    """Never jump to another segment's anchor or to a future checkpoint."""
    out = Path(out)
    size = plan["epoch_size"]
    lower, upper = segment["anchor_epoch"] * size, target * size
    anchor = out / "anchors" / f"epoch-{segment['anchor_epoch']:03d}"
    candidates = [(lower, anchor)]
    paths = [out / "checkpoints" / f"epoch-{e:03d}" for e in segment["target_epochs"] if e <= target]
    paths += sorted((out / "recovery").glob("step-*"))
    for path in paths:
        if not re.fullmatch(r"(?:epoch|step)-\d+", path.name) or not (path / "complete.json").exists():
            continue
        marker = read_json(path / "complete.json")
        if lower < marker["step"] <= upper:
            candidates.append((marker["step"], path))
    step, selected = max(candidates, key=lambda item: item[0])
    validate_checkpoint(selected, metadata, step)
    return step, selected


def publish_target(out, metadata, epoch, selected, step):
    """Promote a completed pause checkpoint by same-filesystem rename (no copy)."""
    out, selected = Path(out).resolve(), Path(selected).resolve()
    destination = out / "checkpoints" / f"epoch-{epoch:03d}"
    expected = epoch * metadata["config"]["expected_train_size"]
    if step != expected:
        raise ValueError("Cannot publish an incomplete epoch")
    if destination.exists():
        validate_checkpoint(destination, metadata, expected)
        return destination
    if selected.parent != out / "recovery" or selected.is_symlink():
        raise ValueError("Only this replay's recovery checkpoint can be promoted")
    validate_checkpoint(selected, metadata, expected)
    destination.parent.mkdir(parents=True, exist_ok=True)
    selected.rename(destination)
    if os.name == "posix":
        for parent in (selected.parent, destination.parent):
            fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    return destination
