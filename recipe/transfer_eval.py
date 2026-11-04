"""Frozen evaluation-only BBH/LiveCodeBench panels and balanced partitions."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import shutil

from .common import file_hash, iter_jsonl, read_json, write_json
from .transfer_data import BBH_TASKS


def load_panel(directory, training_sha256=None):
    directory = Path(directory).resolve()
    manifest = read_json(directory / "manifest.json")
    if manifest.get("format_version") not in ("transfer-panel-v1", "transfer-evaluation-panel-v1"):
        raise ValueError("Expected a completed transfer panel, not the old MATH500/GPQA panel")
    plan = read_json(directory / "plan.json")
    if training_sha256 and plan["training_sha256"] != training_sha256:
        raise ValueError("Transfer panel was checked against a different training subset")
    # Only selected inputs are needed here; never read/cache entire candidate pools.
    for name in ("panel.jsonl", "plan.json"):
        if file_hash(directory / name) != manifest["files"].get(name):
            raise ValueError(f"Transfer panel changed: {name}")
    panel = list(iter_jsonl(directory / "panel.jsonl"))
    if (not panel or len({p["sample_id"] for p in panel}) != len(panel)
            or len({p["question_id"] for p in panel}) != len(panel)):
        raise ValueError("Empty or duplicate transfer panel")
    for row in panel:
        if not isinstance(row["sample_id"], str) or not re.fullmatch(r"(?:bbh|lcb)-[0-9a-f]{64}", row["sample_id"]):
            raise ValueError("Invalid transfer sample ID")
        if row["task"] not in ("bbh", "livecodebench"):
            raise ValueError("Unsupported transfer task")
        rel = Path(row["reference_file"])
        path = (directory / rel).resolve()
        if rel.is_absolute() or not path.is_relative_to(directory / "references"):
            raise ValueError("Reference path escapes the prepared panel")
        if file_hash(path) != row["reference_sha256"] or manifest["files"].get(rel.as_posix()) != row["reference_sha256"]:
            raise ValueError(f"Selected reference changed: {row['sample_id']}")
        if row["task"] == "bbh" and row["subtask"] not in BBH_TASKS:
            raise ValueError("Unexpected BBH subtask")
    return panel, manifest


def copy_panel(source, destination, training_sha256):
    panel, manifest = load_panel(source, training_sha256)
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    files = ["panel.jsonl", "plan.json"] + [p["reference_file"] for p in panel]
    for name in files:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    write_json(destination / "manifest.json", {
        "format_version": "transfer-evaluation-panel-v1",
        "source_manifest_sha256": file_hash(source / "manifest.json"),
        "sources": manifest.get("sources"),
        "files": {name: file_hash(destination / name) for name in files}})
    load_panel(destination, training_sha256)
    return panel


def partition(panel):
    if Counter(p["task"] for p in panel) != {"bbh": 100, "livecodebench": 100}:
        raise ValueError("This four-group protocol requires 100 BBH + 100 LiveCodeBench")
    if Counter(p["subtask"] for p in panel if p["task"] == "bbh") != {t: 25 for t in BBH_TASKS}:
        raise ValueError("Expected 25 questions per BBH subtask")
    groups = [[] for _ in range(4)]
    # Rotating round-robin keeps each group at 25 BBH, 6/7 per subtask.
    bbh = [p for task in BBH_TASKS for p in panel if p["task"] == "bbh" and p["subtask"] == task]
    for i, row in enumerate(bbh):
        groups[i % 4].append(row)
    for i, row in enumerate(p for p in panel if p["task"] == "livecodebench"):
        groups[i % 4].append(row)
    return [{"name": f"group-{i + 1:03d}", "selected_ids": [r["sample_id"] for r in rows],
             "task_counts": dict(Counter(r["task"] for r in rows)),
             "bbh_subtasks": dict(Counter(r["subtask"] for r in rows if r["task"] == "bbh"))}
            for i, rows in enumerate(groups)]
