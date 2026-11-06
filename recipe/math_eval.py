"""Freeze the run's original MATH500/GPQA questions for resumable standalone evaluation."""
from pathlib import Path
import re

from .common import file_hash, iter_jsonl, read_json, write_json, write_jsonl
from .grouped import partition_panel

FORMAT = "math-gpqa-evaluation-panel-v1"
COUNTS = {"math500": 128, "gpqa": 64}


def freeze_panel(source, destination, training_sha256):
    """Preserve questions, prompts, GPQA choices and reference solutions exactly."""
    source, destination = Path(source), Path(destination)
    panel = list(iter_jsonl(source))
    partition_panel(panel)
    for q in panel:
        if not re.fullmatch(r"(?:math500|gpqa)-[0-9a-f]{64}", q["sample_id"]):
            raise ValueError("Invalid original MATH500/GPQA sample ID")
    destination.mkdir(parents=True, exist_ok=False)
    selected = []
    for question in panel:
        name = f"references/{question['sample_id']}.json"
        write_json(destination / name, question.get("original", question))
        selected.append({**question, "source": question.get("source", f"frozen-run-panel:{question['task']}"),
                         "reference_file": name, "reference_sha256": file_hash(destination / name)})
    write_jsonl(destination / "panel.jsonl", selected)
    write_json(destination / "plan.json", {"training_sha256": training_sha256,
               "math_panel_sha256": file_hash(source), "task_counts": COUNTS,
               "policy": "Same questions, prompts, choice ordering and references; no resampling"})
    names = ["panel.jsonl", "plan.json"] + [q["reference_file"] for q in selected]
    write_json(destination / "manifest.json", {"format_version": FORMAT,
               "files": {name: file_hash(destination / name) for name in names}})
    from .transfer_eval import load_panel
    return load_panel(destination, training_sha256)[0]


def verify_math_origin(directory, math_panel_sha256):
    if read_json(Path(directory) / "plan.json")["math_panel_sha256"] != math_panel_sha256:
        raise ValueError("Evaluation panel differs from this run's frozen MATH500/GPQA panel")
