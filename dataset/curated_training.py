"""Freeze the ten edited answers and prepare a single independent LoRA run."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import os
from pathlib import Path
import re
import shutil
import uuid

from recipe.adapters import validate_finetuning
from recipe.common import (ROOT, digest, environment, file_hash, iter_jsonl, now,
                           read_json, snapshot_code, validate_prepared, write_json, write_jsonl)
from recipe.data import OverlapIndex, adapt_sft
from recipe.encoding import encode_example, inspect_example
from dataset.likelihood_training import tree_hashes, verify_bundle
from dataset.likelihood_utils import describe

EPOCHS = list(range(0, 129, 16))
SIZE = 10
DEFAULT_CURATION = ROOT / "dataset/math-cot-20k/manual-curation/high-low-5-v1"
EXTRA_CODE = (
    "__init__.py", "curated_training.py", "train_curated.py", "score_curated.py",
    "likelihood_utils.py", "likelihood_training.py", "train_likelihood_extremes.py",
    "CURATED_TRAINING.md",
)


def training_config(original, seed):
    config = deepcopy(original)
    validate_finetuning(config["finetuning"])
    if config["finetuning"]["base_quantization"] != "none":
        raise ValueError("Curated likelihood/training requires the same BF16 Base, not NF4")
    config.update(seed=seed, expected_train_size=SIZE, epochs=128,
                  checkpoint_epochs=EPOCHS, recovery_every_steps=10, keep_recovery=2)
    return config


def validate_response(text):
    """A complete reasoning section followed by exactly one complete boxed answer."""
    if (text.count("<think>") != 1 or text.count("</think>") != 1
            or not text.startswith("<think>") or text.count("**Final Answer**") != 1):
        raise ValueError("Expected one think section and one Final Answer")
    reasoning, final = text.split("**Final Answer**")
    if not reasoning.rstrip().endswith("</think>") or not reasoning[7:].split("</think>")[0].strip():
        raise ValueError("Reasoning must be nonempty and closed before Final Answer")
    final = final.strip()
    if not final.startswith("\\boxed{") or final.count("\\boxed{") != 1:
        raise ValueError("Final Answer must contain only one boxed")
    depth = 1
    for i, char in enumerate(final[len("\\boxed{"):], len("\\boxed{")):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        if depth == 0:
            if i != len(final) - 1 or not final[len("\\boxed{"):-1].strip():
                raise ValueError("Content after boxed or empty answer")
            return
    raise ValueError("Unclosed boxed")


def load_curated(directory):
    directory = Path(directory).resolve()
    selection = read_json(directory / "selection.json")
    originals = list(iter_jsonl(directory / "originals.jsonl"))
    by_row = {r["row_index"]: r for r in originals}
    records = selection["records"]
    if (len(records) != SIZE or len(by_row) != SIZE or len(originals) != SIZE
            or len({r["row_index"] for r in records}) != SIZE
            or Counter(r["group"] for r in records) != {"high": 5, "low": 5}):
        raise ValueError("Require ten unique reviewed candidates: five high and five low")
    rows = []
    for record in records:
        index = record["row_index"]
        if index not in selection["actual_training_lists"][record["group"]]:
            raise ValueError("Candidate not in its historical high/low training group")
        original = by_row[index]["raw_record"]
        parent = adapt_sft(original, index)
        if (parent["sample_id"] != record["sample_id"]
                or parent["question_id"] != record["question_id"]
                or digest(original["response"].encode()) != record["response_sha256"]):
            raise ValueError(f"Original provenance mismatch: row {index}")
        path = (directory / record["editable_response_file"]).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Editable response path escapes curation directory")
        # Git on Windows may use CRLF. The actual text is canonically LF on both hosts.
        response = path.read_text(encoding="utf-8")
        if digest(response.encode()) != record["current_edit"]["response_sha256"]:
            raise ValueError(f"Unreviewed response change: {path}; update the reviewed selection first")
        validate_response(response)
        row = adapt_sft({**original, "response": response}, index)
        if row["sample_id"] == parent["sample_id"]:
            raise ValueError("Expected an edited answer, not the original response")
        old_length = row["original_metadata"].pop("response_length", None)
        row["curation"] = {
            "historical_group": record["group"], "original_sample_id": parent["sample_id"],
            "original_response_sha256": record["response_sha256"],
            "edited_response_sha256": digest(response.encode()),
            "edit_revision": record["current_edit"]["revision"],
            "original_publisher_response_length": old_length,
            "source_dataset": selection["dataset"], "source_revision": selection["source_revision"],
        }
        rows.append(row)
    if len({r["question_id"] for r in rows}) != SIZE:
        raise ValueError("Duplicate curated questions")
    return rows, selection


def prepare(out, request, check_stop=lambda: None):
    """Tokenize only these ten rows with the existing prepared tokenizer; no weights."""
    from transformers import AutoTokenizer
    out, data, source = Path(out), Path(request["data"]), Path(request["curation"])
    parent = validate_prepared(data)
    model = parent["model"]
    if (model["id"] != "Qwen/Qwen3-4B-Base"
            or not re.fullmatch(r"[0-9a-f]{40}", model.get("revision", ""))):
        raise ValueError("Require the pinned Qwen/Qwen3-4B-Base from original preparation")
    if parent["arguments"]["max_length"] != 10240:
        raise ValueError("Retain the original 10,240-token training limit")
    required = ("holdout.jsonl", "panel.jsonl", "overlap_references.jsonl", "chat_template.jinja")
    if not set(required) <= set(parent["files"]) or not any(k.startswith("tokenizer/") for k in parent["files"]):
        raise ValueError("Prepared manifest must bind evaluation data and tokenizer")
    rows, selection = load_curated(source)
    config = training_config(read_json(request["config"]), request["seed"])
    forbidden = [r for name in required[:3] for r in iter_jsonl(data / name)]
    overlap = OverlapIndex(forbidden)
    tokenizer = AutoTokenizer.from_pretrained(data / "tokenizer", local_files_only=True)
    index, inspections = [], []
    for row in rows:
        check_stop()
        if overlap.match(row):
            raise ValueError(f"Curated training/evaluation overlap: row {row['row_index']}")
        encoded = encode_example(tokenizer, row, 10240)
        if encoded is None:
            raise ValueError(f"Over-length edited answer: row {row['row_index']}; no truncation")
        index.append(describe(row, encoded))
        inspections.append(inspect_example(tokenizer, row, encoded))
        print(f"PREPARE row={row['row_index']} total={len(encoded['input_ids'])} "
              f"supervised={encoded['supervised_tokens']}", flush=True)
    # Inspect actual masked input, reasoning, final answer, and end-token labels.
    for item in inspections[:2]:
        print(f"MASKED PROMPT:\n{item['masked_prompt_text']}\nSUPERVISED:\n{item['supervised_text']}"
              f"\nTAIL: {item['tail_tokens']}", flush=True)
    stage = out / (".initializing-" + uuid.uuid4().hex)
    stage.mkdir(parents=True)
    snapshot_code(stage / "code")
    for name in EXTRA_CODE:
        target = stage / "code" / "dataset" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT / "dataset" / name, target)
    prepared = stage / "prepared" / "train"
    prepared.mkdir(parents=True)
    for name in required:
        shutil.copy2(data / name, prepared / name)
    shutil.copytree(data / "tokenizer", prepared / "tokenizer")
    write_jsonl(prepared / "train.jsonl", rows)
    write_jsonl(prepared / "inspection.jsonl", inspections)
    write_jsonl(stage / "index.jsonl", index)
    write_json(stage / "config.json", config)
    write_json(stage / "environment.json", environment())
    provenance = stage / "curation"
    provenance.mkdir()
    for name in ("selection.json", "originals.jsonl"):
        shutil.copy2(source / name, provenance / name)
    for record in selection["records"]:
        target = provenance / record["editable_response_file"]
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source / record["editable_response_file"], target)
    # Detect source changes during preparation instead of freezing inconsistent copies.
    frozen_rows, _ = load_curated(provenance)
    if frozen_rows != rows or file_hash(provenance / "selection.json") != file_hash(source / "selection.json"):
        raise ValueError("Curated inputs changed during preparation")
    for name in required:
        if file_hash(prepared / name) != parent["files"][name]:
            raise ValueError("Original prepared inputs changed during copying")
    copied_tokens = {"tokenizer/" + k: v for k, v in tree_hashes(prepared / "tokenizer").items()}
    if copied_tokens != {k: v for k, v in parent["files"].items() if k.startswith("tokenizer/")}:
        raise ValueError("Tokenizer changed during preparation")
    write_json(prepared / "manifest.json", {
        "format_version": 1, "arguments": {**parent["arguments"], "train_size": SIZE},
        "model": model, "sources": parent.get("sources", {}),
        "parent_manifest_sha256": file_hash(data / "manifest.json"),
        "stats": {"train": SIZE}, "files": tree_hashes(prepared),
        "curation": {"source_revision": selection["source_revision"],
                     "edited": True, "historical_groups": {"high": 5, "low": 5}},
    })
    plan = {"protocol": "curated-ten-base-likelihood-lora-128-v1", "created": now(),
            "request": request, "model": model, "config": config, "max_length": 10240,
            "updates_per_group": SIZE * 128, "scoring": {"backend": "vllm", "concurrency": 8,
            "max_num_batched_tokens": 1024, "gpu_memory_utilization": 0.85,
            "max_model_len": max(r["total_tokens"] for r in index) + 1,
            "adapter": None, "dtype": "bfloat16"},
            "semantics": "One mixed group of ten; shuffle once per epoch; one example/update; fresh Base LoRA",
            "overlap_audit": {"compared_rows": len(forbidden), "matches": [],
                              "scope": "original holdout, MATH500/GPQA panel and overlap references"},
            "files": tree_hashes(stage)}
    plan["identity"] = digest(plan)
    write_json(stage / "plan.json", plan)
    check_stop()
    if (out / "bundle").exists():
        raise FileExistsError(out / "bundle")
    os.replace(stage, out / "bundle")
    return verify_bundle(out)
