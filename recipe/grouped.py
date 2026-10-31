"""Fixed panel partitions and loss-result reuse; no model or dataset imports."""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path

from .common import digest, file_hash, iter_jsonl, read_json, write_json, write_jsonl

GENERATION_EPOCHS = [64, 32, 16, 8, 4, 2, 1]
MAX_NEW_TOKENS = 8192


def select_panel(panel, ids):
    if (not isinstance(ids, list) or not ids or any(not isinstance(s, str) or not s for s in ids)
            or len(ids) != len(set(ids))):
        raise ValueError("Panel IDs must be a nonempty JSON list of unique strings")
    by_id = {r["sample_id"]: r for r in panel}
    if len(by_id) != len(panel) or not set(ids).issubset(by_id):
        raise ValueError("Duplicate panel entries or unknown panel IDs")
    return [by_id[s] for s in ids]


def partition_panel(panel):
    """Preserve saved task order; each group has 32 MATH500 and 16 GPQA."""
    if Counter(r["task"] for r in panel) != {"math500": 128, "gpqa": 64}:
        raise ValueError("Grouped evaluation requires exactly 128 MATH500 + 64 GPQA questions")
    select_panel(panel, [r["sample_id"] for r in panel])
    math = [r for r in panel if r["task"] == "math500"]
    gpqa = [r for r in panel if r["task"] == "gpqa"]
    return [{"name": f"group-{i + 1:03d}",
             "selected_ids": [r["sample_id"] for r in math[i * 32:(i + 1) * 32] + gpqa[i * 16:(i + 1) * 16]],
             "task_counts": {"math500": 32, "gpqa": 16}} for i in range(4)]


def teacher_forcing_files(run, source, epochs, metadata):
    """Require complete, matching original fitting results; never rewrite them."""
    run, source = Path(run).resolve(), Path(source).resolve()
    samples = {s: {r["sample_id"]: r for r in iter_jsonl(run / "data" / f"{s}.jsonl")}
               for s in ("train", "holdout")}
    hashes = {}
    for epoch in epochs:
        directory = source / f"epoch-{epoch:03d}"
        identity = read_json(directory / "identity.json")
        if (identity["run_sha256"] != file_hash(run / "run.json") or identity["epoch"] != epoch
                or identity["checkpoint"]["config_sha256"] != metadata["config_sha256"]
                or identity["checkpoint"]["step"] != epoch * len(samples["train"])):
            raise ValueError(f"Teacher-forcing identity differs from this run: {directory}")
        paths = [directory / "identity.json"]
        for split, expected in samples.items():
            selected = identity["selected_ids"][split]
            if (len(selected) != len(set(selected)) or set(selected) != set(expected)
                    or identity["data_sha256"][split] != file_hash(run / "data" / f"{split}.jsonl")):
                raise ValueError(f"Teacher-forcing sample selection differs: {directory}/{split}")
            path = directory / split / "samples.jsonl"
            rows = list(iter_jsonl(path))
            if len(rows) != len(expected) or {r["sample_id"] for r in rows} != set(expected):
                raise ValueError(f"Incomplete/duplicate teacher-forcing results: {path}")
            summary = directory / split / "summary.json"
            if read_json(summary)["samples"] != len(expected):
                raise ValueError(f"Incomplete teacher-forcing summary: {summary}")
            paths.extend([path, summary])
            for row in rows:
                if row["question_id"] != expected[row["sample_id"]]["question_id"]:
                    raise ValueError(f"Teacher-forcing question mismatch: {path}")
                count = row["n_tokens"]
                if (type(count) is not int or count < 1 or type(row["correct_tokens"]) is not int
                        or not 0 <= row["correct_tokens"] <= count or not math.isfinite(row["mean_loss"])
                        or row["mean_loss"] < 0
                        or not math.isclose(row["mean_loss"], row["nll_sum"] / count, rel_tol=1e-5, abs_tol=1e-6)
                        or not math.isclose(row["token_accuracy"], row["correct_tokens"] / count, abs_tol=1e-6)
                        or count != expected[row["sample_id"]].get("n_supervised", count)):
                    raise ValueError(f"Invalid teacher-forcing metrics: {path}")
                if not row.get("token_file"):
                    raise ValueError(f"Missing token evidence reference: {path}")
                token = (path.parent / row["token_file"]).resolve()
                if not token.is_relative_to(path.parent.resolve()):
                    raise ValueError(f"Teacher-forcing token path escapes its split: {token}")
                paths.append(token)
        hashes.update({p.relative_to(source).as_posix(): file_hash(p) for p in paths})
    return hashes


def merge_groups(job, plan):
    """Merge complete groups into ordinary full-panel response files for regrading."""
    job = Path(job)
    panel_ids = plan["panel_ids"]
    for epoch in plan["generation_epochs"]:
        rows, sources, protocol = {}, [], None
        for group in plan["groups"]:
            directory = job / "groups" / group["name"] / "generation" / f"epoch-{epoch:03d}"
            identity = read_json(directory / "identity.json")
            path = directory / "responses.jsonl"
            checksum = file_hash(path)
            members = list(iter_jsonl(path))
            complete = read_json(directory / "complete.json")
            if (identity["selected_ids"] != group["selected_ids"] or identity["epoch"] != epoch
                    or identity["run_sha256"] != plan["run_sha256"]
                    or identity["panel_sha256"] != plan["panel_sha256"]
                    or identity["settings"] != plan["settings"]
                    or identity["checkpoint"] != read_json(Path(plan["run"]) / "checkpoints" / f"epoch-{epoch:03d}" / "complete.json")
                    or identity["code_sha256"] != plan["code_sha256"]
                    or complete["samples"] != len(members) or complete["responses_sha256"] != checksum
                    or len(members) != len(group["selected_ids"])
                    or {r["sample_id"] for r in members} != set(group["selected_ids"])):
                raise ValueError(f"Incomplete or incompatible generation group: {directory}")
            signature = {k: v for k, v in identity.items() if k != "selected_ids"}
            if protocol is not None and protocol != signature:
                raise ValueError("Groups used different generation protocols/checkpoints")
            protocol = signature
            for row in members:
                if (row["sample_id"] in rows or row["epoch"] != epoch
                        or row["generated_tokens"] != len(row["generated_token_ids"])
                        or row["generated_tokens"] > MAX_NEW_TOKENS):
                    raise ValueError("Duplicate/mismatched/over-budget generated response")
                rows[row["sample_id"]] = row
            sources.append({"group": group["name"], "responses": str(path), "sha256": checksum,
                            "identity_sha256": file_hash(directory / "identity.json")})
        if set(rows) != set(panel_ids):
            raise ValueError("Merged groups do not cover the full fixed panel")
        out = job / "generation" / f"epoch-{epoch:03d}"
        write_json(out / "identity.json", {**protocol, "selected_ids": panel_ids})
        write_jsonl(out / "responses.jsonl", [rows[s] for s in panel_ids])
        write_json(out / "group_sources.json", sources)
        write_json(out / "complete.json", {"samples": len(rows), "responses_sha256": file_hash(out / "responses.jsonl")})
    write_json(job / "generation" / "merge_manifest.json", {"plan_sha256": digest(plan), "samples": len(panel_ids),
                                                               "epochs": plan["generation_epochs"]})
