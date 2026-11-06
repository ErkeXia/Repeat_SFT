"""Read saved grading evidence and describe observed losses of correctness. CPU only."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from recipe.common import digest, file_hash, read_json

TASKS = {"math500": "MATH500", "gpqa": "GPQA", "bbh": "BBH", "livecodebench": "LiveCodeBench"}


class Evidence:
    def __init__(self):
        self.files = {}
        self.notes = []

    def read(self, path, lines=False):
        path = Path(path).resolve()
        data = path.read_bytes()
        sha = digest(data)
        if str(path) in self.files and self.files[str(path)] != sha:
            raise ValueError(f"Source changed while reading: {path}")
        self.files[str(path)] = sha
        text = data.decode("utf-8")
        return [json.loads(s) for s in text.splitlines() if s.strip()] if lines else json.loads(text)

    def verify(self):
        for path, sha in self.files.items():
            if file_hash(path) != sha:
                raise ValueError(f"Source changed while analyzing; use a new snapshot after grading settles: {path}")


def discover(run, jobs=(), grades=()):
    """Known live result locations only; never search backups, code, analysis or weights."""
    run = Path(run).resolve()
    explicit = bool(jobs or grades)
    jobs, grades = set(map(Path, jobs)), set(map(Path, grades))
    if not explicit:
        for job in (run / "postprocess").glob("*"):
            if (job / "grouped_pipeline.json").is_file() or (job / "setup/plan.json").is_file() or (job / "suite/plan.json").is_file():
                jobs.add(job)
        for pattern in ("grades*/grading_manifest.json", "evaluations/grades*/grading_manifest.json",
                        "postprocess/*/grades*/grading_manifest.json"):
            for manifest in run.glob(pattern):
                if manifest.parent.parent not in jobs:
                    grades.add(manifest.parent)
    expanded = set()
    for job in jobs:
        if (job / "suite/plan.json").is_file():
            plan = read_json(job / "suite/plan.json")
            if plan.get("format_version") != "replay-two-panels-v1" or plan.get("panels") != ["math-gpqa", "bbh-livecodebench"]:
                raise ValueError(f"Unknown intermediate evaluation suite: {job}")
            expanded.update(job / name for name in plan["panels"] if (job / name / "setup/plan.json").is_file())
        else:
            expanded.add(job)
    return sorted(p.resolve() for p in expanded), sorted(p.resolve() for p in grades)


def protocol(identity, grading):
    return {
        "generation": {k: identity.get(k) for k in (
            "run_sha256", "panel_sha256", "settings", "generation_config", "backend", "backend_config", "seed_policy", "code_sha256")},
        "grading": {k: grading.get(k) for k in ("grader_version", "code_sha256", "image", "config", "judge_config")},
    }


def load_sources(run, jobs, grades, evidence):
    run = Path(run).resolve()
    run_hash = file_hash(run / "run.json")
    evidence.read(run / "run.json")
    series = {}

    def panel_read(path):
        rows = evidence.read(path, lines=True)
        panel = {r["sample_id"]: r for r in rows}
        if len(panel) != len(rows) or any(r["task"] not in TASKS for r in rows):
            raise ValueError(f"Duplicate IDs or unsupported validation task: {path}")
        return panel, evidence.files[str(path.resolve())]

    def collection(origin, panel, planned, signature):
        sid = digest([str(origin), signature])[:20]
        if sid not in series:
            series[sid] = {"series_id": sid, "name": Path(origin).name, "origin": str(origin),
                           "protocol": signature, "panel": panel, "planned_epochs": sorted(set(planned)),
                           "records": {}, "sources": defaultdict(list)}
        series[sid]["planned_epochs"] = sorted(set(series[sid]["planned_epochs"]) | set(planned))
        return series[sid]

    def ingest(origin, panel, panel_hash, planned, directory, epoch, identity, grading, expected_hash=None):
        path = directory / f"epoch-{epoch:03d}" / "graded.jsonl"
        if identity.get("run_sha256") != run_hash or identity.get("panel_sha256") != panel_hash:
            raise ValueError(f"Run or panel identity mismatch: {path}")
        if identity.get("epoch") != epoch:
            raise ValueError(f"Epoch identity mismatch: {path}")
        selected = identity["selected_ids"]
        if len(set(selected)) != len(selected) or not set(selected) <= panel.keys():
            raise ValueError(f"Unexpected selected question IDs: {path}")
        rows = evidence.read(path, lines=True)
        sha = evidence.files[str(path.resolve())]
        if expected_hash is not None and sha != expected_hash:
            raise ValueError(f"Completed grades checksum mismatch: {path}")
        marker = path.parent / "complete.json"
        if marker.exists():
            if evidence.read(marker) != {"samples": len(rows), "graded_sha256": sha} or len(rows) != len(selected):
                raise ValueError(f"Invalid grading completion: {path}")
        actual = set()
        target = collection(origin, panel, planned, protocol(identity, grading))
        for row in rows:
            qid = row["sample_id"]
            if qid in actual or qid not in selected:
                raise ValueError(f"Duplicate/unexpected question: {path}: {qid}")
            actual.add(qid)
            question = panel[qid]
            if (row["epoch"] != epoch or any(row.get(k) != question.get(k) for k in
                    ("task", "question_id", "messages", "gold_answer"))):
                raise ValueError(f"Graded question differs from panel: {path}: {qid}")
            if row.get("correct") is not None and type(row["correct"]) is not bool:
                raise ValueError(f"Invalid correctness: {path}: {qid}")
            if row.get("grader_version") != grading.get("grader_version"):
                raise ValueError(f"Grader version differs: {path}: {qid}")
            if not isinstance(row.get("raw_response"), str):
                raise ValueError(f"Full original answer is missing: {path}: {qid}")
            if type(row.get("generated_tokens")) is not int or row["generated_tokens"] < 0:
                raise ValueError(f"Invalid generated token count: {path}: {qid}")
            tokens = row.get("generated_token_ids")
            if tokens is not None and (not isinstance(tokens, list) or len(tokens) != row["generated_tokens"]
                                       or any(type(t) is not int or t < 0 for t in tokens)):
                raise ValueError(f"Invalid generated token count: {path}: {qid}")
            key = epoch, qid
            if key in target["records"] and target["records"][key] != row:
                raise ValueError(f"Conflicting answers for one epoch/question; do not mix evaluations: {key}")
            target["records"][key] = row
            target["sources"][key].append({"graded_file": str(path.resolve()), "graded_sha256": sha})

    def manifest_dir(origin, directory, panel, panel_hash, planned):
        manifest = evidence.read(directory / "grading_manifest.json")
        if "sources" not in manifest:
            raise ValueError(f"Use --job for merged transfer grades: {directory}")
        seen = set()
        for source in manifest["sources"]:
            identity = source["identity"]
            epoch = identity["epoch"]
            if epoch in seen:
                raise ValueError(f"Duplicate epoch in grading manifest: {directory}")
            seen.add(epoch)
            ingest(origin, panel, panel_hash, planned or sorted(s["identity"]["epoch"] for s in manifest["sources"]),
                   directory, epoch, identity, manifest, source.get("graded_sha256"))

    for job in jobs:
        grouped = job / "grouped_pipeline.json"
        if grouped.exists():
            from report_examples import grading_directories
            plan = evidence.read(grouped)
            panel_path = run / "data/panel.jsonl"
            panel, panel_hash = panel_read(panel_path)
            if plan["run_sha256"] != run_hash or plan["panel_sha256"] != panel_hash:
                raise ValueError(f"Job does not belong to this run/panel: {job}")
            try:
                directories, _ = grading_directories(job, plan)
            except ValueError as exc:
                if not str(exc).startswith("No completed grading results yet"):
                    raise
                directories = []
            before = len(series)
            for directory, _ in directories:
                manifest_dir(job, directory, panel, panel_hash, plan["generation_epochs"])
        elif (job / "setup/plan.json").exists():
            plan = evidence.read(job / "setup/plan.json")
            panel_path = job / "setup/panel/panel.jsonl"
            panel, panel_hash = panel_read(panel_path)
            if plan["run_sha256"] != run_hash or plan["panel_sha256"] != panel_hash:
                raise ValueError(f"Job does not belong to this run/panel: {job}")
            before = len(series)
            # Read one committed epoch at a time, even while another epoch is grading.
            # Group originals are authoritative; merged copies are not counted again.
            for group in plan["groups"]:
                directory = job / "groups" / group["name"] / "grades"
                for path in sorted(directory.glob("epoch-*/graded.jsonl")):
                    marker = path.parent / "complete.json"
                    if not marker.exists():
                        evidence.notes.append(f"Skipped uncommitted grading file: {path}")
                        continue
                    binding = evidence.read(path.parent / "identity.json")
                    identity = binding["generation_identity"]
                    if path.parent.name != f"epoch-{identity['epoch']:03d}":
                        raise ValueError(f"Graded directory/epoch mismatch: {path}")
                    if identity["selected_ids"] != group["selected_ids"]:
                        raise ValueError(f"Transfer group membership differs: {path}")
                    ingest(job, panel, panel_hash, plan["generation_epochs"], directory, identity["epoch"], identity, binding)
        else:
            raise ValueError(f"Unknown job layout; pass a grading directory using --grades: {job}")
        if len(series) == before:
            collection(job, panel, plan["generation_epochs"], {"status": "no_completed_grades"})
            evidence.notes.append(f"No completed grades yet; missing observations are not failures: {job}")

    for directory in grades:
        panel, panel_hash = panel_read(run / "data/panel.jsonl")
        manifest_dir(directory, directory, panel, panel_hash, [])
    return list(series.values())


def state(row):
    if row is None or row.get("grading_error") is not None or type(row.get("correct")) is not bool:
        return None
    return row["correct"]


def behavior(row):
    text = row["raw_response"]
    blocks = [re.sub(r"\s+", " ", s).strip() for s in text.splitlines()]
    counts = Counter(s for s in blocks if len(s) >= 40)
    repeated = [(s, n) for s, n in counts.items() if n > 1]
    repeated.sort(key=lambda x: -(x[1] - 1) * len(x[0]))
    tokens = row.get("generated_token_ids")
    fraction = None
    if tokens is not None and len(tokens) >= 16:
        ngrams = {tuple(tokens[i:i + 16]) for i in range(len(tokens) - 15)}
        fraction = 1 - len(ngrams) / (len(tokens) - 15)
    return {"generated_tokens": row.get("generated_tokens"), "stop_reason": row.get("stop_reason"),
            **{k: row.get(k) for k in ("length_truncated", "reached_length_limit", "normal_chat_end", "ended_with_eos",
                                      "extractable", "extraction_method", "grading_status")},
            "repeated_16gram_fraction": fraction,
            "repeated_long_line_occurrences": sum(n - 1 for _, n in repeated),
            "most_repeated_line_excerpt": repeated[0][0][:400] if repeated else None,
            "most_repeated_line_count": repeated[0][1] if repeated else 0,
            "answer_marker_count": len(re.findall(r"\\(?:boxed|fbox)\s*\{|\bfinal\s+answer\b", text, re.I)),
            "unclosed_think": text.rfind("<think>") > text.rfind("</think>")}


def clues(before, after):
    tags = []
    if before["length_truncated"] is False and after["length_truncated"] is True:
        tags.append("new_truncation")
    elif before["length_truncated"] is True and after["length_truncated"] is True:
        tags.append("both_truncated")
    if before["extractable"] is True and after["extractable"] is False:
        tags.append("lost_answer_extraction")
    if before["ended_with_eos"] is True and after["ended_with_eos"] is False:
        tags.append("lost_eos")
    if after["unclosed_think"]:
        tags.append("unclosed_think")
    a, b = before["generated_tokens"], after["generated_tokens"]
    if a and b is not None and b >= 2 * a:
        tags.append("length_at_least_doubled")
    if a and b is not None and b <= a / 2:
        tags.append("length_at_least_halved")
    a, b = before["repeated_16gram_fraction"], after["repeated_16gram_fraction"]
    if a is not None and b is not None and b >= .5 and b - a >= .2:
        tags.append("increased_token_repetition")
    if after["repeated_long_line_occurrences"] > before["repeated_long_line_occurrences"]:
        tags.append("more_repeated_long_lines")
    if after["answer_marker_count"] > max(1, before["answer_marker_count"]):
        tags.append("more_answer_markers")
    if after["grading_status"] not in (None, "graded", "passed"):
        tags.append("code_" + after["grading_status"])
    elif after["extractable"] is True:
        tags.append("extractable_but_incorrect")
    return tags


def analyze_series(series, epochs):
    panel, records = series["panel"], series["records"]
    base = {k: series[k] for k in ("series_id", "name", "origin")}
    coverage, pairs, events, questions = [], [], [], []
    for task in sorted({q["task"] for q in panel.values()}):
        ids = sorted(qid for qid, q in panel.items() if q["task"] == task)
        for epoch in epochs:
            rows = [records.get((epoch, sid)) for sid in ids]
            states = [state(r) for r in rows]
            graded = sum(r is not None for r in rows)
            valid = sum(v is not None for v in states)
            coverage.append({**base, "task": task, "epoch": epoch, "expected_questions": len(ids),
                             "saved_grades": graded, "valid_grades": valid,
                             "correct": states.count(True), "incorrect": states.count(False),
                             "missing": len(ids) - graded, "grading_unknown": graded - valid,
                             "complete": valid == len(ids), "accuracy_on_valid": states.count(True) / valid if valid else None,
                             "status": "complete" if valid == len(ids) else "partial" if graded else
                             "no_completed_grades" if epoch in series["planned_epochs"] else "not_requested_in_this_series"})
        observed = [e for e in epochs if any(state(records.get((e, sid))) is not None for sid in ids)]
        for i, before in enumerate(observed):
            for after in observed[i + 1:]:
                counts = Counter()
                for sid in ids:
                    a, b = state(records.get((before, sid))), state(records.get((after, sid)))
                    key = "unknown" if a is None or b is None else {
                        (True, False): "lost", (False, True): "gained", (True, True): "both_correct",
                        (False, False): "both_incorrect"}[a, b]
                    counts[key] += 1
                matched = len(ids) - counts["unknown"]
                pairs.append({**base, "task": task, "before_epoch": before, "after_epoch": after,
                              "expected_questions": len(ids), "matched": matched,
                              **{k: counts[k] for k in ("lost", "gained", "both_correct", "both_incorrect", "unknown")},
                              "paired_accuracy_delta": (counts["gained"] - counts["lost"]) / matched if matched else None})
    for qid, question in panel.items():
        available = [(e, records[e, qid]) for e in epochs if (e, qid) in records]
        valid = [(e, r) for e, r in available if state(r) is not None]
        details = {e: behavior(r) for e, r in available}
        local_events = []
        for index in range(1, len(valid)):
            before, a = valid[index - 1]
            after, b = valid[index]
            if state(a) is not True or state(b) is not False:
                continue
            recovery = next((e for e, r in valid[index + 1:] if state(r) is True), None)
            wrong_epochs = [e for e, r in valid[index:] if (recovery is None or e < recovery) and state(r) is False]
            event = {**base, "task": question["task"], "subtask": question.get("subtask"), "sample_id": qid,
                     "before_epoch": before, "after_epoch": after, "recovered_at_epoch": recovery,
                     "wrong_observed_epochs_until_recovery": wrong_epochs,
                     "unknown_epochs_between": [e for e in epochs if before < e < after and state(records.get((e, qid))) is None],
                     "no_observed_recovery": recovery is None, "clues": clues(details[before], details[after]),
                     "before_tokens": a.get("generated_tokens"), "after_tokens": b.get("generated_tokens"),
                     "before_extracted_answer": a.get("extracted_answer"), "after_extracted_answer": b.get("extracted_answer")}
            local_events.append(event)
        if local_events:
            filename = digest([series["series_id"], qid])[:24]
            for event in local_events:
                event["question_report"] = f"questions/{filename}.html"
                events.append(event)
            questions.append({"file_id": filename, **base, "question": question, "events": local_events,
                              "timeline": [{"epoch": e, "state": "missing" if (e, qid) not in records else
                                            "grading_unknown" if state(records[e, qid]) is None else
                                            "correct" if state(records[e, qid]) else "incorrect"} for e in epochs],
                              "answers": [{"epoch": e, "graded_response": r, "behavior": details[e],
                                           "sources": series["sources"][e, qid]} for e, r in available]})
    return {"coverage": coverage, "pairs": pairs, "events": events, "questions": questions}
