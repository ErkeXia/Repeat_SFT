"""Dependency-free adapters and deterministic selection for evaluation-only panels."""
from __future__ import annotations

from collections import Counter
from datetime import date, datetime
import json
import re

from .common import digest
from .encoding import question_id

BBH_REPO = "suzgunmirac/BIG-Bench-Hard"
BBH_REVISION = "9ee07bd481feebf959a6b59d61ea57bdcf30964d"
LCB_REPO = "livecodebench/code_generation_lite"
LCB_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
BBH_TASKS = (
    "logical_deduction_five_objects", "tracking_shuffled_objects_five_objects",
    "boolean_expressions", "web_of_lies",
)
PROMPT_VERSION = "transfer-zero-shot-v1"


def lcb_files(version):
    """Explicit upstream v1..v6 mapping; never follow a moving release_latest."""
    match = re.fullmatch(r"release_v([1-6])|v([1-6])(?:_v([1-6]))?", version)
    if not match:
        raise ValueError("Use release_v1..release_v6, v1..v6, or a range such as v5_v6")
    if match[1]:
        first, last = 1, int(match[1])
    else:
        first, last = int(match[2]), int(match[3] or match[2])
    if first > last:
        raise ValueError("LiveCodeBench version range is reversed")
    return ["test.jsonl" if n == 1 else f"test{n}.jsonl" for n in range(first, last + 1)]


def quotas(size, tasks=BBH_TASKS):
    if size < len(tasks):
        raise ValueError(f"BBH size must be at least {len(tasks)} to include every subtask")
    return {task: size // len(tasks) + (i < size % len(tasks)) for i, task in enumerate(tasks)}


def text_field(row, name, empty=False):
    value = row.get(name)
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"Missing/non-text field: {name}")
    return value


def json_field(row, name, expected):
    value = row.get(name)
    value = json.loads(value) if isinstance(value, str) else value
    if not isinstance(value, expected):
        raise ValueError(f"Invalid {name}: expected {expected.__name__}")
    return value


def adapt_bbh(raw, task, location):
    question, target = text_field(raw, "input"), text_field(raw, "target").strip()
    if task == "boolean_expressions":
        valid, answer_format = {"True", "False"}, "True or False"
        gold = target
    elif task == "web_of_lies":
        valid, answer_format = {"Yes", "No"}, "Yes or No"
        gold = target
    else:
        valid, answer_format = {f"({c})" for c in "ABCDE"}, "one option letter A, B, C, D, or E"
        gold = target.strip("()")
    if target not in valid:
        raise ValueError(f"Unexpected BBH target in {task}: {target!r}")
    qid = question_id(question)
    instruction = ("\n\nReason through the problem. Finish with exactly one line in the form "
                   f"'Final answer: <answer>', where <answer> is {answer_format}.")
    panel = {"sample_id": "bbh-" + digest([task, qid]), "question_id": qid,
             "task": "bbh", "subtask": task, "source": BBH_REPO, "question": question,
             "messages": [{"role": "user", "content": question + instruction}],
             "gold_answer": gold, "answer_format": answer_format,
             "grading_method": "bbh_exact_answer", "source_location": location}
    return {"panel": panel, "original": raw}


def adapt_lcb(raw, location):
    question = text_field(raw, "question_content")
    title, platform = text_field(raw, "question_title"), text_field(raw, "platform")
    upstream_id = text_field(raw, "question_id")
    starter = text_field(raw, "starter_code", empty=True)
    contest_date = text_field(raw, "contest_date")
    parsed_date = datetime.fromisoformat(contest_date.replace("Z", "+00:00")).date()
    difficulty = text_field(raw, "difficulty")
    if platform not in {"leetcode", "codeforces", "atcoder"} or difficulty not in {"easy", "medium", "hard"}:
        raise ValueError("Unexpected LiveCodeBench platform/difficulty")
    public_tests = json_field(raw, "public_test_cases", list)
    if any(not isinstance(t, dict) or not {"input", "output", "testtype"} <= t.keys()
           or t["testtype"] not in {"stdin", "functional"} for t in public_tests):
        raise ValueError("Invalid LiveCodeBench public tests")
    metadata = json_field(raw, "metadata", dict)
    private = raw.get("private_test_cases")
    if not isinstance(private, (str, list)) or not private:
        raise ValueError("Missing LiveCodeBench private tests")
    # Private tests may be compressed pickles upstream. Preserve the opaque value;
    # data preparation must not unpickle it or execute any dataset/model code.
    func_name = metadata.get("func_name")
    if func_name is not None and not isinstance(func_name, str):
        raise ValueError("Invalid LiveCodeBench function name")
    functional = bool(starter.strip() or func_name or
                      (public_tests and public_tests[0]["testtype"] == "functional"))
    instruction = ("Solve this programming problem in Python 3. You may explain your reasoning first. "
                   "Finish with one complete solution inside a single ```python code block. ")
    instruction += ("Use the required function/class interface; do not add a test driver."
                    if functional else "Read from standard input and write to standard output.")
    prompt = instruction + "\n\n" + title + "\n\n" + question
    if starter.strip():
        prompt += "\n\nRequired starter code:\n```python\n" + starter + "\n```"
    elif func_name:
        prompt += "\n\nRequired function name: " + func_name
    panel = {"sample_id": "lcb-" + digest([platform, upstream_id]),
             "question_id": question_id(question), "task": "livecodebench",
             "source": LCB_REPO, "upstream_question_id": upstream_id,
             "question_title": title, "question": question, "platform": platform,
             "difficulty": difficulty, "contest_date": contest_date,
             "contest_day": parsed_date.isoformat(), "starter_code": starter,
             "messages": [{"role": "user", "content": prompt}],
             "gold_answer": None, "answer_format": "python3_code_block",
             "grading_method": "execution_public_and_private_tests",
             "interface": "functional" if functional else "stdin",
             "source_location": location}
    return {"panel": panel, "original": raw}


def rank(seed, row):
    return digest(["transfer-selection-v1", seed, row["panel"]["sample_id"]])


def select_stream(rows, size, seed, overlap, start_date=None, end_date=None, progress=None):
    """Keep bottom-k stable hash ranks; bounded by k complete source records.

    Selection is uniform pseudorandom within each supplied candidate pool, not
    biased toward early rows or toward short test cases. No length filtering.
    """
    selected, identities, questions, rejected = {}, {}, {}, []
    stats = Counter()
    for row in rows:
        stats["read"] += 1
        if progress:
            progress(stats["read"])
        p = row["panel"]
        sid, qid = p["sample_id"], p["question_id"]
        fingerprint = digest(row["original"])
        if sid in identities:
            if identities[sid] != fingerprint:
                raise ValueError(f"Conflicting duplicate sample: {sid}")
            stats["duplicate"] += 1
            continue
        identities[sid] = fingerprint
        if qid in questions:
            if questions[qid] != p["gold_answer"]:
                raise ValueError(f"Conflicting answers for question: {qid}")
            stats["duplicate"] += 1
            continue
        questions[qid] = p["gold_answer"]
        day = date.fromisoformat(p["contest_day"]) if p["task"] == "livecodebench" else None
        if day and ((start_date and day < start_date) or (end_date and day > end_date)):
            stats["outside_date_window"] += 1
            continue
        match = overlap.match(p)
        if match:
            stats["training_overlap"] += 1
            rejected.append({"sample_id": sid, "source_location": p["source_location"], **match})
            continue
        stats["eligible"] += 1
        key = rank(seed, row)
        if len(selected) < size:
            selected[key] = row
        elif key < max(selected):
            del selected[max(selected)]
            selected[key] = row
    return [selected[k] for k in sorted(selected)], dict(stats), rejected, identities
