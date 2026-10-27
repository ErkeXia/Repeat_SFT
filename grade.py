"""CPU-only regrading from saved responses; never reload a language model."""
from __future__ import annotations

import argparse
from collections import defaultdict
import statistics

from recipe.common import environment, file_hash, fresh_dir, iter_jsonl, read_json, snapshot_code, write_json, write_jsonl
from recipe.grading import GRADER_VERSION, grade_one
from recipe.paths import artifact_path


def summarize(rows):
    def accuracy(subset):
        return sum(r["correct"] is True for r in subset) / len(subset) if subset else None

    normal = [r for r in rows if r["normal_chat_end"]]
    abnormal = [r for r in rows if not r["normal_chat_end"]]
    extractable = [r for r in rows if r["extractable"]]
    return {"n": len(rows), "correct": sum(r["correct"] is True for r in rows),
            "accuracy": accuracy(rows), "grading_errors": sum(r["grading_error"] is not None for r in rows),
            "normal_chat_end_rate": len(normal) / len(rows),
            "eos_end_rate": sum(r["ended_with_eos"] for r in rows) / len(rows),
            "length_limit_rate": sum(r["reached_length_limit"] for r in rows) / len(rows),
            "length_truncated_rate": sum(r["length_truncated"] for r in rows) / len(rows),
            "extractable_rate": len(extractable) / len(rows),
            "mean_generated_tokens": statistics.mean(r["generated_tokens"] for r in rows),
            "median_generated_tokens": statistics.median(r["generated_tokens"] for r in rows),
            "accuracy_given_normal_chat_end": accuracy(normal),
            "accuracy_given_abnormal_end": accuracy(abnormal),
            "accuracy_given_extractable": accuracy(extractable),
            "normal_chat_end_n": len(normal), "abnormal_end_n": len(abnormal)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--responses", required=True, help="One responses.jsonl or a generation root directory")
    p.add_argument("--out", required=True, help="NEW directory; use a new name when changing grading rules")
    p.add_argument("--allow-partial", action="store_true", help="Explicitly label incomplete panels for debugging")
    args = p.parse_args()
    import math_verify  # Missing symbolic verifier must fail, not silently degrade.
    source = artifact_path(args.responses)
    files = [source] if source.is_file() else sorted(source.glob("epoch-*/responses.jsonl"))
    if not files:
        raise ValueError("No response files")
    out = fresh_dir(artifact_path(args.out))
    code_hash = snapshot_code(out / "code")
    provenance, summaries = [], []
    for file in files:
        rows = list(iter_jsonl(file))
        identity = read_json(file.parent / "identity.json")
        expected = set(identity["selected_ids"])
        actual = {r["sample_id"] for r in rows}
        if len(actual) != len(rows) or not actual.issubset(expected):
            raise ValueError("Duplicate/unexpected response IDs")
        complete = actual == expected
        if not complete and not args.allow_partial:
            raise ValueError("Incomplete evaluation panel; use --allow-partial only for debugging")
        graded = [{**row, **grade_one(row["raw_response"], row["gold_answer"], row["task"]),
                   "grader_version": GRADER_VERSION} for row in rows]
        write_jsonl(out / file.parent.name / "graded.jsonl", graded)
        by_task = defaultdict(list)
        for row in graded:
            by_task[row["task"]].append(row)
        for task, members in by_task.items():
            summary = {"epoch": identity["epoch"], "task": task, "complete_panel": complete,
                       "expected_total_questions": len(expected), **summarize(members)}
            summaries.append(summary)
        provenance.append({"responses": str(file.resolve()), "sha256": file_hash(file), "identity": identity})
    write_jsonl(out / "summary.jsonl", summaries)
    write_json(out / "grading_manifest.json", {"grader_version": GRADER_VERSION, "code_sha256": code_hash,
                                               "sources": provenance, "environment": environment(),
                                               "errors_counted_in_denominator": True})
    for row in summaries:
        print(row)


if __name__ == "__main__":
    main()
