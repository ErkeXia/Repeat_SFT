"""Resumable math/GPQA/BBH grading and Docker-isolated LiveCodeBench test execution."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import os
import signal

from grade import summarize
from recipe.common import (ROOT, append_jsonl, digest, environment, file_hash, iter_jsonl,
                           read_json, recover_jsonl, write_json, write_jsonl)
from recipe.evaluation import bind_output
from recipe.lcb_docker import DEFAULT_IMAGE, DockerJudge
from recipe.paths import artifact_path, configure_runtime
from recipe.pipeline import Pipeline
from recipe.transfer_eval import load_panel
from recipe.transfer_grading import GRADER_VERSION, extract_bbh, extract_code, test_payload

MATH_GPQA_GRADER_VERSION = "math-gpqa-final-v1"


def grade_answer(response, question, reference, judge, grader_version=GRADER_VERSION):
    result = {**response, "question": question["question"], "source": question["source"],
              "reference_file": question["reference_file"], "reference_sha256": question["reference_sha256"],
              "grader_version": grader_version, "grading_error": None,
              "response_sha256": digest(response)}
    for field in ("subtask", "platform", "difficulty", "contest_date", "starter_code", "interface"):
        if field in question:
            result[field] = question[field]
    if question["task"] in ("math500", "gpqa"):
        from recipe.grading import GRADER_VERSION as MATH_GRADER_VERSION, grade_one
        result.update(grade_one(response["raw_response"], question["gold_answer"], question["task"]))
        result.update(math_gpqa_grader_version=MATH_GRADER_VERSION,
                      grading_status="grading_error" if result["grading_error"] is not None else "graded")
    elif question["task"] == "bbh":
        answer, method = extract_bbh(response["raw_response"], question["subtask"])
        result.update(extracted_answer=answer, extraction_method=method, extractable=answer is not None,
                      correct=answer == question["gold_answer"], grading_status="graded")
    elif question["task"] == "livecodebench":
        code, method = extract_code(response["raw_response"])
        # Validate public + hidden tests even if code extraction failed. Broken
        # dataset records are infrastructure errors, never model errors.
        tests = test_payload(reference, question, judge.config)
        details = (judge.run(code, tests) if code is not None else
                   {"correct": False, "status": "no_code", "total_tests": len(tests["cases"]),
                    "tested": 0, "passed": 0, "test_results": []})
        result.update(extracted_answer=code, extraction_method=method, extractable=code is not None,
                      correct=details["correct"], grading_status=details["status"], code_execution=details,
                      public_tests=sum(c["split"] == "public" for c in tests["cases"]),
                      hidden_tests=sum(c["split"] == "private" for c in tests["cases"]))
    else:
        raise ValueError(f"Unsupported grading task: {question['task']}")
    return result


def grade_directory(source, panel_dir, out, judge, stop_check=lambda: None):
    source, panel_dir, out = Path(source), Path(panel_dir), Path(out)
    panel, panel_manifest = load_panel(panel_dir)
    math_panel = panel_manifest["format_version"] == "math-gpqa-evaluation-panel-v1"
    version = MATH_GPQA_GRADER_VERSION if math_panel else GRADER_VERSION
    if math_panel:
        import math_verify  # Infrastructure errors must not become model errors.
    elif judge is None:
        raise ValueError("LiveCodeBench grading requires the Docker judge")
    image, judge_config = (judge.image, judge.config) if judge is not None else (None, None)
    questions = {p["sample_id"]: p for p in panel}
    files = sorted(source.glob("epoch-*/responses.jsonl"))
    if not files:
        raise ValueError("No completed generation files")
    summaries, sources = [], []
    for file in files:
        stop_check()
        identity = read_json(file.parent / "identity.json")
        complete = read_json(file.parent / "complete.json")
        rows = list(iter_jsonl(file))
        selected = identity["selected_ids"]
        if (identity["panel_sha256"] != file_hash(panel_dir / "panel.jsonl")
                or identity.get("transfer_manifest_sha256") != file_hash(panel_dir / "manifest.json")
                or identity.get("backend") != "vllm" or len(selected) != len(set(selected))
                or len(rows) != len(selected) or {r["sample_id"] for r in rows} != set(selected)
                or not set(selected).issubset(questions)
                or complete != {"samples": len(rows), "responses_sha256": file_hash(file)}):
            raise ValueError("Incomplete/changed/mismatched transfer generation")
        epoch = identity["epoch"]
        directory = out / f"epoch-{epoch:03d}"
        binding = {"generation_identity": identity, "responses_sha256": file_hash(file),
                   "grader_version": version, "image": image, "judge_config": judge_config}
        bind_output(directory, binding)
        log = directory / "graded.jsonl"
        previous = recover_jsonl(log)
        done = {r["sample_id"]: r for r in previous}
        if len(done) != len(previous) or not set(done).issubset(selected):
            raise ValueError("Duplicate/unexpected graded answer")
        if (directory / "complete.json").exists():
            if read_json(directory / "complete.json") != {"samples": len(rows), "graded_sha256": file_hash(log)}:
                raise ValueError("Completed grades changed")
        for response in rows:
            stop_check()
            sid = response["sample_id"]
            question = questions[sid]
            if (response["epoch"] != epoch or response.get("backend") != "vllm"
                    or any(response[k] != question[k] for k in ("task", "question_id", "messages", "gold_answer"))):
                raise ValueError("Generated answer no longer matches the frozen question")
            if sid in done:
                if done[sid]["response_sha256"] != digest(response) or done[sid]["reference_sha256"] != question["reference_sha256"]:
                    raise ValueError("Graded source answer/reference changed")
                continue
            reference = read_json(panel_dir / question["reference_file"])
            result = grade_answer(response, question, reference, judge, version)
            append_jsonl(log, result)
            done[sid] = result
            print(f"grade epoch={epoch} {len(done)}/{len(rows)} task={question['task']} correct={result['correct']}", flush=True)
        write_json(directory / "complete.json", {"samples": len(rows), "graded_sha256": file_hash(log)})
        for task in ("math500", "gpqa", "bbh", "livecodebench"):
            members = [r for r in done.values() if r["task"] == task]
            if members:
                summaries.append({"epoch": epoch, "task": task, "complete_panel": True,
                                  "grading_status_counts": dict(Counter(r["grading_status"] for r in members)), **summarize(members)})
        sources.append({"epoch": epoch, "responses": str(file.resolve()), "sha256": file_hash(file),
                        "identity": identity, "graded_sha256": file_hash(log)})
    write_jsonl(out / "summary.jsonl", summaries)
    write_json(out / "grading_manifest.json", {"grader_version": version, "sources": sources,
               "panel_sha256": file_hash(panel_dir / "panel.jsonl"), "image": image,
               "config": judge_config, "environment": environment()})
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--responses", required=True)
    p.add_argument("--panel-dir", required=True)
    p.add_argument("--out", required=True, help="Separate directory; same command resumes unchanged grades")
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--judge-config", default=str(ROOT / "configs" / "lcb_grading.json"))
    args = p.parse_args()
    configure_runtime()
    out = artifact_path(args.out)
    source, panel = artifact_path(args.responses), artifact_path(args.panel_dir)
    if out.is_relative_to(source) or source.is_relative_to(out) or out.is_relative_to(panel) or panel.is_relative_to(out):
        p.error("Grades must have a separate output directory")
    with Pipeline(out) as pipeline:
        if os.name == "posix":
            os.environ["REPEAT_SFT_PIPELINE_LOCK_FD"] = str(pipeline.fd)
        # Unwind Docker.communicate before cleanup; do not wait inside a handler.
        def stop(signum, _frame):
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        _, manifest = load_panel(panel)
        judge = (None if manifest["format_version"] == "math-gpqa-evaluation-panel-v1"
                 else DockerJudge(args.image, read_json(args.judge_config)))
        grade_directory(source, panel, out, judge)


if __name__ == "__main__":
    main()
