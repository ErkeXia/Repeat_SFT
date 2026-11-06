"""Sparse, heterogeneous saved validation results; no GPU, Docker or downloads."""
from __future__ import annotations

import json
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from analyze_regressions import build_report
from recipe.common import ROOT, file_hash, read_json, write_json, write_jsonl
from recipe.regressions import analyze_series, behavior, clues, discover, state
from test_report_examples import FullAnswerParser

TEST_TMP = ROOT / ".test_tmp"
TEST_TMP.mkdir(exist_ok=True)


def question(sid, task):
    return {"sample_id": sid, "question_id": sid, "task": task, "question": f"Question {sid} <tag> & ?",
            "messages": [{"role": "user", "content": f"Question {sid} <tag> & ?"}],
            "gold_answer": None if task == "livecodebench" else "B" if task == "gpqa" else "2",
            "original": {"solution": "Full reference reasoning"}}


def response(q, epoch, correct):
    return {**{k: q[k] for k in ("sample_id", "question_id", "task", "messages", "gold_answer")},
            "epoch": epoch, "correct": correct, "grader_version": "test-v1",
            "grading_error": "verifier failed" if correct is None else None,
            "raw_response": f"<think>reasoning</think><script>never execute</script>\nAnswer {epoch}\n" + "full tail " * 600,
            "generated_tokens": 40, "generated_token_ids": list(range(39)) + [2],
            "length_truncated": False, "reached_length_limit": False, "ended_with_eos": True,
            "normal_chat_end": True, "stop_reason": "chat_end", "extractable": True,
            "extracted_answer": q["gold_answer"] if correct else "wrong", "extraction_method": "test"}


def start_run(root):
    run = root / "run"
    write_json(run / "run.json", {"synthetic": True})
    panel = [question("m", "math500"), question("g", "gpqa"), question("missing-m", "math500")]
    write_jsonl(run / "data/panel.jsonl", panel)
    return run, panel


def identity(run, panel_path, epoch, selected, limit=8192):
    return {"run_sha256": file_hash(run / "run.json"), "panel_sha256": file_hash(panel_path), "epoch": epoch,
            "selected_ids": selected, "settings": {"max_new_tokens": limit}, "backend": "vllm",
            "generation_config": {"temperature": .6}, "backend_config": {"concurrency": 8}, "seed_policy": "fixed"}


def math_job(run, panel, values, limits=None, standalone=False):
    job = run / "postprocess/math-job"
    selected = [q["sample_id"] for q in panel[:2]]
    plan = {"run": str(run), "run_sha256": file_hash(run / "run.json"),
            "panel_sha256": file_hash(run / "data/panel.jsonl"), "panel_ids": [q["sample_id"] for q in panel],
            "generation_epochs": [4, 8, 16, 32, 64], "groups": [{"name": "group-001", "selected_ids": selected}]}
    if not standalone:
        write_json(job / "grouped_pipeline.json", plan)
    grades = run / "grades-early" if standalone else job / "groups/group-001/grades"
    sources = []
    by_id = {q["sample_id"]: q for q in panel}
    for epoch, states in values.items():
        rows = [response(by_id[sid], epoch, correct) for sid, correct in states.items()]
        raw_path = grades.parent / "generation" / f"epoch-{epoch:03d}" / "responses.jsonl"
        write_jsonl(raw_path, rows)
        ident = identity(run, run / "data/panel.jsonl", epoch, selected, (limits or {}).get(epoch, 8192))
        write_json(raw_path.parent / "identity.json", ident)
        write_jsonl(grades / f"epoch-{epoch:03d}/graded.jsonl", rows)
        sources.append({"identity": ident, "responses": str(raw_path), "sha256": file_hash(raw_path)})
    if values:
        write_json(grades / "grading_manifest.json", {"sources": sources, "grader_version": "test-v1", "code_sha256": "code"})
    return job, grades


def transfer_job(run, values, partial=False):
    job = run / "postprocess/transfer-job"
    panel = [question("b", "bbh"), question("l", "livecodebench")]
    panel[0]["subtask"] = "boolean_expressions"
    panel_path = job / "setup/panel/panel.jsonl"
    write_jsonl(panel_path, panel)
    groups = [{"name": f"group-{i+1:03d}", "selected_ids": [q["sample_id"]]} for i, q in enumerate(panel)]
    write_json(job / "setup/plan.json", {"run": str(run), "run_sha256": file_hash(run / "run.json"),
               "panel_sha256": file_hash(panel_path), "generation_epochs": [8, 16, 32], "groups": groups})
    for group, q in zip(groups, panel):
        for epoch, correct in values.items():
            directory = job / "groups" / group["name"] / "grades" / f"epoch-{epoch:03d}"
            row = response(q, epoch, correct)
            if q["task"] == "livecodebench":
                row["grading_status"] = "passed" if correct else "runtime_error"
                row["code_execution"] = {"correct": correct, "status": row["grading_status"],
                                         "test_results": [{"stderr": "ZeroDivisionError example", "actual_output": ""}]}
            write_jsonl(directory / "graded.jsonl", [row])
            write_json(directory / "identity.json", {"generation_identity": identity(run, panel_path, epoch, group["selected_ids"]),
                        "grader_version": "test-v1", "image": {"id": "fixed-image"}, "judge_config": {"case_timeout_seconds": 6}})
            write_json(directory / "complete.json", {"samples": 1, "graded_sha256": file_hash(directory / "graded.jsonl")})
        if partial:
            directory = job / "groups" / group["name"] / "grades/epoch-016"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "graded.jsonl").write_text('{"torn_record":', encoding="utf-8")
    # A merged duplicate is deliberately invalid: analysis must use committed group originals only.
    write_json(job / "grades/grading_manifest.json", {"group_manifests": {}})
    write_jsonl(job / "grades/epoch-032/graded.jsonl", [{"duplicate": "must not be read"}])
    return job


class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEST_TMP)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.run, self.panel = start_run(self.root)

    def test_sparse_per_dataset_epochs_and_partial_groups_end_to_end(self):
        math_job(self.run, self.panel, {4: {"g": True}, 8: {"m": True, "g": False},
                                      16: {"m": None}, 32: {"m": False}, 64: {"m": True}})
        transfer_job(self.run, {8: True, 32: False}, partial=True)
        out = self.run / "analysis-regressions-v1"
        source_hashes = {str(p): file_hash(p) for p in self.run.rglob("*") if p.is_file()}
        results, manifest = build_report(self.run, out)
        self.assertEqual(len(results["events"]), 4)
        events = {e["task"]: e for e in results["events"]}
        self.assertEqual((events["math500"]["before_epoch"], events["math500"]["after_epoch"]), (8, 32))
        self.assertEqual(events["math500"]["unknown_epochs_between"], [16])
        self.assertEqual(events["math500"]["recovered_at_epoch"], 64)
        self.assertEqual((events["gpqa"]["before_epoch"], events["gpqa"]["after_epoch"]), (4, 8))
        self.assertEqual((events["bbh"]["before_epoch"], events["bbh"]["after_epoch"]), (8, 32))
        self.assertIn("code_runtime_error", events["livecodebench"]["clues"])
        cov = {(r["task"], r["epoch"]): r for r in results["coverage"]}
        self.assertEqual(cov["gpqa", 32]["incorrect"], 0)
        self.assertEqual(cov["gpqa", 32]["missing"], 1)
        self.assertEqual(cov["math500", 16]["grading_unknown"], 1)
        self.assertEqual(cov["bbh", 4]["status"], "not_requested_in_this_series")
        pair = next(r for r in results["pairs"] if r["task"] == "math500" and r["before_epoch"] == 8 and r["after_epoch"] == 32)
        self.assertEqual((pair["matched"], pair["lost"], pair["unknown"]), (1, 1, 1))
        self.assertEqual(pair["paired_accuracy_delta"], -1)
        self.assertEqual(len(manifest["notes"]), 2)
        for p, sha in source_hashes.items():
            self.assertEqual(file_hash(p), sha)
        for item in results["questions"]:
            saved = read_json(out / "questions" / (item["file_id"] + ".json"))
            self.assertEqual(saved["answers"], item["answers"])
            html = (out / "questions" / (item["file_id"] + ".html")).read_text(encoding="utf-8")
            self.assertNotIn("<script>never execute</script>", html)
            parser = FullAnswerParser()
            parser.feed(html)
            for answer in item["answers"]:
                self.assertIn(answer["graded_response"]["raw_response"], parser.responses)
        lcb = next(q for q in results["questions"] if q["question"]["task"] == "livecodebench")
        self.assertIn("ZeroDivisionError example", (out / "questions" / (lcb["file_id"] + ".html")).read_text(encoding="utf-8"))
        with self.assertRaises(FileExistsError):
            build_report(self.run, out)

    def test_recovery_relapse_and_all_later_wrong_pairs(self):
        math_job(self.run, self.panel, {4: {"m": True}, 8: {"m": False}, 16: {"m": False},
                                      32: {"m": True}, 64: {"m": False}})
        result, _ = build_report(self.run, self.run / "analysis")
        self.assertEqual([(e["before_epoch"], e["after_epoch"]) for e in result["events"]], [(4, 8), (32, 64)])
        self.assertEqual(result["events"][0]["wrong_observed_epochs_until_recovery"], [8, 16])
        self.assertEqual(result["events"][0]["recovered_at_epoch"], 32)
        self.assertTrue(result["events"][1]["no_observed_recovery"])
        pair = next(r for r in result["pairs"] if r["task"] == "math500" and r["before_epoch"] == 4 and r["after_epoch"] == 16)
        self.assertEqual(pair["lost"], 1)

    def test_protocols_are_separate_and_epoch_filter_is_respected(self):
        job, _ = math_job(self.run, self.panel, {4: {"m": True}, 8: {"m": False}}, limits={8: 30000})
        result, manifest = build_report(self.run, self.run / "analysis", jobs=[job], epochs=[4, 8])
        self.assertEqual(len(manifest["series"]), 2)
        self.assertEqual(result["events"], [])
        self.assertEqual(result["pairs"], [])
        self.assertEqual({r["epoch"] for r in result["coverage"]}, {4, 8})

    def test_no_grades_are_unknown_and_no_regressions(self):
        math_job(self.run, self.panel, {})
        result, _ = build_report(self.run, self.run / "analysis")
        self.assertEqual(result["events"], [])
        self.assertTrue(all(r["incorrect"] == 0 and r["valid_grades"] == 0 for r in result["coverage"]))
        self.assertIn("未发现答对→答错", (self.run / "analysis/report.html").read_text(encoding="utf-8"))

    def test_duplicate_and_changed_completion_fail_before_output(self):
        _, grades = math_job(self.run, self.panel, {4: {"m": True}})
        path = grades / "epoch-004/graded.jsonl"
        row = json.loads(path.read_text(encoding="utf-8"))
        write_jsonl(path, [row, row])
        out = self.run / "analysis"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_report(self.run, out)
        self.assertFalse(out.exists())
        job = transfer_job(self.run, {8: True})
        marker = job / "groups/group-001/grades/epoch-008/complete.json"
        write_json(marker, {"samples": 1, "graded_sha256": "changed"})
        with self.assertRaisesRegex(ValueError, "completion"):
            build_report(self.run, out, jobs=[job])
        self.assertFalse(out.exists())

    def test_standalone_grades_and_explicit_selection(self):
        _, grades = math_job(self.run, self.panel, {0: {"m": True}, 8: {"m": False}}, standalone=True)
        transfer_job(self.run, {8: True, 32: False})
        result, _ = build_report(self.run, self.run / "analysis", grades=[grades])
        self.assertEqual(len(result["events"]), 1)
        self.assertEqual(result["events"][0]["before_epoch"], 0)
        self.assertEqual(discover(self.run)[1], [grades])

    def test_behavior_flags_are_observations_and_malformed_grading_is_unknown(self):
        q = self.panel[0]
        a, b = response(q, 8, True), response(q, 32, False)
        b.update(generated_token_ids=[7]*200, generated_tokens=200, length_truncated=True,
                 ended_with_eos=False, extractable=False, raw_response="<think>" + ("repeated reasoning step " * 4 + "\n") * 4)
        tags = clues(behavior(a), behavior(b))
        for tag in ("new_truncation", "lost_eos", "lost_answer_extraction", "increased_token_repetition",
                    "length_at_least_doubled", "more_repeated_long_lines", "unclosed_think"):
            self.assertIn(tag, tags)
        self.assertIsNone(state({"correct": False, "grading_error": "parse failed"}))
        self.assertIsNone(state(None))

    def test_source_panel_mismatch_and_output_in_job_are_rejected(self):
        job, grades = math_job(self.run, self.panel, {4: {"m": True}, 8: {"m": False}})
        with self.assertRaisesRegex(ValueError, "separate NEW"):
            build_report(self.run, job / "report-new")
        rows = read_json(grades / "grading_manifest.json")
        rows["sources"][0]["identity"]["panel_sha256"] = "other-panel"
        write_json(grades / "grading_manifest.json", rows)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            build_report(self.run, self.run / "analysis")

    def test_help_does_not_need_gpu_or_data(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "analyze_regressions.py"), "--help"],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--grades", result.stdout)

    def test_existing_math_grouped_fixture_is_supported(self):
        from test_report_examples import fixture
        job = fixture(self.root / "existing-math", completed_groups=2)
        run = job.parent.parent
        result, _ = build_report(run, run / "analysis-regressions")
        self.assertEqual(result["events"], [])
        self.assertTrue(all(r["valid_grades"] == 2 and r["expected_questions"] == 4 for r in result["coverage"]))

    def test_real_transfer_grader_artifacts_are_supported(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from grade_transfer import grade_directory
        import transfer_evaluation
        from test_replay import fixture
        from test_transfer_evaluation import FakeJudge, panel_fixture, synthetic_responses
        run = fixture(self.root / "existing-transfer")
        panel_dir = self.root / "existing-transfer/panel"
        panel_fixture(panel_dir, file_hash(run / "data/train.jsonl"))
        job = run / "postprocess/transfer"
        job.mkdir(parents=True)
        with patch.object(transfer_evaluation, "DockerJudge", FakeJudge):
            plan = transfer_evaluation.prepare(run, panel_dir, job, SimpleNamespace(epochs=None, image=None, judge_config=None))
        group = plan["groups"][0]
        synthetic_responses(job, group, [8, 32])
        directory = job / "groups" / group["name"]
        with contextlib.redirect_stdout(io.StringIO()):
            grade_directory(directory / "generation", job / "setup/panel", directory / "grades", FakeJudge())
            result, _ = build_report(run, run / "analysis-regressions")
        self.assertEqual(result["events"], [])
        coverage = {(r["task"], r["epoch"]): r for r in result["coverage"]}
        self.assertEqual(coverage["bbh", 8]["valid_grades"], 25)
        self.assertEqual(coverage["livecodebench", 32]["valid_grades"], 25)
        self.assertEqual(coverage["livecodebench", 16]["valid_grades"], 0)


if __name__ == "__main__":
    unittest.main()
