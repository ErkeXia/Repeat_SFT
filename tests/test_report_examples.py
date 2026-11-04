"""Saved-answer report tests: true categories, same-question reuse and full text."""
from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from recipe.common import ROOT, file_hash, iter_jsonl, read_json, write_json, write_jsonl
from report_examples import CATEGORIES, build_report, load_evidence, select_examples

EPOCHS = [4, 8, 16, 32, 64]
TEST_TMP = ROOT / ".test_tmp"
TEST_TMP.mkdir(exist_ok=True)


def fixture(root, *, truncated_only=False, completed_groups=4):
    run = root / "run"
    panel = [{"sample_id": f"q-{i}", "question_id": f"question-{i}", "task": "math500" if i % 2 == 0 else "gpqa",
              "question": f"Question {i} <tag> & symbols", "gold_answer": "2" if i % 2 == 0 else "B",
              "messages": [{"role": "user", "content": f"Question {i}\nA. first\nB. second"}],
              "choices": ["first", "second", "third", "fourth"] if i % 2 else [],
              "original": {"solution": "Reference reasoning"}} for i in range(8)]
    write_json(run / "run.json", {"fixture": True})
    write_jsonl(run / "data/panel.jsonl", panel)
    job = run / "postprocess" / "job"
    groups = [{"name": f"group-{i + 1:03d}", "selected_ids": [q["sample_id"] for q in panel[i * 2:i * 2 + 2]]}
              for i in range(4)]
    plan = {"run": str(run), "run_sha256": file_hash(run / "run.json"),
            "panel_sha256": file_hash(run / "data/panel.jsonl"), "panel_ids": [q["sample_id"] for q in panel],
            "generation_epochs": EPOCHS[::-1], "settings": {"max_new_tokens": 8}, "generation_backend": "vllm",
            "code_sha256": "frozen-generator", "groups": groups}
    write_json(job / "grouped_pipeline.json", plan)
    for gi, group in enumerate(groups[:completed_groups]):
        directory = job / "groups" / group["name"]
        grades = directory / "grades"
        sources = []
        for epoch in EPOCHS:
            rows = []
            for q in panel[gi * 2:gi * 2 + 2]:
                i = int(q["sample_id"][2:])
                truncated = truncated_only or i % 4 < 2
                ids = [7] * 8 if truncated else [7, 7, 2]
                # Includes HTML-like tokens and a long tail that must survive unmodified.
                response = f"<think>Question {i}</think>\n\\boxed{{2}}\n<script>alert(1)</script> &\n" + "tail " * 400
                rows.append({"sample_id": q["sample_id"], "question_id": q["question_id"], "task": q["task"],
                             "epoch": epoch, "messages": q["messages"], "gold_answer": q["gold_answer"],
                             "generated_token_ids": ids, "generated_tokens": len(ids), "prompt_token_ids": [10, 11],
                             "raw_response": response, "response_text": response, "seed": 42,
                             "reached_length_limit": truncated, "length_truncated": truncated,
                             "normal_chat_end": not truncated, "ended_with_eos": not truncated,
                             "stop_reason": "length" if truncated else "chat_end", "backend": "vllm"})
            identity = {"epoch": epoch, "run_sha256": plan["run_sha256"], "panel_sha256": plan["panel_sha256"],
                        "settings": plan["settings"], "code_sha256": plan["code_sha256"], "selected_ids": group["selected_ids"],
                        "backend": "vllm", "backend_config": {"concurrency": 8}, "generation_config": {"max_tokens": 8},
                        "seed_policy": "fixed"}
            raw = directory / "generation" / f"epoch-{epoch:03d}" / "responses.jsonl"
            write_jsonl(raw, rows)
            write_json(raw.parent / "identity.json", identity)
            graded = [{**r, "grader_version": "saved-test-grader", "correct": int(r["sample_id"][2:]) % 2 == 0,
                       "extracted_answer": "2", "extractable": True, "extraction_method": "boxed", "grading_error": None}
                      for r in rows]
            write_jsonl(grades / f"epoch-{epoch:03d}/graded.jsonl", graded)
            sources.append({"responses": str(raw), "sha256": file_hash(raw), "identity": identity})
        write_json(grades / "grading_manifest.json", {"sources": sources, "grader_version": "saved-test-grader",
                                                      "code_sha256": "frozen-grader"})
    return job


def hashes(path):
    return {str(p): file_hash(p) for p in path.rglob("*") if p.is_file()}


class FullAnswerParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_answer = False
        self.responses = []

    def handle_starttag(self, tag, attrs):
        if tag == "details" and dict(attrs).get("class") == "answer":
            self.in_answer = True
        if tag == "pre" and self.in_answer:
            self.responses.append("")
            self.in_answer = "pre"

    def handle_endtag(self, tag):
        if tag == "pre" and self.in_answer:
            self.in_answer = True
        if tag == "details":
            self.in_answer = False

    def handle_data(self, data):
        if self.in_answer == "pre":
            self.responses[-1] += data


class ExampleReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEST_TMP)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def test_twenty_slots_reuse_four_questions_and_preserve_full_evidence(self):
        job = fixture(self.root)
        before = hashes(job)
        out = self.root / "report"
        manifest = build_report(job, out, EPOCHS)
        self.assertEqual((manifest["slot_count"], manifest["selected_count"], manifest["distinct_questions"]), (20, 20, 4))
        slots = list(iter_jsonl(out / "selected_examples.jsonl"))
        for category, _, truncated, correct in CATEGORIES:
            members = [s for s in slots if s["category"] == category]
            self.assertEqual(len({s["sample_id"] for s in members}), 1)
            for slot in members:
                self.assertIs(slot["graded_response"]["length_truncated"], truncated)
                self.assertIs(slot["graded_response"]["correct"], correct)
                self.assertIn("question", slot["question_record"])
                self.assertEqual(slot["generated_tokens"], len(slot["graded_response"]["generated_token_ids"]))
        html = (out / "report.html").read_text(encoding="utf-8")
        parser = FullAnswerParser()
        parser.feed(html)
        self.assertEqual(parser.responses, [s["graded_response"]["raw_response"] for s in slots])
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html)
        for article in html.split("<article ")[1:]:
            self.assertIn("</pre></details></article>", article)
        self.assertEqual(before, hashes(job))
        saved = hashes(out)
        with self.assertRaises(FileExistsError):
            build_report(job, out, EPOCHS)
        self.assertEqual(saved, hashes(out))

    def test_all_truncated_keeps_ten_explicit_empty_slots(self):
        job = fixture(self.root, truncated_only=True)
        manifest = build_report(job, self.root / "report", EPOCHS)
        self.assertEqual((manifest["selected_count"], manifest["missing_slots"]), (10, 10))
        slots = list(iter_jsonl(self.root / "report/selected_examples.jsonl"))
        empty = [s for s in slots if s["status"] != "selected"]
        self.assertTrue(all(s["status"] == "category_empty" and s["candidate_count"] == 0 for s in empty))
        self.assertTrue(all("sample_id" not in s for s in empty))

    def test_partial_groups_and_grader_errors_are_not_incorrect_answers(self):
        job = fixture(self.root, completed_groups=1)
        path = job / "groups/group-001/grades/epoch-004/graded.jsonl"
        rows = list(iter_jsonl(path))
        rows[1].update(correct=None, grading_error="TimeoutError: fixture")
        write_jsonl(path, rows)
        manifest = build_report(job, self.root / "report", EPOCHS)
        self.assertEqual(manifest["coverage"][0], {"epoch": 4, "observed": 2, "expected": 8, "unknown": 1, "complete": False})
        slots = list(iter_jsonl(self.root / "report/selected_examples.jsonl"))
        failed = next(s for s in slots if s["epoch"] == 4 and s["category"] == "truncated_incorrect")
        self.assertEqual(failed["status"], "no_candidate_in_available_results")
        self.assertEqual(failed["candidate_count"], 0)

    def test_greedy_reuses_question_even_when_its_category_changes(self):
        records = {}
        for epoch_index, epoch in enumerate(EPOCHS):
            for i, (category, _, truncated, correct) in enumerate(CATEGORIES):
                sid = f"q-{(i + epoch_index) % 4}"
                records[epoch, sid] = {"row": {"correct": correct, "length_truncated": truncated, "grading_error": None}}
        candidates, choices = select_examples(records, EPOCHS, 42)
        _, reordered = select_examples(dict(reversed(list(records.items()))), EPOCHS, 42)
        self.assertEqual(choices, reordered)
        self.assertEqual(len(choices), 20)
        self.assertEqual(len(set(choices.values())), 4)
        self.assertTrue(all(sid in candidates[cell] for cell, sid in choices.items()))

    def test_retry_marker_and_merged_grades_do_not_duplicate_groups(self):
        job = fixture(self.root)
        first = job / "groups/group-001"
        retry = first / "grades-retry-test"
        (first / "grades").rename(retry)
        write_json(job / "grouped_state/group-001-grading.json", {"output": str(retry)})
        _, _, records, _, mode = load_evidence(job, EPOCHS)
        self.assertEqual((len(records), mode), (40, "completed_group_grades"))
        # Simulate a later successful merged grading result; groups still exist.
        plan = read_json(job / "grouped_pipeline.json")
        sources = []
        for epoch in EPOCHS:
            rows = [item["row"] for (e, _), item in records.items() if e == epoch]
            stripped = [{k: v for k, v in r.items() if k not in
                         {"grader_version", "correct", "extracted_answer", "extractable", "extraction_method", "grading_error"}}
                        for r in rows]
            raw = job / "generation" / f"epoch-{epoch:03d}/responses.jsonl"
            identity = read_json(first / "generation" / f"epoch-{epoch:03d}/identity.json")
            identity["selected_ids"] = plan["panel_ids"]
            write_jsonl(raw, stripped)
            write_json(raw.parent / "identity.json", identity)
            write_jsonl(job / "grades" / f"epoch-{epoch:03d}/graded.jsonl", rows)
            sources.append({"responses": str(raw), "sha256": file_hash(raw), "identity": identity})
        write_json(job / "grades/grading_manifest.json", {"sources": sources, "grader_version": "saved-test-grader", "code_sha256": "frozen-grader"})
        _, _, merged, _, mode = load_evidence(job, EPOCHS)
        self.assertEqual((len(merged), mode), (40, "full_panel"))
        merged_retry = job / "grades-retry-final"
        (job / "grades").rename(merged_retry)
        write_json(job / "grouped_complete.json", {"grades": str(merged_retry)})
        _, _, merged, _, mode = load_evidence(job, EPOCHS)
        self.assertEqual((len(merged), mode), (40, "full_panel"))

    def test_completed_source_corruption_duplicate_and_protocol_mix_are_rejected(self):
        job = fixture(self.root)
        raw = job / "groups/group-001/generation/epoch-004/responses.jsonl"
        original = raw.read_bytes()
        raw.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "responses changed"):
            load_evidence(job, EPOCHS)
        raw.write_bytes(original)
        graded = job / "groups/group-001/grades/epoch-004/graded.jsonl"
        rows = list(iter_jsonl(graded))
        write_jsonl(graded, rows + [rows[0]])
        with self.assertRaisesRegex(ValueError, "Duplicate/unexpected"):
            load_evidence(job, EPOCHS)
        write_jsonl(graded, rows)
        manifest = job / "groups/group-002/grades/grading_manifest.json"
        m = read_json(manifest)
        m["code_sha256"] = "different-grader"
        write_json(manifest, m)
        with self.assertRaisesRegex(ValueError, "mixed generation/grading"):
            load_evidence(job, EPOCHS)

    def test_eos_exactly_at_cap_is_not_truncated_and_text_changes_are_rejected(self):
        job = fixture(self.root)
        raw = job / "groups/group-002/generation/epoch-004/responses.jsonl"
        grade = job / "groups/group-002/grades/epoch-004/graded.jsonl"
        original, graded = list(iter_jsonl(raw)), list(iter_jsonl(grade))
        for row in (original[0], graded[0]):
            row.update(generated_token_ids=[7] * 7 + [2], generated_tokens=8, reached_length_limit=True)
        write_jsonl(raw, original)
        write_jsonl(grade, graded)
        path = job / "groups/group-002/grades/grading_manifest.json"
        manifest = read_json(path)
        manifest["sources"][0]["sha256"] = file_hash(raw)
        write_json(path, manifest)
        _, _, records, _, _ = load_evidence(job, EPOCHS)
        candidates, _ = select_examples(records, EPOCHS, 42)
        self.assertIn("q-2", candidates[4, "not_truncated_correct"])
        graded[0]["raw_response"] = "Edited answer"
        write_jsonl(grade, graded)
        with self.assertRaisesRegex(ValueError, "modified the original"):
            load_evidence(job, EPOCHS)

    def test_no_completed_grades_does_not_create_output(self):
        job = fixture(self.root, completed_groups=0)
        out = self.root / "report"
        with self.assertRaisesRegex(ValueError, "No completed grading"):
            build_report(job, out, EPOCHS)
        self.assertFalse(out.exists())

    def test_cli_uses_saved_grades_without_model_dependencies(self):
        job = fixture(self.root, completed_groups=2)
        out = self.root / "cli-report"
        result = subprocess.run([sys.executable, "-B", str(ROOT / "report_examples.py"),
                                 "--job", str(job), "--out", str(out)], cwd=ROOT,
                                capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("20/20", result.stdout)
        self.assertTrue((out / "report.html").is_file())
        self.assertEqual(read_json(out / "manifest.json")["epochs"], EPOCHS)


if __name__ == "__main__":
    unittest.main()
