"""Synthetic saved experiments: checkpoint timing, joins, missingness and portable evidence."""
from __future__ import annotations

import csv
from html.parser import HTMLParser
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from analyze_run import Analysis
from generate import stop_details
from recipe.analysis import (correctness_milestones, fit_milestones, paired_changes,
                             tail_stable_epoch, task_milestones)
from recipe.common import (ROOT, digest, file_hash, iter_jsonl, read_json, stable_seed, write_json, write_jsonl)

TEST_TMP = ROOT / ".test_tmp"
TEST_TMP.mkdir(exist_ok=True)
EPOCHS = [0, 1, 2, 4, 8]


def experiment(root):
    """Only JSON and tiny synthetic NPZs; no model, tokenizer or dataset downloads."""
    import numpy as np
    run = root / "run"
    write_json(run / "tokenizer" / "tokenizer_config.json", {"fixture": "tokenizer metadata, no pretrained model"})
    train = [{"sample_id": f"t{i}", "question_id": f"qt{i}", "question": f"Training question {i}",
              "source": "synthetic", "response": "Full reasoning step one. Step two. Final answer: 2",
              "answer": "2", "messages": [{"role": "user", "content": f"Training question {i}"}],
              "n_supervised": i * 1000} for i in (1, 2)]
    holdout = [{**train[0], "sample_id": "h1", "question_id": "qh1", "question": "Unseen holdout question"}]
    panel = [{"sample_id": sid, "question_id": "q" + sid, "task": task,
              "question": "Question " + sid + (' <script>alert("data")</script>' if sid == "m1" else ''),
              "gold_answer": "A" if task == "gpqa" else "2",
              "messages": [{"role": "user", "content": "Prompt " + sid}],
              "original": {"solution": "Reference reasoning, then answer 2"} if task == "math500" else {"Correct Answer": "Choice one"}}
             for sid, task in (("m1", "math500"), ("m2", "math500"), ("g1", "gpqa"))]
    for split, rows in (("train", train), ("holdout", holdout), ("panel", panel)):
        write_jsonl(run / "data" / f"{split}.jsonl", rows)
    metadata = {"config": {"checkpoint_epochs": EPOCHS, "epochs": 8, "expected_train_size": 2},
                "config_sha256": "fixture-config", "model": {"id": "synthetic", "revision": "fixture"},
                "tokenizer_files": {"tokenizer_config.json": file_hash(run / "tokenizer" / "tokenizer_config.json")},
                "data_files": {s + ".jsonl": file_hash(run / "data" / f"{s}.jsonl") for s in ("train", "holdout", "panel")}}
    write_json(run / "run.json", metadata)
    losses = {"t1": [1, .2, .105, .102, .1], "t2": [.8] * 5, "h1": [1, .8, .7, .65, .6]}
    accuracies = {"t1": [.5, .9, .99, .991, .992], "t2": [.5] * 5, "h1": [.4, .5, .6, .65, .7]}
    correct = {"m1": [False, False, True, False, True], "m2": [True] * 5, "g1": [False, False, False, True, True]}
    sources = []
    grades = run / "grades-v1"
    for index, epoch in enumerate(EPOCHS):
        checkpoint = {"config_sha256": "fixture-config", "step": 2 * epoch}
        fit = run / "evaluations" / "teacher_forcing" / f"epoch-{epoch:03d}"
        identity = {"epoch": epoch, "run_sha256": file_hash(run / "run.json"), "checkpoint": checkpoint,
                    "chunk_size": 128, "selected_ids": {"train": [r["sample_id"] for r in train], "holdout": ["h1"]},
                    "data_sha256": {s: file_hash(run / "data" / f"{s}.jsonl") for s in ("train", "holdout")},
                    "code_sha256": "fixture-evaluator"}
        write_json(fit / "identity.json", identity)
        for split, rows in (("train", train), ("holdout", holdout)):
            metrics = []
            for row in rows:
                sid, count = row["sample_id"], row["n_supervised"]
                loss, acc = losses[sid][index], accuracies[sid][index]
                ncorrect = round(count * acc)
                token = fit / split / "tokens" / f"{sid}.npz"
                token.parent.mkdir(parents=True, exist_ok=True)
                predicted = np.zeros(count, dtype=np.int64)
                predicted[ncorrect:] = 1
                np.savez_compressed(token, positions=np.arange(count), nll=np.full(count, loss),
                                    target_ids=np.zeros(count, dtype=np.int64), predicted_ids=predicted)
                metrics.append({"sample_id": sid, "question_id": row["question_id"], "source": "synthetic",
                    "n_tokens": count, "correct_tokens": ncorrect, "token_accuracy": ncorrect / count,
                    "nll_sum": loss * count, "mean_loss": loss, "end_token_loss": loss,
                    "end_token_correct": ncorrect == count, "token_file": f"tokens/{sid}.npz"})
            write_jsonl(fit / split / "samples.jsonl", metrics)
        gen = run / "evaluations" / "generation" / f"epoch-{epoch:03d}"
        raw_rows, graded = [], []
        for question in panel:
            sid = question["sample_id"]
            ok = correct[sid][index]
            normal = ok or (sid == "m1" and epoch == 4)
            extractable = not (sid == "m1" and epoch in (0, 1))
            ids = [7] * (4 if normal else 31) + [2 if normal else 7]
            answer = question["gold_answer"] if ok else "B" if sid == "g1" else "3"
            raw = {"sample_id": sid, "question_id": question["question_id"], "task": question["task"],
                   "epoch": epoch, "seed": stable_seed(2026, sid), "messages": question["messages"],
                   "gold_answer": question["gold_answer"], "generated_token_ids": ids, "prompt_token_ids": [1, 8],
                   "raw_response": "Reasoning step. \\boxed{" + answer + "}" + ("<|im_end|>" if normal else ""),
                   "response_text": "Reasoning step. Final answer " + answer,
                   "generated_tokens": len(ids), "prompt_tokens": 2, "generation_seconds": 1.5,
                   **stop_details(ids, 3, 2, 32)}
            raw_rows.append(raw)
            graded.append({**raw, "correct": ok, "extractable": extractable,
                           "extracted_answer": answer if extractable else None,
                           "grading_error": None, "extraction_method": "fixture", "grader_version": "fixture-v1"})
        write_jsonl(gen / "responses.jsonl", raw_rows)
        identity = {"run_sha256": file_hash(run / "run.json"), "epoch": epoch, "checkpoint": checkpoint,
                    "settings": {"seed": 2026, "temperature": .6, "max_new_tokens": 32},
                    "generation_config": {"do_sample": True}, "selected_ids": [r["sample_id"] for r in panel],
                    "panel_sha256": file_hash(run / "data" / "panel.jsonl"), "seed_policy": "fixture-fixed-question-seed"}
        write_json(gen / "identity.json", identity)
        write_jsonl(grades / gen.name / "graded.jsonl", graded)
        sources.append({"responses": str(gen / "responses.jsonl"), "sha256": file_hash(gen / "responses.jsonl"), "identity": identity})
    write_json(grades / "grading_manifest.json", {"sources": sources, "grader_version": "fixture-v1", "code_sha256": "fixture-grader"})
    return run


def arguments(run, out, **overrides):
    args = dict(run=str(run), out=str(out), grades=None, teacher_forcing=None, epochs=None, allow_partial=False,
                loss_tolerance=.01, accuracy_tolerance=.005, saturation_min_points=3,
                token_accuracy_target=.99, correct_min_points=2)
    args.update(overrides)
    return SimpleNamespace(**args)


class TrajectoryTests(unittest.TestCase):
    def test_plateau_uses_total_tail_range_future_evidence_and_missingness(self):
        epochs = [0, 1, 2, 4, 8, 16]
        self.assertEqual(tail_stable_epoch(epochs, [1, .3, .104, .102, .1, .1], .01, 3), 2)
        self.assertIsNone(tail_stable_epoch(epochs, [1, .4, .3, .2, .106, .1], .01, 3))
        self.assertIsNone(tail_stable_epoch(epochs, [.1, .108, .116, .124, .132, .14], .01, 3))
        self.assertIsNone(tail_stable_epoch(epochs, [.1, .1, .1, .1, None, .1], .01, 3))
        # A high-loss plateau is stable, but is not falsely labeled as memorization.
        result = fit_milestones(epochs, [{"mean_loss": 5, "token_accuracy": .1}] * 6)
        self.assertEqual(result["joint_tail_stable_epoch"], 0)
        self.assertIsNone(result["first_accuracy_target_epoch"])

    def test_first_correct_baseline_regression_persistence_and_unknown(self):
        obs = lambda states: [{"correct": s, "grading_error": None} if s is not None else None for s in states]
        m = correctness_milestones(EPOCHS, obs([False, False, True, False, True]))
        self.assertEqual(m["first_observed_correct_epoch"], 2)
        self.assertEqual(m["first_correct_after_last_observed_wrong_epoch"], 1)
        self.assertEqual(m["regression_intervals"], [(2, 4)])
        self.assertIsNone(m["sustained_correct_epoch"])
        self.assertEqual(correctness_milestones(EPOCHS, obs([False, False, False, True, True]))["sustained_correct_epoch"], 4)
        self.assertTrue(correctness_milestones(EPOCHS, obs([True] * 5))["first_correct_is_baseline"])
        missing = correctness_milestones(EPOCHS, obs([False, None, False, False, False]))
        self.assertIsNone(missing["never_correct_on_evaluated_checkpoints"])
        self.assertEqual(missing["unknown_epochs"], [1])

    def test_task_improvement_needs_base_and_tracks_reversals(self):
        curves = [{"epoch": e, "accuracy": a, "complete": True} for e, a in zip(EPOCHS, [.5, .75, .5, .75, 1])]
        result = task_milestones(EPOCHS, curves)
        self.assertEqual(result["first_observed_above_base_epoch"], 1)
        self.assertEqual(result["sustained_above_base_epoch"], 4)
        self.assertEqual(result["best_epochs"], [8])
        self.assertIsNone(task_milestones(EPOCHS[1:], curves[1:])["first_observed_above_base_epoch"])

    def test_paired_behavior_groups_overlap_and_missing_does_not_shrink_denominator(self):
        def row(correct, normal, extractable):
            return {"correct": correct, "normal_chat_end": normal, "extractable": extractable, "grading_error": None}
        before = {"a": row(False, False, False), "b": row(True, True, True), "c": row(False, True, True)}
        after = {"a": row(True, True, True), "b": row(False, True, True), "c": row(True, True, True)}
        paired = paired_changes(list(before), before, after)
        self.assertEqual(paired["gained_n"], 2)
        self.assertEqual(paired["lost_n"], 1)
        self.assertEqual(paired["gained_with_normal_end_improvement_n"], 1)
        self.assertEqual(paired["gained_with_extraction_improvement_n"], 1)
        self.assertEqual(paired["gained_with_unchanged_end_and_extraction_n"], 1)
        self.assertAlmostEqual(paired["accuracy_change"], 1 / 3)
        del after["b"]
        self.assertIsNone(paired_changes(list(before), before, after)["accuracy_change"])


class SavedAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEST_TMP)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = experiment(self.root)

    def analyze(self, name="analysis", **options):
        analysis = Analysis(arguments(self.run, self.root / name, **options))
        analysis.execute()
        return analysis.out

    def test_end_to_end_values_evidence_links_and_read_only_inputs(self):
        hashes = {p: file_hash(p) for p in self.run.rglob("*") if p.is_file()}
        out = self.analyze()
        summary = read_json(out / "summary.json")
        train_zero = next(r for r in summary["fit_curves"] if r["split"] == "train" and r["epoch"] == 0)
        self.assertAlmostEqual(train_zero["sample_mean_loss"], .9)
        self.assertAlmostEqual(train_zero["token_weighted_loss"], 2600 / 3000)
        with (out / "fit_saturation.csv").open(encoding="utf-8") as f:
            saturation = {r["sample_id"]: r for r in csv.DictReader(f)}
        self.assertEqual(saturation["t1"]["joint_tail_stable_epoch"], "2")
        self.assertEqual(saturation["t2"]["loss_tail_stable_epoch"], "0")
        self.assertEqual(saturation["t2"]["first_accuracy_target_epoch"], "")
        self.assertEqual(saturation["h1"]["joint_tail_stable_epoch"], "")
        with (out / "question_milestones.csv").open(encoding="utf-8") as f:
            questions = {r["sample_id"]: r for r in csv.DictReader(f)}
        self.assertEqual(questions["m1"]["first_observed_correct_epoch"], "2")
        self.assertEqual(questions["m1"]["regression_intervals"], "[[2, 4]]")
        self.assertEqual(questions["g1"]["sustained_correct_epoch"], "4")
        g4 = next(r for r in summary["downstream_curves"] if r["task"] == "gpqa" and r["epoch"] == 4)
        self.assertEqual(g4["accuracy"], 1)
        self.assertEqual(g4["normal_chat_end_rate"], 1)
        dossier = read_json(out / questions["m1"]["evidence_file"])
        response = read_json(out / dossier["trajectory"][2]["response_file"])
        self.assertIn("Reasoning step", response["raw_response"])
        self.assertTrue(response["generated_token_ids"])
        self.assertEqual(dossier["sample"]["original"]["solution"], "Reference reasoning, then answer 2")
        fit_dossier = read_json(out / saturation["t1"]["evidence_file"])
        self.assertIn("Step two", fit_dossier["sample"]["response"])
        token = out / fit_dossier["trajectory"][2]["token_file"]
        original = self.run / "evaluations" / "teacher_forcing" / "epoch-002" / "train" / "tokens" / "t1.npz"
        self.assertEqual(token.read_bytes(), original.read_bytes())
        self.assertEqual(read_json(out / "analysis_status.json")["status"], "complete")
        self.assertEqual((out / "inputs" / "tokenizer" / "tokenizer_config.json").read_bytes(),
                         (self.run / "tokenizer" / "tokenizer_config.json").read_bytes())
        for path, checksum in hashes.items():
            self.assertEqual(file_hash(path), checksum)
        class Links(HTMLParser):
            def handle_starttag(self, tag, attrs):
                if tag == "a":
                    self.hrefs.append(dict(attrs)["href"])
        for path in out.rglob("*.html"):
            parser = Links()
            parser.hrefs = []
            parser.feed(path.read_text(encoding="utf-8"))
            for href in parser.hrefs:
                target = (path.parent / href).resolve()
                self.assertTrue(target.is_relative_to(out.resolve()), href)
                self.assertTrue(target.is_file(), str(target))
        question_html = (out / questions["m1"]["report_file"]).read_text(encoding="utf-8")
        self.assertNotIn('<script>alert("data")</script>', question_html)
        self.assertIn('&lt;script&gt;', question_html)
        with self.assertRaises(FileExistsError):
            Analysis(arguments(self.run, out))

    def test_missing_checkpoint_fails_and_partial_keeps_unknown(self):
        missing = self.run / "grades-v1" / "epoch-002" / "graded.jsonl"
        missing.unlink()
        with self.assertRaisesRegex(ValueError, "Missing graded answers"):
            self.analyze("strict")
        out = self.analyze("partial", allow_partial=True)
        summary = read_json(out / "summary.json")
        self.assertTrue(all(r["accuracy"] is None for r in summary["downstream_curves"] if r["epoch"] == 2))
        dossier = read_json(out / "evidence" / "questions" / (digest("m2") + ".json"))
        self.assertEqual(dossier["milestones"]["unknown_epochs"], [2])
        self.assertIsNone(dossier["milestones"]["never_correct_on_evaluated_checkpoints"])
        self.assertTrue(summary["issues"])

    def test_grading_error_is_unknown_and_not_a_loss_transition(self):
        path = self.run / "grades-v1" / "epoch-002" / "graded.jsonl"
        rows = list(iter_jsonl(path))
        rows[0].update(correct=None, grading_error="fixture verifier timeout")
        write_jsonl(path, rows)
        with self.assertRaisesRegex(ValueError, "Grading error"):
            self.analyze("strict")
        out = self.analyze("partial", allow_partial=True)
        dossier = read_json(out / "evidence" / "questions" / (digest("m1") + ".json"))
        self.assertEqual(dossier["milestones"]["unknown_epochs"], [2])
        self.assertEqual(dossier["milestones"]["regression_intervals"], [])

    def test_legacy_grades_with_missing_originals_remain_portable(self):
        for path in (self.run / "evaluations" / "generation").glob("epoch-*/responses.jsonl"):
            path.unlink()
        out = self.analyze()
        self.assertTrue(read_json(out / "summary.json")["issues"])
        response = read_json(out / "evidence" / "responses" / "epoch-008" / (digest("g1") + ".json"))
        self.assertTrue(response["correct"])
        self.assertTrue(response["raw_response"])

    def test_inconsistent_raw_answer_and_generation_settings_are_rejected(self):
        path = self.run / "grades-v1" / "epoch-002" / "graded.jsonl"
        rows = list(iter_jsonl(path))
        rows[0]["raw_response"] += "unexpected modification"
        write_jsonl(path, rows)
        with self.assertRaisesRegex(ValueError, "differs from original"):
            self.analyze("corrupted")
        rows[0]["raw_response"] = next(iter_jsonl(self.run / "evaluations" / "generation" / "epoch-002" / "responses.jsonl"))["raw_response"]
        write_jsonl(path, rows)
        manifest_path = self.run / "grades-v1" / "grading_manifest.json"
        manifest = read_json(manifest_path)
        manifest["sources"][-1]["identity"]["settings"]["temperature"] = .9
        write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "different prompts/panels/settings"):
            self.analyze("settings")

    def test_changed_original_hash_or_frozen_dataset_is_rejected(self):
        raw = self.run / "evaluations" / "generation" / "epoch-002" / "responses.jsonl"
        original = raw.read_bytes()
        raw.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "Original responses hash"):
            self.analyze("raw_hash")
        raw.write_bytes(original)
        train = self.run / "data" / "train.jsonl"
        train.write_bytes(train.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "Frozen run data changed"):
            self.analyze("changed_data")

    def test_duplicate_teacher_forcing_and_checkpoint_identity_are_rejected(self):
        path = self.run / "evaluations" / "teacher_forcing" / "epoch-002" / "train" / "samples.jsonl"
        rows = list(iter_jsonl(path))
        write_jsonl(path, rows + rows[:1])
        with self.assertRaisesRegex(ValueError, "Duplicate/unexpected teacher-forcing sample"):
            self.analyze("duplicate")
        write_jsonl(path, rows)
        identity_path = path.parent.parent / "identity.json"
        identity = read_json(identity_path)
        identity["checkpoint"]["config_sha256"] = "different-experiment"
        write_json(identity_path, identity)
        with self.assertRaisesRegex(ValueError, "Checkpoint identity mismatch"):
            self.analyze("wrong_checkpoint")

    def test_cli_runs_without_importing_torch_or_transformers(self):
        target = self.root / "cli"
        source = """import runpy, sys
class NoHeavyImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'transformers', 'datasets', 'numpy'):
            raise AssertionError('Heavy import in CPU saved-result analysis: ' + fullname)
sys.meta_path.insert(0, NoHeavyImports())
sys.argv = ['analyze_run.py', '--run', sys.argv[1], '--out', sys.argv[2]]
runpy.run_path('analyze_run.py', run_name='__main__')
"""
        result = subprocess.run([sys.executable, "-B", "-c", source, str(self.run), str(target)],
                                cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((target / "report.html").exists())


if __name__ == "__main__":
    unittest.main()
