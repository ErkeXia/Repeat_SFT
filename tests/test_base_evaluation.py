"""Offline Base-only routing, fixed panels, interruption and result integrity."""
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import base_evaluation as suite
import grade_transfer
import transfer_evaluation as pipeline
from recipe.common import ROOT, file_hash, iter_jsonl, read_json
from recipe.replay import tree_hashes
from recipe.transfer_eval import load_panel
from test_intermediate_evaluation import generated, replay_fixture
from test_transfer_evaluation import FakeJudge, panel_fixture

TEST_TMP = ROOT / ".test_tmp"
TEST_TMP.mkdir(exist_ok=True)


class BaseEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEST_TMP)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.run, self.replay, self.math_panel = replay_fixture(self.root)
        self.panel_dir = self.root / "transfer"
        self.transfer_panel = panel_fixture(self.panel_dir, file_hash(self.run / "data/train.jsonl"))
        self.job = self.run / "postprocess/b"
        self.job.mkdir(parents=True)
        self.args = SimpleNamespace(image=None, judge_config=None)
        self.judge = FakeJudge()
        for p in (patch.dict(sys.modules, {"math_verify": ModuleType("math_verify")}),
                  patch("recipe.grading.math_equal", side_effect=lambda gold, answer: str(gold) == answer)):
            p.start()
            self.addCleanup(p.stop)

    def prepare(self):
        plan = suite.prepare(self.run, self.panel_dir, self.job, self.args)
        with patch.object(pipeline, "DockerJudge", return_value=self.judge):
            self.plans = suite.prepare_panels(self.job, plan)
        return plan

    def test_only_zero_four_benchmarks_fixed_prompts_and_separate_outputs(self):
        old = {name: tree_hashes(self.run / name) for name in ("data", "checkpoints", "evaluations")}
        plan = self.prepare()
        self.assertEqual(plan["epochs"], [0])
        self.assertEqual(plan["model_mode"], "base_only_no_adapter")
        self.assertEqual(plan["responses"], 392)
        self.assertEqual(suite.verify(self.job), plan)
        for name, expected, size in (("math-gpqa", self.math_panel, 48), ("bbh-livecodebench", self.transfer_panel, 50)):
            child = self.plans[name]
            self.assertEqual(child["generation_epochs"], [0])
            self.assertTrue(child["base_only"])
            self.assertEqual(len(child["groups"]), 4)
            self.assertTrue(all(len(g["selected_ids"]) == size for g in child["groups"]))
            actual, _ = load_panel(self.job / name / "setup/panel")
            self.assertEqual([q["messages"] for q in actual], [q["messages"] for q in expected])
            self.assertEqual([q["sample_id"] for q in actual], [q["sample_id"] for q in expected])
        self.assertEqual(self.prepare(), plan)
        self.assertEqual(old, {name: tree_hashes(self.run / name) for name in old})
        args = SimpleNamespace(mode="math", epochs="0", base_only=False, image=None, judge_config=None)
        with self.assertRaisesRegex(ValueError, "separate job names"):
            pipeline.prepare(self.run, self.run / "data", self.job / "math-gpqa", args)

    def test_replay_or_nonzero_epoch_cannot_be_used_as_base(self):
        with self.assertRaisesRegex(ValueError, "original BF16 LoRA run"):
            suite.prepare(self.replay, self.panel_dir, self.job, self.args)
        args = SimpleNamespace(mode="math", epochs="4", base_only=True, image=None, judge_config=None)
        with self.assertRaisesRegex(ValueError, "epoch 0 exclusively"):
            pipeline.prepare(self.run, self.run / "data", self.job, args)
        self.assertFalse((self.job / "suite").exists())

    def test_grader_failure_stops_before_gpu_and_resume_keeps_frozen_math(self):
        plan = suite.prepare(self.run, self.panel_dir, self.job, self.args)
        worker = SimpleNamespace(check_stop=lambda: None, invoke=lambda *a: self.fail("GPU started before graders"))
        with patch.object(pipeline, "DockerJudge", side_effect=RuntimeError("Docker unavailable")):
            with self.assertRaisesRegex(RuntimeError, "Docker unavailable"):
                suite.execute(self.job, worker)
        original = tree_hashes(self.job / "math-gpqa/setup")
        with patch.object(pipeline, "DockerJudge", return_value=self.judge):
            suite.prepare_panels(self.job, plan)
        self.assertEqual(tree_hashes(self.job / "math-gpqa/setup"), original)

    def test_resume_preserves_answers_grades_and_eight_group_order(self):
        self.prepare()
        calls, fail = [], [True]
        def child_invoke(command, cwd):
            script = Path(command[3]).name
            out = Path(command[command.index("--out") + 1])
            child = out.parents[2]
            name = child.name
            group = next(g for g in self.plans[name]["groups"] if g["name"] == out.parent.name)
            calls.append((name, script, group["name"]))
            if name == "bbh-livecodebench":
                self.assertTrue((self.job / "math-gpqa/complete.json").exists())
            if script == "generate_vllm.py":
                self.assertIn("--base-only", command)
                self.assertEqual(command[command.index("--epochs") + 1], "0")
                stop = name == "math-gpqa" and group["name"] == "group-002" and fail[0]
                fail[0] = False if stop else fail[0]
                generated(child, group, [0], stop)
            else:
                grade_transfer.grade_directory(out.parent / "generation", child / "setup/panel", out,
                                                None if name == "math-gpqa" else self.judge)
        def invoke(command, cwd):
            pipeline.execute(Path(command[-1]), SimpleNamespace(invoke=child_invoke, check_stop=lambda: None))
        worker = SimpleNamespace(invoke=invoke, check_stop=lambda: None)
        with self.assertRaisesRegex(RuntimeError, "interrupted generation"):
            suite.execute(self.job, worker)
        self.assertFalse((self.job / "bbh-livecodebench/groups").exists())
        self.assertFalse((self.job / "complete.json").exists())
        first = tree_hashes(self.job / "math-gpqa/groups/group-001")
        log = self.job / "math-gpqa/groups/group-002/generation/epoch-000/responses.jsonl"
        prefix = log.read_bytes()
        suite.execute(self.job, worker)
        self.assertTrue(log.read_bytes().startswith(prefix))
        self.assertEqual(tree_hashes(self.job / "math-gpqa/groups/group-001"), first)
        self.assertEqual(self.judge.calls, 100)
        self.assertEqual(read_json(self.job / "complete.json")["responses"], 392)
        summary = read_json(self.job / "summary.json")
        self.assertEqual(summary["completed_panels"], ["math-gpqa", "bbh-livecodebench"])
        all_rows = [r for r in summary["curves"] if r["subset"] == "all"]
        self.assertEqual({r["task"] for r in all_rows}, {"math500", "gpqa", "bbh", "livecodebench"})
        self.assertEqual(sum(int(r["n"]) for r in all_rows), 392)
        self.assertTrue((self.job / "curves.csv").is_file())
        count = len(calls)
        suite.execute(self.job, worker)
        self.assertEqual(len(calls), count)
        self.assertEqual(self.judge.calls, 100)

    def test_help_without_gpu_and_no_epoch_selection(self):
        r = subprocess.run([sys.executable, "-B", str(ROOT / "base_evaluation.py"), "--help"],
                           capture_output=True, text=True, timeout=20)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--prepare-only", r.stdout)
        self.assertNotIn("--epochs", r.stdout)


if __name__ == "__main__":
    unittest.main()
