"""Offline four-benchmark high/low setup, ordering, overlap checks and durable resume."""
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))

import grade_transfer
import transfer_evaluation
from dataset import evaluate_likelihood_extremes as suite
from dataset.likelihood_training import training_config, tree_hashes
from recipe.common import digest, file_hash, iter_jsonl, read_json, write_json, write_jsonl
from recipe.data import adapt_gpqa, adapt_math, adapt_sft
from recipe.transfer_eval import load_panel
from test_replay import make_checkpoint
from test_transfer_evaluation import FakeJudge, panel_fixture
from test_intermediate_evaluation import generated


def fixture(root):
    experiment = root / "e"
    config = training_config(read_json(ROOT / "configs/baseline.json"), 42)
    model = {"id": "Qwen/Qwen3-4B-Base", "revision": "a" * 40}
    math = [adapt_math({"problem": f"Geometry task {i}: find a triangle area", "answer": "2", "solution": "Two"}, i)
            for i in range(128)]
    math += [adapt_gpqa({"Question": f"Quantum theory experiment number {i}", "Correct Answer": "yes",
                       "Incorrect Answer 1": "no", "Incorrect Answer 2": "maybe", "Incorrect Answer 3": "never"}, i, 42)
             for i in range(64)]
    for group in suite.GROUPS:
        run = experiment / group
        train = [adapt_sft({"question": f"Training-only arithmetic {group} item {i}",
                            "response": f"Reasoning for {i}. Answer {i}", "answer": str(i)}, i) for i in range(100)]
        write_jsonl(run / "data/train.jsonl", train)
        write_jsonl(run / "data/holdout.jsonl", [{"sample_id": "heldout", "question": "heldout"}])
        write_jsonl(run / "data/panel.jsonl", math)
        write_json(run / "tokenizer/tokenizer.json", {"synthetic": True})
        metadata = {"config": config, "model": model, "max_length": 10240, "config_sha256": digest([group, config]),
                    "data_files": tree_hashes(run / "data"), "tokenizer_files": tree_hashes(run / "tokenizer")}
        write_json(run / "run.json", metadata)
        for epoch in config["checkpoint_epochs"]:
            make_checkpoint(run / "checkpoints" / f"epoch-{epoch:03d}", metadata, epoch)
        write_jsonl(experiment / "bundle/prepared" / group / "train.jsonl", train)
    plan = {"config": config, "model": model, "files": tree_hashes(experiment / "bundle")}
    plan["identity"] = digest(plan)
    write_json(experiment / "bundle/plan.json", plan)
    panel_dir = root / "t"
    transfer = panel_fixture(panel_dir, "old-400-example-training-checksum")
    return experiment, panel_dir, math, transfer


class LikelihoodBenchmarksTests(unittest.TestCase):
    def setUp(self):
        tmp = ROOT / ".test_tmp"
        tmp.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=tmp)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.experiment, self.panel, self.math, self.transfer = fixture(self.root)
        self.job = self.experiment / "postprocess/x"
        self.job.mkdir(parents=True)
        self.args = SimpleNamespace(job_name="x", epochs=None, image=None, judge_config=None)
        self.judge = FakeJudge()
        for context in (patch.dict(sys.modules, {"math_verify": ModuleType("math_verify")}),
                        patch.object(suite, "DockerJudge", return_value=self.judge),
                        patch.object(transfer_evaluation, "DockerJudge", return_value=self.judge),
                        patch("recipe.grading.math_equal", side_effect=lambda gold, answer: str(gold) == answer)):
            context.start()
            self.addCleanup(context.stop)

    def prepare(self):
        plan = suite.prepare(self.experiment, self.panel, self.job, self.args)
        self.plans = suite.prepare_panels(self.job, plan)
        return plan

    def test_defaults_two_runs_four_panels_identical_questions_and_no_fit_requirement(self):
        original = tree_hashes(self.panel)
        training = {g: tree_hashes(self.experiment / g / "data") for g in suite.GROUPS}
        plan = self.prepare()
        self.assertEqual(plan["epochs"], [16,14,12,10,8,4,2,1])
        self.assertEqual(plan["responses"], 6272)
        self.assertEqual([(r["group"], r["panel"]) for r in plan["routes"]],
                         [(g, p) for g in ("high", "low") for p in ("math-gpqa", "bbh-livecodebench")])
        for route, p in zip(plan["routes"], self.plans):
            self.assertEqual(p["generation_epochs"], plan["epochs"])
            self.assertEqual(len(p["groups"]), 4)
            self.assertTrue(all(len(g["selected_ids"]) == route["group_size"] for g in p["groups"]))
            child = Path(route["path"])
            new, _ = load_panel(child / "setup/panel", file_hash(self.experiment / route["group"] / "data/train.jsonl"))
            expected = self.math if route["mode"] == "math" else self.transfer
            for row, original_row in zip(new, expected):
                for key in ("sample_id", "question_id", "question", "messages", "gold_answer"):
                    self.assertEqual(row[key], original_row[key])
            if route["mode"] == "math":
                self.assertIsNone(p["image"])
                self.assertEqual(p["evaluation_kind"], "math")
            else:
                self.assertEqual(file_hash(child / "setup/panel/panel.jsonl"), file_hash(self.panel / "panel.jsonl"))
                for q in new:
                    self.assertEqual(file_hash(child / "setup/panel" / q["reference_file"]), q["reference_sha256"])
        self.assertEqual(tree_hashes(self.panel), original)
        self.assertEqual({g: tree_hashes(self.experiment / g / "data") for g in suite.GROUPS}, training)
        self.assertEqual(self.prepare(), plan)

    def test_nonmatching_training_hash_is_rebound_only_after_overlap_checks(self):
        with self.assertRaisesRegex(ValueError, "different training subset"):
            load_panel(self.panel, file_hash(self.experiment / "high/data/train.jsonl"))
        plan = self.prepare()
        for g in suite.GROUPS:
            rebound = self.job / "suite/panels" / g
            load_panel(rebound, file_hash(self.experiment / g / "data/train.jsonl"))
            info = read_json(rebound / "plan.json")
            self.assertEqual(info["previous_training_sha256"], "old-400-example-training-checksum")
            self.assertEqual(info["overlap_check"]["matches"], [])
            self.assertEqual(info["overlap_check"]["checked_bbh_livecodebench"], 200)
        self.args.epochs = "1"
        with self.assertRaisesRegex(ValueError, "inputs differ"):
            suite.prepare(self.experiment, self.panel, self.job, self.args)

    def test_overlap_stops_before_docker_or_gpu_and_preserves_panel(self):
        before = tree_hashes(self.panel)
        with patch.object(suite, "overlap_checks", wraps=suite.overlap_checks) as audit, \
                patch.object(suite, "DockerJudge", side_effect=AssertionError("Docker must not start")):
            # Inject the same question through the overlap detector's training input
            # without modifying trusted fixture hashes/checkpoints.
            def contaminated(training, math, transfer):
                return suite.overlap_checks_original({**training, "low": [transfer[0], *training["low"][1:]]}, math, transfer)
            original = audit._mock_wraps
            with patch.object(suite, "overlap_checks_original", original, create=True):
                audit.side_effect = contaminated
                with self.assertRaisesRegex(ValueError, "overlap; stopped"):
                    suite.prepare(self.experiment, self.panel, self.job, self.args)
        self.assertFalse((self.job / "suite").exists())
        report = read_json(next(self.job.glob("overlap-rejections-*.json")))
        self.assertEqual(report["low"]["matches"][0]["evaluation_sample_id"], self.transfer[0]["sample_id"])
        self.assertEqual(tree_hashes(self.panel), before)

    def test_missing_checkpoint_and_mismatched_panel_fail_before_setup(self):
        (self.experiment / "low/checkpoints/epoch-014/complete.json").unlink()
        with self.assertRaises(FileNotFoundError):
            suite.prepare(self.experiment, self.panel, self.job, self.args)
        self.assertFalse((self.job / "suite").exists())
        self.args.epochs = "1,16"
        run = self.experiment / "low"
        altered = list(reversed(self.math))
        write_jsonl(run / "data/panel.jsonl", altered)
        meta = read_json(run / "run.json")
        meta["data_files"]["panel.jsonl"] = file_hash(run / "data/panel.jsonl")
        write_json(run / "run.json", meta)
        with self.assertRaisesRegex(ValueError, "same frozen math"):
            suite.prepare(self.experiment, self.panel, self.job, self.args)

    def test_optional_epoch_zero_and_frozen_code_tamper(self):
        self.args.epochs = "0,1,16"
        plan = self.prepare()
        self.assertEqual(plan["epochs"], [16,1,0])
        self.assertEqual(plan["responses"], 2352)
        code = self.job / "suite/code/dataset/evaluate_likelihood_extremes.py"
        code.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Frozen suite input/code"):
            suite.verify(self.job)

    def test_sixteen_group_barriers_resume_answers_and_final_reports(self):
        self.args.epochs = "1,16"
        plan = self.prepare()
        before = {g: tree_hashes(self.experiment / g / "checkpoints") for g in suite.GROUPS}
        routes = {r["path"]: (i, r, p) for i, (r, p) in enumerate(zip(plan["routes"], self.plans))}
        calls, fail = [], [True]

        def child_invoke(command, cwd):
            script = Path(command[3]).name
            out = Path(command[command.index("--out") + 1])
            child = out.parents[2]
            i, route, p = routes[str(child)]
            group = next(g for g in p["groups"] if g["name"] == out.parent.name)
            for earlier in plan["routes"][:i]:
                self.assertTrue((Path(earlier["path"]) / "complete.json").exists())
            calls.append((route["group"], route["panel"], group["name"], script))
            if script == "generate_vllm.py":
                stop = i == 0 and group["name"] == "group-002" and fail[0]
                if stop:
                    fail[0] = False
                generated(child, group, p["generation_epochs"], stop)
            else:
                judge = None if route["mode"] == "math" else self.judge
                grade_transfer.grade_directory(out.parent / "generation", child / "setup/panel", out, judge)

        def invoke(command, cwd):
            transfer_evaluation.execute(Path(command[-1]), SimpleNamespace(invoke=child_invoke, check_stop=lambda: None))

        worker = SimpleNamespace(invoke=invoke, check_stop=lambda: None)
        with self.assertRaisesRegex(RuntimeError, "interrupted generation"):
            suite.execute(self.job, worker)
        first = Path(plan["routes"][0]["path"])
        completed = tree_hashes(first / "groups/group-001")
        pending = first / "groups/group-002/generation/epoch-016/responses.jsonl"
        prefix = pending.read_bytes()
        self.assertFalse((Path(plan["routes"][1]["path"]) / "groups").exists())
        suite.execute(self.job, worker)
        self.assertEqual(tree_hashes(first / "groups/group-001"), completed)
        self.assertTrue(pending.read_bytes().startswith(prefix))
        self.assertEqual(self.judge.calls, 400)  # 100 LCB x 2 epochs x 2 runs, each graded once.
        self.assertEqual(read_json(self.job / "complete.json")["responses"], 1568)
        self.assertEqual(len(read_json(self.job / "summary.json")["completed_panels"]), 4)
        self.assertTrue((self.job / "curves.csv").exists())
        self.assertTrue((self.job / "report.html").exists())
        count = len(calls)
        suite.execute(self.job, worker)
        self.assertEqual(len(calls), count)
        self.assertEqual(before, {g: tree_hashes(self.experiment / g / "checkpoints") for g in suite.GROUPS})


if __name__ == "__main__":
    unittest.main()
