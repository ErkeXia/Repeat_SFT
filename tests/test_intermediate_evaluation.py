"""Four-task replay evaluation: fixed panels, checkpoint routing and durable resume."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import generate_vllm
import grade_transfer
import transfer_evaluation as pipeline
from analyze_regressions import build_report as regression_report
from generate import stop_details
from recipe.common import ROOT, append_jsonl, digest, file_hash, iter_jsonl, read_json, write_json, write_jsonl
from recipe.data import adapt_gpqa, adapt_math
from recipe.math_eval import COUNTS, FORMAT
import intermediate_evaluation as suite
from recipe.replay import DEFAULT_EPOCHS, IDENTITY_KEYS, initialize, plan_replay, tree_hashes
from recipe.transfer_eval import load_panel
from test_replay import fixture, make_checkpoint
from test_transfer_evaluation import FakeJudge, panel_fixture

TEST_TMP = ROOT / ".test_tmp"
TEST_TMP.mkdir(exist_ok=True)


def replay_fixture(root):
    source = fixture(root)
    panel = [adapt_math({"problem": f"Synthetic math problem {i}: compute 1+1", "answer": "2", "solution": "1+1=2"}, i)
             for i in range(128)]
    panel += [adapt_gpqa({"Question": f"Synthetic science question {i}", "Correct Answer": "yes",
                         "Incorrect Answer 1": "no", "Incorrect Answer 2": "maybe", "Incorrect Answer 3": "never"}, i, 42)
              for i in range(64)]
    write_jsonl(source / "data/panel.jsonl", panel)
    metadata = read_json(source / "run.json")
    metadata["data_files"] = tree_hashes(source / "data")
    metadata["config_sha256"] = digest({k: metadata[k] for k in IDENTITY_KEYS})
    write_json(source / "run.json", metadata)
    for path in (source / "checkpoints").glob("epoch-*/complete.json"):
        marker = read_json(path)
        write_json(path, {**marker, "config_sha256": metadata["config_sha256"]})
    run = root / "r"
    initialize(plan_replay(source, run, DEFAULT_EPOCHS))
    for epoch in DEFAULT_EPOCHS:
        make_checkpoint(run / "checkpoints" / f"epoch-{epoch:03d}", metadata, epoch)
    return source, run, panel


def generated(job, group, epochs, interrupt=False):
    plan = read_json(job / "setup/plan.json")
    panel, _ = load_panel(job / "setup/panel")
    by_id = {q["sample_id"]: q for q in panel}
    for epoch in epochs:
        directory = job / "groups" / group["name"] / "generation" / f"epoch-{epoch:03d}"
        identity = {"run_sha256": plan["run_sha256"], "panel_sha256": plan["panel_sha256"],
                    "epoch": epoch, "checkpoint": read_json(Path(plan["run"]) / "checkpoints" / f"epoch-{epoch:03d}/complete.json"),
                    "code_sha256": plan["code_sha256"], "settings": plan["settings"], "backend": "vllm",
                    "transfer_manifest_sha256": file_hash(job / "setup/panel/manifest.json"),
                    "selected_ids": group["selected_ids"]}
        write_json(directory / "identity.json", identity)
        log = directory / "responses.jsonl"
        done = {r["sample_id"] for r in iter_jsonl(log)} if log.exists() else set()
        for sid in group["selected_ids"]:
            if sid in done:
                continue
            q = by_id[sid]
            if q["task"] in ("math500", "gpqa"):
                # Deliberate later loss on math, so the regression reader is tested too.
                answer = "999" if q["task"] == "math500" and epoch > 6 else q["gold_answer"]
                text = "\\boxed{" + answer + "}"
            elif q["task"] == "bbh":
                text = f"Final answer: {q['gold_answer']}"
            else:
                code = "class Solution:\n def solve(self,n): return n+1" if q["interface"] == "functional" else "print(int(input())+1)"
                text = "```python\n" + code + "\n```"
            row = {k: q[k] for k in ("sample_id", "question_id", "task", "messages", "gold_answer")}
            append_jsonl(log, {**row, "epoch": epoch, "backend": "vllm", "raw_response": text + "<|im_end|>",
                              "generated_tokens": 2, "generated_token_ids": [7, 2], **stop_details([7, 2], 3, 2, 8192)})
            done.add(sid)
            if interrupt and len(done) == 7:
                raise RuntimeError("synthetic interrupted generation")
        write_json(directory / "complete.json", {"samples": len(done), "responses_sha256": file_hash(log)})


class IntermediateEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEST_TMP)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source, self.run, self.math_panel = replay_fixture(self.root)
        self.panel_dir = self.root / "transfer-panel"
        self.transfer_panel = panel_fixture(self.panel_dir, file_hash(self.run / "data/train.jsonl"))
        # Keep synthetic paths below Windows MAX_PATH despite atomic-save suffixes.
        self.job = self.run / "postprocess/i"
        self.job.mkdir(parents=True)
        self.args = SimpleNamespace(epochs=None, image=None, judge_config=None)
        self.judge = FakeJudge()
        module = patch.dict(sys.modules, {"math_verify": ModuleType("math_verify")})
        module.start()
        self.addCleanup(module.stop)
        grading = patch("recipe.grading.math_equal", side_effect=lambda gold, answer: str(gold) == answer)
        self.math_equal = grading.start()
        self.addCleanup(grading.stop)

    def prepare(self):
        plan = suite.prepare(self.run, self.panel_dir, self.job, self.args)
        with patch.object(pipeline, "DockerJudge", return_value=self.judge):
            self.plans = suite.prepare_panels(self.job, plan)
        return plan

    def test_default_ten_nodes_two_separate_panels_and_original_prompts_preserved(self):
        old = tree_hashes(self.source)
        data = tree_hashes(self.run / "data")
        plan = self.prepare()
        self.assertEqual(plan["epochs"], sorted(DEFAULT_EPOCHS, reverse=True))
        self.assertEqual(plan["panels"], ["math-gpqa", "bbh-livecodebench"])
        self.assertEqual(sum(len(p["panel_ids"]) for p in self.plans.values()) * len(plan["epochs"]), 3920)
        for name, expected, size, group_size in (("math-gpqa", self.math_panel, 192, 48),
                                                ("bbh-livecodebench", self.transfer_panel, 200, 50)):
            child = self.job / name
            child_plan = self.plans[name]
            self.assertEqual(child_plan["generation_epochs"], plan["epochs"])
            self.assertEqual(len(child_plan["panel_ids"]), size)
            self.assertEqual(len(child_plan["groups"]), 4)
            self.assertTrue(all(len(g["selected_ids"]) == group_size for g in child_plan["groups"]))
            panel, _ = load_panel(child / "setup/panel")
            for copied, original in zip(panel, expected):
                for key in ("sample_id", "question_id", "messages", "gold_answer"):
                    self.assertEqual(copied[key], original[key])
        self.assertIsNone(self.plans["math-gpqa"]["image"])
        self.assertEqual(self.plans["bbh-livecodebench"]["image"], self.judge.image)
        self.assertEqual(tree_hashes(self.source), old)
        self.assertEqual(tree_hashes(self.run / "data"), data)
        self.assertEqual(self.prepare(), plan)
        self.args.epochs = "6"
        with self.assertRaisesRegex(ValueError, "inputs differ"):
            self.prepare()

    def test_new_entry_requires_replay_and_rejects_missing_or_old_nodes(self):
        with self.assertRaisesRegex(ValueError, "requires the replay run"):
            suite.prepare(self.source, self.panel_dir, self.job, self.args)
        with self.assertRaisesRegex(ValueError, "not replay checkpoints"):
            pipeline.prepare(self.run, self.panel_dir, self.job, self.args)
        self.args.epochs = "8,32"
        with self.assertRaisesRegex(ValueError, "not saved targets"):
            self.prepare()
        self.args.epochs = "6,30"
        (self.run / "checkpoints/epoch-030/complete.json").unlink()
        with self.assertRaisesRegex(ValueError, "Checkpoint not ready"):
            self.prepare()
        self.assertFalse((self.job / "suite").exists())

    def test_panel_training_mismatch_is_rejected(self):
        manifest = read_json(self.panel_dir / "manifest.json")
        write_json(self.panel_dir / "plan.json", {"training_sha256": "other-training-subset"})
        manifest["files"]["plan.json"] = file_hash(self.panel_dir / "plan.json")
        write_json(self.panel_dir / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "different training subset"):
            self.prepare()

    def test_both_preflights_finish_before_any_generation_and_retry_keeps_math_setup(self):
        plan = suite.prepare(self.run, self.panel_dir, self.job, self.args)
        worker = SimpleNamespace(check_stop=lambda: None, invoke=lambda *a: self.fail("GPU started before preflight"))
        with patch.dict(sys.modules, {"math_verify": None}), patch.object(pipeline, "DockerJudge") as judge:
            with self.assertRaises(ModuleNotFoundError):
                suite.execute(self.job, worker)
            judge.assert_not_called()
        with patch.object(pipeline, "DockerJudge", side_effect=RuntimeError("Docker unavailable")):
            with self.assertRaisesRegex(RuntimeError, "Docker unavailable"):
                suite.execute(self.job, worker)
        math_setup = tree_hashes(self.job / "math-gpqa/setup")
        with patch.object(pipeline, "DockerJudge", return_value=self.judge):
            suite.prepare_panels(self.job, plan)
        self.assertEqual(tree_hashes(self.job / "math-gpqa/setup"), math_setup)

    def test_vllm_uses_new_adapters_and_common_context_within_each_panel(self):
        self.args.epochs = "6,30"
        self.prepare()
        class Tokenizer:
            eos_token_id = 3
            def convert_tokens_to_ids(self, token): return 2
            def decode(self, ids, **kw): return "<|im_end|>" if ids == [2] else "tokens"
            def apply_chat_template(self, messages, **kw): return [1] * (len(messages[0]["content"]) + 1)
        module = ModuleType("transformers")
        module.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer())
        module.AutoConfig = SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(
            model_type="qwen3", max_position_embeddings=32768,
            to_dict=lambda: {"model_type": "qwen3", "max_position_embeddings": 32768}))
        for name, total, count in (("math-gpqa", 192, 48), ("bbh-livecodebench", 200, 50)):
            options = []
            for group in self.plans[name]["groups"][:2]:
                ids = self.root / (name + group["name"] + "-ids.json")
                write_json(ids, group["selected_ids"])
                args = SimpleNamespace(run=str(self.run), out=str(self.root / (name + group["name"])), epochs="6,30",
                    panel_dir=str(self.job / name / "setup/panel"), sample_ids=str(ids), limit_per_task=None, progress_label="test",
                    config=str(ROOT / "configs/generation.json"), engine_config=str(ROOT / "configs/vllm.json"))
                with patch.dict(sys.modules, {"transformers": module}), patch.object(generate_vllm, "runtime_identity", return_value={"test": True}):
                    prepared = generate_vllm.prepare(args)
                self.assertEqual([c["epoch"] for c in prepared["checkpoints"]], [30, 6])
                self.assertTrue(all(c["checkpoint"].is_relative_to(self.run / "checkpoints") for c in prepared["checkpoints"]))
                self.assertEqual(len(prepared["panel"]), count)
                self.assertEqual(len(prepared["prompts"]), total)
                self.assertEqual(prepared["options"]["max_num_seqs"], 8)
                options.append(prepared["options"])
            self.assertEqual(options[0], options[1])

    def test_eight_groups_resume_preserves_all_saved_results_and_regression_compatibility(self):
        self.args.epochs = "6,30"
        self.prepare()
        preserved, weights = tree_hashes(self.source), tree_hashes(self.run / "checkpoints")
        calls, fail = [], [True]
        def child_invoke(command, cwd):
            script = Path(command[3]).name
            out = Path(command[command.index("--out") + 1])
            child = out.parents[2]
            name = child.name
            child_plan = self.plans[name]
            group = next(g for g in child_plan["groups"] if g["name"] == out.parent.name)
            calls.append((name, script, group["name"]))
            if name == "bbh-livecodebench":
                self.assertTrue((self.job / "math-gpqa/complete.json").exists())
            if script == "generate_vllm.py":
                stop = name == "math-gpqa" and group["name"] == "group-002" and fail[0]
                fail[0] = False if stop else fail[0]
                generated(child, group, child_plan["generation_epochs"], stop)
            else:
                judge = None if name == "math-gpqa" else self.judge
                grade_transfer.grade_directory(out.parent / "generation", child / "setup/panel", out, judge)
        def invoke(command, cwd):
            pipeline.execute(Path(command[-1]), SimpleNamespace(invoke=child_invoke, check_stop=lambda: None))
        worker = SimpleNamespace(invoke=invoke, check_stop=lambda: None)
        with self.assertRaisesRegex(RuntimeError, "interrupted generation"):
            suite.execute(self.job, worker)
        self.assertFalse((self.job / "math-gpqa/groups/group-003").exists())
        self.assertFalse((self.job / "bbh-livecodebench/groups").exists())
        partial, _ = regression_report(self.run, self.run / "analysis-partial")
        untested = [r for r in partial["coverage"] if r["task"] in ("bbh", "livecodebench")]
        self.assertTrue(all(r["valid_grades"] == 0 and r["incorrect"] == 0 and r["accuracy_on_valid"] is None for r in untested))
        first = tree_hashes(self.job / "math-gpqa/groups/group-001")
        pending = self.job / "math-gpqa/groups/group-002/generation/epoch-030/responses.jsonl"
        prefix = pending.read_bytes()
        suite.execute(self.job, worker)
        self.assertEqual(tree_hashes(self.job / "math-gpqa/groups/group-001"), first)
        self.assertTrue(pending.read_bytes().startswith(prefix))
        self.assertEqual(self.judge.calls, 200)
        self.assertEqual(read_json(self.job / "complete.json")["responses"], 784)
        self.assertEqual(read_json(self.job / "math-gpqa/complete.json")["responses"], 384)
        self.assertEqual(read_json(self.job / "bbh-livecodebench/complete.json")["responses"], 400)
        self.assertEqual(read_json(self.job / "math-gpqa/groups/group-001/grades/grading_manifest.json")["grader_version"],
                         grade_transfer.MATH_GPQA_GRADER_VERSION)
        result, _ = regression_report(self.run, self.run / "analysis-regressions")
        self.assertEqual(len(result["events"]), 128)
        self.assertEqual({(r["before_epoch"], r["after_epoch"]) for r in result["events"]}, {(6, 30)})
        count = len(calls)
        suite.execute(self.job, worker)
        self.assertEqual(len(calls), count)
        self.assertEqual(tree_hashes(self.source), preserved)
        self.assertEqual(tree_hashes(self.run / "checkpoints"), weights)

    def test_per_question_grading_resume_for_both_panels(self):
        self.args.epochs = "6"
        self.prepare()
        for name in ("math-gpqa", "bbh-livecodebench"):
            child = self.job / name
            group = self.plans[name]["groups"][0]
            generated(child, group, [6])
            directory = child / "groups" / group["name"]
            log = directory / "grades/epoch-006/graded.jsonl"
            if name == "math-gpqa":
                def stop_check():
                    if log.exists() and len(list(iter_jsonl(log))) == 9:
                        raise RuntimeError("interrupted math grading")
                with self.assertRaisesRegex(RuntimeError, "interrupted math grading"):
                    grade_transfer.grade_directory(directory / "generation", child / "setup/panel", directory / "grades", None, stop_check)
                self.assertEqual(len(list(iter_jsonl(log))), 9)
                judge = None
            else:
                original = self.judge.run
                def fail(code, tests):
                    if self.judge.calls == 2:
                        raise RuntimeError("interrupted Docker")
                    return original(code, tests)
                with patch.object(self.judge, "run", side_effect=fail):
                    with self.assertRaisesRegex(RuntimeError, "interrupted Docker"):
                        grade_transfer.grade_directory(directory / "generation", child / "setup/panel", directory / "grades", self.judge)
                self.assertEqual(len(list(iter_jsonl(log))), 27)
                judge = self.judge
            prefix = log.read_bytes()
            grade_transfer.grade_directory(directory / "generation", child / "setup/panel", directory / "grades", judge)
            self.assertTrue(log.read_bytes().startswith(prefix))
            self.assertEqual(len(list(iter_jsonl(log))), len(group["selected_ids"]))
            self.assertTrue(all(r["correct"] for r in iter_jsonl(log)))
        self.assertEqual(self.math_equal.call_count, 32)
        self.assertEqual(self.judge.calls, 25)

    def test_cli_help_is_available_without_gpu_or_volume(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "intermediate_evaluation.py"), "--help"],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--epochs", result.stdout)


if __name__ == "__main__":
    unittest.main()
