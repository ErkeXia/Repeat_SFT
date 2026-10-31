"""No-download tests for grouped scheduling, replacement, capped output and reuse."""
from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import sys
import subprocess
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import analyze_run
import generate
import grouped_evaluation as pipeline
from recipe.common import ROOT, file_hash, iter_jsonl, read_json, stable_seed, write_json, write_jsonl
from recipe.grouped import GENERATION_EPOCHS, MAX_NEW_TOKENS, merge_groups, partition_panel, select_panel, teacher_forcing_files

TEST_TMP = ROOT / ".test_tmp"
TEST_TMP.mkdir(exist_ok=True)
FIT_EPOCHS = [0, 1, 2, 4, 8, 16, 32, 64]


def fixture(root):
    run = root / "run"
    panel = [{"sample_id": f"{task}-{i:03d}", "question_id": f"q-{task}-{i}", "task": task,
              "question": f"Synthetic {task} question {i}", "gold_answer": "2" if task == "math500" else "A",
              "messages": [{"role": "user", "content": f"Synthetic {task} prompt {i}"}],
              "original": {"solution": "Reference steps then 2"} if task == "math500" else {}}
             for task, count in (("math500", 128), ("gpqa", 64)) for i in range(count)]
    training = [{"sample_id": "t", "question_id": "qt", "source": "synthetic", "question": "Training question",
                 "response": "Full reasoning; final answer 2", "n_supervised": 3}]
    holdout = [{**training[0], "sample_id": "h", "question_id": "qh"}]
    for split, rows in (("train", training), ("holdout", holdout), ("panel", panel)):
        write_jsonl(run / "data" / f"{split}.jsonl", rows)
    write_json(run / "tokenizer" / "tokenizer_config.json", {"fixture": True})
    metadata = {"config": {"checkpoint_epochs": FIT_EPOCHS, "expected_train_size": 1, "attention_implementation": "sdpa"},
                "config_sha256": "synthetic-config",
                "data_files": {s + ".jsonl": file_hash(run / "data" / (s + ".jsonl")) for s in ("train", "holdout", "panel")},
                "tokenizer_files": {"tokenizer_config.json": file_hash(run / "tokenizer" / "tokenizer_config.json")}}
    write_json(run / "run.json", metadata)
    job = run / "postprocess" / "old-job"
    source = job / "teacher_forcing"
    for epoch in FIT_EPOCHS:
        marker = {"config_sha256": metadata["config_sha256"], "step": epoch}
        write_json(run / "checkpoints" / f"epoch-{epoch:03d}" / "complete.json", marker)
        directory = source / f"epoch-{epoch:03d}"
        write_json(directory / "identity.json", {"epoch": epoch, "run_sha256": file_hash(run / "run.json"),
                   "checkpoint": marker, "chunk_size": 128, "code_sha256": "original-frozen-evaluator",
                   "selected_ids": {"train": ["t"], "holdout": ["h"]},
                   "data_sha256": {s: file_hash(run / "data" / (s + ".jsonl")) for s in ("train", "holdout")}})
        for split, question in (("train", training[0]), ("holdout", holdout[0])):
            token = directory / split / "tokens" / (question["sample_id"] + ".npz")
            token.parent.mkdir(parents=True, exist_ok=True)
            token.write_bytes(b"synthetic token evidence; copied without loading")
            write_jsonl(directory / split / "samples.jsonl", [{"sample_id": question["sample_id"],
                        "question_id": question["question_id"], "source": "synthetic", "mean_loss": .2,
                        "nll_sum": .6, "token_accuracy": 1., "correct_tokens": 3, "n_tokens": 3,
                        "end_token_loss": .1, "end_token_correct": True, "token_file": "tokens/" + token.name}])
            write_json(directory / split / "summary.json", {"samples": 1})
    return run, job, source, panel


def args(**updates):
    return SimpleNamespace(**{"reset_generation": False, "teacher_forcing": None, **updates})


def hashes(directory):
    return {str(p): file_hash(p) for p in directory.rglob("*") if p.is_file()}


def generate_group(job, plan, group):
    questions = select_panel(list(iter_jsonl(Path(plan["run"]) / "data" / "panel.jsonl")), group["selected_ids"])
    for epoch in plan["generation_epochs"]:
        out = job / "groups" / group["name"] / "generation" / f"epoch-{epoch:03d}"
        identity = {"epoch": epoch, "run_sha256": plan["run_sha256"], "panel_sha256": plan["panel_sha256"],
                    "checkpoint": read_json(Path(plan["run"]) / "checkpoints" / f"epoch-{epoch:03d}" / "complete.json"),
                    "settings": plan["settings"], "generation_config": {"max_new_tokens": 8192},
                    "selected_ids": group["selected_ids"], "seed_policy": "fixed", "code_sha256": plan["code_sha256"]}
        write_json(out / "identity.json", identity)
        rows = [{"sample_id": q["sample_id"], "question_id": q["question_id"], "task": q["task"], "epoch": epoch,
                 "messages": q["messages"], "gold_answer": q["gold_answer"],
                 "seed": stable_seed(plan["settings"]["seed"], q["sample_id"]),
                 "generated_token_ids": [7, 2], "prompt_token_ids": [1, 8], "generated_tokens": 2, "prompt_tokens": 2,
                 "raw_response": "Reasoning then \\boxed{" + q["gold_answer"] + "}<|im_end|>",
                 "response_text": "Reasoning then answer " + q["gold_answer"], "generation_seconds": .01,
                 **generate.stop_details([7, 2], 3, 2, 8192)} for q in questions]
        write_jsonl(out / "responses.jsonl", rows)
        write_json(out / "complete.json", {"samples": len(rows), "responses_sha256": file_hash(out / "responses.jsonl")})


def fake_grade(source, out):
    sources = []
    for file in sorted(source.glob("epoch-*/responses.jsonl")):
        rows = [{**r, "extracted_answer": r["gold_answer"], "extractable": True, "correct": True,
                 "extraction_method": "synthetic", "grading_error": None, "grader_version": "synthetic"}
                for r in iter_jsonl(file)]
        write_jsonl(out / file.parent.name / "graded.jsonl", rows)
        sources.append({"responses": str(file), "sha256": file_hash(file), "identity": read_json(file.parent / "identity.json")})
    write_json(out / "grading_manifest.json", {"sources": sources, "grader_version": "synthetic"})


class GroupedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEST_TMP)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.run, self.job, self.source, self.panel = fixture(self.root)

    def test_four_balanced_disjoint_fixed_groups(self):
        groups = partition_panel(self.panel)
        ids = [s for g in groups for s in g["selected_ids"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {r["sample_id"] for r in self.panel})
        self.assertEqual(groups, partition_panel(self.panel))
        for group in groups:
            rows = select_panel(self.panel, group["selected_ids"])
            self.assertEqual(len(rows), 48)
            self.assertEqual(Counter(r["task"] for r in rows), {"math500": 32, "gpqa": 16})
        with self.assertRaises(ValueError):
            select_panel(self.panel, ["unknown"])
        with self.assertRaises(ValueError):
            select_panel(self.panel, [ids[0], ids[0]])

    def test_replace_archives_generation_preserves_every_fitting_file(self):
        before = hashes(self.source)
        write_json(self.job / "generation" / "epoch-001" / "old.json", {"old": True})
        write_json(self.job / "grades" / "old.json", {"old": True})
        write_json(self.job / "state" / "loss-0.done", {"keep": True})
        write_json(self.job / "state" / "generation-1.done", {"old": True})
        with self.assertRaisesRegex(ValueError, "--reset-generation"):
            pipeline.prepare_job(self.run, self.job, args())
        self.assertTrue((self.job / "generation" / "epoch-001" / "old.json").exists())
        plan = pipeline.prepare_job(self.run, self.job, args(reset_generation=True))
        backup = Path(plan["previous_generation_backup"])
        self.assertTrue((backup / "generation" / "epoch-001" / "old.json").exists())
        self.assertFalse((self.job / "generation").exists())
        self.assertFalse((self.job / "state" / "generation-1.done").exists())
        self.assertTrue((self.job / "state" / "loss-0.done").exists())
        self.assertEqual(hashes(self.source), before)
        self.assertEqual(plan["generation_epochs"], [64, 32, 16, 8, 4, 2, 1])
        self.assertEqual(plan["fit_epochs"], FIT_EPOCHS)
        self.assertEqual(plan, pipeline.prepare_job(self.run, self.job, args()))
        self.assertEqual(hashes(self.source), before)

    def test_invalid_fitting_cannot_reset_old_generation(self):
        old = self.job / "generation" / "old.json"
        write_json(old, {"keep": True})
        missing = self.source / "epoch-064" / "holdout" / "samples.jsonl"
        missing.unlink()
        with self.assertRaisesRegex(ValueError, "No complete matching"):
            pipeline.prepare_job(self.run, self.job, args(reset_generation=True, teacher_forcing=str(self.source)))
        self.assertTrue(old.exists())
        self.assertFalse((self.job / "generation-backups").exists())

    def test_frozen_source_tampering_is_rejected(self):
        pipeline.prepare_job(self.run, self.job, args())
        with (self.job / "grouped_code" / "generate.py").open("a", encoding="utf-8") as f:
            f.write("\n# changed\n")
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            pipeline.prepare_job(self.run, self.job, args())

    def test_merge_full_panel_preserves_raw_rows_and_rejects_changed_protocol(self):
        plan = pipeline.prepare_job(self.run, self.job, args())
        for group in plan["groups"]:
            generate_group(self.job, plan, group)
        merge_groups(self.job, plan)
        for epoch in GENERATION_EPOCHS:
            out = self.job / "generation" / f"epoch-{epoch:03d}"
            rows = list(iter_jsonl(out / "responses.jsonl"))
            self.assertEqual([r["sample_id"] for r in rows], plan["panel_ids"])
            self.assertEqual(len(rows), 192)
            original = next(iter_jsonl(self.job / "groups" / "group-001" / "generation" / f"epoch-{epoch:03d}" / "responses.jsonl"))
            self.assertEqual(rows[0], original)
        identity_path = self.job / "groups" / "group-003" / "generation" / "epoch-064" / "identity.json"
        identity = read_json(identity_path)
        identity["settings"]["temperature"] = .9
        write_json(identity_path, identity)
        with self.assertRaisesRegex(ValueError, "incompatible|different generation protocols"):
            merge_groups(self.job, plan)

    def test_worker_group_barrier_failure_resume_and_full_reports(self):
        before = hashes(self.source)
        plan = pipeline.prepare_job(self.run, self.job, args())
        calls = []
        fail = [True]

        def invoke(command, cwd):
            script = Path(command[3]).name
            flags = dict(zip(command[4::2], command[5::2]))
            out = Path(flags["--out"])
            if script == "generate.py":
                group = next(g for g in plan["groups"] if g["name"] == out.parent.name)
                calls.append((script, group["name"], flags["--epochs"]))
                if group["name"] == "group-002":
                    self.assertTrue((self.job / "groups" / "group-001" / "complete.json").exists())
                    if fail[0]:
                        fail[0] = False
                        raise RuntimeError("synthetic interrupted generation")
                generate_group(self.job, plan, group)
            elif script == "grade.py":
                calls.append((script, out.parent.name))
                fake_grade(Path(flags["--responses"]), out)
            elif script == "analyze_run.py":
                calls.append((script, out.parent.name))
                with patch.object(sys, "argv", [script, *map(str, command[4:])]):
                    analyze_run.main()
            else:
                self.fail("Unexpected stage: " + script)

        with patch.object(pipeline, "invoke", side_effect=invoke):
            with self.assertRaisesRegex(RuntimeError, "synthetic interrupted"):
                pipeline.execute(self.job)
            first = hashes(self.job / "groups" / "group-001" / "generation")
            pipeline.execute(self.job)
            self.assertEqual(hashes(self.job / "groups" / "group-001" / "generation"), first)
            finished_calls = len(calls)
            pipeline.execute(self.job)
            self.assertEqual(len(calls), finished_calls)
        generation = [r for r in calls if r[0] == "generate.py"]
        self.assertEqual([r[1] for r in generation], ["group-001", "group-002", "group-002", "group-003", "group-004"])
        self.assertTrue(all(r[2] == "64,32,16,8,4,2,1" for r in generation))
        self.assertEqual(hashes(self.source), before)
        summary = read_json(self.job / "analysis" / "summary.json")
        self.assertEqual(summary["epochs"], FIT_EPOCHS)
        self.assertEqual(summary["generation_epochs"], sorted(GENERATION_EPOCHS))
        self.assertTrue(all(r["complete"] for r in summary["downstream_curves"]))
        self.assertTrue(all(r["baseline_accuracy"] is None for r in summary["task_milestones"]))
        self.assertEqual(len(summary["fit_curves"]), 16)
        self.assertTrue((self.job / "grouped_complete.json").exists())
        for group in plan["groups"]:
            summary = read_json(self.job / "groups" / group["name"] / "analysis" / "summary.json")
            self.assertTrue(all(r["complete"] for r in summary["downstream_curves"]))
            self.assertEqual({r["expected_n"] for r in summary["downstream_curves"]}, {32, 16})
        # Exercise the actual bootstrap -> frozen worker subprocess on a saved job.
        result = subprocess.run([sys.executable, "-B", str(ROOT / "grouped_evaluation.py"), str(self.run), self.job.name],
                                env={**os.environ, "REPEAT_SFT_ROOT": str(self.root)},
                                capture_output=True, text=True, encoding="utf-8", timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ALL DONE", result.stdout)
        self.assertEqual(hashes(self.source), before)

    def test_lock_prevents_concurrent_replacement(self):
        with pipeline.job_lock(self.job):
            with self.assertRaisesRegex(RuntimeError, "already running"):
                with pipeline.job_lock(self.job):
                    self.fail("Concurrent job obtained the lock")

    def test_changed_fitting_evidence_prevents_gpu_stages(self):
        pipeline.prepare_job(self.run, self.job, args())
        token = self.source / "epoch-001" / "train" / "tokens" / "t.npz"
        token.write_bytes(b"changed evidence")
        with patch.object(pipeline, "invoke") as invoke:
            with self.assertRaisesRegex(ValueError, "teacher-forcing evidence changed"):
                pipeline.execute(self.job)
            invoke.assert_not_called()

    def test_generate_exact_cap_and_question_resume_without_model_downloads(self):
        import torch
        out = self.root / "generation"
        selected = self.root / "ids.json"
        write_json(selected, [r["sample_id"] for r in self.panel[:2]])
        module = ModuleType("transformers")

        class Config(SimpleNamespace):
            def to_dict(self):
                return vars(self)

        class Tokenizer:
            eos_token_id = 3
            pad_token_id = 3
            bos_token_id = 1
            def convert_tokens_to_ids(self, token):
                return 2
            def apply_chat_template(self, *a, **kw):
                return [1, 8]
            def decode(self, ids, **kw):
                return "synthetic response " + str(len(ids))

        module.GenerationConfig = Config
        module.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer())
        calls = []
        interrupt = [True]
        real_tensor = torch.tensor

        class Model:
            config = SimpleNamespace(max_position_embeddings=32768)
            def generate(self, input_ids, generation_config, **kw):
                calls.append(generation_config.max_new_tokens)
                if len(calls) == 2 and interrupt[0]:
                    interrupt[0] = False
                    raise RuntimeError("synthetic interrupted question")
                ids = torch.full((1, generation_config.max_new_tokens), 7, dtype=torch.long)
                return torch.cat([input_ids, ids], dim=1)

        def cpu_tensor(*a, **kw):
            kw.pop("device", None)
            return real_tensor(*a, **kw)

        def snapshot(path):
            Path(path).mkdir(parents=True)
            return "synthetic"

        command = ["generate.py", "--run", str(self.run), "--epochs", "64", "--sample-ids", str(selected),
                   "--out", str(out), "--config", str(ROOT / "configs" / "generation.json")]
        with patch.dict(sys.modules, {"transformers": module}), patch.object(sys, "argv", command), \
                patch.object(generate, "configure_runtime"), patch.object(generate, "seed_everything"), \
                patch.object(generate, "load_model", return_value=Model()), patch.object(generate, "snapshot_code", side_effect=snapshot), \
                patch.object(generate, "environment", return_value={}), patch.object(torch, "tensor", side_effect=cpu_tensor):
            with self.assertRaisesRegex(RuntimeError, "synthetic interrupted question"):
                generate.main()
            saved = out / "epoch-064" / "responses.jsonl"
            self.assertEqual(len(list(iter_jsonl(saved))), 1)
            first = saved.read_bytes()
            generate.main()
            self.assertTrue(saved.read_bytes().startswith(first))
        self.assertEqual(calls, [8192, 8192, 8192])
        rows = list(iter_jsonl(saved))
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["generated_tokens"], 8192)
            self.assertEqual(row["stop_reason"], "length")
            self.assertTrue(row["length_truncated"])
            self.assertFalse(row["normal_chat_end"])
            self.assertFalse(row["ended_with_eos"])
            self.assertEqual(row["generated_token_ids"][-1], 7)


if __name__ == "__main__":
    unittest.main()
