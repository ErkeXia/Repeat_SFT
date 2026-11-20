"""CPU state-continuity and real trainer pause/resume checks; no pretrained model."""
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import io
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import train
from dataset import continue_training as launcher
from recipe import checkpoints
from recipe.common import (ROOT, capture_rng, digest, epoch_order, file_hash, iter_jsonl,
                           lr_for_update, read_json, restore_rng, seed_everything, write_json, write_jsonl)
from recipe.extension import BOOTSTRAP, initialize, plan_extension, verify_extension
from recipe.replay import IDENTITY_KEYS, tree_hashes


class Tokenizer:
    pad_token_id = 0

    def save_pretrained(self, path):
        write_json(Path(path) / "tokenizer.json", {"synthetic": True})


class Tiny(torch.nn.Linear):
    def __init__(self, meta):
        super().__init__(1, 1)
        self.meta = meta
        self.peft_config = {"test": None}
        self._recipe_model_spec = {"format_version": 1, "compute_dtype": "bfloat16",
                                   "finetuning": meta["config"]["finetuning"], "base_model": meta["model"]}

    def save_pretrained(self, path, **_):
        ft = self.meta["config"]["finetuning"]
        # Synthetic adapter bytes; the CPU loader below reads them with torch.
        torch.save(self.state_dict(), Path(path) / "adapter_model.safetensors")
        write_json(Path(path) / "adapter_config.json", {"peft_type": "LORA", "task_type": "CAUSAL_LM",
                   "r": ft["rank"], "lora_alpha": ft["alpha"], "lora_dropout": ft["dropout"],
                   "target_modules": ft["target_modules"], "bias": "none",
                   "base_model_name_or_path": self.meta["model"]["id"], "revision": self.meta["model"]["revision"]})


def objects(meta):
    cfg = meta["config"]
    model = Tiny(meta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], betas=tuple(cfg["betas"]),
                                 eps=cfg["adam_epsilon"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda completed: lr_for_update(completed + 1, 1.0, cfg["warmup_steps"]))
    return model, optimizer, scheduler


def loss(model, batch, *_):
    noise = (torch.rand(()) + random.random() + np.random.random()) * .01
    return (model(batch["x"]) - (batch["x"] * .25 + noise)).square().mean()


def updates(meta, model, optimizer, scheduler, start, end):
    cfg = meta["config"]
    for step in range(start, end):
        index = epoch_order(10, cfg["seed"], step // 10)[step % 10]
        optimizer.zero_grad(set_to_none=True)
        loss(model, {"x": torch.tensor([[index / 10]])}).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
        optimizer.step()
        scheduler.step()


class CpuInput:
    def __init__(self, tensor):
        self.tensor = tensor

    def to(self, device):
        assert device == "cuda"
        return self.tensor


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / ".test_tmp"
        parent.mkdir(exist_ok=True)
        temp = tempfile.TemporaryDirectory(dir=parent, prefix="continue-")
        self.root = Path(temp.name).resolve()
        self.addCleanup(temp.cleanup)
        if os.name == "nt":
            rename = Path.rename
            def retry(path, target):
                assert path.resolve().is_relative_to(self.root) and Path(target).resolve().is_relative_to(self.root)
                for i in range(21):
                    try:
                        return rename(path, target)
                    except PermissionError:
                        if i == 20:
                            raise
                        time.sleep(.1)
            context = patch.object(Path, "rename", retry)
            context.start()
            self.addCleanup(context.stop)
        self.source, self.out = self.root / "source", self.root / "continued"
        cfg = read_json(ROOT / "configs/baseline.json")
        cfg.update(epochs=128, expected_train_size=10, checkpoint_epochs=[0, 128],
                   recovery_every_steps=10, keep_recovery=2)
        write_jsonl(self.source / "data/train.jsonl", [{"sample_id": str(i), "question_id": str(i), "index": i}
                                                     for i in range(10)])
        for name in ("holdout.jsonl", "overlap_references.jsonl", "panel.jsonl"):
            write_jsonl(self.source / "data" / name, [])
        Tokenizer().save_pretrained(self.source / "tokenizer")
        code = self.source / "attempts/original/code"
        code.mkdir(parents=True)
        (code / "train.py").write_bytes((ROOT / "train.py").read_bytes())
        files = tree_hashes(code)
        write_json(code / "manifest.json", {"files": files, "code_sha256": digest(files)})
        self.meta = {"config": cfg, "model": {"id": "synthetic-base", "revision": "pinned"},
                     "semantics": {"batch_size": 1, "adapter_dtype": "float32"}, "max_length": 10240,
                     "data_files": tree_hashes(self.source / "data"), "tokenizer_files": tree_hashes(self.source / "tokenizer"),
                     "code_sha256": digest(files)}
        self.meta["config_sha256"] = digest({k: self.meta[k] for k in IDENTITY_KEYS})
        write_json(self.source / "run.json", self.meta)
        seed_everything(42)
        self.model, self.optimizer, self.scheduler = objects(self.meta)
        updates(self.meta, self.model, self.optimizer, self.scheduler, 0, 1280)
        self.anchor = self.source / "checkpoints/epoch-128"
        budget = patch.object(checkpoints, "checkpoint_estimate", return_value={"required_bytes": 0, "reserve_bytes": 0})
        budget.start()
        self.addCleanup(budget.stop)
        checkpoints.save_checkpoint(self.anchor, self.model, Tokenizer(), 1280, self.meta["config_sha256"],
                                    self.optimizer, self.scheduler, {"epoch_index": 128, "position": 0,
                                    "order": epoch_order(10, 42, 128)})
        write_json(self.source / "evaluations/old.json", {"retain": True})
        self.before = tree_hashes(self.source)

    def prepare(self, additional=128, interval=16):
        plan = plan_extension(self.source, self.out, additional, interval)
        initialize(plan)
        return verify_extension(self.out)

    def assert_state_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for k in left:
                self.assert_state_equal(left[k], right[k])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(left, right)

    def invoke_cpu_trainer(self, command, cwd):
        self.assertEqual(Path(cwd), self.out / BOOTSTRAP)
        self.assertEqual(Path(command[3]), self.out / BOOTSTRAP / "train.py")
        meta = read_json(self.out / "run.json")
        def load(path, **_):
            model = Tiny(meta)
            model.load_state_dict(torch.load(Path(path) / "adapter_model.safetensors", weights_only=True))
            return model
        with ExitStack() as stack:
            patches = [patch.object(sys, "argv", list(map(str, command[3:]))),
                       patch.dict(sys.modules, {"bitsandbytes": SimpleNamespace(optim=SimpleNamespace(AdamW8bit=torch.optim.AdamW)),
                           "transformers": SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer()))}),
                       patch.object(train, "configure_runtime"), patch.object(train, "load_model", side_effect=load),
                       patch.object(train, "snapshot_code", return_value=meta["code_sha256"]),
                       patch.object(train, "encode_example", side_effect=lambda t, r, m: {"input_ids": [r["index"]], "supervised_tokens": 1}),
                       patch.object(train, "pad_batch", side_effect=lambda e, p: {"x": CpuInput(torch.tensor([[e[0]["input_ids"][0] / 10]]))}),
                       patch.object(train, "training_loss", side_effect=loss),
                       patch.object(train, "assert_adapter_training", return_value={"trainable_parameters": 2}),
                       patch.object(torch.cuda, "get_device_name", return_value="CPU test"),
                       patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(total_memory=0)),
                       patch.object(train, "environment", return_value={}),
                       patch("recipe.data.OverlapIndex", return_value=SimpleNamespace(match=lambda _: None)),
                       patch.object(train.signal, "signal")]
            for context in patches:
                stack.enter_context(context)
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            train.main()

    def test_default_budget_and_exact_state_copy(self):
        plan, meta = self.prepare()
        self.assertEqual(plan["target_epochs"], [144, 160, 176, 192, 208, 224, 240, 256])
        self.assertEqual((plan["start_step"], plan["total_steps"], plan["additional_updates"]), (1280, 2560, 1280))
        self.assertEqual({k for k in meta["config"] if meta["config"][k] != self.meta["config"][k]}, {"epochs", "checkpoint_epochs"})
        a = torch.load(self.anchor / "training_state.pt", weights_only=False)
        b = torch.load(self.out / "checkpoints/epoch-128/training_state.pt", weights_only=False)
        self.assertNotEqual(a.pop("config_sha256"), b.pop("config_sha256"))
        self.assert_state_equal(a, b)
        self.assertEqual(tree_hashes(self.source), self.before)
        self.assertEqual(file_hash(self.anchor / "adapter_model.safetensors"), file_hash(self.out / "checkpoints/epoch-128/adapter_model.safetensors"))

    def test_actual_training_loop_pause_resume_matches_uninterrupted_cpu(self):
        plan, meta = self.prepare(additional=3, interval=1)
        restore_rng(torch.load(self.anchor / "training_state.pt", weights_only=False)["rng"])
        updates(self.meta, self.model, self.optimizer, self.scheduler, 1280, 1310)
        expected_rng = capture_rng()
        pipeline = SimpleNamespace(invoke=self.invoke_cpu_trainer)
        self.assertFalse(launcher.execute(self.out, pipeline, stop_after=1293))
        self.assertEqual(read_json(self.out / "progress.json")["step"], 1293)
        saved = tree_hashes(self.out / "checkpoints/epoch-129")
        self.assertTrue(launcher.execute(self.out, pipeline))
        self.assertEqual(tree_hashes(self.out / "checkpoints/epoch-129"), saved)
        final = self.out / "checkpoints/epoch-131"
        self.assert_state_equal(self.model.state_dict(), torch.load(final / "adapter_model.safetensors", weights_only=True))
        state = torch.load(final / "training_state.pt", weights_only=False)
        self.assert_state_equal(self.optimizer.state_dict(), state["optimizer"])
        self.assert_state_equal(self.scheduler.state_dict(), state["scheduler"])
        self.assert_state_equal(expected_rng, state["rng"])
        self.assertEqual(state["cursor"], {"epoch_index": 131, "position": 0, "order": epoch_order(10, 42, 131)})
        rows = [r for p in sorted((self.out / "logs").glob("*.jsonl")) for r in iter_jsonl(p)]
        self.assertEqual([r["step"] for r in rows], list(range(1281, 1311)))
        self.assertTrue(all(r["learning_rate"] == 1e-5 for r in rows))
        self.assertEqual([r["sample_id"] for r in rows], [str(epoch_order(10, 42, i // 10)[i % 10]) for i in range(1280, 1310)])
        with patch.object(pipeline, "invoke", side_effect=AssertionError("finished run must not restart")):
            self.assertTrue(launcher.execute(self.out, pipeline))
        self.assertEqual(tree_hashes(self.source), self.before)

    def test_bad_source_state_is_rejected_without_publishing(self):
        state = torch.load(self.anchor / "training_state.pt", weights_only=False)
        state["cursor"]["position"] = 1
        torch.save(state, self.anchor / "training_state.pt")
        with self.assertRaisesRegex(ValueError, "cursor"):
            self.prepare()
        self.assertFalse(self.out.exists())

    def test_changed_anchor_or_run_is_rejected_on_resume(self):
        self.prepare()
        with (self.out / "checkpoints/epoch-128/training_state.pt").open("ab") as f:
            f.write(b"modified")
        with self.assertRaisesRegex(ValueError, "anchor changed"):
            verify_extension(self.out)
        self.assertEqual(tree_hashes(self.source), self.before)

    def test_incomplete_source_nested_output_and_overwrite_rejected(self):
        with self.assertRaisesRegex(ValueError, "separate"):
            plan_extension(self.source, self.source / "child")
        plan, _ = self.prepare()
        with self.assertRaises(FileExistsError):
            initialize(plan)
        (self.anchor / "training_state.pt").unlink()
        with self.assertRaisesRegex(ValueError, "full training state"):
            plan_extension(self.source, self.root / "another")

    def test_prepare_resume_does_not_depend_on_source_directory(self):
        self.prepare()
        self.source.rename(self.root / "source-moved")
        plan, _ = verify_extension(self.out)
        self.assertEqual(plan["total_steps"], 2560)

    def test_read_only_plan_and_changed_resume_request(self):
        command = ["continue_training.py", "--source-run", str(self.source), "--out", str(self.out), "--plan-only"]
        with patch.object(sys, "argv", command), patch.object(launcher, "initialize", side_effect=AssertionError("read only")):
            launcher.main()
        self.assertFalse(self.out.exists())
        self.assertEqual(tree_hashes(self.source), self.before)
        plan, _ = self.prepare()
        with self.assertRaisesRegex(ValueError, "inputs differ"):
            launcher.check_request(plan, self.source, SimpleNamespace(additional_epochs=256, checkpoint_every=16))


if __name__ == "__main__":
    unittest.main()
