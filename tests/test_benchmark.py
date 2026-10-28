"""Benchmark accounting, real tiny-model updates and failure isolation; no Hub/GPU calls."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import benchmark_train as bench
from recipe.common import (ROOT, epoch_order, file_hash, iter_jsonl, lr_for_update,
                           read_json, seed_everything, write_json, write_jsonl)
from recipe.encoding import pad_batch
import test_offline

TEST_TMP = ROOT / ".test_tmp"
TEST_TMP.mkdir(exist_ok=True)


class BenchmarkTests(unittest.TestCase):
    def test_four_cases_cover_identical_400_examples_in_identical_order(self):
        self.assertEqual([(c["batch_size"], c["attention"]) for c in bench.CASES],
                         [(1, "sdpa"), (4, "sdpa"), (1, "flash_attention_2"), (4, "flash_attention_2")])
        order = epoch_order(400, 42, 0)
        for case in bench.CASES:
            batches = bench.batch_indices(order, case["batch_size"])
            self.assertEqual([i for batch in batches for i in batch], order)
            self.assertEqual(len(batches), 400 // case["batch_size"])
        with self.assertRaisesRegex(ValueError, "exactly 400"):
            bench.validate_rows([])
        rows = [{"sample_id": str(i), "question_id": str(i)} for i in range(400)]
        rows[-1]["question_id"] = rows[0]["question_id"]
        with self.assertRaisesRegex(ValueError, "Duplicate question_id"):
            bench.validate_rows(rows)

    def test_warmup_excludes_same_examples_not_same_number_of_updates(self):
        def records(batch):
            return [{"sample_start": i, "batch_size": batch, "update_seconds": 2 * batch,
                     "input_tokens": 10 * batch, "padded_tokens": 12 * batch, "supervised_tokens": 6 * batch}
                    for i in range(0, 400, batch)]
        for batch in (1, 4):
            stats = bench.summarize_steps(records(batch), 16)
            self.assertEqual(stats["all_updates"]["examples"], 400)
            steady = stats["after_timing_warmup"]
            self.assertEqual(steady["examples"], 384)
            self.assertEqual(steady["update_seconds"], 768)
            self.assertEqual(steady["nonpadding_tokens_per_second"], 5)
            self.assertAlmostEqual(steady["padding_fraction"], 1 / 6)
        config = read_json(ROOT / "configs" / "baseline.json")
        for bad in (-4, 2, 400):
            with self.assertRaisesRegex(ValueError, "multiple of 4"):
                bench.validate_config(config, bad)

    def test_kernel_evidence_distinguishes_sdpa_flash_math_and_external(self):
        names = ["aten::_scaled_dot_product_flash_attention", "aten::_scaled_dot_product_attention_math", "aten::mm"]
        result = bench.attention_evidence(names)
        self.assertEqual(result["inferred_backends"], ["pytorch_flash", "pytorch_math"])
        self.assertNotIn("aten::mm", result["operator_names"])
        self.assertEqual(bench.attention_evidence(["flash_attn::_flash_attn_varlen_forward"])["inferred_backends"],
                         ["external_flash_attn"])
        self.assertTrue(bench.attention_evidence([])["inferred_backends"][0].startswith("unclassified"))

    def test_tiny_updates_match_independent_full_logits_with_unequal_lengths(self):
        import torch
        import torch.nn.functional as F
        seed_everything(17)
        config = dict(read_json(ROOT / "configs" / "baseline.json"), logit_chunk_size=2, warmup_steps=2)
        rows = [{"sample_id": str(i), "question_id": f"q{i}"} for i in range(8)]
        encoded = []
        for i in range(8):
            ids = [1, 4] + [7] * (i % 3 + 1) + [2]
            encoded.append({"input_ids": ids, "labels": [-100, -100] + ids[2:],
                            "attention_mask": [1] * len(ids), "supervised_tokens": len(ids) - 2})
        order = epoch_order(8, 42, 0)
        for batch_size in (1, 4):
            model = test_offline.ModelTests.model()
            reference = copy.deepcopy(model)
            optimizers = [torch.optim.AdamW(m.parameters(), lr=config["learning_rate"], weight_decay=0)
                          for m in (model, reference)]
            schedulers = [torch.optim.lr_scheduler.LambdaLR(
                opt, lambda completed: lr_for_update(completed + 1, 1.0, config["warmup_steps"])) for opt in optimizers]
            with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp, patch("builtins.print"):
                metrics = bench.train_epoch(model, optimizers[0], schedulers[0], SimpleNamespace(pad_token_id=3),
                    rows, encoded, order, {"name": "test", "batch_size": batch_size}, config, Path(tmp), 4, device="cpu")
                logs = list(iter_jsonl(Path(tmp) / "steps.jsonl"))
                self.assertEqual([s for r in logs for s in r["sample_ids"]], [str(i) for i in order])
                self.assertEqual(metrics["all_updates"]["examples"], 8)
                self.assertEqual(metrics["after_timing_warmup"]["examples"], 4)
                self.assertEqual(metrics["all_updates"]["input_tokens"], sum(len(e["input_ids"]) for e in encoded))
                self.assertEqual(read_json(Path(tmp) / "progress.json")["status"], "training_complete")
            for step, indices in enumerate(bench.batch_indices(order, batch_size)):
                batch = pad_batch([encoded[i] for i in indices], 3)
                optimizers[1].zero_grad(set_to_none=True)
                hidden = reference.model(**batch).last_hidden_state
                logits = reference.lm_head(hidden[:, :-1]).reshape(-1, 32)
                loss = F.cross_entropy(logits, batch["labels"][:, 1:].reshape(-1), ignore_index=-100)
                self.assertAlmostEqual(loss.item(), logs[step]["loss"], places=5)
                self.assertAlmostEqual(optimizers[1].param_groups[0]["lr"], logs[step]["learning_rate"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(reference.parameters(), config["max_grad_norm"])
                optimizers[1].step()
                schedulers[1].step()
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_preparation_freezes_inputs_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            root = Path(tmp)
            data = root / "prepared"
            (data / "tokenizer").mkdir(parents=True)
            (data / "tokenizer" / "tokenizer.json").write_text("synthetic", encoding="utf-8")
            (data / "chat_template.jinja").write_text("synthetic", encoding="utf-8")
            rows = [{"sample_id": str(i), "question_id": str(i), "response": "full reasoning"} for i in range(400)]
            write_jsonl(data / "train.jsonl", rows)
            files = {p.relative_to(data).as_posix(): file_hash(p) for p in data.rglob("*") if p.is_file()}
            write_json(data / "manifest.json", {"files": files, "model": {"id": "Qwen/Qwen3-4B-Base", "revision": "pinned"},
                                                "arguments": {"max_length": 10240}})
            args = SimpleNamespace(data=str(data), out=str(root / "output"), config=None, seed=7, warmup_samples=None,
                                   save_weights=None, kernel_probe=None)
            with patch.object(bench, "snapshot_code", return_value="source-hash"):
                out, manifest = bench.prepare_run(args)
                with self.assertRaises(FileExistsError):
                    bench.prepare_run(args)
            with patch.object(bench, "verified_code", return_value=out / "code"):
                verified, _ = bench.verify_run(out)
                self.assertEqual(verified, manifest)
                self.assertEqual(verified["config"]["epochs"], 1)
                self.assertEqual(verified["config"]["seed"], 7)
                self.assertEqual(list(iter_jsonl(out / "data" / "train.jsonl")), rows)
                self.assertEqual(read_json(out / "order.json")["indices"], epoch_order(400, 7, 0))
                (out / "data" / "train.jsonl").write_text("changed", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Frozen benchmark input changed"):
                    bench.verify_run(out)

    def test_failures_continue_and_resume_only_retries_failed_case_without_overwrite(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            out = Path(tmp)
            manifest = {"identity": "benchmark-identity", "warmup_samples": 16}
            calls = []
            fail = True

            def invoke(command, attempt):
                case_name = command[command.index("--worker") + 1]
                calls.append(case_name)
                case = next(c for c in bench.CASES if c["name"] == case_name)
                if case_name == "b4-sdpa" and fail:
                    return -9  # Also verify a hard-killed worker with no result.json.
                write_json(attempt / "result.json", {"case": case_name, "identity": manifest["identity"],
                    "attempt": str(attempt), "batch_size": case["batch_size"], "attention": case["attention"],
                    "status": "complete", "metrics": {"epoch_wall_seconds": 100 if case_name == "b1-sdpa" else 50,
                                                         "all_updates": {"updates": 400 // case["batch_size"]}}})
                return 0

            with patch.object(bench, "invoke_worker", side_effect=invoke), patch("builtins.print"):
                self.assertEqual(bench.run_cases(out, manifest, ROOT), 1)
                self.assertEqual(calls, [c["name"] for c in bench.CASES])
                originals = {p: p.read_bytes() for p in out.glob("*/attempt-*/result.json")}
                fail = False
                calls.clear()
                self.assertEqual(bench.run_cases(out, manifest, ROOT), 0)
                self.assertEqual(calls, ["b4-sdpa"])
                for path, content in originals.items():
                    self.assertEqual(path.read_bytes(), content)
            summary = read_json(out / "summary.json")
            self.assertEqual([r["speedup_vs_b1_sdpa"] for r in summary["table"]], [1, 2, 2, 2])
            self.assertEqual(len(list((out / "b4-sdpa").glob("attempt-*"))), 2)
            self.assertTrue((out / "summary.csv").exists())

    def test_interruption_records_failure_and_stops_matrix(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            out = Path(tmp)
            with patch.object(bench, "invoke_worker", side_effect=KeyboardInterrupt), patch("builtins.print"):
                self.assertEqual(bench.run_cases(out, {"identity": "id", "warmup_samples": 16}, ROOT), 130)
            table = read_json(out / "summary.json")["table"]
            self.assertEqual([r["status"] for r in table], ["interrupted", "pending", "pending", "pending"])


if __name__ == "__main__":
    unittest.main()
