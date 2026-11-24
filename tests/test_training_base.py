"""Offline: exact selected texts, no answer leakage/LoRA, durable resume, group reports."""
import asyncio
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import solve_training_base as suite
from generate_vllm import run_checkpoints
from recipe.common import ROOT, append_jsonl, file_hash, iter_jsonl, read_json, write_json, write_jsonl
from recipe.data import adapt_sft


def training_row(i):
    return adapt_sft({"question": f"Compute {i}+1. <script>example</script>", "answer": str(i + 1),
                      "response": f"REFERENCE_ONLY_SECRET_{i}. Final answer: {i + 1}",
                      "data_source": "test-source"}, i)


class Tokenizer:
    eos_token_id = 3
    def convert_tokens_to_ids(self, _):
        return 2
    def decode(self, ids, **_):
        return "<|im_end|>" if ids == [2] else r"Some reasoning. \boxed{1}<|im_end|>"
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        assert [m["role"] for m in messages] == ["user"]
        assert "REFERENCE_ONLY_SECRET" not in str(messages)
        return [101] + [ord(c) + 100 for c in messages[0]["content"]]


class Engine:
    def __init__(self, blocked=False):
        self.requests, self.active, self.peak, self.closed = [], 0, 0, False
        self.blocked = blocked
    async def generate(self, prompt, sampling_params, request_id, lora_request):
        assert lora_request is None
        assert set(prompt) == {"prompt_token_ids"}
        self.requests.append(request_id)
        self.active += 1
        self.peak = max(self.peak, self.active)
        first = len(self.requests) == 1
        try:
            if self.blocked and not first:
                await asyncio.Event().wait()
            await asyncio.sleep(0)
            yield SimpleNamespace(request_id=request_id, prompt_token_ids=prompt["prompt_token_ids"],
                                  finished=True, outputs=[SimpleNamespace(token_ids=[7, 2],
                                  finish_reason="stop", stop_reason=2)])
        finally:
            self.active -= 1
    async def abort(self, _):
        pass
    def shutdown(self):
        self.closed = True


class SelectionTests(unittest.TestCase):
    def test_shared_question_reuses_generation_but_keeps_both_references(self):
        a, b = training_row(0), training_row(1)
        panel = suite.build_panel({"random": [a], "high": [deepcopy(a)], "low": [b]},
                                  {"random": 1, "high": 1, "low": 1})
        self.assertEqual(len(panel), 2)
        self.assertEqual(panel[0]["groups"], ["random", "high"])
        self.assertEqual(len(panel[0]["references"]), 2)
        self.assertEqual(panel[0]["messages"], a["messages"])
        self.assertNotIn(a["response"], str(panel[0]["messages"]))
        self.assertEqual(panel[0]["references"][1]["sample"], a)
        changed_prompt = deepcopy(a)
        changed_prompt["messages"][0]["content"] += " Use algebra."
        different = suite.build_panel({"random": [a], "high": [changed_prompt], "low": [b]},
                                      {"random": 1, "high": 1, "low": 1})
        self.assertEqual(len(different), 3)
        bad = deepcopy(a)
        bad["answer"] = "conflicting gold"
        with self.assertRaisesRegex(ValueError, "Conflicting gold"):
            suite.build_panel({"random": [a], "high": [bad], "low": [b]},
                              {"random": 1, "high": 1, "low": 1})

    def test_wrong_size_duplicate_tampered_text_or_answer_turn_is_rejected(self):
        groups = {"random": [training_row(0)], "high": [training_row(1)], "low": [training_row(2)]}
        sizes = {k: 1 for k in groups}
        with self.assertRaisesRegex(ValueError, "400 distinct"):
            suite.build_panel(groups)
        groups["low"] = [groups["low"][0], groups["low"][0]]
        with self.assertRaisesRegex(ValueError, "distinct"):
            suite.build_panel(groups, {**sizes, "low": 2})
        groups["low"] = [training_row(2)]
        groups["low"][0]["response"] += "Changed"
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            suite.build_panel(groups, sizes)
        groups["low"] = [training_row(2)]
        groups["low"][0]["messages"].append({"role": "assistant", "content": "the answer"})
        with self.assertRaisesRegex(ValueError, "question-only"):
            suite.build_panel(groups, sizes)


class TrainingBaseTests(unittest.TestCase):
    def setUp(self):
        (ROOT / ".test_tmp").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".test_tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.sources = {k: self.root / k for k in suite.GROUP_SIZES}
        self.groups = {"random": [training_row(i) for i in range(400)],
                       "high": [training_row(i) for i in range(390, 490)],
                       "low": [training_row(i) for i in range(480, 580)]}
        for group, source in self.sources.items():
            write_jsonl(source / "data/train.jsonl", self.groups[group])
            write_json(source / "tokenizer/tokenizer_config.json", {"fake": True})
            write_json(source / "run.json", {"model": {"id": "Qwen/Qwen3-4B-Base", "revision": "a" * 40},
                       "config": read_json(ROOT / "configs/baseline.json"),
                       "data_files": {"train.jsonl": file_hash(source / "data/train.jsonl")},
                       "tokenizer_files": {"tokenizer_config.json": file_hash(source / "tokenizer/tokenizer_config.json")}})
        self.out = self.root / "output"
        self.out.mkdir()
        module = ModuleType("transformers")
        module.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer())
        def model_config(*a, **kw):
            self.assertEqual(kw["revision"], "a" * 40)
            return SimpleNamespace(model_type="qwen3", max_position_embeddings=32768,
                                   to_dict=lambda: {"model_type": "qwen3", "max_position_embeddings": 32768})
        module.AutoConfig = SimpleNamespace(from_pretrained=model_config)
        for p in (patch.dict(sys.modules, {"transformers": module, "math_verify": ModuleType("math_verify")}),
                  patch.object(suite, "runtime_identity", return_value={"test": True}),
                  patch("recipe.grading.math_equal", side_effect=lambda gold, pred: gold == pred),
                  patch.object(suite.importlib.metadata, "version", return_value="offline-test")):
            p.start()
            self.addCleanup(p.stop)

    def prepare(self):
        return suite.prepare(self.out, self.sources)

    def generated(self, prepared, count=None):
        from recipe.vllm_generation import response_record
        panel = prepared["panel"][:count] if count is not None else prepared["panel"]
        for q in panel:
            sid = q["sample_id"]
            result = SimpleNamespace(request_id=sid, finished=True, prompt_token_ids=prepared["prompts"][sid],
                                     outputs=[SimpleNamespace(token_ids=[7, 2], finish_reason="stop", stop_reason=2)])
            record = response_record(result, sid, q, prepared["prompts"][sid], 0,
                                     suite.stable_seed(prepared["settings"]["seed"], sid), Tokenizer(), 3, 2, 8192, .01)
            append_jsonl(self.out / "generation/epoch-000/responses.jsonl", record)

    def test_exact_frozen_inputs_base_only_and_generation_budget(self):
        before = {str(p): file_hash(p) for s in self.sources.values() for p in s.rglob("*") if p.is_file()}
        plan = self.prepare()
        self.assertEqual(plan["group_sizes"], {"random": 400, "high": 100, "low": 100})
        self.assertEqual((plan["group_memberships"], plan["unique_prompts"], plan["shared_prompts"]), (600, 580, 20))
        self.assertEqual(self.prepare(), plan)
        self.assertEqual(suite.verify_setup(self.out), plan)
        p = suite.generation_inputs(self.out)
        self.assertIsNone(p["checkpoints"][0]["checkpoint"])
        self.assertFalse(p["options"]["enable_lora"])
        self.assertEqual(p["options"]["dtype"], "bfloat16")
        self.assertEqual(p["options"]["max_num_seqs"], 8)
        self.assertEqual(p["options"]["max_model_len"], max(map(len, p["prompts"].values())) + 8192)
        self.assertEqual(before, {name: file_hash(name) for name in before})
        self.assertEqual(list(iter_jsonl(self.out / "setup/training/high/train.jsonl")), self.groups["high"])
        changed = self.root / "changed.json"
        write_json(changed, {**p["settings"], "max_new_tokens": 4096})
        with self.assertRaisesRegex(ValueError, "settings changed"):
            suite.prepare(self.out, self.sources, config=changed)
        (self.out / "setup/panel.jsonl").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "input changed"):
            suite.verify_setup(self.out)

    def test_interruption_resume_torn_write_and_skip_completed_without_engine(self):
        self.prepare()
        prepared = suite.generation_inputs(self.out)
        blocked = Engine(blocked=True)
        async def interrupted():
            task = asyncio.create_task(run_checkpoints(prepared, lambda _: blocked, None, SimpleNamespace, "FINAL"))
            log = self.out / "generation/epoch-000/responses.jsonl"
            for _ in range(1000):
                if log.exists():
                    break
                await asyncio.sleep(.001)
            self.assertTrue(log.exists())
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        asyncio.run(interrupted())
        self.assertTrue(blocked.closed)
        log = self.out / "generation/epoch-000/responses.jsonl"
        prefix = log.read_bytes()
        self.assertEqual(len(list(iter_jsonl(log))), 1)
        with log.open("ab") as stream:
            stream.write(b'{"torn":')
        resumed = suite.generation_inputs(self.out)
        self.assertEqual(resumed["checkpoints"][0]["pending"], 579)
        self.assertEqual(log.read_bytes(), prefix)
        self.assertTrue(list(log.parent.glob("responses.jsonl.torn-*")))
        engine = Engine()
        asyncio.run(run_checkpoints(resumed, lambda _: engine, None, SimpleNamespace, "FINAL"))
        self.assertEqual(len(engine.requests), 579)
        self.assertLessEqual(engine.peak, 8)
        self.assertTrue(log.read_bytes().startswith(prefix))
        finished = suite.generation_inputs(self.out)
        self.assertEqual(finished["checkpoints"][0]["pending"], 0)
        before = file_hash(log)
        asyncio.run(run_checkpoints(finished, lambda _: self.fail("Reloaded model for completed job"), None, None, None))
        self.assertEqual(file_hash(log), before)

    def test_partial_grading_resume_full_reports_and_changed_response_rejected(self):
        self.prepare()
        prepared = suite.generation_inputs(self.out)
        self.generated(prepared, 395)
        with self.assertRaisesRegex(ValueError, "Generation incomplete"):
            suite.saved_responses(self.out)
        suite.grade_saved(self.out, allow_partial=True)
        summary = read_json(self.out / "report/summary.json")
        self.assertEqual([(r["completed"], r["missing"]) for r in summary["groups"]], [(395, 5), (5, 95), (0, 100)])
        self.assertFalse((self.out / "grades/complete.json").exists())
        prefix = (self.out / "grades/graded.jsonl").read_bytes()
        rest = {**prepared, "panel": prepared["panel"][395:]}
        self.generated(rest)
        suite.grade_saved(self.out)
        self.assertTrue((self.out / "grades/graded.jsonl").read_bytes().startswith(prefix))
        summary = read_json(self.out / "report/summary.json")
        self.assertEqual([r["completed"] for r in summary["groups"]], [400, 100, 100])
        self.assertEqual([r["correct"] for r in summary["groups"]], [1, 0, 0])
        self.assertEqual(summary["completed_unique_prompts"], 580)
        html = (self.out / "report/report.html").read_text(encoding="utf-8")
        self.assertNotIn("<script>example</script>", html)
        self.assertIn("&lt;script&gt;example&lt;/script&gt;", html)
        self.assertIn("REFERENCE_ONLY_SECRET_0", html)
        high = list(iter_jsonl(self.out / "report/high.jsonl"))
        self.assertEqual(len(high), 100)
        self.assertEqual(high[0]["references"][1]["sample"]["row_index"], 390)
        before = file_hash(self.out / "grades/graded.jsonl")
        with patch.object(suite, "grade_one", side_effect=AssertionError("Regraded completed answer")):
            suite.grade_saved(self.out)
        self.assertEqual(file_hash(self.out / "grades/graded.jsonl"), before)
        log = self.out / "generation/epoch-000/responses.jsonl"
        rows = list(iter_jsonl(log))
        rows[0]["raw_response"] = "changed answer"
        write_jsonl(log, rows)
        with self.assertRaisesRegex(ValueError, "changed source responses"):
            suite.grade_saved(self.out)

    def test_different_model_or_changed_training_file_cannot_prepare(self):
        meta = read_json(self.sources["high"] / "run.json")
        meta["model"]["revision"] = "b" * 40
        write_json(self.sources["high"] / "run.json", meta)
        with self.assertRaisesRegex(ValueError, "differs"):
            self.prepare()
        (self.sources["low"] / "data/train.jsonl").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Frozen run input changed"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
