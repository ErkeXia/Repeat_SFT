"""No GPU/model downloads: fake pinned vLLM API, concurrent writes and recovery."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path
import tempfile
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import generate_vllm
from generate_vllm import run_checkpoints
from recipe.common import ROOT, file_hash, iter_jsonl, read_json, stable_seed, write_json
from recipe.vllm_generation import (completed_rows, engine_options, generate_epoch, response_record,
                                   sampling_options, validate_config)

TEST_TMP = ROOT / ".test_tmp"
TEST_TMP.mkdir(exist_ok=True)


class Tokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "full reasoning " + " ".join(str(i) for i in ids if not skip_special_tokens or i not in (2, 3))


def output(request_id, prompt, ids=None, reason="stop", stop=2):
    return SimpleNamespace(request_id=request_id, prompt_token_ids=prompt, finished=True,
                           outputs=[SimpleNamespace(token_ids=ids or [7, 2], finish_reason=reason, stop_reason=stop)])


class FakeEngine:
    def __init__(self, fail_id=None, blocked=False):
        self.active = 0
        self.peak = 0
        self.requests = []
        self.aborted = []
        self.closed = False
        self.fail_id = fail_id
        self.blocked = blocked

    async def generate(self, prompt, sampling_params, request_id, lora_request):
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.requests.append((request_id, list(prompt["prompt_token_ids"]), sampling_params, lora_request))
        try:
            index = int(request_id.split("q-")[1])
            if self.blocked and index != 0:
                await asyncio.Event().wait()
            await asyncio.sleep(0 if index == 0 else .01 * (1 + index % 3))
            if self.fail_id is not None and request_id.endswith(self.fail_id):
                raise RuntimeError("synthetic interrupted request")
            yield output(request_id, prompt["prompt_token_ids"])
        finally:
            self.active -= 1

    async def abort(self, request_id):
        self.aborted.append(request_id)

    def shutdown(self):
        self.closed = True


class VllmConfigTests(unittest.TestCase):
    def test_engine_and_sampling_preserve_pinned_base_bf16_and_eight_requests(self):
        cfg = read_json(ROOT / "configs" / "vllm.json")
        baseline = read_json(ROOT / "configs" / "baseline.json")
        settings = read_json(ROOT / "configs" / "generation.json")
        metadata = {"config": baseline, "model": {"id": "Qwen/Qwen3-4B-Base", "revision": "immutable-sha"}}
        prompts = {"short": [1] * 20, "long": [1] * 5000}
        opts = engine_options(metadata, cfg, prompts, 32768, settings)
        self.assertEqual(opts["max_model_len"], 5000 + 8192)
        self.assertEqual(opts["revision"], "immutable-sha")
        self.assertEqual(opts["max_num_seqs"], 8)
        self.assertEqual(opts["max_loras"], 1)
        self.assertEqual(opts["lora_dtype"], "bfloat16")
        self.assertTrue(opts["skip_tokenizer_init"])
        self.assertFalse(opts["enable_prefix_caching"])
        self.assertIsNone(opts["quantization"])
        sampling = sampling_options(settings, 3, 2)
        self.assertEqual((sampling["temperature"], sampling["top_p"], sampling["top_k"]), (.6, .95, 20))
        self.assertEqual(sampling["max_tokens"], 8192)
        self.assertEqual(sampling["stop_token_ids"], [2, 3])
        self.assertIsNone(sampling["truncate_prompt_tokens"])
        with self.assertRaisesRegex(ValueError, "requires concurrency"):
            validate_config({**cfg, "max_num_seqs": 1})
        with self.assertRaisesRegex(ValueError, "Do not truncate"):
            engine_options(metadata, {**cfg, "max_model_len": 8192}, prompts, 32768, settings)
        with self.assertRaisesRegex(ValueError, "Do not truncate"):
            engine_options(metadata, cfg, prompts, 10000, settings)
        nf4 = deepcopy(metadata)
        nf4["config"]["finetuning"]["base_quantization"] = "nf4"
        with self.assertRaisesRegex(ValueError, "BF16 LoRA only"):
            engine_options(nf4, cfg, prompts, 32768, settings)


class VllmPreparationTests(unittest.TestCase):
    def setUp(self):
        from test_grouped_evaluation import fixture
        self.temp = tempfile.TemporaryDirectory(dir=TEST_TMP)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.run, _, _, self.panel = fixture(self.root)
        metadata = read_json(self.run / "run.json")
        ft, base = metadata["config"]["finetuning"], metadata["model"]
        self.ckpt = self.run / "checkpoints" / "epoch-064"
        write_json(self.ckpt / "recipe_model.json", {"format_version": 1, "compute_dtype": "bfloat16",
                   "base_model": base, "finetuning": ft})
        write_json(self.ckpt / "adapter_config.json", {"peft_type": "LORA", "task_type": "CAUSAL_LM",
                   "r": ft["rank"], "lora_alpha": ft["alpha"], "lora_dropout": ft["dropout"],
                   "target_modules": ft["target_modules"], "bias": "none",
                   "base_model_name_or_path": base["id"], "revision": base["revision"]})
        (self.ckpt / "adapter_model.safetensors").write_bytes(b"synthetic adapter, no real model")
        marker = read_json(self.ckpt / "complete.json")
        marker["model_files"] = {n: file_hash(self.ckpt / n) for n in
                                  ("recipe_model.json", "adapter_config.json", "adapter_model.safetensors")}
        write_json(self.ckpt / "complete.json", marker)
        class FakeTokenizer(Tokenizer):
            eos_token_id = 3
            def convert_tokens_to_ids(self, token):
                return 2
            def decode(self, ids, **kw):
                return "<|im_end|>" if ids == [2] else super().decode(ids, **kw)
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return [101] + [ord(c) + 100 for c in messages[0]["content"]]
        module = ModuleType("transformers")
        module.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *a, **kw: FakeTokenizer())
        def model_config(*a, **kw):
            self.assertEqual(kw["revision"], base["revision"])
            return SimpleNamespace(model_type="qwen3", max_position_embeddings=32768,
                                   to_dict=lambda: {"model_type": "qwen3", "max_position_embeddings": 32768})
        module.AutoConfig = SimpleNamespace(from_pretrained=model_config)
        self.module = module

    def args(self, out, ids=None, config=None):
        return SimpleNamespace(run=str(self.run), out=str(out), epochs="64", sample_ids=str(ids) if ids else None,
                               limit_per_task=None, progress_label="", config=str(ROOT / "configs/generation.json"),
                               engine_config=str(config or ROOT / "configs/vllm.json"))

    def prepare(self, args):
        with patch.dict(sys.modules, {"transformers": self.module}), \
                patch.object(generate_vllm, "runtime_identity", return_value={"test_runtime": True}):
            return generate_vllm.prepare(args)

    def test_frozen_group_protocol_common_context_and_separate_backend_identity(self):
        ids = self.root / "ids.json"
        write_json(ids, [r["sample_id"] for r in self.panel[:48]])
        a = self.prepare(self.args(self.root / "group1", ids))
        write_json(ids, [r["sample_id"] for r in self.panel[48:96]])
        b = self.prepare(self.args(self.root / "group2", ids))
        self.assertEqual(a["options"], b["options"])
        self.assertEqual(len(a["panel"]), 48)
        self.assertEqual(len(a["prompts"]), 192)
        left = read_json(self.root / "group1/epoch-064/identity.json")
        right = read_json(self.root / "group2/epoch-064/identity.json")
        self.assertEqual(left["backend"], "vllm")
        self.assertEqual(left["backend_config"], right["backend_config"])
        self.assertNotEqual(left["selected_ids"], right["selected_ids"])
        self.assertTrue((self.root / "group1/epoch-064/code/requirements-vllm.txt").is_file())

    def test_changed_engine_settings_or_adapter_weights_are_rejected(self):
        out = self.root / "generation"
        self.prepare(self.args(out))
        cfg = read_json(ROOT / "configs/vllm.json")
        config = self.root / "other-engine.json"
        write_json(config, {**cfg, "gpu_memory_utilization": .9})
        with self.assertRaisesRegex(ValueError, "different input/config/code"):
            self.prepare(self.args(out, config=config))
        (self.ckpt / "adapter_model.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Adapter checkpoint changed"):
            self.prepare(self.args(self.root / "changed-adapter"))

    def test_replay_nodes_are_accepted_without_rewriting_training_configuration(self):
        import shutil
        metadata = read_json(self.run / "run.json")
        frozen_config = deepcopy(metadata["config"])
        metadata["replay"] = {"target_epochs": [6, 7]}
        write_json(self.run / "run.json", metadata)
        ckpt = self.run / "checkpoints" / "epoch-006"
        shutil.copytree(self.ckpt, ckpt)
        marker = read_json(ckpt / "complete.json")
        marker["step"] = 6
        write_json(ckpt / "complete.json", marker)
        args = self.args(self.root / "replay-generation")
        args.epochs = "6"
        prepared = self.prepare(args)
        self.assertEqual([item["epoch"] for item in prepared["checkpoints"]], [6])
        self.assertEqual(read_json(self.run / "run.json")["config"], frozen_config)
        args.epochs = "64"
        with self.assertRaisesRegex(ValueError, "not a saved training checkpoint"):
            self.prepare(args)


class VllmGenerationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEST_TMP)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.panel = [{"sample_id": f"q-{i}", "question_id": f"question-{i}", "task": "math500",
                       "messages": [{"role": "user", "content": f"Question {i}"}], "gold_answer": "2"}
                      for i in range(19)]
        self.prompts = {r["sample_id"]: [101, i + 1000] for i, r in enumerate(self.panel)}
        self.settings = {**read_json(ROOT / "configs" / "generation.json"), "max_new_tokens": 8}
        self.tokenizer = Tokenizer()

    async def generate(self, engine, out, epoch=64):
        out.mkdir(parents=True, exist_ok=True)
        await generate_epoch(engine, "adapter", self.panel, self.prompts, out, epoch, self.settings,
                             self.tokenizer, 3, 2, SimpleNamespace, "FINAL_ONLY")

    def prepared(self, epochs):
        checkpoints = []
        for epoch in epochs:
            out = self.root / f"epoch-{epoch:03d}"
            out.mkdir(exist_ok=True)
            done = completed_rows(out, self.panel, self.prompts, epoch, self.settings, 3, 2)
            checkpoints.append({"epoch": epoch, "out": out, "checkpoint": self.root / f"adapter-{epoch}",
                                "pending": len(self.panel) - len(done)})
        return {"checkpoints": checkpoints, "options": {"max_num_seqs": 8}, "panel": self.panel,
                "prompts": self.prompts, "settings": self.settings, "tokenizer": self.tokenizer,
                "eos_id": 3, "end_id": 2, "progress_label": "group=1/4"}

    async def test_eight_in_flight_full_tokens_per_question_seeds_and_no_duplicates(self):
        engine = FakeEngine()
        out = self.root / "generation"
        await self.generate(engine, out)
        self.assertEqual(engine.peak, 8)
        rows = list(iter_jsonl(out / "responses.jsonl"))
        self.assertEqual(len(rows), len(self.panel))
        self.assertEqual(len({r["sample_id"] for r in rows}), len(self.panel))
        for row in rows:
            self.assertEqual(row["prompt_token_ids"], self.prompts[row["sample_id"]])
            self.assertEqual(row["generated_token_ids"], [7, 2])
            self.assertTrue(row["normal_chat_end"])
            self.assertEqual(row["seed"], stable_seed(self.settings["seed"], row["sample_id"]))
        for request, prompt, params, _ in engine.requests:
            self.assertEqual(params.seed, stable_seed(self.settings["seed"], request.split(":")[1]))
            self.assertEqual(params.output_kind, "FINAL_ONLY")
        self.assertEqual(read_json(out / "complete.json")["samples"], len(self.panel))

    async def test_same_base_is_reused_across_checkpoints_and_finished_run_never_loads(self):
        engines = []
        def factory(options):
            self.assertEqual(options["max_num_seqs"], 8)
            engines.append(FakeEngine())
            return engines[-1]
        lora = lambda name, number, path: (name, number, path)
        await run_checkpoints(self.prepared([64, 32, 16, 8, 4]), factory, lora, SimpleNamespace, "FINAL_ONLY")
        self.assertEqual(len(engines), 1)
        self.assertTrue(engines[0].closed)
        self.assertEqual(len(engines[0].requests), len(self.panel) * 5)
        self.assertEqual({r[3][1] for r in engines[0].requests}, {65, 33, 17, 9, 5})
        originals = {p: p.read_bytes() for p in self.root.glob("epoch-*/responses.jsonl")}
        await run_checkpoints(self.prepared([64, 32, 16, 8, 4]), factory, lora, SimpleNamespace, "FINAL_ONLY")
        self.assertEqual(len(engines), 1)
        self.assertEqual(originals, {p: p.read_bytes() for p in originals})

    async def test_failure_torn_tail_and_resume_preserve_finished_answers(self):
        out = self.root / "generation"
        failed = FakeEngine(fail_id="q-1")
        with self.assertRaisesRegex(RuntimeError, "synthetic interrupted"):
            await self.generate(failed, out)
        saved = out / "responses.jsonl"
        prefix = saved.read_bytes()
        done = {r["sample_id"] for r in iter_jsonl(saved)}
        self.assertIn("q-0", done)
        self.assertNotIn("q-1", done)
        self.assertFalse((out / "complete.json").exists())
        self.assertEqual(failed.active, 0)
        with saved.open("ab") as f:
            f.write(b'{"sample_id":"half-written')
        torn = saved.read_bytes()
        resumed = FakeEngine()
        await self.generate(resumed, out)
        self.assertTrue(saved.read_bytes().startswith(prefix))
        self.assertEqual(next(out.glob("responses.jsonl.torn-*")).read_bytes(), torn)
        self.assertEqual({r[0].split(":")[1] for r in resumed.requests},
                         {r["sample_id"] for r in self.panel} - done)
        self.assertEqual(len(list(iter_jsonl(saved))), len(self.panel))

    async def test_cancellation_aborts_pending_requests_shuts_down_and_keeps_saved_row(self):
        engine = FakeEngine(blocked=True)
        prepared = self.prepared([64])
        task = asyncio.create_task(run_checkpoints(prepared, lambda options: engine,
                                   lambda *a: a, SimpleNamespace, "FINAL_ONLY"))
        log = self.root / "epoch-064" / "responses.jsonl"
        try:
            for _ in range(100):
                if log.exists():
                    break
                await asyncio.sleep(.01)
            self.assertTrue(log.exists())
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(len(list(iter_jsonl(log))), 1)
        self.assertTrue(engine.closed)
        self.assertEqual(engine.active, 0)
        self.assertTrue(engine.aborted)
        self.assertFalse((log.parent / "complete.json").exists())

    def test_stop_semantics_and_raw_tokens_are_preserved(self):
        row = self.panel[0]
        prompt = self.prompts[row["sample_id"]]
        for ids, finish, stop, expected in (([7, 2], "stop", 2, "chat_end"),
                                           ([7, 3], "stop", None, "base_eos"),
                                           ([7] * 8, "length", None, "length"),
                                           ([7] * 7 + [2], "length", None, "chat_end")):
            r = response_record(output("r", prompt, ids, finish, stop), "r", row, prompt, 64, 1,
                                self.tokenizer, 3, 2, 8, .1)
            self.assertEqual(r["generated_token_ids"], ids)
            self.assertEqual(r["stop_reason"], expected)
            self.assertEqual(r["length_truncated"], expected == "length")
        for bad in (output("wrong", prompt), output("r", [999]),
                    output("r", prompt, [7] * 9, "length", None),
                    output("r", prompt, [7], "abort", None), output("r", prompt, [7, 2], "stop", "string")):
            with self.assertRaises(ValueError):
                response_record(bad, "r", row, prompt, 64, 1, self.tokenizer, 3, 2, 8, .1)

    async def test_resume_rejects_modified_completed_output(self):
        out = self.root / "generation"
        await self.generate(FakeEngine(), out)
        with (out / "responses.jsonl").open("ab") as f:
            f.write(b"\n")
        with self.assertRaisesRegex(ValueError, "Completed vLLM responses changed"):
            completed_rows(out, self.panel, self.prompts, 64, self.settings, 3, 2)


if __name__ == "__main__":
    unittest.main()
