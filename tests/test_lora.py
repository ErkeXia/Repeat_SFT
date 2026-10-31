"""CPU-only PEFT integration with tiny random Qwen models; never download weights."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from recipe.adapters import (adapter_spec, assert_adapter_training, checkpoint_state_dict,
                             trainable_parameters, validate_finetuning)
from recipe.common import ROOT, read_json, restore_rng, seed_everything, write_json

SETTINGS = read_json(ROOT / "configs" / "qlora_nf4.json")["finetuning"]
BASE = {"id": "Qwen/Qwen3-4B-Base", "revision": "a" * 40}
HAS_PEFT = all(importlib.util.find_spec(n) for n in ("torch", "transformers", "peft", "safetensors"))


class ProtocolTests(unittest.TestCase):
    def test_both_presets_are_explicit_lora_and_same_optimization(self):
        baseline = read_json(ROOT / "configs" / "baseline.json")
        q = read_json(ROOT / "configs" / "qlora_nf4.json")
        b = read_json(ROOT / "configs" / "lora_bf16.json")
        self.assertEqual(baseline, b)
        self.assertEqual(q["finetuning"]["base_quantization"], "nf4")
        self.assertEqual(b["finetuning"]["base_quantization"], "none")
        validate_finetuning(q["finetuning"])
        validate_finetuning(b["finetuning"])
        b["finetuning"]["base_quantization"] = "nf4"
        self.assertEqual(q, b)
        self.assertEqual(q["expected_train_size"] * q["epochs"], 25600)

    def test_full_parameter_and_invalid_lora_settings_rejected(self):
        for settings in (None, {}, dict(SETTINGS, method="full"), dict(SETTINGS, rank=0),
                         dict(SETTINGS, target_modules=["lm_head"]), dict(SETTINGS, dropout=1),
                         dict(SETTINGS, base_quantization="8bit")):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                validate_finetuning(settings)


@unittest.skipUnless(HAS_PEFT, "Install declared PEFT/Transformers deps for tiny CPU integration tests")
class LoraIntegrationTests(unittest.TestCase):
    def setUp(self):
        (ROOT / ".test_tmp").mkdir(exist_ok=True)
        self.env = patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        import torch
        self.torch = torch
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.old_threads)
        self.settings = dict(SETTINGS, rank=2, alpha=4, base_quantization="none")
        self.batch = {"input_ids": torch.tensor([[1, 4, 5, 6, 7, 2, 0]]),
                      "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0]]),
                      "labels": torch.tensor([[-100, -100, -100, 6, 7, 2, -100]])}

    def base(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        self.torch.manual_seed(7)
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=37, hidden_size=32, intermediate_size=48,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=128, bos_token_id=1, eos_token_id=2, pad_token_id=0,
            attention_dropout=0.0))
        model.config._attn_implementation = "eager"
        model.requires_grad_(False)
        return model

    def model(self):
        from recipe.modeling import attach_adapter
        model = attach_adapter(self.base(), self.settings, BASE)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return model.train()

    def optimizer(self, model):
        opt = self.torch.optim.AdamW(trainable_parameters(model), lr=0.001, weight_decay=0)
        scheduler = self.torch.optim.lr_scheduler.LambdaLR(opt, lambda n: min(1, (n + 1) / 3))
        return opt, scheduler

    def update(self, model, opt, scheduler):
        from recipe.modeling import training_loss
        opt.zero_grad(set_to_none=True)
        loss = training_loss(model, self.batch, chunk_size=2)
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(trainable_parameters(model), 1.0)
        opt.step()
        scheduler.step()
        return loss.item()

    def save(self, folder, model, opt=None, scheduler=None, step=0):
        from recipe.checkpoints import save_checkpoint
        class Tokenizer:
            def save_pretrained(self, path):
                write_json(Path(path) / "tokenizer_config.json", {"synthetic": True})
        return save_checkpoint(folder, model, Tokenizer(), step, "identity", opt, scheduler,
                               {"epoch_index": 0, "position": step, "order": [0, 1, 2, 3]})

    def test_zero_adapter_equals_base_and_chunked_loss_includes_only_answer(self):
        from recipe.modeling import attach_adapter, teacher_forcing, training_loss
        base = self.base().eval()
        with self.torch.no_grad():
            before = base(input_ids=self.batch["input_ids"], attention_mask=self.batch["attention_mask"]).logits
        model = attach_adapter(base, self.settings, BASE).eval()
        after = model(**self.batch)
        self.assertTrue(self.torch.equal(before, after.logits))
        self.assertTrue(self.torch.allclose(training_loss(model, self.batch, 2), after.loss, atol=1e-6))
        metrics = teacher_forcing(model, self.batch, 2)
        self.assertEqual(metrics["n_tokens"], 3)
        self.assertEqual(metrics["target_ids"].tolist(), [6, 7, 2])
        self.assertEqual(metrics["positions"].tolist(), [3, 4, 5])

    def test_only_adapters_update_and_real_peft_gradients_match_full_logits(self):
        from recipe.modeling import training_loss
        model = self.model()
        frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        training_loss(model, self.batch, 2).backward()
        chunk_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
        model.zero_grad(set_to_none=True)
        model(**self.batch).loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.assertTrue(self.torch.allclose(chunk_grads[n], p.grad, atol=1e-6), n)
            else:
                self.assertIsNone(p.grad, n)
        opt, sched = self.optimizer(model)
        self.update(model, opt, sched)
        for n, p in model.named_parameters():
            if n in frozen:
                self.assertTrue(self.torch.equal(frozen[n], p), n)
        self.assertTrue(any(p.abs().sum() > 0 for n, p in model.named_parameters() if "lora_B." in n))
        assert_adapter_training(model)
        next(model.get_base_model().get_input_embeddings().parameters()).requires_grad_(True)
        with self.assertRaises(ValueError):
            assert_adapter_training(model)

    def test_bf16_base_uses_fp32_trainable_adapters_and_chunked_backward(self):
        from recipe.modeling import attach_adapter, teacher_forcing
        base = self.base().to(dtype=self.torch.bfloat16)
        model = attach_adapter(base, self.settings, BASE)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        self.assertTrue(all(p.dtype == self.torch.float32 for p in trainable_parameters(model)))
        self.assertEqual(model.get_base_model().lm_head.weight.dtype, self.torch.bfloat16)
        opt, sched = self.optimizer(model)
        self.assertGreater(self.update(model, opt, sched), 0)
        model.eval()
        self.assertEqual(teacher_forcing(model, self.batch, 2)["n_tokens"], 3)

    def test_adapter_save_reload_exact_resume_and_generation(self):
        from recipe.modeling import load_model
        from recipe.storage import checkpoint_estimate, tensor_storage_bytes
        from recipe.checkpoints import latest_recovery
        model = self.model()
        opt, sched = self.optimizer(model)
        seed_everything(19)
        self.update(model, opt, sched)
        with tempfile.TemporaryDirectory(dir=ROOT / ".test_tmp") as tmp:
            path = Path(tmp) / "checkpoints" / "epoch-001"
            self.save(path, model, opt, sched, step=1)
            self.assertEqual(latest_recovery(tmp), path)
            self.assertFalse((path / "model.safetensors").exists())
            self.assertTrue((path / "adapter_model.safetensors").exists())
            spec = adapter_spec(path, finetuning=self.settings, base_model=BASE)
            self.assertEqual(spec["base_model"], BASE)
            expected_bytes = tensor_storage_bytes(checkpoint_state_dict(model))
            self.assertEqual(checkpoint_estimate(model, opt)["weight_bytes"], expected_bytes)
            self.assertLess(expected_bytes, tensor_storage_bytes(model.state_dict()))
            later = [self.update(model, opt, sched) for _ in range(2)]
            with patch("torch.cuda.is_available", return_value=True), \
                 patch("torch.cuda.is_bf16_supported", return_value=True), \
                 patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=lambda *a, **kw: self.base()) as loading:
                restored = load_model(path, training=True, finetuning=self.settings, base_model=BASE)
            self.assertEqual(loading.call_args.args[0], BASE["id"])
            self.assertEqual(loading.call_args.kwargs["revision"], BASE["revision"])
            new_opt, new_sched = self.optimizer(restored)
            state = self.torch.load(path / "training_state.pt", weights_only=False)
            new_opt.load_state_dict(state["optimizer"])
            new_sched.load_state_dict(state["scheduler"])
            restore_rng(state["rng"])
            self.assertEqual(state["cursor"]["position"], 1)
            self.assertEqual(later, [self.update(restored, new_opt, new_sched) for _ in range(2)])
            for n, p in checkpoint_state_dict(model).items():
                self.assertTrue(self.torch.equal(p, checkpoint_state_dict(restored)[n]), n)
            # Exercise the inference adapter loader, not just training reload.
            with patch("torch.cuda.is_available", return_value=True), \
                 patch("torch.cuda.is_bf16_supported", return_value=True), \
                 patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=lambda *a, **kw: self.base()):
                inference = load_model(path, finetuning=self.settings, base_model=BASE)
            self.assertFalse(any(p.requires_grad for p in inference.parameters()))
            self.assertFalse(inference.training)
            generated = inference.generate(input_ids=self.batch["input_ids"][:, :3], max_new_tokens=2,
                                           do_sample=False, use_cache=True)
            self.assertGreater(generated.shape[1], 3)

    def test_checkpoint_rejects_wrong_base_quantization_and_tampering(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".test_tmp") as tmp:
            path = Path(tmp) / "adapter"
            self.save(path, self.model())
            with self.assertRaisesRegex(ValueError, "configuration"):
                adapter_spec(path, finetuning=dict(self.settings, base_quantization="nf4"))
            with self.assertRaisesRegex(ValueError, "Base"):
                adapter_spec(path, base_model=dict(BASE, revision="wrong"))
            with (path / "adapter_model.safetensors").open("ab") as f:
                f.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "changed"):
                adapter_spec(path)

    def test_nf4_preparation_keeps_large_frozen_layers_bf16_and_norms_fp32(self):
        from recipe.modeling import prepare_base
        model = self.base().to(dtype=self.torch.bfloat16)
        model.is_loaded_in_4bit = True
        prepared = prepare_base(model, SETTINGS)
        self.assertEqual(prepared.get_input_embeddings().weight.dtype, self.torch.bfloat16)
        self.assertEqual(prepared.get_output_embeddings().weight.dtype, self.torch.bfloat16)
        self.assertEqual(prepared.model.layers[0].input_layernorm.weight.dtype, self.torch.float32)
        self.assertFalse(any(p.requires_grad for p in prepared.parameters()))

    def test_nf4_loader_and_base_scorer_use_identical_quantization_without_training_adapter(self):
        from recipe.modeling import load_model
        for training in (False, True):
            with patch("torch.cuda.is_available", return_value=True), \
                 patch("torch.cuda.is_bf16_supported", return_value=True), \
                 patch("transformers.BitsAndBytesConfig", side_effect=lambda **kw: kw), \
                 patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=lambda *a, **kw: self.base()) as loading, \
                 patch("recipe.modeling.prepare_base", side_effect=lambda m, s: m):
                model = load_model(BASE["id"], revision=BASE["revision"], finetuning=SETTINGS, training=training)
            kw = loading.call_args.kwargs
            self.assertEqual(kw["device_map"], {"": 0})
            self.assertEqual(kw["quantization_config"]["bnb_4bit_quant_type"], "nf4")
            self.assertTrue(kw["quantization_config"]["bnb_4bit_use_double_quant"])
            self.assertEqual(bool(getattr(model, "peft_config", None)), training)
            if training:
                assert_adapter_training(model)


if __name__ == "__main__":
    unittest.main()
