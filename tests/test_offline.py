"""Offline invariants: synthetic tokens/tiny random weights only, zero Hub calls."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import random
import re
import tempfile
from types import SimpleNamespace
import unittest

from recipe.common import (ROOT, append_jsonl, capture_rng, epoch_order, lr_for_update,
                           read_json, recover_jsonl, restore_rng, seed_everything, write_json)
from recipe.data import OverlapIndex, adapt_gpqa, adapt_sft, matched_loss_groups
from recipe.encoding import encode_example, pad_batch, question_id
from recipe.grading import extract_answer, grade_one, last_boxed
from generate import stop_details

TEST_TMP = ROOT / ".test_tmp"
TEST_TMP.mkdir(exist_ok=True)


class FakeTokenizer:
    """Character tokenizer with atomic delimiters; not a pretrained tokenizer."""
    specials = {"<|im_start|>": 1, "<|im_end|>": 2, "<|endoftext|>": 3}
    pad_token_id = eos_token_id = 3

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return self.encode(text) if tokenize else text

    def encode(self, text, add_special_tokens=False):
        pieces = re.split("(" + "|".join(re.escape(k) for k in self.specials) + ")", text)
        result = []
        for part in pieces:
            result.extend([self.specials[part]] if part in self.specials else [ord(c) + 10 for c in part])
        return result

    def decode(self, ids, skip_special_tokens=False):
        reverse = {v: k for k, v in self.specials.items()}
        return "".join(reverse[i] if i in reverse else chr(i - 10) for i in ids)

    def convert_tokens_to_ids(self, text):
        return self.specials.get(text)

    def save_pretrained(self, path):
        write_json(Path(path) / "tokenizer.json", self.specials)


def example(question="What is 2+2?", response="<think>2 plus 2 equals 4.</think>\n\\boxed{4}"):
    return adapt_sft({"question": question, "response": response, "answer": "4", "data_source": "synthetic"}, 0)


class EncodingTests(unittest.TestCase):
    def test_reasoning_final_end_supervised_user_not(self):
        tok, row = FakeTokenizer(), example()
        encoded = encode_example(tok, row)
        self.assertIn(row["response"], encoded["supervised_text"])
        self.assertTrue(encoded["supervised_text"].endswith("<|im_end|>"))
        self.assertNotIn(row["question"], encoded["supervised_text"])
        self.assertTrue(all(v == -100 for v in encoded["labels"][:encoded["prompt_length"]]))
        self.assertEqual(encoded["labels"][encoded["assistant_end"]], 2)
        self.assertEqual(encoded["labels"][-1], -100)  # Template's trailing newline.

    def test_overlength_is_filtered_without_truncation(self):
        tok, row = FakeTokenizer(), example()
        n = len(encode_example(tok, row)["input_ids"])
        self.assertIsNotNone(encode_example(tok, row, n))
        self.assertIsNone(encode_example(tok, row, n - 1))

    def test_padding_mask_with_eos_as_pad(self):
        tok = FakeTokenizer()
        examples = [encode_example(tok, example()), encode_example(tok, example(response="4"))]
        batch = pad_batch(examples, tok.pad_token_id)
        self.assertTrue(bool(batch["labels"][1, len(examples[1]["input_ids"]):].eq(-100).all()))
        self.assertEqual(batch["labels"][1, examples[1]["assistant_end"]].item(), 2)

    def test_corrupted_template_fails(self):
        class Broken(FakeTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                copied = copy.deepcopy(messages)
                if copied[-1]["role"] == "assistant":
                    copied[-1]["content"] = "Only a final answer"
                return super().apply_chat_template(copied, **kwargs)
        with self.assertRaises(ValueError):
            encode_example(Broken(), example())

    def test_embedded_delimiter_fails(self):
        with self.assertRaises(ValueError):
            encode_example(FakeTokenizer(), example(response="4<|im_end|>"))


class DataTests(unittest.TestCase):
    def test_response_field_not_answer(self):
        row = example()
        self.assertIn("<think>", row["response"])
        self.assertEqual(row["answer"], "4")

    def test_normalization_preserves_math_distinctions(self):
        self.assertEqual(question_id("x + 1"), question_id("x+1"))
        self.assertNotEqual(question_id("x+1"), question_id("x-1"))
        self.assertNotEqual(question_id("A+a"), question_id("a+a"))

    def test_overlap_exact_and_near(self):
        row = example(question="A sufficiently long mathematical question about a triangle with sides three four and five.")
        index = OverlapIndex([row])
        self.assertEqual(index.match(row)["method"], "normalized_exact")
        similar = example(question=row["question"] + " ")
        self.assertIsNotNone(index.match(similar))
        near = example(question=row["question"] + "?")
        self.assertEqual(index.match(near)["method"], "char7_jaccard")
        self.assertIsNone(index.match(example(question="Unrelated question about prime numbers")))

    def test_gpqa_shuffle_is_stable_and_gold_tracks(self):
        raw = {"Question": "What is correct?", "Correct Answer": "correct", **{f"Incorrect Answer {i}": f"wrong{i}" for i in (1, 2, 3)}}
        a, b = adapt_gpqa(raw, 0, 7), adapt_gpqa(raw, 0, 7)
        self.assertEqual(a, b)
        self.assertEqual(a["choices"]["ABCD".index(a["gold_answer"])], "correct")

    def test_matched_groups_disjoint_and_same_strata(self):
        rows = [{"sample_id": str(i), "source": "a" if i % 2 else "b", "n_supervised": 100 + (i % 3) * 100} for i in range(180)]
        scores = {str(i): i for i in range(180)}
        groups, _ = matched_loss_groups(rows, scores, 24, seed=6, length_bin=100)
        from collections import Counter
        counts = [Counter((r["source"], r["n_supervised"] // 100) for r in group) for group in groups.values()]
        self.assertEqual(counts[0], counts[1])
        self.assertEqual(counts[1], counts[2])
        ids = [r["sample_id"] for group in groups.values() for r in group]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(len(x) == 24 for x in groups.values()))

    def test_no_silent_matching_relaxation(self):
        rows = [{"sample_id": str(i), "source": str(i), "n_supervised": 10} for i in range(9)]
        with self.assertRaises(ValueError):
            matched_loss_groups(rows, {str(i): i for i in range(9)}, 1)


class BehaviorTests(unittest.TestCase):
    def test_boxed_nested_fraction(self):
        self.assertEqual(last_boxed(r"\boxed{1} then \boxed{\frac{1}{2}}"), r"\frac{1}{2}")

    def test_unfinished_reasoning_not_final(self):
        self.assertEqual(extract_answer(r"<think>Maybe \boxed{A}", "gpqa"), (None, "unfinished_think"))
        self.assertEqual(extract_answer(r"<think>\boxed{B}</think>\boxed{A}", "gpqa"), ("A", "boxed_letter"))

    def test_gpqa_no_incidental_letter_extraction(self):
        self.assertIsNone(extract_answer("A molecule contains B atoms.", "gpqa")[0])
        self.assertEqual(extract_answer("The final answer is (C).", "gpqa")[0], "C")
        self.assertTrue(grade_one(r"\boxed{D}", "D", "gpqa")["correct"])

    def test_stop_causes_and_last_token_at_cap(self):
        self.assertEqual(stop_details([8, 2], 3, 2, 8)["stop_reason"], "chat_end")
        self.assertEqual(stop_details([8, 3], 3, 2, 8)["stop_reason"], "base_eos")
        self.assertTrue(stop_details([8, 9], 3, 2, 2)["length_truncated"])
        result = stop_details([8, 2], 3, 2, 2)
        self.assertTrue(result["reached_length_limit"])
        self.assertFalse(result["length_truncated"])

    def test_shuffle_and_warmup(self):
        self.assertEqual(epoch_order(400, 42, 2), epoch_order(400, 42, 2))
        self.assertNotEqual(epoch_order(400, 42, 1), epoch_order(400, 42, 2))
        self.assertEqual(sorted(epoch_order(400, 42, 0)), list(range(400)))
        self.assertAlmostEqual(lr_for_update(1, 1e-5, 100), 1e-7)
        self.assertEqual(lr_for_update(100, 1e-5, 100), 1e-5)
        self.assertEqual(lr_for_update(25600, 1e-5, 100), 1e-5)

    def test_torn_jsonl_preserved_and_repaired(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            path = Path(tmp) / "log.jsonl"
            append_jsonl(path, {"step": 1})
            with path.open("ab") as f:
                f.write(b'{"step":')
            self.assertEqual(recover_jsonl(path), [{"step": 1}])
            self.assertEqual(len(list(Path(tmp).glob("*.torn-*"))), 1)


class ModelTests(unittest.TestCase):
    @staticmethod
    def model(dropout=0.0):
        import torch
        class TinyBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(32, 12)
                self.dropout = torch.nn.Dropout(dropout)
            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(last_hidden_state=self.dropout(self.embedding(input_ids)))
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = TinyBackbone()
                self.lm_head = torch.nn.Linear(12, 32)
            def save_pretrained(self, path, **kwargs):
                torch.save(self.state_dict(), Path(path) / "weights.pt")
        return TinyModel()

    @staticmethod
    def batch():
        import torch
        return {"input_ids": torch.tensor([[1, 4, 7, 8, 2, 3]]),
                "labels": torch.tensor([[-100, -100, 7, 8, 2, -100]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0]])}

    def test_chunked_loss_and_gradients_equal_full_logits(self):
        import torch
        from recipe.modeling import training_loss, teacher_forcing
        seed_everything(4)
        model = self.model()
        other = copy.deepcopy(model)
        batch = self.batch()
        actual = training_loss(model, batch, chunk_size=2)
        hidden = other.model(**batch).last_hidden_state
        logits = other.lm_head(hidden)[:, :-1].reshape(-1, 32)
        expected = torch.nn.functional.cross_entropy(logits, batch["labels"][:, 1:].reshape(-1), ignore_index=-100)
        self.assertAlmostEqual(actual.item(), expected.item(), places=6)
        actual.backward()
        expected.backward()
        for a, b in zip(model.parameters(), other.parameters()):
            torch.testing.assert_close(a.grad, b.grad)
        scores = teacher_forcing(model, batch, chunk_size=2)
        self.assertEqual(scores["positions"].tolist(), [2, 3, 4])
        self.assertEqual(scores["target_ids"].tolist(), [7, 8, 2])
        self.assertAlmostEqual(scores["mean_loss"], expected.item(), places=6)

    def test_checkpoint_optimizer_rng_and_scheduler_resume(self):
        import numpy as np
        import torch
        from recipe.checkpoints import latest_recovery, save_checkpoint
        from recipe.modeling import training_loss
        seed_everything(91)
        model = self.model(dropout=0.25)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: lr_for_update(n + 1, 1.0, 3))

        def update(m, opt, sched):
            opt.zero_grad()
            loss = training_loss(m, self.batch(), 2)
            loss.backward()
            opt.step()
            sched.step()
            return loss.item()

        update(model, optimizer, scheduler)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            path = Path(tmp) / "recovery" / "step-1"
            save_checkpoint(path, model, FakeTokenizer(), 1, "test", optimizer, scheduler,
                            {"epoch_index": 0, "position": 1, "order": [0, 1]})
            self.assertEqual(latest_recovery(tmp), path)
            expected_loss = update(model, optimizer, scheduler)
            state = torch.load(path / "training_state.pt", weights_only=False)
            restored = self.model(dropout=0.25)
            restored.load_state_dict(torch.load(path / "weights.pt", weights_only=True))
            opt = torch.optim.AdamW(restored.parameters(), lr=1e-3)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda n: lr_for_update(n + 1, 1.0, 3))
            opt.load_state_dict(state["optimizer"])
            sched.load_state_dict(state["scheduler"])
            restore_rng(state["rng"])
            actual_loss = update(restored, opt, sched)
            self.assertEqual(actual_loss, expected_loss)
            for a, b in zip(model.parameters(), restored.parameters()):
                self.assertTrue(torch.equal(a, b))
            self.assertEqual(optimizer.param_groups[0]["lr"], opt.param_groups[0]["lr"])
            rng = capture_rng()
            expected = (random.random(), np.random.random(), torch.rand(1).item())
            restore_rng(rng)
            self.assertEqual(expected, (random.random(), np.random.random(), torch.rand(1).item()))


if __name__ == "__main__":
    unittest.main()
