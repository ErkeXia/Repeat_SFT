"""Offline tests: synthetic logits/tokenizer/Parquet, no downloads or pretrained models."""
import asyncio
import copy
import math
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from recipe.common import ROOT, digest, file_hash, read_json, write_json
from recipe.data import adapt_sft
from dataset.likelihood_utils import (describe, encode_full, extract_logprobs, iter_rows,
                                      load_scores, metrics, save_score)
from dataset.report_likelihood import aggregate, build_report
from dataset.score_likelihood import prepare_index, score_pending


class Tokenizer:
    specials = {"<|im_start|>": 1, "<|im_end|>": 2, "<|endoftext|>": 3}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = ''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return self.encode(text) if tokenize else text

    def encode(self, text, add_special_tokens=False):
        parts = re.split('(' + '|'.join(re.escape(x) for x in self.specials) + ')', text)
        return [token for part in parts for token in
                ([self.specials[part]] if part in self.specials else [ord(c) + 10 for c in part])]

    def decode(self, ids, skip_special_tokens=False):
        reverse = {v: k for k, v in self.specials.items()}
        return ''.join(reverse[i] if i in reverse else chr(i - 10) for i in ids)

    def convert_tokens_to_ids(self, text):
        return self.specials[text]


def raw(response="<think>2+2=4.</think>\n\\boxed{4}"):
    return {"question": "2+2? <script>alert(1)</script>\r\n", "response": response,
            "answer": "4", "data_source": "synthetic", "response_length": len(response)}


def output(encoded, logprob=-0.5):
    ids = encoded["input_ids"]
    return SimpleNamespace(finished=True, prompt_token_ids=ids,
                           prompt_logprobs=[None] + [{token: SimpleNamespace(logprob=logprob)} for token in ids[1:]])


class LikelihoodTests(unittest.TestCase):
    def setUp(self):
        temp_root = ROOT / ".test_tmp"
        temp_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_root, prefix="likelihood-")
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.tokenizer = Tokenizer()

    def job(self, rows):
        pq.write_table(pa.Table.from_pylist(rows), self.out / "source.parquet")
        spec = {"selected_rows": len(rows), "context_length": 32768, "model": {"id": "synthetic", "revision": "test"},
                "dataset": {"id": "synthetic", "rows": len(rows)}, "request": {"concurrency": 1},
                "ranking": "mean_logprob descending; row_index tie"}
        job = {"spec": spec, "identity": digest(spec)}
        write_json(self.out / "job.json", job)
        index = prepare_index(self.out, job, self.tokenizer)
        return job, index

    def test_logprobs_match_causal_cross_entropy_and_mask(self):
        import torch
        import torch.nn.functional as F
        # Score ids[2], ids[3]. Prompt ids[0:2] and trailing newline ids[4] are excluded.
        ids = [3, 4, 1, 2, 0]
        logits = torch.tensor([[1., 2., 4., 0., -1.], [0., 3., 1., -1., 2.],
                               [2., 4., 1., 3., -2.], [8., 0., 0., 0., 0.], [1., 1., 1., 1., 1.]])
        full = F.log_softmax(logits, dim=-1)
        encoded = {"input_ids": ids, "prompt_length": 2, "assistant_end": 3}
        result = SimpleNamespace(finished=True, prompt_token_ids=ids,
                 prompt_logprobs=[None] + [{ids[i]: SimpleNamespace(logprob=full[i-1, ids[i]].item())}
                                          for i in range(1, len(ids))])
        values = extract_logprobs(result, encoded)
        reference = F.cross_entropy(logits[1:3], torch.tensor(ids[2:4])).item()
        self.assertAlmostEqual(metrics(values)["mean_nll"], reference, places=6)
        self.assertEqual(values[-1], full[2, 2].item())  # Assistant EOS is scored.
        self.assertEqual(len(values), 2)

    def test_full_reasoning_and_end_but_not_prompt_or_newline(self):
        row = adapt_sft(raw(), 0)
        encoded = encode_full(self.tokenizer, row, 32768)
        selected = self.tokenizer.decode(encoded["input_ids"][encoded["prompt_length"]:encoded["assistant_end"] + 1])
        self.assertEqual(selected, row["response"] + "<|im_end|>")
        self.assertEqual(encoded["labels"][-1], -100)
        self.assertTrue(all(x == -100 for x in encoded["labels"][:encoded["prompt_length"]]))

    def test_long_answers_not_filtered_or_truncated(self):
        row = adapt_sft(raw("x" * 12000), 0)
        encoded = encode_full(self.tokenizer, row, 32768)
        self.assertGreater(len(encoded["input_ids"]), 10240)
        self.assertEqual(encoded["supervised_tokens"], 12001)
        with self.assertRaisesRegex(ValueError, "no sample was truncated"):
            encode_full(self.tokenizer, row, len(encoded["input_ids"]))

    def test_missing_gold_invalid_values_and_wrong_alignment_rejected(self):
        encoded = encode_full(self.tokenizer, adapt_sft(raw(), 0), 32768)
        for bad in (float('nan'), float('inf'), 0.001):
            with self.assertRaises(ValueError):
                extract_logprobs(output(encoded, bad), encoded)
        result = output(encoded)
        result.prompt_logprobs[encoded["assistant_end"]] = {}
        with self.assertRaisesRegex(ValueError, "Missing reference"):
            extract_logprobs(result, encoded)
        result = output(encoded)
        result.prompt_token_ids = result.prompt_token_ids[:-1]
        with self.assertRaises(ValueError):
            extract_logprobs(result, encoded)

    def test_interrupted_payload_retried_committed_record_never_overwritten(self):
        job, index = self.job([raw()])
        row = next(iter_rows(self.out / "source.parquet"))
        encoded = encode_full(self.tokenizer, row, 32768)
        values = extract_logprobs(output(encoded), encoded)
        with patch("dataset.likelihood_utils.write_json", side_effect=OSError("simulated interrupted commit")):
            with self.assertRaises(OSError):
                save_score(self.out, job["identity"], index[0], encoded, values)
        self.assertEqual(load_scores(self.out, job, index), {})
        save_score(self.out, job["identity"], index[0], encoded, values)
        self.assertEqual(len(load_scores(self.out, job, index)), 1)
        with self.assertRaises(FileExistsError):
            save_score(self.out, job["identity"], index[0], encoded, values)
        payload = self.out / "tokens/row-00000.npz"
        payload.write_bytes(payload.read_bytes() + b"corruption")
        with self.assertRaisesRegex(ValueError, "Damaged token"):
            load_scores(self.out, job, index)

    def test_inference_interruption_resume_duplicate_questions_kept(self):
        job, index = self.job([raw(), raw(), raw("different solution")])
        self.assertEqual(index[0]["sample_id"], index[1]["sample_id"])

        class Engine:
            def __init__(self, fail=False):
                self.fail, self.calls = fail, []
            async def generate(self, prompt, params, rid):
                self.calls.append(rid)
                await asyncio.sleep(0)
                if self.fail and rid == "row-00001":
                    raise RuntimeError("simulated GPU interruption")
                yield output({"input_ids": prompt["prompt_token_ids"]})
            async def abort(self, rid):
                pass

        engine = Engine(fail=True)
        with self.assertRaisesRegex(RuntimeError, "interruption"):
            asyncio.run(score_pending(engine, None, self.out, job, self.tokenizer, index, {}))
        done = load_scores(self.out, job, index)
        self.assertEqual(set(done), {0})
        saved_hash = file_hash(self.out / "records/row-00000.json")
        engine = Engine()
        asyncio.run(score_pending(engine, None, self.out, job, self.tokenizer, index, done))
        self.assertEqual(engine.calls, ["row-00001", "row-00002"])
        self.assertEqual(len(load_scores(self.out, job, index)), 3)
        self.assertEqual(file_hash(self.out / "records/row-00000.json"), saved_hash)

    def test_identity_and_preflight_corruption_rejected(self):
        job, index = self.job([raw()])
        row = next(iter_rows(self.out / "source.parquet"))
        encoded = encode_full(self.tokenizer, row, 32768)
        save_score(self.out, job["identity"], index[0], encoded, [-1.] * encoded["supervised_tokens"])
        with self.assertRaisesRegex(ValueError, "different experiment"):
            load_scores(self.out, {**job, "identity": "wrong"}, index)
        with (self.out / "index.jsonl").open("a") as stream:
            stream.write("{}\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            prepare_index(self.out, job, self.tokenizer)

    def test_report_205_rows_exact_ranking_averages_full_text(self):
        job, index = self.job([raw(f"<think>line {i}\r\n</think>\\boxed{{4}}") for i in range(205)])
        for row, desc in zip(iter_rows(self.out / "source.parquet"), index):
            encoded = encode_full(self.tokenizer, row, 32768)
            save_score(self.out, job["identity"], desc, encoded, [-0.01 * (row["row_index"] + 1)] * desc["n_tokens"])
        with patch("dataset.report_likelihood.DATA_SHA256", file_hash(self.out / "source.parquet")), \
                patch("dataset.report_likelihood.DATA_ROWS", 205):
            summary = build_report(self.out)
        self.assertTrue(summary["complete_corpus"])
        self.assertEqual(summary["highest"]["row_index"], 0)
        self.assertEqual(summary["lowest"]["row_index"], 204)
        expected = sum(math.exp(-0.01 * i) for i in range(1, 206)) / 205
        self.assertAlmostEqual(summary["all"]["mean_normalized_likelihood"], expected, places=12)
        self.assertAlmostEqual(summary["top100"]["mean_normalized_likelihood"],
                               sum(math.exp(-0.01 * i) for i in range(1, 101)) / 100, places=12)
        self.assertAlmostEqual(summary["bottom100"]["mean_normalized_likelihood"],
                               sum(math.exp(-0.01 * i) for i in range(106, 206)) / 100, places=12)
        from recipe.common import iter_jsonl
        top = list(iter_jsonl(self.out / "report/top100.jsonl"))
        bottom = list(iter_jsonl(self.out / "report/bottom100.jsonl"))
        self.assertEqual([x["row_index"] for x in top], list(range(100)))
        self.assertEqual([x["row_index"] for x in bottom], list(range(105, 205)))
        self.assertEqual(top[0]["response"], raw("<think>line 0\r\n</think>\\boxed{4}")["response"])
        with (self.out / "report/report.html").open(encoding="utf-8", newline="") as stream:
            html = stream.read()
        self.assertNotIn("<script>alert", html)
        self.assertIn("&lt;script&gt;alert", html)
        self.assertIn("&lt;think&gt;line 0\r\n&lt;/think&gt;", html)
        self.assertEqual(html.count("<details "), 200)

    def test_partial_report_explicit_and_normalized_ranking_not_length_bias(self):
        job, index = self.job([raw("a"), raw("long response " * 50), raw("pending")])
        for row, desc, lp in zip(iter_rows(self.out / "source.parquet"), index, [-1., -0.5]):
            encoded = encode_full(self.tokenizer, row, 32768)
            save_score(self.out, job["identity"], desc, encoded, [lp] * desc["n_tokens"])
        with patch("dataset.report_likelihood.DATA_SHA256", file_hash(self.out / "source.parquet")):
            with self.assertRaisesRegex(ValueError, "Only 2/3"):
                build_report(self.out)
            summary = build_report(self.out, allow_partial=True)
        self.assertFalse(summary["complete_corpus"])
        self.assertFalse(summary["complete_selected"])
        self.assertEqual(summary["highest"]["row_index"], 1)  # Longer, but easier per token.
        self.assertEqual(read_json(self.out / "report/missing_rows.json"), [2])

    def test_sequence_likelihood_mean_stays_in_log_space(self):
        rows = [{**metrics([-100.] * 10000), "n_tokens": 10000} for _ in range(2)]
        self.assertAlmostEqual(aggregate(rows)["mean_sequence_likelihood_log10"], -1000000 / math.log(10))


if __name__ == "__main__":
    unittest.main()
