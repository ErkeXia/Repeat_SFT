"""Offline adapter routing, fixed-answer scores, reports and interruption recovery."""
import asyncio
import csv
import os
from pathlib import Path
import shutil
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np

from dataset import score_curated as base
from dataset import score_curated_checkpoints as scorer
from dataset.curated_likelihood_report import build_comparison
from dataset.likelihood_training import tree_hashes
from dataset.tests import test_curated_training as fixtures
from dataset.tests.test_likelihood import output
from recipe.common import ROOT, digest, file_hash, iter_jsonl, read_json, write_json
from recipe.replay import IDENTITY_KEYS

sys.path.insert(0, str(ROOT / "tests"))
from test_replay import make_checkpoint


class LoRA:
    def __init__(self, name, epoch, path):
        self.name, self.epoch, self.path = name, epoch, path


class Engine:
    def __init__(self, fail_epoch=None):
        self.calls, self.aborted, self.fail_epoch = [], [], fail_epoch

    async def generate(self, prompt, sampling, request_id, lora_request=None):
        epoch = lora_request.epoch if lora_request else 0
        self.calls.append((request_id, epoch))
        await asyncio.sleep(0)
        if epoch == self.fail_epoch:
            raise RuntimeError("simulated interruption")
        yield output({"input_ids": prompt["prompt_token_ids"]}, logprob={0: -.5, 64: -.25, 128: -.125}[epoch])

    async def abort(self, request_id):
        self.aborted.append(request_id)


class CheckpointLikelihoodTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CuratedTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.experiment = self.fixture.root, self.fixture.out
        if os.name == "nt":
            original_rename = Path.rename
            def retry(path, target):
                assert path.resolve().is_relative_to(self.root) and Path(target).resolve().is_relative_to(self.root)
                for attempt in range(21):
                    try:
                        return original_rename(path, target)
                    except PermissionError:
                        if attempt == 20:
                            raise
                        time.sleep(.1)
            context = patch.object(Path, "rename", retry)
            context.start()
            self.addCleanup(context.stop)
        self.original = self.fixture.prepare()
        self.rows, self.index, self.tokenizer = self.fixture.rows, self.fixture.index, self.fixture.tokenizer
        self.packages = {name: '0.11.0' if name == 'vllm' else 'synthetic'
                         for name in ('vllm', 'torch', 'transformers', 'numpy')}
        with patch.object(base.importlib.metadata, 'version', side_effect=lambda name: self.packages.get(name, 'synthetic')):
            job = base.initialize(self.experiment / "base_likelihood", self.original)
        asyncio.run(base.score_pending(Engine(), None, self.experiment / "base_likelihood", job,
                                      self.rows, self.tokenizer, self.index, {}))
        base.build_report(self.experiment / "base_likelihood", job, self.rows, self.index)
        run = self.experiment / "train"
        shutil.copytree(self.experiment / "bundle/prepared/train", run / "data",
                        ignore=shutil.ignore_patterns("tokenizer"))
        shutil.copytree(self.experiment / "bundle/prepared/train/tokenizer", run / "tokenizer")
        meta = {"model": self.original["model"], "config": self.original["config"], "max_length": 10240,
                "data_files": tree_hashes(run / "data"), "tokenizer_files": tree_hashes(run / "tokenizer"),
                "semantics": {"batch_size": 1}}
        meta["config_sha256"] = digest({k: meta[k] for k in IDENTITY_KEYS})
        write_json(run / "run.json", meta)
        for epoch in (64, 128):
            make_checkpoint(run / "checkpoints" / f"epoch-{epoch:03d}", meta, epoch)
        self.before = tree_hashes(self.experiment)
        self.out = self.root / "scores"
        self.out.mkdir()
        context = patch.object(scorer, "versions", return_value=self.packages)
        context.start()
        self.addCleanup(context.stop)

    def prepare(self):
        return scorer.prepare(self.experiment, self.out, [64, 128])

    def test_real_frozen_rows_base_reuse_adapter_routing_and_report(self):
        plan = self.prepare()
        self.assertEqual(plan["epochs"], [64, 128])
        options = scorer.engine_options(plan, self.out)
        self.assertTrue(options["enable_lora"])
        self.assertFalse(options["enable_prefix_caching"])
        self.assertEqual(options["max_num_seqs"], 8)
        self.assertEqual(options["lora_dtype"], "bfloat16")
        self.assertEqual(options["max_model_len"], max(r['total_tokens'] for r in self.index) + 1)
        engine = Engine()
        asyncio.run(scorer.score_models(engine, None, LoRA, self.out, plan, self.tokenizer))
        self.assertEqual([e for _, e in engine.calls], [64] * 10 + [128] * 10)
        self.assertTrue(scorer.finish(self.out, plan))
        summary = read_json(self.out / "report/summary.json")
        self.assertEqual([s['mean_sample_nll'] for s in summary['models']], [.5, .25, .125])
        samples = list(iter_jsonl(self.out / "report/samples.jsonl"))
        self.assertEqual(len(samples), 30)
        self.assertEqual(len({s['sample_id'] for s in samples}), 10)
        for sample in samples:
            original = next(r for r in self.rows if r['sample_id'] == sample['sample_id'])
            self.assertEqual(sample['response'], original['response'])
            with np.load(self.out / sample['token_file']) as tokens:
                self.assertEqual(len(tokens['logprobs']), sample['n_tokens'])
                self.assertEqual(int(tokens['input_ids'][int(tokens['assistant_end'])]), 2)
        text = (self.out / 'report/report.html').read_text(encoding='utf-8')
        self.assertIn('Epoch 64', text)
        self.assertIn('Epoch 128', text)
        self.assertIn('&lt;think&gt;', text)
        self.assertNotIn('<think>', text)
        with (self.out / 'report/samples.csv').open(encoding='utf-8') as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 30)
        self.assertEqual(tree_hashes(self.experiment), self.before)

    def test_interruption_keeps_finished_epoch_and_skips_committed_rows(self):
        plan = self.prepare()
        failed = Engine(fail_epoch=128)
        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            asyncio.run(scorer.score_models(failed, None, LoRA, self.out, plan, self.tokenizer))
        self.assertTrue(failed.aborted)
        early = tree_hashes(self.out / "epochs/epoch-064")
        self.assertFalse(scorer.finish(self.out, plan))
        self.assertFalse((self.out / 'complete.json').exists())
        self.assertEqual(read_json(self.out / 'report/summary.json')['models'][-1]['status'], 'pending')
        path, job = scorer.epoch_job(self.out, plan, 128)
        asyncio.run(base.score_pending(Engine(), None, path, job, self.rows[:3], self.tokenizer, self.index, {},
                                      lora_request=LoRA('epoch128',128,'synthetic'), label='TEST'))
        committed = tree_hashes(path / 'records')
        resumed = Engine()
        asyncio.run(scorer.score_models(resumed, None, LoRA, self.out, plan, self.tokenizer))
        self.assertEqual(len(resumed.calls), 7)
        self.assertTrue(all(epoch == 128 for _, epoch in resumed.calls))
        self.assertEqual(tree_hashes(self.out / "epochs/epoch-064"), early)
        for name, expected in committed.items():
            self.assertEqual(file_hash(path / 'records' / name), expected)
        self.assertTrue(scorer.finish(self.out, plan))
        skipped = Engine()
        asyncio.run(scorer.score_models(skipped, None, LoRA, self.out, plan, self.tokenizer))
        self.assertFalse(skipped.calls)
        self.assertEqual(tree_hashes(self.experiment), self.before)

    def test_missing_checkpoint_and_wrong_base_identity_rejected(self):
        file = self.experiment / 'train/checkpoints/epoch-128/complete.json'
        saved = file.read_bytes()
        file.unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepare()
        self.assertFalse((self.out / 'setup').exists())
        file.write_bytes(saved)
        job = read_json(self.experiment / 'base_likelihood/job.json')
        job['spec']['bundle_identity'] = 'wrong'
        job['identity'] = digest(job['spec'])
        write_json(self.experiment / 'base_likelihood/job.json', job)
        with self.assertRaisesRegex(ValueError, 'Base scores'):
            self.prepare()

    def test_source_checkpoint_tamper_and_changed_epochs_rejected(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, 'inputs differ'):
            scorer.prepare(self.experiment, self.out, [64])
        (self.experiment / 'train/checkpoints/epoch-064/adapter_model.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            scorer.verify(self.out)

    def test_committed_token_corruption_is_not_silently_rescored(self):
        plan = self.prepare()
        asyncio.run(scorer.score_models(Engine(), None, LoRA, self.out, plan, self.tokenizer))
        token = next((self.out / 'epochs/epoch-064/tokens').glob('*.npz'))
        token.write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'Damaged token'):
            build_comparison(self.out, plan)

    def test_version_difference_is_rejected_before_gpu_work(self):
        with patch.object(scorer, 'versions', return_value={**self.packages, 'torch': 'different'}):
            with self.assertRaisesRegex(ValueError, 'same scoring package versions'):
                self.prepare()
        self.assertFalse((self.out / 'setup').exists())


if __name__ == '__main__':
    unittest.main()
