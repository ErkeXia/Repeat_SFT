"""Offline curation/scoring and real subprocess recovery tests; no models/downloads."""
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from recipe.common import ROOT, digest, file_hash, iter_jsonl, read_json, write_json, write_jsonl
from recipe.data import adapt_sft
from recipe.pipeline import Pipeline
from dataset import curated_training as data
from dataset import train_curated as launcher
from dataset import score_curated as scorer
from dataset.likelihood_training import tree_hashes
from dataset.likelihood_utils import load_scores
from dataset.tests.test_likelihood import Tokenizer, output


# This child only writes tiny protocol-compatible fixtures, never model weights.
FAKE_TRAIN = '''import argparse, os, shutil
from pathlib import Path
from recipe.common import read_json, write_json, file_hash, digest
p=argparse.ArgumentParser()
p.add_argument('--run');p.add_argument('--data');p.add_argument('--config');p.add_argument('--resume');p.add_argument('--stop-after',type=int)
a=p.parse_args();run=Path(a.run);code=Path(__file__).resolve().parent
if a.resume:
    assert Path(a.resume).is_relative_to(run)
    meta=read_json(run/'run.json')
else:
    run.mkdir()
    if os.environ.get('CURATED_TEST_FAIL_INIT'):
        (run/'partial.txt').write_text('preserve this attempt')
        raise SystemExit(17)
    prepared=Path(a.data)
    shutil.copytree(prepared/'tokenizer',run/'tokenizer')
    (run/'data').mkdir()
    for name in ('train.jsonl','holdout.jsonl','panel.jsonl','overlap_references.jsonl','manifest.json','chat_template.jinja'):
        shutil.copy2(prepared/name,run/'data'/name)
    meta={'config':read_json(a.config),'model':read_json(prepared/'manifest.json')['model'],
          'prepared_manifest_sha256':file_hash(prepared/'manifest.json'),'max_length':10240,
          'code_sha256':read_json(code/'manifest.json')['code_sha256'],
          'data_files':{x.name:file_hash(x) for x in (run/'data').iterdir()},
          'tokenizer_files':{x.name:file_hash(x) for x in (run/'tokenizer').iterdir()}}
    meta['config_sha256']=digest(meta)
    write_json(run/'run.json',meta)
def save(path,step):
    if path.exists():return
    path.mkdir(parents=True)
    ft=meta['config']['finetuning'];model=meta['model']
    (path/'adapter_model.safetensors').write_bytes(b'fake adapter')
    (path/'training_state.pt').write_bytes(b'fake optimizer, rng, cursor')
    write_json(path/'recipe_model.json',{'format_version':1,'compute_dtype':'bfloat16','finetuning':ft,'base_model':model})
    write_json(path/'adapter_config.json',{'peft_type':'LORA','task_type':'CAUSAL_LM','r':ft['rank'],
        'lora_alpha':ft['alpha'],'lora_dropout':ft['dropout'],'target_modules':ft['target_modules'],
        'bias':'none','base_model_name_or_path':model['id'],'revision':model['revision']})
    files={x.name:file_hash(x) for x in path.iterdir() if x.name!='training_state.pt'}
    write_json(path/'complete.json',{'step':step,'config_sha256':meta['config_sha256'],
        'has_training_state':True,'weights_format':'lora_adapter','model_files':files})
size=meta['config']['expected_train_size'];target=a.stop_after or size*meta['config']['epochs']
for e in meta['config']['checkpoint_epochs']:
    if e*size<=target:save(run/'checkpoints'/f'epoch-{e:03d}',e*size)
if a.stop_after:save(run/'recovery'/f'step-{target:08d}',target)
'''


class Engine:
    def __init__(self):
        self.calls = []

    async def generate(self, prompt, sampling, rid):
        self.calls.append(rid)
        await asyncio.sleep(0)
        yield output({"input_ids": prompt["prompt_token_ids"]})

    async def abort(self, rid):
        pass


class CuratedTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / ".test_tmp"
        parent.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent, prefix="curated-")
        self.root = Path(self.temp.name).resolve()
        assert self.root.is_relative_to(parent.resolve())
        self.addCleanup(self.temp.cleanup)
        self.out = self.root / "experiment"
        self.out.mkdir()
        self.tokenizer = Tokenizer()

    def prepare(self, overlap=False):
        prepared = self.root / "data"
        prepared.mkdir()
        (prepared / "tokenizer").mkdir()
        (prepared / "tokenizer/test.json").write_text('{}')
        (prepared / "chat_template.jinja").write_text('synthetic test template')
        for name in ("holdout.jsonl", "panel.jsonl", "overlap_references.jsonl"):
            write_jsonl(prepared / name, [])
        if overlap:
            write_jsonl(prepared / "holdout.jsonl", [data.load_curated(data.DEFAULT_CURATION)[0][0]])
        write_json(prepared / "manifest.json", {"model": {"id": "Qwen/Qwen3-4B-Base", "revision": "a" * 40},
                   "arguments": {"max_length": 10240}, "files": tree_hashes(prepared)})
        request = {"curation": str(data.DEFAULT_CURATION), "data": str(prepared),
                   "config": str(ROOT / "configs/baseline.json"), "seed": 42, "vllm_python": sys.executable}
        module = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: self.tokenizer))
        with patch.dict(sys.modules, {"transformers": module}), patch('builtins.print'):
            plan = data.prepare(self.out, request)
        self.rows = list(iter_jsonl(self.out / "bundle/prepared/train/train.jsonl"))
        self.index = list(iter_jsonl(self.out / "bundle/index.jsonl"))
        self.plan = plan
        return plan

    def job(self):
        spec = {"bundle_identity": self.plan["identity"], "model": self.plan["model"], "scoring": self.plan["scoring"]}
        job = {"identity": digest(spec), "spec": spec}
        write_json(self.out / "base_likelihood/job.json", job)
        return job

    def score(self, rows=None, done=None):
        job = self.job()
        engine = Engine()
        with patch('builtins.print'):
            asyncio.run(scorer.score_pending(engine, None, self.out / "base_likelihood", job,
                                            self.rows if rows is None else rows, self.tokenizer,
                                            self.index, done or {}, concurrency=8))
        return job, engine

    def finish_scores(self):
        job, _ = self.score()
        scorer.build_report(self.out / "base_likelihood", job, self.rows, self.index)
        self.assertTrue(launcher.scoring_complete(self.out, self.plan))

    def fake_trainer(self):
        code = self.out / "bundle/code"
        (code / "train.py").write_text(FAKE_TRAIN, encoding="utf-8")
        manifest = read_json(code / "manifest.json")
        manifest["files"]["train.py"] = file_hash(code / "train.py")
        manifest["code_sha256"] = digest(manifest["files"])
        write_json(code / "manifest.json", manifest)
        plan = read_json(self.out / "bundle/plan.json")
        plan.pop("identity")
        plan["files"] = tree_hashes(self.out / "bundle")
        plan["files"].pop("plan.json")
        plan["identity"] = digest(plan)
        write_json(self.out / "bundle/plan.json", plan)
        self.plan = data.verify_bundle(self.out)

    def test_real_ten_texts_fresh_ids_and_prompt_preservation(self):
        rows, selection = data.load_curated(data.DEFAULT_CURATION)
        originals = {r['row_index']: r['raw_record'] for r in iter_jsonl(data.DEFAULT_CURATION / 'originals.jsonl')}
        self.assertEqual(len(rows), 10)
        self.assertEqual(len({r['sample_id'] for r in rows}), 10)
        for row in rows:
            old = adapt_sft(originals[row['row_index']], row['row_index'])
            self.assertNotEqual(row['sample_id'], old['sample_id'])
            for key in ('question', 'question_id', 'answer', 'messages'):
                self.assertEqual(row[key], old[key])
            self.assertNotIn('response_length', row['original_metadata'])
            self.assertNotIn('initial_likelihood', row)

    def test_exact_update_schedule_and_no_original_config_mutation(self):
        original = read_json(ROOT / 'configs/baseline.json')
        copied = deepcopy(original)
        config = data.training_config(original, 17)
        self.assertEqual(original, copied)
        self.assertEqual(config['epochs'] * config['expected_train_size'], 1280)
        self.assertEqual(config['checkpoint_epochs'], [0,16,32,48,64,80,96,112,128])
        self.assertEqual(config['warmup_steps'], 100)
        self.assertEqual(config['recovery_every_steps'], 10)
        self.assertEqual(config['seed'], 17)
        original['finetuning']['base_quantization'] = 'nf4'
        with self.assertRaises(ValueError):
            data.training_config(original, 42)

    def test_response_structure_nested_boxed_and_reject_tail(self):
        data.validate_response('<think>reason</think>\n**Final Answer**\n\\boxed{\\frac{3}{4}}\n')
        for text in ('<think>r</think>\n**Final Answer**\n\\boxed{A} then more',
                     '<think>r\n**Final Answer**\n\\boxed{A}',
                     '<think>r</think>\n**Final Answer**\n\\boxed{A}\\boxed{B}'):
            with self.assertRaises(ValueError):
                data.validate_response(text)

    def test_prepare_and_frozen_provenance_tamper(self):
        plan = self.prepare()
        self.assertEqual(data.verify_bundle(self.out), plan)
        inspections = list(iter_jsonl(self.out / 'bundle/prepared/train/inspection.jsonl'))
        self.assertEqual(len(inspections), 10)
        for item in inspections:
            self.assertIn('<think>', item['supervised_text'])
            self.assertIn('**Final Answer**', item['supervised_text'])
            self.assertTrue(item['supervised_text'].endswith('<|im_end|>'))
            self.assertNotIn('**Final Answer**', item['masked_prompt_text'])
        path = self.out / 'bundle/prepared/train/train.jsonl'
        path.write_bytes(path.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'changed'):
            data.verify_bundle(self.out)

    def test_overlap_rejected_before_bundle_commit(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.prepare(overlap=True)
        self.assertFalse((self.out / 'bundle').exists())

    def test_overlength_rejected_without_truncation(self):
        with patch.object(data, 'encode_example', return_value=None):
            with self.assertRaisesRegex(ValueError, 'no truncation'):
                self.prepare()
        self.assertFalse((self.out / 'bundle').exists())

    def test_scoring_resume_retains_records_and_masks_prompt(self):
        self.prepare()
        job, first = self.score(rows=self.rows[:3])
        scores_dir = self.out / 'base_likelihood'
        saved = tree_hashes(scores_dir / 'records')
        done = load_scores(scores_dir, job, self.index)
        self.assertEqual(len(done), 3)
        self.assertFalse(launcher.scoring_complete(self.out, self.plan))
        job, second = self.score(done=done)
        self.assertEqual(len(second.calls), 7)
        self.assertFalse(set(first.calls) & set(second.calls))
        for name, value in saved.items():
            self.assertEqual(file_hash(scores_dir / 'records' / name), value)
        summary = scorer.build_report(scores_dir, job, self.rows, self.index)
        self.assertEqual(summary['all']['samples'], 10)
        self.assertAlmostEqual(summary['all']['mean_sample_nll'], .5)
        self.assertEqual(summary['historical_groups']['high']['samples'], 5)
        self.assertTrue(launcher.scoring_complete(self.out, self.plan))
        for desc in self.index:
            record = read_json(scores_dir / 'records' / f"row-{desc['row_index']:05d}.json")
            with np.load(scores_dir / record['token_file']) as tokens:
                self.assertEqual(len(tokens['logprobs']), desc['n_tokens'])
                self.assertEqual(int(tokens['input_ids'][desc['assistant_end']]), 2)
        self.assertEqual(len(list(iter_jsonl(scores_dir / 'samples.jsonl'))), 10)
        self.assertIn(self.rows[0]['question'].replace('&','&amp;').replace('<','&lt;').replace('>','&gt;'),
                      (scores_dir / 'report.html').read_text(encoding='utf-8'))
        payload = scores_dir / record['token_file']
        payload.write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            launcher.scoring_complete(self.out, self.plan)

    def test_real_child_pause_resume_complete_and_skip(self):
        self.prepare()
        self.fake_trainer()
        self.finish_scores()
        with Pipeline(self.out) as pipeline:
            self.assertFalse(launcher.execute(self.out, self.plan, pipeline, stop_after=20))
            commands = list(iter_jsonl(self.out / 'commands.jsonl'))
            self.assertIn('--data', commands[-1]['command'])
            self.assertNotIn('--resume', commands[-1]['command'])
            self.assertFalse((self.out / 'complete.json').exists())
            self.assertTrue(launcher.execute(self.out, self.plan, pipeline))
            commands = list(iter_jsonl(self.out / 'commands.jsonl'))
            self.assertIn('--resume', commands[-1]['command'])
            self.assertTrue(commands[-1]['command'][-1].endswith('step-00000020'))
            count = len(commands)
            self.assertTrue(launcher.execute(self.out, self.plan, pipeline))
            self.assertEqual(len(list(iter_jsonl(self.out / 'commands.jsonl'))), count)
        self.assertEqual(read_json(self.out / 'complete.json')['updates'], 1280)
        for e in data.EPOCHS:
            self.assertEqual(read_json(self.out / 'train/checkpoints' / f'epoch-{e:03d}/complete.json')['step'], e*10)
        (self.out / 'train/checkpoints/epoch-048/adapter_model.safetensors').write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            launcher.training_complete(self.out, self.plan)

    def test_score_then_train_order_and_score_only(self):
        self.prepare()
        self.fake_trainer()
        with Pipeline(self.out) as pipeline:
            real_invoke = pipeline.invoke

            def invoke(command, cwd):
                if any(str(x).endswith('score_curated.py') for x in command):
                    self.assertFalse((self.out / 'train').exists())
                    self.assertEqual(command[0], self.plan['request']['vllm_python'])
                    self.finish_scores()  # Synthetic logprobs stand in for the GPU worker.
                else:
                    self.assertTrue(launcher.scoring_complete(self.out, self.plan))
                    real_invoke(command, cwd)

            with patch.object(pipeline, 'invoke', side_effect=invoke):
                self.assertFalse(launcher.execute(self.out, self.plan, pipeline, score_only=True))
                self.assertFalse((self.out / 'train').exists())
                self.assertTrue(launcher.execute(self.out, self.plan, pipeline))
        commands = list(iter_jsonl(self.out / 'commands.jsonl'))
        self.assertEqual([r['stage'] for r in commands], ['base_likelihood', 'training'])

    def test_uninitialized_attempt_archived_not_deleted(self):
        self.prepare()
        self.fake_trainer()
        self.finish_scores()
        with Pipeline(self.out) as pipeline:
            with patch.dict(os.environ, {'CURATED_TEST_FAIL_INIT': '1'}):
                with self.assertRaises(RuntimeError):
                    launcher.execute(self.out, self.plan, pipeline)
            self.assertTrue(launcher.execute(self.out, self.plan, pipeline))
        saved = list((self.out / 'incomplete_initializations').glob('train-*/partial.txt'))
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0].read_text(), 'preserve this attempt')

    def test_training_never_starts_when_scoring_fails(self):
        self.prepare()
        fake = SimpleNamespace(check_stop=lambda: None, invoke=lambda *a: (_ for _ in ()).throw(RuntimeError('score failed')))
        with self.assertRaisesRegex(RuntimeError, 'score failed'):
            launcher.execute(self.out, self.plan, fake)
        self.assertFalse((self.out / 'train').exists())
        self.assertEqual(read_json(self.out / 'progress.json')['stage'], 'base_likelihood')


if __name__ == '__main__':
    unittest.main()
