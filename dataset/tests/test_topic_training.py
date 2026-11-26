"""Offline selection, frozen inputs and subprocess stop/resume; no downloads."""
from collections import Counter
from copy import deepcopy
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from dataset import train_topic_balanced as job
from dataset.topic_selection import DEFAULT_TOPICS, classify, select_rows
from dataset.tests.test_likelihood import Tokenizer
from dataset.tests.test_curated_high5 import FAKE_TRAIN
from dataset.likelihood_training import tree_hashes, verify_bundle
from recipe.common import ROOT, digest, file_hash, iter_jsonl, read_json, validate_prepared, write_json, write_jsonl
from recipe.data import adapt_sft
from recipe.encoding import encode_example
from recipe.extension import plan_extension
from recipe.pipeline import Pipeline


QUESTIONS = [
    'Solve the quadratic equation x squared minus 5x plus 6 equals zero.',
    'Find the inradius of an equilateral triangle with side length 6.',
    'Find the greatest common divisor of 32 and 48.',
    'Find the probability of obtaining two heads with two fair coins.',
    'Compute the definite integral of x from zero to one.',
]


def sample(topic, i, tokenizer, question=None):
    question = question or QUESTIONS[DEFAULT_TOPICS.index(topic)]
    raw = {'question': question + ' Instance ' + digest([topic, i]),
           'response': '<think>Complete original reasoning.</think>\nFinal answer: \\boxed{2}',
           'answer': '2', 'data_source': 'synthetic'}
    row = adapt_sft(raw, DEFAULT_TOPICS.index(topic) * 100 + i)
    encoded = encode_example(tokenizer, row)
    row.update(n_tokens=len(encoded['input_ids']), n_supervised=encoded['supervised_tokens'],
               verification={'correct': True})
    return row


class TopicTrainingTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / '.test_tmp'
        parent.mkdir(exist_ok=True)
        temp = tempfile.TemporaryDirectory(dir=parent, prefix='topics-')
        self.root = Path(temp.name).resolve()
        self.assertTrue(self.root.is_relative_to(parent.resolve()))
        self.addCleanup(temp.cleanup)
        self.tokenizer = Tokenizer()
        self.rows = [sample(t, i, self.tokenizer) for t in DEFAULT_TOPICS for i in range(12)]
        self.data, self.panel, self.out = (self.root / name for name in ('data', 'panel', 'experiment'))
        self.out.mkdir()
        (self.data / 'tokenizer').mkdir(parents=True)
        (self.data / 'tokenizer/test.json').write_text('{}', encoding='utf-8')
        (self.data / 'chat_template.jinja').write_text('test template', encoding='utf-8')
        for name in ('holdout.jsonl', 'panel.jsonl', 'overlap_references.jsonl'):
            write_jsonl(self.data / name, [])
        write_jsonl(self.data / 'pool.jsonl', self.rows)
        write_json(self.data / 'manifest.json', {'model': {'id': 'Qwen/Qwen3-4B-Base', 'revision': 'a' * 40},
                   'arguments': {'max_length': 10240}, 'files': tree_hashes(self.data)})
        (self.panel / 'references').mkdir(parents=True)
        write_json(self.panel / 'references/test.json', {'input': 'not a training question'})
        write_json(self.panel / 'plan.json', {'training_sha256': 'original random run'})
        # One overlapping transfer question must be excluded despite being eligible.
        row = self.rows[0]
        write_jsonl(self.panel / 'panel.jsonl', [{'sample_id': 'bbh-' + row['question_id'],
                    'question_id': row['question_id'], 'question': row['question'], 'task': 'bbh',
                    'subtask': 'boolean_expressions', 'reference_file': 'references/test.json',
                    'reference_sha256': file_hash(self.panel / 'references/test.json')}])
        write_json(self.panel / 'manifest.json', {'format_version': 'transfer-panel-v1', 'files': tree_hashes(self.panel)})
        self.request = {'data': str(self.data), 'panel_dir': str(self.panel),
                        'config': str(ROOT / 'configs/baseline.json'), 'seed': 42}
        self.before = tree_hashes(self.data), tree_hashes(self.panel)
        if os.name == 'nt':
            rename = Path.rename
            def retry(path, target):
                assert path.resolve().is_relative_to(self.root) and Path(target).resolve().is_relative_to(self.root)
                for attempt in range(21):
                    try:
                        return rename(path, target)
                    except PermissionError:
                        if attempt == 20:
                            raise
                        time.sleep(.1)
            context = patch.object(Path, 'rename', retry)
            context.start()
            self.addCleanup(context.stop)

    def prepare(self):
        loader = SimpleNamespace(from_pretrained=lambda path, **kw: self.tokenizer if kw == {'local_files_only': True} else None)
        with patch.dict(sys.modules, {'transformers': SimpleNamespace(AutoTokenizer=loader)}), patch('builtins.print'):
            self.plan = job.prepare(self.out, self.request)
        return self.plan

    def fake_trainer(self):
        code = self.out / 'bundle/code'
        (code / 'train.py').write_text(FAKE_TRAIN, encoding='utf-8')
        manifest = read_json(code / 'manifest.json')
        manifest['files']['train.py'] = file_hash(code / 'train.py')
        manifest['code_sha256'] = digest(manifest['files'])
        write_json(code / 'manifest.json', manifest)
        plan = read_json(self.out / 'bundle/plan.json')
        plan.pop('identity')
        plan['files'] = tree_hashes(self.out / 'bundle')
        plan['files'].pop('plan.json')
        plan['identity'] = digest(plan)
        write_json(self.out / 'bundle/plan.json', plan)
        self.plan = verify_bundle(self.out)

    def test_conservative_multitopic_and_unknown_cases(self):
        for topic, question in zip(DEFAULT_TOPICS, QUESTIONS):
            self.assertEqual(classify(question)['topic'], topic)
        self.assertEqual(classify('Count the integer solutions of a quadratic equation divisible by 3')['status'], 'ambiguous')
        self.assertEqual(classify('Two parabolas: write their distance as m/n for relatively prime m,n')['status'], 'ambiguous')
        self.assertEqual(classify('Compute 3+4')['status'], 'unclassified')
        self.assertEqual(classify('Find the next term of this sequence')['topic'], 'sequences')
        self.assertNotEqual(classify('Find integral values of a and b')['topic'], 'calculus')
        self.assertEqual(classify('Find the equation of the plane through A, B and C')['status'], 'ambiguous')

    def test_balanced_deterministic_selection_ignores_response_vocabulary(self):
        rows = deepcopy(self.rows)
        for row in rows:
            row['response'] += ' integral prime triangle probability equation'
            row['sample_id'] = digest([row['question_id'], row['response']])
        first, audit, stats = select_rows(rows, [], 42)
        second, _, _ = select_rows(list(reversed(rows)), [], 42)
        self.assertEqual(first, second)
        self.assertEqual(stats['selected_counts'], {t: 10 for t in DEFAULT_TOPICS})
        self.assertEqual(len({r['question_id'] for r in first}), 50)
        different, _, _ = select_rows(rows, [], 43)
        self.assertNotEqual({r['sample_id'] for r in first}, {r['sample_id'] for r in different})

    def test_exclusions_and_no_silent_quota_relaxation(self):
        rows = deepcopy(self.rows)
        rows[0]['n_tokens'] = 10241
        rows[1]['verification'] = {'correct': False}
        _, audit, _ = select_rows(rows, [], 42)
        self.assertEqual(Counter(r['disposition'] for r in audit)['ineligible_length_or_verification'], 2)
        with self.assertRaisesRegex(ValueError, 'need 10'):
            select_rows(rows, [rows[2]], 42)
        extra = sample('geometry', 99, self.tokenizer, 'A triangle ![figure](https://example.test/a.png)')
        _, audit, _ = select_rows(self.rows + [self.rows[0], extra], [], 42)
        self.assertEqual(audit[-2]['disposition'], 'duplicate')
        self.assertEqual(audit[-1]['disposition'], 'external_image')
        bad = deepcopy(self.rows)
        bad[0]['response'] += 'changed'
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            select_rows(bad, [], 42)

    def test_frozen_full_answers_all_benchmarks_overlap_and_supervision(self):
        self.prepare()
        rows = list(iter_jsonl(self.out / 'bundle/prepared/train/train.jsonl'))
        self.assertEqual(len(rows), 50)
        self.assertNotIn(self.rows[0]['sample_id'], {r['sample_id'] for r in rows})
        originals = {r['sample_id']: r for r in self.rows}
        for row in rows:
            self.assertEqual({k: v for k, v in row.items() if k != 'topic_selection'}, originals[row['sample_id']])
        manifest = validate_prepared(self.out / 'bundle/prepared/train')
        self.assertEqual(manifest['parent_manifest_sha256'], file_hash(self.data / 'manifest.json'))
        for inspection in iter_jsonl(self.out / 'bundle/prepared/train/inspection.jsonl'):
            self.assertTrue(inspection['supervised_text'].endswith('<|im_end|>'))
            self.assertIn('Complete original reasoning.', inspection['supervised_text'])
            self.assertNotIn('Complete original reasoning.', inspection['masked_prompt_text'])
        self.assertEqual(self.plan['config']['checkpoint_epochs'], list(range(17)))
        self.assertEqual(self.plan['updates_per_group'], 800)
        self.assertEqual(self.plan['config']['recovery_every_steps'], 25)
        self.assertEqual(self.plan['config']['warmup_steps'], 100)
        self.assertEqual(self.before, (tree_hashes(self.data), tree_hashes(self.panel)))
        self.assertFalse((self.out / 'train').exists())
        html = (self.out / 'bundle/review.html').read_text(encoding='utf-8')
        self.assertEqual(html.count('<article>'), 50)
        self.assertNotIn('<think>', html)

    def test_subprocess_pause_resume_retains_every_epoch_and_can_extend(self):
        self.prepare()
        self.fake_trainer()
        with Pipeline(self.out) as pipeline:
            self.assertFalse(job.execute(self.out, self.plan, pipeline, stop_after=77))
            checkpoint_before = tree_hashes(self.out / 'train/checkpoints/epoch-001')
            state = torch.load(self.out / 'train/recovery/step-00000077/training_state.pt', weights_only=False)
            self.assertEqual((state['cursor']['epoch_index'], state['cursor']['position']), (1, 27))
            self.assertTrue(job.execute(self.out, self.plan, pipeline))
            calls = list(iter_jsonl(self.out / 'commands.jsonl'))
            self.assertNotIn('--resume', calls[0]['command'])
            self.assertIn('--resume', calls[1]['command'])
            self.assertTrue(calls[1]['command'][-1].endswith('step-00000077'))
            self.assertEqual(checkpoint_before, tree_hashes(self.out / 'train/checkpoints/epoch-001'))
            self.assertTrue(job.execute(self.out, self.plan, pipeline))
            self.assertEqual(len(list(iter_jsonl(self.out / 'commands.jsonl'))), 2)
        for epoch in range(17):
            path = self.out / 'train/checkpoints' / f'epoch-{epoch:03d}'
            self.assertEqual(read_json(path / 'complete.json')['step'], epoch * 50)
            self.assertTrue((path / 'training_state.pt').exists())
        extension = plan_extension(self.out / 'train', self.root / 'extended/train', additional_epochs=16, checkpoint_every=1)
        self.assertEqual(extension['target_epochs'], list(range(17, 33)))
        self.assertEqual(extension['additional_updates'], 800)

    def test_frozen_inputs_detect_edits_and_resume_without_original_source(self):
        self.prepare()
        self.data.rename(self.root / 'moved-data')
        self.assertEqual(verify_bundle(self.out), self.plan)
        path = self.out / 'bundle/code/dataset/topic_selection.py'
        path.write_bytes(path.read_bytes() + b'\n# changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            verify_bundle(self.out)

    def test_config_preserved_and_quantization_rejected(self):
        config = read_json(ROOT / 'configs/baseline.json')
        original = deepcopy(config)
        changed = job.training_config(config, 17)
        self.assertEqual(config, original)
        self.assertEqual(changed['seed'], 17)
        self.assertEqual(changed['learning_rate'], 1e-5)
        config['finetuning']['base_quantization'] = 'nf4'
        with self.assertRaisesRegex(ValueError, 'without base quantization'):
            job.training_config(config, 42)

    def test_interrupted_prepare_owned_progress_can_retry(self):
        write_json(self.out / 'progress.json', {'protocol': job.PROTOCOL, 'request': self.request, 'status': 'failed'})
        self.prepare()
        self.assertEqual(verify_bundle(self.out), self.plan)


if __name__ == '__main__':
    unittest.main()
