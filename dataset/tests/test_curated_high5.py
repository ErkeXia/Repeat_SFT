"""Frozen five-answer selection, real launcher subprocesses and extension state."""
from copy import deepcopy
import os
from pathlib import Path
import shutil
import time
import unittest
from unittest.mock import patch

import torch

from dataset import train_curated_high5 as high
from dataset.tests import test_curated_training as fixtures
from dataset.likelihood_training import tree_hashes, verify_bundle
from recipe.common import digest, file_hash, iter_jsonl, read_json, validate_prepared, write_json
from recipe.extension import initialize, plan_extension, verify_extension
from recipe.pipeline import Pipeline


# Small synthetic artifacts with real torch serialization; no pretrained weights.
FAKE_TRAIN = fixtures.FAKE_TRAIN.replace(
    "import argparse, os, shutil", "import argparse, os, shutil, torch"
).replace(
    "from recipe.common import read_json, write_json, file_hash, digest",
    "from recipe.common import read_json, write_json, file_hash, digest, capture_rng, epoch_order, lr_for_update"
).replace(
    "    meta['config_sha256']=digest(meta)",
    "    meta['semantics']={'batch_size':1,'adapter_dtype':'float32'}\n"
    "    meta['config_sha256']=digest({k:meta[k] for k in ('config','model','data_files','tokenizer_files','semantics')})"
).replace(
    "    write_json(run/'run.json',meta)",
    "    write_json(run/'run.json',meta)\n"
    "    shutil.copytree(code,run/'attempts/original/code')"
).replace(
    "    (path/'training_state.pt').write_bytes(b'fake optimizer, rng, cursor')",
    "    cfg=meta['config'];size=cfg['expected_train_size']\n"
    "    torch.save({'step':step,'config_sha256':meta['config_sha256'],\n"
    "        'optimizer':{'state':{'test':torch.tensor([step])},'param_groups':[{'lr':lr_for_update(step+1,cfg['learning_rate'],cfg['warmup_steps'])}]},\n"
    "        'scheduler':{'last_epoch':step},'rng':capture_rng(),\n"
    "        'cursor':{'epoch_index':step//size,'position':step%size,'order':epoch_order(size,cfg['seed'],step//size)}},path/'training_state.pt')"
)


class HighFiveTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CuratedTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.source = self.fixture.root, self.fixture.out
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
        self.parent = self.fixture.prepare()
        self.before = tree_hashes(self.source)
        self.out = self.root / 'high5'
        self.out.mkdir()
        self.request = {'source_experiment': str(self.source), 'seed': 42}

    def prepare(self):
        self.plan = high.prepare(self.out, self.request)
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

    def test_five_exact_original_high_texts_and_64_epoch_protocol(self):
        original = deepcopy(self.parent['config'])
        plan = self.prepare()
        prepared = self.out / 'bundle/prepared/train'
        selected = list(iter_jsonl(prepared / 'train.jsonl'))
        expected = [r for r in self.fixture.rows if r['curation']['historical_group'] == 'high']
        self.assertEqual(selected, expected)
        self.assertEqual([r['row_index'] for r in selected], [6430, 2988, 4561, 14120, 1432])
        self.assertEqual(plan['config']['checkpoint_epochs'], [0,16,32,48,64])
        self.assertEqual(plan['updates_per_group'], 320)
        self.assertEqual(plan['config']['learning_rate'], 1e-5)
        self.assertEqual(plan['config']['warmup_steps'], 100)
        self.assertEqual(plan['config']['recovery_every_steps'], 5)
        self.assertEqual(self.parent['config'], original)
        manifest = validate_prepared(prepared)
        self.assertEqual(manifest['stats']['train'], 5)
        self.assertEqual(manifest['curation']['historical_groups'], {'high':5, 'low':0})
        inspections = list(iter_jsonl(prepared / 'inspection.jsonl'))
        self.assertEqual(len(inspections), 5)
        for inspection in inspections:
            self.assertTrue(inspection['supervised_text'].endswith('<|im_end|>'))
            self.assertIn('<think>', inspection['supervised_text'])
            self.assertNotIn('**Final Answer**', inspection['masked_prompt_text'])
        self.assertFalse((self.out / 'train').exists())
        self.assertEqual(tree_hashes(self.source), self.before)

    def test_actual_child_pause_resume_and_later_extension(self):
        self.prepare()
        self.fake_trainer()
        with Pipeline(self.out) as pipeline:
            self.assertFalse(high.execute(self.out, self.plan, pipeline, stop_after=7))
            first = list(iter_jsonl(self.out / 'commands.jsonl'))[-1]['command']
            self.assertIn('--data', first)
            self.assertNotIn('--resume', first)
            epoch0 = tree_hashes(self.out / 'train/checkpoints/epoch-000')
            state = torch.load(self.out / 'train/recovery/step-00000007/training_state.pt', weights_only=False)
            self.assertEqual(state['cursor']['epoch_index'], 1)
            self.assertEqual(state['cursor']['position'], 2)
            self.assertFalse((self.out / 'complete.json').exists())
            self.assertTrue(high.execute(self.out, self.plan, pipeline))
            last = list(iter_jsonl(self.out / 'commands.jsonl'))[-1]['command']
            self.assertTrue(last[-1].endswith('step-00000007'))
            self.assertEqual(tree_hashes(self.out / 'train/checkpoints/epoch-000'), epoch0)
            count = len(list(iter_jsonl(self.out / 'commands.jsonl')))
            self.assertTrue(high.execute(self.out, self.plan, pipeline))
            self.assertEqual(len(list(iter_jsonl(self.out / 'commands.jsonl'))), count)
        for epoch in high.EPOCHS:
            checkpoint = self.out / 'train/checkpoints' / f'epoch-{epoch:03d}'
            self.assertEqual(read_json(checkpoint / 'complete.json')['step'], epoch * 5)
            self.assertTrue((checkpoint / 'training_state.pt').exists())
        run_before = tree_hashes(self.out)
        extended = self.root / 'extended/train'
        extension = plan_extension(self.out / 'train', extended, additional_epochs=64, checkpoint_every=16)
        self.assertEqual(extension['target_epochs'], [80,96,112,128])
        self.assertEqual((extension['start_step'], extension['total_steps'], extension['additional_updates']), (320,640,320))
        initialize(extension)
        verify_extension(extended)
        copied = torch.load(extended / 'checkpoints/epoch-064/training_state.pt', weights_only=False)
        self.assertEqual(copied['step'], 320)
        self.assertEqual(copied['scheduler']['last_epoch'], 320)
        self.assertEqual(copied['optimizer']['param_groups'][0]['lr'], 1e-5)
        self.assertEqual(copied['cursor']['epoch_index'], 64)
        self.assertEqual(tree_hashes(self.out), run_before)
        self.assertEqual(tree_hashes(self.source), self.before)

    def test_changed_frozen_text_and_wrong_group_size_rejected(self):
        rows = deepcopy(self.fixture.rows)
        rows[0]['curation']['historical_group'] = 'low'
        with self.assertRaisesRegex(ValueError, 'five high'):
            high.selected_rows(rows)
        self.prepare()
        file = self.out / 'bundle/prepared/train/train.jsonl'
        file.write_bytes(file.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'changed'):
            verify_bundle(self.out)

    def test_source_not_required_to_resume_after_freezing(self):
        self.prepare()
        self.source.rename(self.root / 'moved-source')
        self.assertEqual(verify_bundle(self.out), self.plan)

    def test_nested_output_and_unknown_existing_files_rejected(self):
        nested = self.source / 'nested'
        nested.mkdir()
        with self.assertRaisesRegex(ValueError, 'separate'):
            high.prepare(nested, self.request)
        (self.out / 'unknown.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'empty high-five'):
            self.prepare()

    def test_interrupted_initialization_preserved_and_restarted_from_base(self):
        self.prepare()
        self.fake_trainer()
        with Pipeline(self.out) as pipeline:
            with patch.dict(os.environ, {'CURATED_TEST_FAIL_INIT':'1'}):
                with self.assertRaises(RuntimeError):
                    high.execute(self.out, self.plan, pipeline)
            self.assertTrue(high.execute(self.out, self.plan, pipeline))
        archived = list((self.out / 'incomplete_initializations').glob('train-*/partial.txt'))
        self.assertEqual(len(archived), 1)
        self.assertEqual(archived[0].read_text(), 'preserve this attempt')
        commands = list(iter_jsonl(self.out / 'commands.jsonl'))
        self.assertTrue(all('--data' in r['command'] and '--resume' not in r['command'] for r in commands))


if __name__ == '__main__':
    unittest.main()
