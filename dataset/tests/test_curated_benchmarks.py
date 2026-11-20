"""Epoch barriers, complete four-task reports and durable resume, without GPU/Docker."""
import csv
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests'))

import grade_transfer
import transfer_evaluation
from dataset import evaluate_curated as suite
from dataset import curated_benchmark_report as reports
from dataset.curated_training import training_config
from dataset.likelihood_training import tree_hashes
from recipe.common import digest, file_hash, iter_jsonl, read_json, write_json, write_jsonl
from recipe.data import adapt_math, adapt_gpqa, adapt_sft
from recipe.transfer_eval import load_panel
from test_replay import make_checkpoint
from test_transfer_evaluation import FakeJudge, panel_fixture
from test_intermediate_evaluation import generated


class CuratedBenchmarkTests(unittest.TestCase):
    def setUp(self):
        temp_root = ROOT / '.test_tmp'
        temp_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_root)
        self.root = Path(self.temp.name).resolve()
        assert self.root.is_relative_to(temp_root.resolve())
        self.addCleanup(self.temp.cleanup)
        # The production entry is Linux-only. Windows can briefly hold a newly
        # populated snapshot directory open; retry the REAL rename in fixtures.
        if os.name == 'nt':
            original_rename = Path.rename
            def rename(path, target):
                assert path.resolve().is_relative_to(self.root)
                assert Path(target).resolve().is_relative_to(self.root)
                for attempt in range(21):
                    try:
                        return original_rename(path, target)
                    except PermissionError:
                        if attempt == 20:
                            raise
                        time.sleep(.1)
            context = patch.object(Path, 'rename', rename)
            context.start()
            self.addCleanup(context.stop)
        self.run, self.panel, self.job = self.root / 'r', self.root / 't', self.root / 'j'
        self.job.mkdir()
        config = training_config(read_json(ROOT / 'configs/baseline.json'), 42)
        self.math = [adapt_math({'problem': f'Geometry challenge {i}: <b>triangle area</b>', 'answer': '2', 'solution': 'Two'}, i)
                     for i in range(128)]
        self.math += [adapt_gpqa({'Question': f'Quantum theory experiment {i}', 'Correct Answer': 'yes',
                     'Incorrect Answer 1': 'no', 'Incorrect Answer 2': 'maybe', 'Incorrect Answer 3': 'never'}, i, 42)
                      for i in range(64)]
        training = [adapt_sft({'question': f'Unique training arithmetic {i}', 'response': f'Compute {i}', 'answer': str(i)}, i)
                    for i in range(10)]
        write_jsonl(self.run / 'data/train.jsonl', training)
        write_jsonl(self.run / 'data/holdout.jsonl', [])
        write_jsonl(self.run / 'data/panel.jsonl', self.math)
        write_json(self.run / 'tokenizer/tokenizer.json', {'synthetic': True})
        metadata = {'config': config, 'model': {'id': 'Qwen/Qwen3-4B-Base', 'revision': 'a'*40}, 'max_length': 10240,
                    'config_sha256': digest(config), 'data_files': tree_hashes(self.run / 'data'),
                    'tokenizer_files': tree_hashes(self.run / 'tokenizer')}
        write_json(self.run / 'run.json', metadata)
        for epoch in config['checkpoint_epochs']:
            make_checkpoint(self.run / 'checkpoints' / f'epoch-{epoch:03d}', metadata, epoch)
        self.transfer = panel_fixture(self.panel, 'previous-training-hash')
        self.args = SimpleNamespace(epochs=None, image=None, judge_config=None)
        self.judge = FakeJudge()
        for context in (patch.dict(sys.modules, {'math_verify': ModuleType('math_verify')}),
                        patch.object(suite, 'DockerJudge', return_value=self.judge),
                        patch.object(transfer_evaluation, 'DockerJudge', return_value=self.judge),
                        patch('recipe.grading.math_equal', side_effect=lambda gold, answer: str(gold) == answer)):
            context.start()
            self.addCleanup(context.stop)

    def prepare(self, children=True):
        plan = suite.prepare(self.run, self.panel, self.job, self.args)
        if children:
            suite.prepare_panels(self.job, plan)
        return plan

    def test_default_all_nine_checkpoints_and_no_old_data_changes(self):
        before = tree_hashes(self.panel), tree_hashes(self.run)
        plan = self.prepare(children=False)
        self.assertEqual(plan['epochs'], [128,112,96,80,64,48,32,16,0])
        self.assertEqual(plan['responses'], 3528)
        self.assertEqual(plan['engine']['max_num_seqs'], 8)
        self.assertEqual(plan['generation']['max_new_tokens'], 8192)
        load_panel(self.job / 'suite/transfer-panel', file_hash(self.run / 'data/train.jsonl'))
        self.assertEqual(before, (tree_hashes(self.panel), tree_hashes(self.run)))
        self.assertEqual(self.prepare(children=False), plan)

    def test_each_child_has_only_one_epoch_and_same_prompts(self):
        self.args.epochs = '0,16'
        plan = self.prepare()
        for epoch in plan['epochs']:
            for name, expected, size in [('math-gpqa', self.math, 48), ('bbh-livecodebench', self.transfer, 50)]:
                child = self.job / 'epochs' / f'epoch-{epoch:03d}' / name
                info = read_json(child / 'setup/plan.json')
                self.assertEqual(info['generation_epochs'], [epoch])
                self.assertEqual(len(info['groups']), 4)
                self.assertTrue(all(len(g['selected_ids']) == size for g in info['groups']))
                panel, _ = load_panel(child / 'setup/panel')
                for left, right in zip(panel, expected):
                    for key in ('sample_id','question_id','messages','gold_answer','question'):
                        self.assertEqual(left[key], right[key])

    def test_continued_run_new_epochs_work_with_existing_four_task_suite(self):
        metadata = read_json(self.run / 'run.json')
        metadata['config'].update(epochs=256, checkpoint_epochs=list(range(144, 257, 16)))
        metadata['continuation'] = {'start_epoch': 128, 'start_step': 1280}
        metadata['config_sha256'] = digest(metadata['config'])
        write_json(self.run / 'run.json', metadata)
        for epoch in metadata['config']['checkpoint_epochs']:
            make_checkpoint(self.run / 'checkpoints' / f'epoch-{epoch:03d}', metadata, epoch)
        plan = self.prepare(children=False)
        self.assertEqual(plan['epochs'], [256, 240, 224, 208, 192, 176, 160, 144])
        self.assertEqual(plan['responses'], 8 * 392)
        # Exercise both original child panel protocols for a continued epoch.
        suite.prepare_panels(self.job, {**plan, 'epochs': [256]})
        for name, size in [('math-gpqa', 48), ('bbh-livecodebench', 50)]:
            child = self.job / 'epochs/epoch-256' / name
            info = read_json(child / 'setup/plan.json')
            self.assertEqual(info['generation_epochs'], [256])
            self.assertEqual(len(info['groups']), 4)
            self.assertTrue(all(len(g['selected_ids']) == size for g in info['groups']))

    def test_high_five_run_four_requested_checkpoints_and_fixed_panels(self):
        from dataset.train_curated_high5 import training_config as high_config
        source = self.run
        before = tree_hashes(source), tree_hashes(self.panel)
        self.run = self.root / 'high5'
        shutil.copytree(source / 'data', self.run / 'data')
        shutil.copytree(source / 'tokenizer', self.run / 'tokenizer')
        training = list(iter_jsonl(source / 'data/train.jsonl'))[:5]
        write_jsonl(self.run / 'data/train.jsonl', training)
        metadata = read_json(source / 'run.json')
        metadata['config'] = high_config(metadata['config'], 42)
        metadata['data_files'] = tree_hashes(self.run / 'data')
        metadata['config_sha256'] = digest(metadata['config'])
        write_json(self.run / 'run.json', metadata)
        for epoch in metadata['config']['checkpoint_epochs']:
            make_checkpoint(self.run / 'checkpoints' / f'epoch-{epoch:03d}', metadata, epoch)
        self.args.epochs = '16,32,48,64'
        plan = self.prepare()
        self.assertEqual(plan['epochs'], [64,48,32,16])
        self.assertEqual(plan['responses'], 1568)
        self.assertNotIn('0', plan['checkpoints'])
        self.assertEqual(plan['engine']['max_num_seqs'], 8)
        self.assertEqual(plan['generation']['max_new_tokens'], 8192)
        for epoch in plan['epochs']:
            self.assertEqual(read_json(self.run / 'checkpoints' / f'epoch-{epoch:03d}/complete.json')['step'], epoch * 5)
            for name, expected, size in [('math-gpqa', self.math, 48), ('bbh-livecodebench', self.transfer, 50)]:
                child = self.job / 'epochs' / f'epoch-{epoch:03d}' / name
                child_plan = read_json(child / 'setup/plan.json')
                self.assertEqual(child_plan['generation_epochs'], [epoch])
                self.assertEqual(len(child_plan['groups']), 4)
                self.assertTrue(all(len(g['selected_ids']) == size for g in child_plan['groups']))
                questions, _ = load_panel(child / 'setup/panel')
                for actual, original in zip(questions, expected):
                    for key in ('sample_id', 'question_id', 'messages', 'gold_answer'):
                        self.assertEqual(actual[key], original[key])
        self.assertEqual(before, (tree_hashes(source), tree_hashes(self.panel)))
        self.assertEqual(suite.prepare(self.run, self.panel, self.job, self.args), plan)

    def test_missing_checkpoint_and_changed_resume_arguments_fail(self):
        (self.run / 'checkpoints/epoch-112/complete.json').unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepare(children=False)
        self.assertFalse((self.job / 'suite').exists())
        self.args.epochs = '128'
        self.prepare(children=False)
        self.args.epochs = '0,128'
        with self.assertRaisesRegex(ValueError, 'inputs differ'):
            self.prepare(children=False)

    def test_overlap_rejected_before_execution(self):
        original = suite.overlap_checks
        def contaminated(training, math, transfer):
            return original({'train': [transfer[0]]}, math, transfer)
        with patch.object(suite, 'overlap_checks', side_effect=contaminated), \
                patch.object(suite, 'DockerJudge', side_effect=AssertionError('must not start')):
            with self.assertRaisesRegex(ValueError, 'overlap'):
                self.prepare(children=False)
        self.assertTrue(list(self.job.glob('overlap-rejections-*.json')))

    def test_epoch_barrier_incremental_reports_resume_and_four_tables(self):
        self.args.epochs = '0,16'
        plan = self.prepare()
        calls, failed = [], [False]

        def child_invoke(command, cwd):
            script = Path(command[3]).name
            out = Path(command[command.index('--out') + 1])
            child = out.parents[2]
            child_plan = read_json(child / 'setup/plan.json')
            epoch = child_plan['generation_epochs'][0]
            group = next(g for g in child_plan['groups'] if g['name'] == out.parent.name)
            if epoch == 0:
                self.assertTrue((self.job / 'epochs/epoch-016/report/report.html').exists())
                self.assertTrue((self.job / 'epochs/epoch-016/complete.json').exists())
                self.assertEqual(read_json(self.job / 'report/summary.json')['completed_epochs'], [16])
            calls.append((epoch, child.name, group['name'], script))
            if script == 'generate_vllm.py':
                interrupt = epoch == 0 and child.name == 'bbh-livecodebench' and group['name'] == 'group-002' and not failed[0]
                if interrupt:
                    failed[0] = True
                generated(child, group, [epoch], interrupt)
            else:
                grade_transfer.grade_directory(out.parent / 'generation', child / 'setup/panel', out,
                                                None if child.name == 'math-gpqa' else self.judge)

        def invoke(command, cwd):
            transfer_evaluation.execute(Path(command[-1]), SimpleNamespace(invoke=child_invoke, check_stop=lambda: None))

        worker = SimpleNamespace(invoke=invoke, check_stop=lambda: None)
        with patch('builtins.print'):
            with self.assertRaisesRegex(RuntimeError, 'interrupted generation'):
                suite.execute(self.job, worker)
        completed = tree_hashes(self.job / 'epochs/epoch-016')
        pending = self.job / 'epochs/epoch-000/bbh-livecodebench/groups/group-002/generation/epoch-000/responses.jsonl'
        prefix = pending.read_bytes()
        self.assertFalse((self.job / 'epochs/epoch-000/report/report.html').exists())
        partial = read_json(self.job / 'report/summary.json')
        self.assertEqual(partial['completed_epochs'], [16])
        self.assertFalse(partial['complete'])
        self.assertEqual(len(partial['curves']), 4)
        html = (self.job / 'report/report.html').read_text(encoding='utf-8')
        self.assertEqual(html.count('<table>'), 4)
        self.assertIn('未完成', html)
        with patch('builtins.print'):
            suite.execute(self.job, worker)
        self.assertEqual(tree_hashes(self.job / 'epochs/epoch-016'), completed)
        self.assertTrue(pending.read_bytes().startswith(prefix))
        self.assertEqual(self.judge.calls, 200)  # Every LCB answer graded only once.
        self.assertEqual(read_json(self.job / 'complete.json')['responses'], 784)
        summary = read_json(self.job / 'report/summary.json')
        self.assertTrue(summary['complete'])
        self.assertEqual(summary['completed_epochs'], [0,16])
        self.assertEqual(len(summary['curves']), 8)
        self.assertEqual({r['task']:r['n'] for r in summary['curves']}, reports.COUNTS)
        final_html = (self.job / 'report/report.html').read_text(encoding='utf-8')
        self.assertEqual(final_html.count('<table>'), 4)
        for task in reports.TASKS.values():
            self.assertIn(f'<h2>{task}</h2>', final_html)
        epoch_html = (self.job / 'epochs/epoch-016/report/report.html').read_text(encoding='utf-8')
        self.assertEqual(epoch_html.count('<table>'), 1)
        self.assertEqual(epoch_html.count('<details id='), 392)
        self.assertIn('&lt;b&gt;triangle area&lt;/b&gt;', epoch_html)
        self.assertNotIn('<b>triangle area</b>', epoch_html)
        with (self.job / 'report/question_milestones.csv').open(encoding='utf-8', newline='') as stream:
            first = next(r for r in csv.DictReader(stream) if r['task'] == 'math500')
        self.assertEqual(first['first_observed_correct_epoch'], '0')
        self.assertEqual(first['regression_intervals'], '[(0, 16)]')
        count = len(calls)
        with patch('builtins.print'):
            suite.execute(self.job, worker)
            suite.execute(self.job, worker, report_only=True)
        self.assertEqual(len(calls), count)
        file = self.job / 'epochs/epoch-016/report/samples.jsonl'
        file.write_bytes(file.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            suite.completed_epochs(self.job, plan)

    def test_incomplete_or_duplicate_panel_not_accepted_by_report(self):
        self.args.epochs = '16'
        self.prepare()
        directory = self.job / 'epochs/epoch-016'
        with self.assertRaises(FileNotFoundError):
            reports.collect_epoch(directory, 16)
        for name in reports.PANELS:
            child = directory / name
            panel, _ = load_panel(child / 'setup/panel')
            write_json(child / 'complete.json', {'epochs': [16]})
            rows = [{'sample_id': panel[0]['sample_id']}] * len(panel)
            file = child / 'grades/epoch-016/graded.jsonl'
            write_jsonl(file, rows)
            write_json(file.parent / 'complete.json', {'samples':len(rows),'graded_sha256':file_hash(file)})
        with self.assertRaisesRegex(ValueError, 'nonduplicate'):
            reports.collect_epoch(directory, 16)

    def test_frozen_launcher_code_integrity(self):
        self.prepare(children=False)
        (self.job / 'suite/code/dataset/evaluate_curated.py').write_text('modified', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'input/code changed'):
            suite.verify(self.job)

    def test_empty_overview_has_four_tables_and_no_invented_scores(self):
        plan = self.prepare(children=False)
        summary = reports.overview(self.job, plan, [])
        self.assertEqual(summary['curves'], [])
        self.assertFalse(summary['complete'])
        text = (self.job / 'report/report.html').read_text(encoding='utf-8')
        self.assertEqual(text.count('<table>'), 4)
        self.assertEqual(text.count('未完成，未计入统计'), 36)
        self.assertNotIn('href="curves.csv"', text)


if __name__ == '__main__':
    unittest.main()
