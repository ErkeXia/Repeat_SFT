"""vLLM: finish all four benchmarks for one epoch, publish its report, then continue."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import sys
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recipe.adapters import adapter_spec
from recipe.common import checkpoint_path, file_hash, iter_jsonl, now, read_json, snapshot_code, write_json
from recipe.grouped import generation_epochs, partition_panel
from recipe.lcb_docker import DEFAULT_IMAGE, DockerJudge
from recipe.paths import artifact_path, configure_runtime
from recipe.pipeline import Pipeline
from recipe.replay import evaluation_epochs
from recipe.resume import verified_code, verify_run_inputs
from recipe.storage import GIB, require_free_space
from recipe.transfer_eval import load_panel, partition
from recipe.vllm_generation import validate_config
from dataset.likelihood_training import tree_hashes
from dataset.evaluate_likelihood_extremes import freeze_transfer, overlap_checks
from dataset.curated_benchmark_report import PANELS, epoch_report, overview
import transfer_evaluation

EXTRA_CODE = ('__init__.py', 'evaluate_curated.py', 'curated_benchmark_report.py',
              'evaluate_likelihood_extremes.py', 'likelihood_training.py', 'likelihood_utils.py')


def scheduled_epochs(value, order='descending'):
    """Validate with the shared parser, optionally retaining the requested order.

    Other evaluation entries keep their existing descending behavior. The order
    is part of this suite's frozen plan and cannot change on resume.
    """
    descending = generation_epochs(value)
    if order == 'descending':
        return descending
    if order == 'listed':
        return [int(part.strip()) for part in value.split(',')]
    raise ValueError('--epoch-order must be descending or listed')


def verify(job):
    suite = job / 'suite'
    if read_json(suite / 'complete.json')['plan_sha256'] != file_hash(suite / 'plan.json'):
        raise ValueError('Frozen four-benchmark plan changed')
    plan = read_json(suite / 'plan.json')
    verified_code(suite / 'code', plan['code_sha256'])
    for name, expected in plan['files'].items():
        path = (suite / name).resolve()
        if not path.is_relative_to(suite.resolve()) or file_hash(path) != expected:
            raise ValueError(f'Frozen evaluation input/code changed: {name}')
    run = Path(plan['run'])
    metadata = verify_run_inputs(run)
    if file_hash(run / 'run.json') != plan['run_sha256']:
        raise ValueError('Training run changed')
    for epoch, expected in plan['checkpoints'].items():
        checkpoint = checkpoint_path(run, int(epoch))
        if file_hash(checkpoint / 'complete.json') != expected:
            raise ValueError(f'Checkpoint changed: epoch {epoch}')
        adapter_spec(checkpoint, finetuning=metadata['config']['finetuning'], base_model=metadata['model'])
    return plan


def prepare(run, panel_source, job, args):
    order = getattr(args, 'epoch_order', 'descending')
    if (job / 'suite').exists():
        plan = verify(job)
        if (plan['run'] != str(run) or plan['panel_source'] != str(panel_source)
                or order != plan.get('epoch_order', 'descending')
                or args.epochs and scheduled_epochs(args.epochs, order) != plan['epochs']
                or args.image and args.image != plan['image_requested']
                or args.judge_config and read_json(args.judge_config) != read_json(job / 'suite/judge.json')):
            raise ValueError('Resume inputs differ; repeat the original command or use a new --job-name')
        return plan
    if any(p.name != 'pipeline.lock' and not p.name.startswith(('.suite-partial-', 'overlap-rejections-')) for p in job.iterdir()):
        raise ValueError('Nonempty output is not this evaluation; choose a new job name')
    metadata = verify_run_inputs(run)
    if metadata.get('replay') or metadata['config']['finetuning']['base_quantization'] != 'none':
        raise ValueError('This entry expects a standard BF16 LoRA run, not a replay or QLoRA run')
    epochs = scheduled_epochs(args.epochs or ','.join(map(str, evaluation_epochs(metadata))), order)
    if not set(epochs) <= set(evaluation_epochs(metadata)):
        raise ValueError('Requested epochs were not scheduled for this run')
    checkpoints = {}
    for epoch in epochs:
        checkpoint = checkpoint_path(run, epoch)
        marker = read_json(checkpoint / 'complete.json')
        if (marker['config_sha256'] != metadata['config_sha256']
                or marker['step'] != epoch * metadata['config']['expected_train_size']):
            raise ValueError('Checkpoint belongs to another training trajectory')
        adapter_spec(checkpoint, finetuning=metadata['config']['finetuning'], base_model=metadata['model'])
        checkpoints[str(epoch)] = file_hash(checkpoint / 'complete.json')
    math = list(iter_jsonl(run / 'data/panel.jsonl'))
    transfer, _ = load_panel(panel_source)
    partition_panel(math)
    partition(transfer)
    training = list(iter_jsonl(run / 'data/train.jsonl'))
    audit = overlap_checks({'train': training}, math, transfer)['train']
    audit['training_sha256'] = metadata['data_files']['train.jsonl']
    if audit['matches']:
        file = job / ('overlap-rejections-' + uuid.uuid4().hex + '.json')
        write_json(file, audit)
        raise ValueError(f'Training/evaluation overlap; no questions replaced. Inspect {file}')
    import math_verify  # Fail before GPU work if the existing grading dependency is unavailable.
    judge = DockerJudge(args.image or DEFAULT_IMAGE, read_json(args.judge_config or ROOT / 'configs/lcb_grading.json'))
    preflight = judge.preflight()
    generation = read_json(ROOT / 'configs/generation.json')
    engine = validate_config(read_json(ROOT / 'configs/vllm.json'))
    if (generation['max_new_tokens'] != 8192 or engine['max_num_seqs'] != 8
            or (generation['temperature'], generation['top_p'], generation['top_k']) != (0.6, 0.95, 20)):
        raise ValueError('Expected existing 8192-token, temperature .6/top-p .95/top-k 20, concurrency 8 protocol')
    selected_bytes = sum((panel_source / q['reference_file']).stat().st_size for q in transfer)
    require_free_space(job, selected_bytes * (2 * len(epochs) + 1) + 4 * GIB)
    stage = job / ('.suite-partial-' + uuid.uuid4().hex)
    stage.mkdir()
    code_hash = snapshot_code(stage / 'code')
    for name in EXTRA_CODE:
        target = stage / 'code/dataset' / name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT / 'dataset' / name, target)
    write_json(stage / 'judge.json', judge.config)
    write_json(stage / 'overlap_checks.json', audit)
    freeze_transfer(panel_source, stage / 'transfer-panel', metadata['data_files']['train.jsonl'], audit)
    plan = {'format_version': 'curated-epoch-first-four-benchmarks-v1', 'created_at': now(),
            'run': str(run), 'run_sha256': file_hash(run / 'run.json'), 'panel_source': str(panel_source),
            'source_panel_manifest_sha256': file_hash(panel_source / 'manifest.json'),
            'epochs': epochs, 'checkpoints': checkpoints, 'code_sha256': code_hash,
            'generation': generation, 'engine': engine, 'image_requested': args.image or DEFAULT_IMAGE,
            'image': judge.image, 'preflight': preflight, 'questions_per_epoch': 392,
            'responses': 392 * len(epochs), 'epoch_order': order,
            'execution_order': f'{order} epoch order; both panels/all groups; epoch report; next epoch',
            'files': tree_hashes(stage)}
    write_json(stage / 'plan.json', plan)
    write_json(stage / 'complete.json', {'plan_sha256': file_hash(stage / 'plan.json')})
    stage.rename(job / 'suite')
    return plan


def prepare_panels(job, plan, check_stop=lambda: None):
    """Single-epoch child jobs keep the original generation/grading/resume implementation."""
    run = Path(plan['run'])
    for epoch in plan['epochs']:
        for name in PANELS:
            check_stop()
            child = job / 'epochs' / f'epoch-{epoch:03d}' / name
            child.mkdir(parents=True, exist_ok=True)
            math = name == 'math-gpqa'
            fresh = not (child / 'setup').exists()
            args = SimpleNamespace(mode='math' if math else 'transfer', epochs=str(epoch),
                                   image=plan['image']['id'] if fresh and not math else None,
                                   judge_config=str(job / 'suite/judge.json') if not math else None)
            source = run / 'data' if math else job / 'suite/transfer-panel'
            child_plan = transfer_evaluation.prepare(run, source, child, args)
            if (child_plan['generation_epochs'] != [epoch] or child_plan['settings'] != plan['generation']
                    or read_json(child / 'setup/vllm.json') != plan['engine']
                    or not math and child_plan['image'] != plan['image']):
                raise ValueError('Child generation/checkpoint/judge protocol differs from the frozen suite')


def epoch_evidence(directory, epoch):
    paths = [directory / 'report' / name for name in ('report.html', 'curves.csv', 'samples.jsonl', 'summary.json', 'manifest.json')]
    for name in PANELS:
        child = directory / name
        child_plan = transfer_evaluation.verify(child)
        if child_plan['generation_epochs'] != [epoch] or read_json(child / 'complete.json')['epochs'] != [epoch]:
            raise ValueError('Completed epoch contains results for another checkpoint')
        paths.extend(child / name for name in ('complete.json', f'grades/epoch-{epoch:03d}/graded.jsonl',
                     f'grades/epoch-{epoch:03d}/complete.json', f'generation/epoch-{epoch:03d}/responses.jsonl',
                     'grades/grading_manifest.json'))
        for group in child_plan['groups']:
            group_dir = child / 'groups' / group['name']
            paths.append(group_dir / 'complete.json')
            paths.extend(group_dir / rel for rel in transfer_evaluation.group_evidence(group_dir, [epoch]))
    return {p.relative_to(directory).as_posix(): file_hash(p) for p in paths}


def completed_epochs(job, plan):
    completed = []
    for epoch in plan['epochs']:
        directory = job / 'epochs' / f'epoch-{epoch:03d}'
        marker = directory / 'complete.json'
        if marker.exists():
            saved = read_json(marker)
            if (saved['suite_plan_sha256'] != file_hash(job / 'suite/plan.json') or saved['epoch'] != epoch
                    or saved['files'] != epoch_evidence(directory, epoch)):
                raise ValueError(f'Completed epoch evidence changed: {epoch}')
            completed.append(epoch)
    # No later epoch can be declared complete while an earlier scheduled epoch is unfinished.
    if completed != plan['epochs'][:len(completed)]:
        raise ValueError('Completed epoch reports do not form the scheduled prefix')
    return completed


def execute(job, worker, report_only=False):
    plan = verify(job)
    if not report_only:
        prepare_panels(job, plan, worker.check_stop)
    completed = completed_epochs(job, plan)
    overview(job, plan, completed)
    if report_only:
        return
    for epoch in plan['epochs']:
        worker.check_stop()
        if epoch in completed:
            print(f'SKIP epoch={epoch}: all four benchmarks and report verified', flush=True)
            continue
        directory = job / 'epochs' / f'epoch-{epoch:03d}'
        for name in PANELS:
            worker.check_stop()
            child = directory / name
            write_json(job / 'progress.json', {'stage': 'evaluation', 'epoch': epoch, 'panel': name,
                       'completed_epochs': completed, 'requested_epochs': plan['epochs']})
            print(f'EPOCH {epoch}: {name}; finish all four groups before continuing', flush=True)
            code = child / 'setup/code'
            worker.invoke([sys.executable, '-B', '-u', code / 'transfer_evaluation.py', '--worker', child], code)
            if not (child / 'complete.json').exists():
                raise RuntimeError(f'Incomplete panel: epoch={epoch}, {name}')
        worker.check_stop()
        epoch_report(job, epoch, plan)
        write_json(directory / 'complete.json', {'epoch': epoch, 'completed_at': now(),
                   'suite_plan_sha256': file_hash(job / 'suite/plan.json'), 'files': epoch_evidence(directory, epoch)})
        completed.append(epoch)
        overview(job, plan, completed)
        print(f'EPOCH {epoch} SAVED: {directory / "report/report.html"}', flush=True)
    write_json(job / 'complete.json', {'suite_plan_sha256': file_hash(job / 'suite/plan.json'),
               'epochs': completed, 'responses': plan['responses'],
               'epoch_completions': {str(e): file_hash(job / 'epochs' / f'epoch-{e:03d}/complete.json') for e in completed}})
    write_json(job / 'progress.json', {'stage': 'complete', 'completed_epochs': completed})
    print(f'ALL EPOCHS DONE: {job / "report/report.html"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs/curated-10x128-s42/train')
    parser.add_argument('--panel-dir', default='artifacts/transfer-s2026')
    parser.add_argument('--job-name', default='four-benchmarks-epoch-first-v1')
    parser.add_argument('--epochs', help='Default: all configured checkpoints descending, including epoch 0')
    parser.add_argument('--epoch-order', choices=('descending', 'listed'), default='descending',
                        help='listed follows --epochs exactly; default retains descending order')
    parser.add_argument('--image')
    parser.add_argument('--judge-config')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare-only', action='store_true')
    modes.add_argument('--report-only', action='store_true', help='Rebuild overall report from committed epochs; no GPU/Docker')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if os.name != 'posix':
        raise RuntimeError('Run evaluation on the Linux VM; local tests use synthetic data')
    configure_runtime()
    if args.worker:
        job = Path(args.worker).resolve()
        with Pipeline(job, inherited=True) as worker:
            try:
                if args.prepare_only:
                    plan = verify(job)
                    prepare_panels(job, plan, worker.check_stop)
                    overview(job, plan, completed_epochs(job, plan))
                    print('Prepared all checkpoint/panel jobs; repeat without --prepare-only to evaluate', flush=True)
                else:
                    execute(job, worker, args.report_only)
            except BaseException as exc:
                progress = read_json(job / 'progress.json') if (job / 'progress.json').exists() else {}
                progress.update(stage='paused' if isinstance(exc, (SystemExit, KeyboardInterrupt)) else 'failed', error=str(exc))
                write_json(job / 'progress.json', progress)
                raise
        return
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.job_name):
        parser.error('Invalid job name')
    run, panel = artifact_path(args.run), artifact_path(args.panel_dir)
    job = (run / 'postprocess' / args.job_name).resolve()
    if job.parent != (run / 'postprocess').resolve():
        parser.error('Job path escapes run/postprocess')
    with Pipeline(job) as worker:
        if args.report_only and not (job / 'suite').exists():
            raise ValueError('No existing evaluation to report')
        prepare(run, panel, job, args)
        code = job / 'suite/code'
        flags = ['--prepare-only'] if args.prepare_only else ['--report-only'] if args.report_only else []
        worker.invoke([sys.executable, '-B', '-u', code / 'dataset/evaluate_curated.py', '--worker', job, *flags], code)


if __name__ == '__main__':
    main()
