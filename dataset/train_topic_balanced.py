"""SERVER: five mathematical topics x ten questions, fresh Base LoRA for 16 epochs."""
from __future__ import annotations

import argparse
from copy import deepcopy
from html import escape
import os
from pathlib import Path
import re
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recipe.adapters import validate_finetuning
from recipe.common import (append_jsonl, digest, file_hash, iter_jsonl, now, read_json,
                           recover_jsonl, snapshot_code, validate_prepared, write_json, write_jsonl)
from recipe.encoding import encode_example, inspect_example
from recipe.paths import artifact_path, configure_runtime
from recipe.pipeline import Pipeline
from recipe.replay import validate_checkpoint
from recipe.transfer_eval import load_panel
from dataset.likelihood_training import tree_hashes, verify_bundle
from dataset.train_likelihood_extremes import check_run, command_for
from dataset.topic_selection import DEFAULT_TOPICS, TOPICS, VERSION, select_rows

SIZE = 50
EPOCHS = list(range(17))
UPDATES = SIZE * EPOCHS[-1]
PROTOCOL = "five-topics-ten-each-lora-16-v1"
EXTRA_CODE = ("__init__.py", "train_topic_balanced.py", "topic_selection.py",
              "train_likelihood_extremes.py", "likelihood_training.py", "likelihood_utils.py",
              "TOPIC_BALANCED_TRAINING.md")


def training_config(original, seed):
    config = deepcopy(original)
    validate_finetuning(config["finetuning"])
    if config["finetuning"]["base_quantization"] != "none":
        raise ValueError("This experiment requires BF16 Base LoRA, without base quantization")
    config.update(seed=seed, expected_train_size=SIZE, epochs=16,
                  checkpoint_epochs=EPOCHS.copy(), recovery_every_steps=25, keep_recovery=2)
    return config


def review_html(rows, stats):
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
             '<title>五类各十题：训练选题</title><style>body{max-width:1100px;margin:2em auto;padding:1em;font:16px/1.6 sans-serif}'
             'pre{white-space:pre-wrap;overflow-wrap:anywhere}article{border-top:1px solid #bbb;padding:1em 0}'
             'td,th{padding:.4em 1em;text-align:left}details{margin:1em 0}</style>',
             '<h1>五类各十题：50 道训练题</h1><p>题型由题目关键词保守推断，不是官方标签。'
             '交叉命中或未识别的题目不选；请在训练前抽查。解答保留数据集全文，没有改写或删尾。</p>',
             '<p>同一组混合训练：16 轮 × 50 次更新；每轮保存；原始 Base 新建 LoRA。</p>',
             '<table><tr><th>题型</th><th>合格候选（选题间去重前）</th><th>入选</th></tr>']
    for topic in DEFAULT_TOPICS:
        parts.append(f'<tr><td>{TOPICS[topic]}</td><td>{stats["candidate_counts"][topic]}</td><td>10</td></tr>')
    parts.append('</table>')
    for topic in DEFAULT_TOPICS:
        parts.append(f'<h2>{TOPICS[topic]}</h2>')
        for row in rows:
            if row['topic_selection']['topic'] != topic:
                continue
            parts.extend([f'<article><h3>原始行号 {row["row_index"]}</h3>',
                          '<p>' + escape(f'来源：{row["source"]}；总 token：{row["n_tokens"]}；监督 token：{row["n_supervised"]}') + '</p>',
                          '<p>分类依据：' + escape(', '.join(row['topic_selection']['matches'])) + '</p>',
                          '<p>sample_id：' + escape(row['sample_id']) + '</p>',
                          '<h4>题目</h4><pre>' + escape(row['question']) + '</pre>',
                          '<h4>数据集短答案</h4><pre>' + escape(str(row['answer'])) + '</pre>',
                          '<details><summary>完整原解答（点击展开）</summary><pre>' + escape(row['response']) + '</pre></details></article>'])
    return '\n'.join(parts) + '</html>\n'


def prepare(out, request, check_stop=lambda: None):
    data, panel = Path(request['data']), Path(request['panel_dir'])
    for source in (data, panel):
        if out.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(out.resolve()):
            raise ValueError("Use a separate output directory, not inside an input or its parent")
    progress = out / 'progress.json'
    allowed = {'pipeline.lock'}
    if progress.exists():
        previous = read_json(progress)
        if previous.get('protocol') == PROTOCOL and previous.get('request') == request:
            allowed.add('progress.json')
    if any(p.name not in allowed and not p.name.startswith('.initializing-') for p in out.iterdir()):
        raise ValueError("Output is not an empty topic experiment; choose a new --out")
    original = validate_prepared(data)
    original_manifest_sha = file_hash(data / 'manifest.json')
    model = original['model']
    if (original['arguments']['max_length'] != 10240 or model['id'] != 'Qwen/Qwen3-4B-Base'
            or not re.fullmatch(r'[0-9a-f]{40}', model['revision'])):
        raise ValueError("Expected the pinned Qwen Base and original 10,240-token prepared pool")
    if 'pool.jsonl' not in original['files']:
        raise ValueError("--data must be the original prepare_data.py bundle with pool.jsonl")
    config = training_config(read_json(request['config']), request['seed'])
    transfer, transfer_manifest = load_panel(panel)
    references = [r for name in ('holdout.jsonl', 'panel.jsonl', 'overlap_references.jsonl')
                  for r in iter_jsonl(data / name)] + transfer
    print('Classifying eligible pool and checking holdout / four-benchmark overlap...', flush=True)
    rows, audit, stats = select_rows(iter_jsonl(data / 'pool.jsonl'), references, request['seed'], check_stop)
    check_stop()
    # No model loading or remote tokenizer lookup. Use the exact original template.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(data / 'tokenizer', local_files_only=True)
    inspections = []
    for row in rows:
        check_stop()
        encoded = encode_example(tokenizer, row, 10240)
        if (encoded is None or len(encoded['input_ids']) != row['n_tokens']
                or encoded['supervised_tokens'] != row['n_supervised']):
            raise ValueError(f"Prepared token length changed: {row['sample_id']}; do not truncate")
        inspections.append(inspect_example(tokenizer, row, encoded))
    stage = out / ('.initializing-' + uuid.uuid4().hex)
    prepared = stage / 'prepared/train'
    prepared.mkdir(parents=True)
    for name in ('holdout.jsonl', 'panel.jsonl', 'chat_template.jinja'):
        shutil.copy2(data / name, prepared / name)
        if file_hash(prepared / name) != original['files'][name]:
            raise ValueError(f"Source changed during copying: {name}")
    # Include all four benchmarks in the trainer's repeated overlap check.
    write_jsonl(prepared / 'overlap_references.jsonl',
                [{k: r[k] for k in ('sample_id', 'question_id', 'question')} for r in references])
    shutil.copytree(data / 'tokenizer', prepared / 'tokenizer')
    if {'tokenizer/' + k: v for k, v in tree_hashes(prepared / 'tokenizer').items()} != {
            k: v for k, v in original['files'].items() if k.startswith('tokenizer/')}:
        raise ValueError('Source tokenizer changed during copying')
    write_jsonl(prepared / 'train.jsonl', rows)
    write_jsonl(prepared / 'inspection.jsonl', inspections)
    write_jsonl(stage / 'classification.jsonl', audit)
    write_json(stage / 'source_manifest.json', original)
    write_json(stage / 'transfer_manifest.json', transfer_manifest)
    write_json(stage / 'config.json', config)
    selection = {'version': VERSION, 'official_labels': False, 'seed': request['seed'],
                 'topics': {k: TOPICS[k] for k in DEFAULT_TOPICS}, 'per_topic': 10, 'stats': stats,
                 'rule': 'Question text only; unique topic keyword match; uniform seeded sampling per topic; no likelihood ranking',
                 'overlap': 'Original holdout, full original MATH500/GPQA references, selected BBH/LCB panel; normalized exact + char7 Jaccard >= 0.9',
                 'source_pool_sha256': original['files']['pool.jsonl'],
                 'transfer_panel_sha256': transfer_manifest['files']['panel.jsonl'],
                 'selected': [{k: r[k] for k in ('sample_id', 'question_id', 'row_index', 'source', 'topic_selection', 'n_tokens', 'n_supervised')} for r in rows]}
    write_json(stage / 'selection.json', selection)
    (stage / 'review.html').write_text(review_html(rows, stats), encoding='utf-8')
    write_json(prepared / 'manifest.json', {'format_version': 1,
               'arguments': {**original['arguments'], 'train_size': SIZE}, 'model': model,
               'sources': original.get('sources', {}), 'parent_manifest_sha256': original_manifest_sha,
               'selection': selection, 'stats': {'train': SIZE}, 'files': tree_hashes(prepared)})
    snapshot_code(stage / 'code')
    (stage / 'code/dataset').mkdir()
    for name in EXTRA_CODE:
        shutil.copy2(ROOT / 'dataset' / name, stage / 'code/dataset' / name)
    plan = {'protocol': PROTOCOL, 'created_at': now(), 'request': request, 'model': model,
            'config': config, 'max_length': 10240, 'updates_per_group': UPDATES, 'selection': selection,
            'initialization': 'fresh pinned Base and newly initialized LoRA; no prior adapter',
            'files': tree_hashes(stage)}
    plan['identity'] = digest(plan)
    write_json(stage / 'plan.json', plan)
    # Detect inputs changed while selection/tokenization was running.
    validate_prepared(data)
    if file_hash(data / 'manifest.json') != original_manifest_sha:
        raise ValueError('Source manifest changed during preparation')
    if file_hash(panel / 'panel.jsonl') != selection['transfer_panel_sha256']:
        raise ValueError('Transfer panel changed during preparation')
    check_stop()
    stage.rename(out / 'bundle')
    return verify_bundle(out)


def training_complete(out, plan):
    run = out / 'train'
    if not (run / 'run.json').exists():
        return False
    metadata = check_run(out, 'train', plan)
    if not (run / 'checkpoints/epoch-016/complete.json').exists():
        return False
    for epoch in EPOCHS:
        validate_checkpoint(run / 'checkpoints' / f'epoch-{epoch:03d}', metadata, epoch * SIZE)
    return True


def execute(out, plan, pipeline, stop_after=None):
    recover_jsonl(out / 'commands.jsonl')
    pipeline.check_stop()
    if not training_complete(out, plan):
        command = command_for(out, 'train', plan)
        if stop_after is not None:
            command += ['--stop-after', str(stop_after)]
        append_jsonl(out / 'commands.jsonl', {'timestamp': now(), 'job_identity': plan['identity'],
                     'group': 'train', 'command': list(map(str, command))})
        write_json(out / 'progress.json', {'status': 'running', 'job_identity': plan['identity'], 'total_steps': UPDATES})
        pipeline.invoke(command, out / 'bundle/code')
        if not training_complete(out, plan):
            write_json(out / 'progress.json', {'status': 'paused', 'job_identity': plan['identity']})
            print('Paused; repeat the same command without --stop-after to resume', flush=True)
            return False
    result = {'job_identity': plan['identity'], 'updates': UPDATES, 'epochs': EPOCHS,
              'checkpoint_markers': {str(e): file_hash(out / 'train/checkpoints' / f'epoch-{e:03d}' / 'complete.json') for e in EPOCHS}}
    if (out / 'complete.json').exists() and read_json(out / 'complete.json') != result:
        raise ValueError('Previously completed checkpoint markers changed')
    write_json(out / 'complete.json', result)
    write_json(out / 'progress.json', {'status': 'complete', 'job_identity': plan['identity'], 'total_steps': UPDATES})
    print(f'Complete: {out / "train"}; all epochs 0..16 retained', flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='artifacts/data-s42')
    parser.add_argument('--panel-dir', default='artifacts/transfer-s2026', help='Existing BBH/LCB panel for overlap exclusion')
    parser.add_argument('--config', default=str(ROOT / 'configs/baseline.json'))
    parser.add_argument('--out', default='runs/topic-balanced-50x16-s42')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prepare-only', action='store_true', help='Freeze selection/config/code and reading report, no model loading')
    parser.add_argument('--stop-after', type=int, help='Pause at absolute update 1..800; omit on resume')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stop_after is not None and not 1 <= args.stop_after <= UPDATES:
        parser.error('--stop-after must be between 1 and 800')
    if os.name != 'posix':
        raise RuntimeError('Run this training experiment on the Linux VM')
    configure_runtime()
    out = Path(args.worker).resolve() if args.worker else artifact_path(args.out)
    with Pipeline(out, inherited=bool(args.worker)) as pipeline:
        try:
            if args.worker:
                execute(out, verify_bundle(out), pipeline, args.stop_after)
                return
            request = {'data': str(artifact_path(args.data)), 'panel_dir': str(artifact_path(args.panel_dir)),
                       'config': str(Path(args.config).resolve()), 'seed': args.seed}
            if (out / 'bundle').exists():
                plan = verify_bundle(out)
                if plan['protocol'] != PROTOCOL or plan['request'] != request:
                    raise ValueError('Different experiment/seed/inputs; use the original command or a new --out')
            else:
                plan = prepare(out, request, pipeline.check_stop)
            config = plan['config']
            print(f'5 topics x 10 questions x 16 epochs = {UPDATES} updates; checkpoints={EPOCHS}\n'
                  f'Fresh Base LoRA; lr={config["learning_rate"]}; warmup={config["warmup_steps"]} updates\n'
                  f'Review: {out / "bundle/review.html"}\nRun: {out / "train"}', flush=True)
            pipeline.check_stop()
            if args.prepare_only:
                return
            code = out / 'bundle/code'
            command = [sys.executable, '-B', '-u', code / 'dataset/train_topic_balanced.py', '--worker', out]
            if args.stop_after is not None:
                command += ['--stop-after', str(args.stop_after)]
            pipeline.invoke(command, code)
        except BaseException as exc:
            if not (out / 'complete.json').exists():
                write_json(out / 'progress.json', {'status': 'paused' if isinstance(exc, (SystemExit, KeyboardInterrupt)) else 'failed',
                           'error': str(exc), 'timestamp': now(), 'protocol': PROTOCOL,
                           'request': locals().get('request')})
            raise


if __name__ == '__main__':
    main()
