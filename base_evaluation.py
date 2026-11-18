"""SERVER: unadapted BF16 Base, four fixed benchmarks, eight resumable groups."""
from __future__ import annotations

import argparse
import csv
from html import escape
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import uuid

from recipe.adapters import adapter_spec
from recipe.common import ROOT, checkpoint_path, file_hash, iter_jsonl, now, read_json, snapshot_code, write_json
from recipe.grouped import partition_panel
from recipe.lcb_docker import DEFAULT_IMAGE
from recipe.paths import artifact_path, configure_runtime
from recipe.pipeline import Pipeline
from recipe.replay import evaluation_epochs
from recipe.resume import verified_code, verify_run_inputs
from recipe.transfer_eval import load_panel, partition
import transfer_evaluation

PANELS = (("math-gpqa", "math", "MATH500 / GPQA", 192, 48),
          ("bbh-livecodebench", "transfer", "BBH / LiveCodeBench", 200, 50))


def base_origin(run):
    metadata = verify_run_inputs(run)
    if metadata.get("replay") or 0 not in evaluation_epochs(metadata):
        raise ValueError("Use the original BF16 LoRA run with epoch 0, not the intermediate replay run")
    if metadata["config"]["finetuning"]["base_quantization"] != "none":
        raise ValueError("This baseline requires the original non-quantized BF16 Base")
    ckpt = checkpoint_path(run, 0)
    marker = read_json(ckpt / "complete.json")
    if marker["step"] != 0 or marker["config_sha256"] != metadata["config_sha256"]:
        raise ValueError("Epoch 0 does not match the original run")
    # Validate provenance only; the inference engine NEVER loads this adapter.
    adapter_spec(ckpt, finetuning=metadata["config"]["finetuning"], base_model=metadata["model"])
    return metadata, file_hash(ckpt / "complete.json")


def verify(job):
    suite = job / "suite"
    if read_json(suite / "complete.json")["plan_sha256"] != file_hash(suite / "plan.json"):
        raise ValueError("Frozen Base suite plan changed")
    plan = read_json(suite / "plan.json")
    if plan["epochs"] != [0] or plan["model_mode"] != "base_only_no_adapter":
        raise ValueError("Expected unadapted Base at epoch 0 only")
    run = Path(plan["run"])
    metadata, checkpoint_sha = base_origin(run)
    if (file_hash(run / "run.json") != plan["run_sha256"] or metadata["model"] != plan["model"]
            or checkpoint_sha != plan["checkpoint_sha256"]):
        raise ValueError("Original Base identity changed")
    verified_code(suite / "code", plan["code_sha256"])
    if file_hash(suite / "judge.json") != plan["judge_config_sha256"]:
        raise ValueError("Frozen judge settings changed")
    return plan


def prepare(run, panel_source, job, args):
    suite = job / "suite"
    if suite.exists():
        plan = verify(job)
        if (plan["run"] != str(run) or plan["panel_source"] != str(panel_source)
                or args.image and args.image != plan["image_requested"]
                or args.judge_config and read_json(args.judge_config) != read_json(suite / "judge.json")):
            raise ValueError("Resume inputs differ; use the original command or a new job name")
        return plan
    if any(p.name != "pipeline.lock" and not p.name.startswith(".suite-partial-") for p in job.iterdir()):
        raise ValueError("Nonempty output is not a Base suite; choose a new job name")
    metadata, checkpoint_sha = base_origin(run)
    partition_panel(list(iter_jsonl(run / "data/panel.jsonl")))
    panel, _ = load_panel(panel_source, metadata["data_files"]["train.jsonl"])
    partition(panel)
    staged = job / (".suite-partial-" + uuid.uuid4().hex)
    staged.mkdir()
    code_sha = snapshot_code(staged / "code")
    write_json(staged / "judge.json", read_json(args.judge_config or ROOT / "configs/lcb_grading.json"))
    plan = {"format_version": "base-four-benchmarks-v1", "created_at": now(), "run": str(run),
            "run_sha256": file_hash(run / "run.json"), "model": metadata["model"],
            "model_mode": "base_only_no_adapter", "epochs": [0], "checkpoint_sha256": checkpoint_sha,
            "panel_source": str(panel_source), "panel_source_manifest_sha256": file_hash(panel_source / "manifest.json"),
            "code_sha256": code_sha, "image_requested": args.image or DEFAULT_IMAGE,
            "judge_config_sha256": file_hash(staged / "judge.json"), "questions": 392, "responses": 392,
            "panels": [name for name, *_ in PANELS],
            "execution_order": "all four math/GPQA groups, then all four BBH/LCB groups"}
    write_json(staged / "plan.json", plan)
    write_json(staged / "complete.json", {"plan_sha256": file_hash(staged / "plan.json")})
    staged.rename(suite)
    return plan


def prepare_panels(job, plan, stop_check=lambda: None):
    run, panel_source = Path(plan["run"]), Path(plan["panel_source"])
    plans = {}
    for name, mode, _, size, _ in PANELS:
        stop_check()
        child = job / name
        child.mkdir(exist_ok=True)
        new_transfer = mode == "transfer" and not (child / "setup").exists()
        if new_transfer and file_hash(panel_source / "manifest.json") != plan["panel_source_manifest_sha256"]:
            raise ValueError("Transfer source changed since this suite was created")
        args = SimpleNamespace(mode=mode, epochs="0", base_only=True,
                               image=plan["image_requested"] if new_transfer else None,
                               judge_config=str(job / "suite/judge.json") if mode == "transfer" else None)
        source = run / "data" if mode == "math" else panel_source
        saved = transfer_evaluation.prepare(run, source, child, args)
        if not saved.get("base_only") or saved["generation_epochs"] != [0] or len(saved["panel_ids"]) != size:
            raise ValueError("Child job is not the fixed Base-only panel")
        plans[name] = saved
    # Both graders/preflights and all frozen panel setups are ready before any GPU work.
    return plans


def overview(job, plan):
    body = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Base 四任务基线</title>'
    body += '<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:20px}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}</style>'
    body += '<h1>Base 四任务基线 · epoch 0</h1><p>模型：' + escape(plan["model"]["id"])
    body += '；revision：' + escape(plan["model"]["revision"]) + '。BF16，不加载任何 LoRA adapter。</p>'
    body += '<p>每题一次采样，并发 8，最多 8192 新 token；两个面板各分四组，共 392 题。汇总仅纳入完整面板。</p><ul>'
    combined, completed = [], []
    for name, _, label, size, group_size in PANELS:
        child = job / name
        done = (child / "complete.json").exists()
        body += f'<li>{label}：{size} 题，四组各 {group_size} 题；'
        body += f'<a href="{name}/report/report.html">完整报告</a>' if done else '尚未全部完成'
        body += '<ul>'
        for i in range(1, 5):
            if (child / "groups" / f"group-{i:03d}" / "complete.json").exists():
                body += f'<li><a href="{name}/groups/group-{i:03d}/report/report.html">第 {i} 组报告</a></li>'
        body += '</ul></li>'
        if done:
            completed.append(name)
            with (child / "report/curves.csv").open(encoding="utf-8", newline="") as stream:
                combined.extend({"model_mode": plan["model_mode"], "panel": name, **r} for r in csv.DictReader(stream))
    body += '</ul><p><a href="curves.csv">已完成面板的指标 CSV</a> · <a href="summary.json">汇总 JSON</a>；完整题目、回答、token IDs 和判分保存在面板报告及对应数据目录。</p>'
    if combined:
        body += '<table><tr><th>任务</th><th>题数</th><th>正确率</th><th>平均长度</th><th>正常结束率</th><th>截断率</th><th>提取率</th></tr>'
        for r in combined:
            if r["subset"] == "all":
                body += '<tr>' + ''.join(f'<td>{escape(str(r[k]))}</td>' for k in (
                    "task", "n", "accuracy", "mean_generated_tokens", "eos_end_rate", "length_truncated_rate", "extractable_rate")) + '</tr>'
        body += '</table>'
        with (job / "curves.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(combined[0]))
            writer.writeheader()
            writer.writerows(combined)
    (job / "report.html").write_text(body + '</html>', encoding="utf-8")
    write_json(job / "summary.json", {"model": plan["model"], "model_mode": plan["model_mode"], "epochs": [0],
               "expected_responses": 392, "completed_panels": completed, "curves": combined})


def execute(job, worker):
    plan = verify(job)
    prepare_panels(job, plan, worker.check_stop)
    overview(job, plan)
    for index, (name, _, label, _, group_size) in enumerate(PANELS, 1):
        worker.check_stop()
        write_json(job / "progress.json", {"stage": "evaluation", "panel_index": index, "panel": name,
                                           "epochs": [0], "panel_progress": f"{name}/progress.json"})
        print(f"BASE PANEL {index}/2: {label}; four groups of {group_size}; epoch=0; no adapter", flush=True)
        code = job / name / "setup/code"
        try:
            worker.invoke([sys.executable, "-B", "-u", code / "transfer_evaluation.py", "--worker", job / name], code)
        finally:
            overview(job, plan)  # Include completed group links even after interruption.
        if not (job / name / "complete.json").is_file():
            raise RuntimeError(f"Panel did not finish: {name}")
    write_json(job / "complete.json", {"completed_at": now(), "epochs": [0], "questions": 392, "responses": 392,
               "model_mode": plan["model_mode"], "suite_plan_sha256": file_hash(job / "suite/plan.json"),
               "panel_completions": {name: file_hash(job / name / "complete.json") for name, *_ in PANELS}})
    write_json(job / "progress.json", {"stage": "complete", "panels_complete": 2})
    print(f"BASE FOUR BENCHMARKS DONE: {job / 'report.html'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="runs/random-lora-s42", help="Original run providing the pinned Base/tokenizer and fixed panels")
    p.add_argument("--panel-dir", default="artifacts/transfer-s2026")
    p.add_argument("--job-name", default="base-four-benchmarks-vllm-v1")
    p.add_argument("--image", help="Optional Docker image for a new job; resume keeps the pinned image")
    p.add_argument("--judge-config")
    p.add_argument("--prepare-only", action="store_true", help="Freeze inputs and check both graders, without loading a GPU model")
    p.add_argument("--worker", help=argparse.SUPPRESS)
    args = p.parse_args()
    configure_runtime()
    if args.worker:
        job = Path(args.worker).resolve()
        with Pipeline(job, inherited=True) as worker:
            try:
                if args.prepare_only:
                    plan = verify(job)
                    prepare_panels(job, plan, worker.check_stop)
                    overview(job, plan)
                    print("Base panels prepared; repeat without --prepare-only to generate", flush=True)
                else:
                    execute(job, worker)
            except BaseException as exc:
                progress = read_json(job / "progress.json") if (job / "progress.json").exists() else {}
                progress.update(stage="paused" if isinstance(exc, (SystemExit, KeyboardInterrupt)) else "failed", error=str(exc))
                write_json(job / "progress.json", progress)
                raise
        return
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.job_name):
        p.error("Invalid job name")
    run, panel_source = artifact_path(args.run), artifact_path(args.panel_dir)
    job = (run / "postprocess" / args.job_name).resolve()
    if job.parent != (run / "postprocess").resolve():
        p.error("Job path escapes run/postprocess")
    with Pipeline(job) as worker:
        prepare(run, panel_source, job, args)
        worker.check_stop()
        code = job / "suite/code"
        worker.invoke([sys.executable, "-B", "-u", code / "base_evaluation.py", "--worker", job,
                       *(["--prepare-only"] if args.prepare_only else [])], code)


if __name__ == "__main__":
    main()
