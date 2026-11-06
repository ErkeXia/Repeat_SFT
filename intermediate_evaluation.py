"""Replay checkpoints: four MATH500/GPQA groups, then four BBH/LiveCodeBench groups."""
from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import uuid

from recipe.adapters import adapter_spec
from recipe.common import ROOT, checkpoint_path, file_hash, now, read_json, snapshot_code, write_json
from recipe.grouped import generation_epochs
from recipe.lcb_docker import DEFAULT_IMAGE
from recipe.paths import artifact_path, configure_runtime
from recipe.pipeline import Pipeline
from recipe.replay import evaluation_epochs
from recipe.resume import verified_code, verify_run_inputs
from recipe.transfer_eval import load_panel, partition
import transfer_evaluation

PANELS = (("math-gpqa", "replay-math", "MATH500 / GPQA", 192, 48),
          ("bbh-livecodebench", "replay-transfer", "BBH / LiveCodeBench", 200, 50))


def verify(job):
    suite = job / "suite"
    if read_json(suite / "complete.json")["plan_sha256"] != file_hash(suite / "plan.json"):
        raise ValueError("Frozen suite plan changed")
    plan = read_json(suite / "plan.json")
    run = Path(plan["run"])
    if file_hash(run / "run.json") != plan["run_sha256"]:
        raise ValueError("Replay run metadata changed")
    verify_run_inputs(run)
    verified_code(suite / "code", plan["code_sha256"])
    if file_hash(suite / "judge.json") != plan["judge_config_sha256"]:
        raise ValueError("Frozen judge settings changed")
    for epoch, sha in plan["checkpoints"].items():
        if file_hash(checkpoint_path(run, int(epoch)) / "complete.json") != sha:
            raise ValueError(f"Replay checkpoint changed: epoch {epoch}")
    return plan


def prepare(run, panel_source, job, args):
    suite = job / "suite"
    if suite.exists():
        plan = verify(job)
        if (plan["run"] != str(run) or plan["panel_source"] != str(panel_source)
                or (args.epochs and generation_epochs(args.epochs) != plan["epochs"])
                or (args.image and args.image != plan["image_requested"])
                or (args.judge_config and read_json(args.judge_config) != read_json(suite / "judge.json"))):
            raise ValueError("Suite inputs differ; resume with the original command or use a new job name")
        return plan
    if any(p.name != "pipeline.lock" and not p.name.startswith(".suite-partial-") for p in job.iterdir()):
        raise ValueError("Nonempty output is not a prepared intermediate suite; choose a new job name")
    metadata = verify_run_inputs(run)
    if not metadata.get("replay"):
        raise ValueError("This entry requires the replay run containing the new checkpoints")
    epochs = generation_epochs(args.epochs or ",".join(map(str, evaluation_epochs(metadata))))
    if not set(epochs) <= set(evaluation_epochs(metadata)):
        raise ValueError("Requested epochs are not saved targets of this replay run")
    checkpoints = {}
    for epoch in epochs:
        try:
            ckpt = checkpoint_path(run, epoch)
        except FileNotFoundError as exc:
            raise ValueError(f"Checkpoint not ready: epoch {epoch}. Finish replay or select completed nodes with --epochs") from exc
        marker = read_json(ckpt / "complete.json")
        if marker["config_sha256"] != metadata["config_sha256"] or marker["step"] != epoch * metadata["config"]["expected_train_size"]:
            raise ValueError("Checkpoint belongs to a different training trajectory")
        adapter_spec(ckpt, finetuning=metadata["config"]["finetuning"], base_model=metadata["model"])
        checkpoints[str(epoch)] = file_hash(ckpt / "complete.json")
    panel, _ = load_panel(panel_source, metadata["data_files"]["train.jsonl"])
    partition(panel)
    staged = job / (".suite-partial-" + uuid.uuid4().hex)
    staged.mkdir()
    code_sha = snapshot_code(staged / "code")
    write_json(staged / "judge.json", read_json(args.judge_config or ROOT / "configs/lcb_grading.json"))
    plan = {"format_version": "replay-two-panels-v1", "created_at": now(), "run": str(run),
            "run_sha256": file_hash(run / "run.json"), "panel_source": str(panel_source),
            "panel_source_manifest_sha256": file_hash(panel_source / "manifest.json"),
            "epochs": epochs, "checkpoints": checkpoints, "code_sha256": code_sha,
            "image_requested": args.image or DEFAULT_IMAGE, "judge_config_sha256": file_hash(staged / "judge.json"),
            "panels": [name for name, *_ in PANELS], "execution_order": "all four math/GPQA groups, then all four BBH/LCB groups",
            "replay": metadata["replay"]}
    write_json(staged / "plan.json", plan)
    write_json(staged / "complete.json", {"plan_sha256": file_hash(staged / "plan.json")})
    staged.rename(suite)
    return plan


def overview(job, plan):
    body = '<!doctype html><meta charset="utf-8"><title>新节点：两个评测面板</title><h1>新节点：两个评测面板各分四组</h1>'
    body += '<p>生成节点（降序）：' + escape(str(plan["epochs"])) + '。每题一次采样，并发 8，最多 8192 个新 token。</p><ul>'
    for name, _, label, size, group_size in PANELS:
        complete = (job / name / "complete.json").exists()
        body += f'<li>{label}：{size} 题，四组各 {group_size} 题；'
        body += f'<a href="{name}/report/report.html">已完成，打开报告</a>' if complete else "尚未全部完成，可查看子目录中的已完成组报告"
        body += '</li>'
    body += '</ul><p>完整回答和判分分别在两个面板的 generation/ 与 grades/；旧 run 和拟合结果保持原样。</p>'
    (job / "report.html").write_text(body, encoding="utf-8")


def prepare_panels(job, plan, stop_check=lambda: None):
    run, panel_source = Path(plan["run"]), Path(plan["panel_source"])
    # Prepare BOTH panels, including math dependency and Docker preflight, before any GPU generation.
    plans = {}
    for name, mode, _, _, _ in PANELS:
        stop_check()
        child = job / name
        child.mkdir(exist_ok=True)
        if mode == "replay-transfer" and not (child / "setup").exists():
            if file_hash(panel_source / "manifest.json") != plan["panel_source_manifest_sha256"]:
                raise ValueError("Transfer source changed since this suite was created")
        args = SimpleNamespace(mode=mode, epochs=",".join(map(str, plan["epochs"])),
               image=plan["image_requested"] if mode == "replay-transfer" and not (child / "setup").exists() else None,
               judge_config=str(job / "suite/judge.json") if mode == "replay-transfer" else None)
        source = run / "data" if mode == "replay-math" else panel_source
        plans[name] = transfer_evaluation.prepare(run, source, child, args)
    return plans


def execute(job, worker):
    plan = verify(job)
    prepare_panels(job, plan, worker.check_stop)
    overview(job, plan)
    for index, (name, _, label, _, group_size) in enumerate(PANELS, 1):
        worker.check_stop()
        write_json(job / "progress.json", {"stage": "evaluation", "panel_index": index, "panel": name,
                                           "epochs": plan["epochs"], "panel_progress": f"{name}/progress.json"})
        print(f"PANEL {index}/2: {label}; four groups of {group_size}; epochs={plan['epochs']}", flush=True)
        code = job / name / "setup/code"
        worker.invoke([sys.executable, "-B", "-u", code / "transfer_evaluation.py", "--worker", job / name], code)
        if not (job / name / "complete.json").is_file():
            raise RuntimeError(f"Panel did not finish: {name}")
        overview(job, plan)
    write_json(job / "complete.json", {"completed_at": now(), "epochs": plan["epochs"], "questions": 392,
               "responses": 392 * len(plan["epochs"]),
               "panel_completions": {name: file_hash(job / name / "complete.json") for name, *_ in PANELS}})
    write_json(job / "progress.json", {"stage": "complete", "panels_complete": 2})
    print(f"ALL PANELS DONE: {job / 'report.html'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="runs/random-lora-s42-intermediate-v1")
    p.add_argument("--panel-dir", default="artifacts/transfer-s2026")
    p.add_argument("--job-name", default="intermediate-two-panels-vllm-v1")
    p.add_argument("--epochs", help="Default all replay targets, descending")
    p.add_argument("--image", help="Optional Docker image for the new BBH/LCB job")
    p.add_argument("--judge-config", help="Optional LiveCodeBench limits; frozen at first launch")
    p.add_argument("--worker", help=argparse.SUPPRESS)
    args = p.parse_args()
    configure_runtime()
    if args.worker:
        job = Path(args.worker).resolve()
        with Pipeline(job, inherited=True) as worker:
            execute(job, worker)
        return
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.job_name):
        p.error("Invalid job name")
    run, panel_source = artifact_path(args.run), artifact_path(args.panel_dir)
    job = (run / "postprocess" / args.job_name).resolve()
    if job.parent != (run / "postprocess").resolve():
        p.error("Job path escapes run/postprocess")
    with Pipeline(job) as worker:
        prepare(run, panel_source, job, args)
        code = job / "suite/code"
        worker.invoke([sys.executable, "-B", "-u", code / "intermediate_evaluation.py", "--worker", job], code)


if __name__ == "__main__":
    main()
