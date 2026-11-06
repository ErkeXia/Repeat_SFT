"""Four groups of 25 BBH + 25 LiveCodeBench, old checkpoints 64/32/16/8/4."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import sys
import uuid

from recipe.adapters import adapter_spec
from recipe.common import (ROOT, checkpoint_path, digest, file_hash, iter_jsonl, now, read_json,
                           snapshot_code, write_json, write_jsonl)
from recipe.grouped import generation_epochs, merge_groups, partition_panel
from recipe.lcb_docker import DEFAULT_IMAGE, DockerJudge
from recipe.paths import artifact_path, configure_runtime
from recipe.pipeline import Pipeline
from recipe.resume import verified_code, verify_run_inputs
from recipe.replay import evaluation_epochs
from recipe.storage import GIB, require_free_space
from recipe.transfer_eval import copy_panel, load_panel, partition
from recipe.vllm_generation import validate_config
from report_transfer import report


def prepare(run, panel_source, job, args):
    setup = job / "setup"
    mode = getattr(args, "mode", "transfer")
    if mode not in ("transfer", "replay-math", "replay-transfer"):
        raise ValueError("Unknown evaluation mode")
    replay_mode, math_mode = mode != "transfer", mode == "replay-math"
    if setup.exists():
        plan = read_json(setup / "plan.json")
        if plan.get("evaluation_kind", "transfer") != mode:
            raise ValueError("This saved job uses a different panel mode; choose another job name")
        if (plan["run"] != str(run) or plan["panel_source"] != str(panel_source)
                or (args.epochs and generation_epochs(args.epochs) != plan["generation_epochs"])):
            raise ValueError("Resume inputs differ from the frozen transfer job; choose another job name")
        if args.judge_config and read_json(args.judge_config) != plan["preflight"]["config"]:
            raise ValueError("Judge limits differ from the saved job; choose a new job name")
        if args.image and DockerJudge(args.image, plan["preflight"]["config"]).image != plan["image"]:
            raise ValueError("Execution image differs from the saved job")
        verify(job)
        return plan
    if any(p.name != "pipeline.lock" and not p.name.startswith(".setup-partial-") for p in job.iterdir()):
        raise ValueError("Existing non-transfer outputs in this job; choose a new job name")
    metadata = verify_run_inputs(run)
    if metadata.get("replay") and not replay_mode:
        raise ValueError("This transfer experiment uses the original run's checkpoints, not replay checkpoints")
    if replay_mode and not metadata.get("replay"):
        raise ValueError("The intermediate experiment requires the replay run with the new checkpoints")
    if math_mode:
        if panel_source != run / "data" or args.image or args.judge_config:
            raise ValueError("Math/GPQA evaluation uses the run's own panel and no Docker options")
        panel = list(iter_jsonl(run / "data" / "panel.jsonl"))
        groups = partition_panel(panel)
    else:
        panel, _ = load_panel(panel_source, metadata["data_files"]["train.jsonl"])
        groups = partition(panel)
    epochs = generation_epochs(args.epochs or ",".join(map(str, evaluation_epochs(metadata)))) if replay_mode else generation_epochs(args.epochs)
    if not set(epochs).issubset(evaluation_epochs(metadata)):
        raise ValueError("Requested epochs are not saved targets of this run; use the run containing those checkpoints")
    checkpoints = {}
    for epoch in epochs:
        try:
            ckpt = checkpoint_path(run, epoch)
        except FileNotFoundError as exc:
            path = run / "checkpoints" / f"epoch-{epoch:03d}"
            raise ValueError(f"Checkpoint not ready: {path}. Finish replay or use --epochs with completed targets only") from exc
        marker = read_json(ckpt / "complete.json")
        if marker["config_sha256"] != metadata["config_sha256"] or marker["step"] != epoch * metadata["config"]["expected_train_size"]:
            raise ValueError("Checkpoint differs from the selected run")
        adapter_spec(ckpt, finetuning=metadata["config"]["finetuning"], base_model=metadata["model"])
        checkpoints[str(epoch)] = file_hash(ckpt / "complete.json")
    if metadata["config"]["finetuning"]["base_quantization"] != "none":
        raise ValueError("vLLM transfer evaluation requires BF16 LoRA")
    settings = read_json(ROOT / "configs" / "generation.json")
    if settings["max_new_tokens"] != 8192:
        raise ValueError("Transfer protocol uses 8192 new tokens")
    engine = validate_config(read_json(ROOT / "configs" / "vllm.json"))
    if math_mode:
        import math_verify  # Fail before GPU generation if the mathematical grader is missing.
        judge, preflight = None, {"math_verifier": "available", "config": None}
        selected_bytes = (run / "data" / "panel.jsonl").stat().st_size * 2
    else:
        judge = DockerJudge(args.image or DEFAULT_IMAGE, read_json(args.judge_config or ROOT / "configs" / "lcb_grading.json"))
        preflight = judge.preflight()  # Verify execution infrastructure before spending GPU time.
        selected_bytes = sum((panel_source / p["reference_file"]).stat().st_size for p in panel)
    require_free_space(job, selected_bytes * 2 + GIB)
    staged = job / (".setup-partial-" + uuid.uuid4().hex)
    staged.mkdir()
    if math_mode:
        from recipe.math_eval import freeze_panel
        panel = freeze_panel(run / "data" / "panel.jsonl", staged / "panel", metadata["data_files"]["train.jsonl"])
    else:
        copy_panel(panel_source, staged / "panel", metadata["data_files"]["train.jsonl"])
    code_hash = snapshot_code(staged / "code")
    write_json(staged / "generation.json", settings)
    write_json(staged / "vllm.json", engine)
    write_json(staged / "judge.json", judge.config if judge else None)
    files = {name: file_hash(staged / name) for name in ("generation.json", "vllm.json", "judge.json", "panel/manifest.json")}
    plan = {"version": 1, "created_at": now(), "run": str(run), "run_sha256": file_hash(run / "run.json"),
            "panel_source": str(panel_source), "panel_sha256": file_hash(staged / "panel" / "panel.jsonl"),
            "panel_ids": [p["sample_id"] for p in panel], "groups": groups,
            "generation_epochs": epochs, "generation_backend": "vllm", "settings": settings,
            "checkpoints": checkpoints, "code_sha256": code_hash, "image": judge.image if judge else None,
            "preflight": preflight, "setup_files": files}
    if replay_mode:
        plan.update(evaluation_kind=mode, replay=metadata["replay"])
    if math_mode:
        plan["math_panel_sha256"] = metadata["data_files"]["panel.jsonl"]
    write_json(staged / "plan.json", plan)
    write_json(staged / "complete.json", {"plan_sha256": file_hash(staged / "plan.json")})
    staged.rename(setup)
    return plan


def verify(job):
    setup = job / "setup"
    if read_json(setup / "complete.json")["plan_sha256"] != file_hash(setup / "plan.json"):
        raise ValueError("Frozen transfer plan changed")
    plan = read_json(setup / "plan.json")
    run = Path(plan["run"])
    if file_hash(run / "run.json") != plan["run_sha256"]:
        raise ValueError("Original run metadata changed")
    metadata = verify_run_inputs(run)
    verified_code(setup / "code", plan["code_sha256"])
    for name, expected in plan["setup_files"].items():
        if file_hash(setup / name) != expected:
            raise ValueError(f"Frozen evaluation input changed: {name}")
    load_panel(setup / "panel", metadata["data_files"]["train.jsonl"])
    if plan.get("evaluation_kind") in ("replay-math", "replay-transfer"):
        if plan["replay"] != metadata.get("replay") or not set(plan["generation_epochs"]) <= set(evaluation_epochs(metadata)):
            raise ValueError("Replay checkpoint selection or lineage changed")
    if plan.get("evaluation_kind") == "replay-math":
        from recipe.math_eval import verify_math_origin
        verify_math_origin(setup / "panel", metadata["data_files"]["panel.jsonl"])
    for epoch, expected in plan["checkpoints"].items():
        ckpt = checkpoint_path(run, int(epoch))
        if file_hash(ckpt / "complete.json") != expected:
            raise ValueError("Original checkpoint marker changed")
        adapter_spec(ckpt, finetuning=metadata["config"]["finetuning"], base_model=metadata["model"])
    return plan


def group_evidence(directory, epochs):
    files = [directory / "grades" / "grading_manifest.json", directory / "report" / "manifest.json"]
    for epoch in epochs:
        for kind, filename in (("generation", "responses.jsonl"), ("grades", "graded.jsonl")):
            folder = directory / kind / f"epoch-{epoch:03d}"
            files.extend([folder / filename, folder / "complete.json", folder / "identity.json"])
    return {p.relative_to(directory).as_posix(): file_hash(p) for p in files}


def execute(job, pipeline):
    plan = verify(job)
    setup = job / "setup"
    code, panel_dir = setup / "code", setup / "panel"
    epochs = plan["generation_epochs"]
    for index, group in enumerate(plan["groups"], 1):
        pipeline.check_stop()
        directory = job / "groups" / group["name"]
        marker = directory / "complete.json"
        if marker.exists():
            if read_json(marker)["files"] != group_evidence(directory, epochs):
                raise ValueError("Completed group results changed")
            print(f"SKIP group {index}/4: all saved generation and grading retained", flush=True)
            continue
        directory.mkdir(parents=True, exist_ok=True)
        ids = directory / "sample_ids.json"
        if ids.exists() and read_json(ids) != group["selected_ids"]:
            raise ValueError("Frozen group membership changed")
        write_json(ids, group["selected_ids"])
        write_json(job / "progress.json", {"group": index, "stage": "generation", "epochs": epochs})
        tasks = ", ".join(f"{task}={n}" for task, n in group["task_counts"].items())
        print(f"GROUP {index}/4: {tasks}; epochs={epochs}", flush=True)
        pipeline.invoke([sys.executable, "-B", "-u", code / "generate_vllm.py", "--run", plan["run"],
                         "--panel-dir", panel_dir, "--epochs", ",".join(map(str, epochs)),
                         "--sample-ids", ids, "--config", setup / "generation.json", "--engine-config", setup / "vllm.json",
                         "--out", directory / "generation", "--progress-label", f"group={index}/4"], code)
        write_json(job / "progress.json", {"group": index, "stage": "grading", "epochs": epochs})
        judge_options = ["--image", plan["image"]["id"], "--judge-config", setup / "judge.json"] if plan["image"] else []
        pipeline.invoke([sys.executable, "-B", "-u", code / "grade_transfer.py", "--responses", directory / "generation",
                         "--panel-dir", panel_dir, "--out", directory / "grades", *judge_options], code)
        report(panel_dir, directory / "grades", directory / "report", group["selected_ids"])
        write_json(marker, {"completed_at": now(), "files": group_evidence(directory, epochs)})
        print(f"GROUP {index}/4 SAVED: {directory / 'report' / 'report.html'}", flush=True)
    pipeline.check_stop()
    merge_groups(job, plan)
    for epoch in epochs:
        rows = []
        for group in plan["groups"]:
            rows.extend(iter_jsonl(job / "groups" / group["name"] / "grades" / f"epoch-{epoch:03d}" / "graded.jsonl"))
        if len(rows) != len(plan["panel_ids"]) or {r["sample_id"] for r in rows} != set(plan["panel_ids"]):
            raise ValueError("Graded groups do not cover the whole transfer panel")
        directory = job / "grades" / f"epoch-{epoch:03d}"
        write_jsonl(directory / "graded.jsonl", rows)
        write_json(directory / "complete.json", {"samples": len(rows), "graded_sha256": file_hash(directory / "graded.jsonl")})
    write_json(job / "grades" / "grading_manifest.json", {"plan_sha256": digest(plan), "image": plan["image"],
               "group_manifests": {g["name"]: file_hash(job / "groups" / g["name"] / "grades" / "grading_manifest.json") for g in plan["groups"]}})
    report(panel_dir, job / "grades", job / "report")
    size = len(plan["panel_ids"])
    write_json(job / "complete.json", {"completed_at": now(), "epochs": epochs, "questions": size,
               "responses": size * len(epochs), "report": str(job / "report" / "report.html")})
    write_json(job / "progress.json", {"stage": "complete", "groups_complete": 4})
    print(f"ALL DONE: {job / 'report' / 'report.html'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="runs/random-lora-s42")
    p.add_argument("--panel-dir", default="artifacts/transfer-s2026")
    p.add_argument("--job-name", default="transfer-e4plus-vllm-v1")
    p.add_argument("--epochs", help="Default old checkpoints 64,32,16,8,4")
    p.add_argument("--image", help=f"New job default: {DEFAULT_IMAGE}; resume keeps the pinned image ID")
    p.add_argument("--judge-config", help="New job default: configs/lcb_grading.json; resume keeps frozen limits")
    p.add_argument("--worker", help=argparse.SUPPRESS)
    args = p.parse_args()
    configure_runtime()
    if args.worker:
        job = Path(args.worker).resolve()
        with Pipeline(job, inherited=True) as pipeline:
            execute(job, pipeline)
        return
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.job_name):
        p.error("Invalid job name")
    run, panel = artifact_path(args.run), artifact_path(args.panel_dir)
    job = (run / "postprocess" / args.job_name).resolve()
    if job.parent != (run / "postprocess").resolve():
        p.error("Job path escapes run/postprocess")
    with Pipeline(job) as pipeline:
        prepare(run, panel, job, args)
        code = job / "setup" / "code"
        pipeline.invoke([sys.executable, "-B", "-u", code / "transfer_evaluation.py", "--worker", job], code)


if __name__ == "__main__":
    main()
