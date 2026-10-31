"""SERVER: four saved 48-question groups, each evaluated at epochs 64 down to 1."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import uuid

from recipe.common import (ROOT, checkpoint_path, environment, file_hash, iter_jsonl, now, read_json,
                           snapshot_code, write_json, write_jsonl)
from recipe.grouped import GENERATION_EPOCHS, MAX_NEW_TOKENS, merge_groups, partition_panel, select_panel, teacher_forcing_files
from recipe.paths import artifact_path, configure_runtime
from recipe.resume import verified_code, verify_run_inputs

_child = None


def stop(signum, _frame):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if _child is not None and _child.poll() is None:
        _child.terminate()
        _child.wait()
    print("Stopped. Resume with the same command WITHOUT --reset-generation.", flush=True)
    raise SystemExit(128 + signum)


def invoke(command, cwd):
    global _child
    _child = subprocess.Popen([str(x) for x in command], cwd=cwd)
    try:
        result = _child.wait()
    finally:
        _child = None
    if result:
        raise RuntimeError(f"Stage exited {result}: {' '.join(map(str, command))}")


@contextmanager
def job_lock(job):
    """Shares the existing shell launcher's flock on Linux."""
    job.mkdir(parents=True, exist_ok=True)
    with (job / "pipeline.lock").open("a+b") as lock:
        try:
            if os.name == "posix":
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                import msvcrt
                if lock.seek(0, 2) == 0:
                    lock.write(b"\0")
                    lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError(f"This job is already running: {job}") from exc
        try:
            yield
        finally:
            if os.name == "posix":
                fcntl.flock(lock, fcntl.LOCK_UN)
            else:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def replacement_paths(job):
    names = {"generation", "groups", "grouped_code", "grouped_pipeline.json", "grouped_launcher.sh",
             "grouped_state", "grouped_complete.json", "grouped_progress.json", "grouped_environment.json",
             "grades_path.txt", "analysis_path.txt"}
    paths = [p for p in job.iterdir() if p.name in names or re.fullmatch(r"(?:grades|analysis)(?:-retry-.*)?", p.name)]
    state = job / "state"
    if state.is_dir():
        paths += [p for p in state.iterdir() if re.fullmatch(r"(?:generation-\d+|grading|analysis)\.done", p.name)]
    return sorted(paths)


def archive_generation(job, fit_source):
    """Replace only generation/derived reports. Never move fitting results or weights."""
    job = job.resolve()
    paths = replacement_paths(job)
    for path in paths:
        target = path.resolve()
        if not target.is_relative_to(job) or target == job:
            raise ValueError(f"Replacement path escapes the selected job: {path}")
        if fit_source.resolve().is_relative_to(target) or target.is_relative_to(fit_source.resolve()):
            raise ValueError(f"Replacement would move fitting evidence: {path}")
    if not paths:
        return None
    backup = (job / "generation-backups" / uuid.uuid4().hex).resolve()
    if not backup.is_relative_to(job):
        raise ValueError("Generation backup path escapes the selected job")
    backup.mkdir(parents=True, exist_ok=False)
    write_json(backup / "replacement.json", {"created_at": now(), "teacher_forcing_preserved": str(fit_source),
                                            "paths": [p.relative_to(job).as_posix() for p in paths]})
    for path in paths:
        target = backup / path.relative_to(job)
        target.parent.mkdir(parents=True, exist_ok=True)
        path.rename(target)
    print(f"Previous generation and derived outputs replaced; backup: {backup}", flush=True)
    return backup


def locate_fitting(run, job, explicit, epochs, metadata):
    candidates = ([artifact_path(explicit)] if explicit else
                  [job / "teacher_forcing", run / "evaluations" / "teacher_forcing"] +
                  [p / "teacher_forcing" for p in sorted((run / "postprocess").glob("*")) if p != job])
    errors = []
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
            return candidate, teacher_forcing_files(run, candidate, epochs, metadata)
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"{candidate}: {exc}")
    raise ValueError("No complete matching teacher-forcing results. Supply --teacher-forcing PATH. "
                     "No fitting results were changed.\n" + "\n".join(errors))


def prepare_job(run, job, args):
    metadata = verify_run_inputs(run)
    path = job / "grouped_pipeline.json"
    if path.exists() and not args.reset_generation:
        plan = read_json(path)
        if plan["run"] != str(run) or plan["run_sha256"] != file_hash(run / "run.json"):
            raise ValueError("Grouped job belongs to a different run")
        if args.teacher_forcing and str(artifact_path(args.teacher_forcing)) != plan["teacher_forcing"]:
            raise ValueError("Teacher-forcing source changed; resume with the saved source")
        verified_code(job / "grouped_code", plan["code_sha256"])
        return plan
    panel = list(iter_jsonl(run / "data" / "panel.jsonl"))
    groups = partition_panel(panel)
    fit_epochs = sorted(set(metadata["config"]["checkpoint_epochs"]))
    if not set(GENERATION_EPOCHS).issubset(fit_epochs):
        raise ValueError("This run does not contain all seven requested training checkpoints")
    checkpoints = {}
    for epoch in fit_epochs:
        ckpt = checkpoint_path(run, epoch)
        marker = read_json(ckpt / "complete.json")
        if (marker["config_sha256"] != metadata["config_sha256"]
                or marker["step"] != epoch * metadata["config"]["expected_train_size"]):
            raise ValueError(f"Checkpoint differs from this run: {ckpt}")
        checkpoints[str(epoch)] = file_hash(ckpt / "complete.json")
    settings = read_json(ROOT / "configs" / "generation.json")
    if settings["max_new_tokens"] != MAX_NEW_TOKENS:
        raise ValueError("Grouped protocol requires max_new_tokens=8192")
    source, hashes = locate_fitting(run, job, args.teacher_forcing, fit_epochs, metadata)
    old_outputs = replacement_paths(job)
    if old_outputs and not args.reset_generation:
        raise ValueError("This job contains older generation outputs. Use --reset-generation once to replace them.")
    backup = archive_generation(job, source) if args.reset_generation else None
    # Stage the source copy; publish its manifest only when initialization is complete.
    staged = job / (".grouped-code-" + uuid.uuid4().hex)
    code_hash = snapshot_code(staged)
    staged.rename(job / "grouped_code")
    launcher = ROOT / "run_grouped_analysis.sh"
    if launcher.exists():
        shutil.copy2(launcher, job / "grouped_launcher.sh")
    group_files = {}
    for group in groups:
        directory = job / "groups" / group["name"]
        write_json(directory / "sample_ids.json", group["selected_ids"])
        write_jsonl(directory / "panel.jsonl", select_panel(panel, group["selected_ids"]))
        for name in ("sample_ids.json", "panel.jsonl"):
            group_files[(directory / name).relative_to(job).as_posix()] = file_hash(directory / name)
    plan = {"version": 1, "created_at": now(), "run": str(run), "run_sha256": file_hash(run / "run.json"),
            "panel_sha256": file_hash(run / "data" / "panel.jsonl"), "panel_ids": [r["sample_id"] for r in panel],
            "generation_epochs": GENERATION_EPOCHS, "fit_epochs": fit_epochs, "settings": settings,
            "groups": groups, "group_files": group_files, "code_sha256": code_hash, "checkpoints": checkpoints,
            "launcher_sha256": file_hash(job / "grouped_launcher.sh") if launcher.exists() else None,
            "teacher_forcing": str(source), "teacher_forcing_files": hashes,
            "previous_generation_backup": str(backup) if backup else None,
            "stop_policy": "Stop at EOS or exactly 8192 new tokens; capped output is length-truncated, never a fabricated EOS.",
            "execution_order": "group -> descending checkpoint -> fixed question order"}
    write_json(path, plan)
    write_json(job / "grouped_environment.json", environment())
    return plan


def execute(job):
    plan = read_json(job / "grouped_pipeline.json")
    run = Path(plan["run"])
    metadata = verify_run_inputs(run)
    if file_hash(run / "run.json") != plan["run_sha256"]:
        raise ValueError("Run identity changed")
    code = verified_code(job / "grouped_code", plan["code_sha256"])
    if plan.get("launcher_sha256") and file_hash(job / "grouped_launcher.sh") != plan["launcher_sha256"]:
        raise ValueError("Saved grouped launcher changed")
    for name, checksum in plan["group_files"].items():
        if file_hash(job / name) != checksum:
            raise ValueError(f"Saved question group changed: {name}")
    for epoch, checksum in plan["checkpoints"].items():
        if file_hash(checkpoint_path(run, int(epoch)) / "complete.json") != checksum:
            raise ValueError(f"Checkpoint identity changed: epoch {epoch}")
    source = Path(plan["teacher_forcing"])
    if teacher_forcing_files(run, source, plan["fit_epochs"], metadata) != plan["teacher_forcing_files"]:
        raise ValueError("Saved teacher-forcing evidence changed")
    print(f"Job: {job}\nReusing fitting evidence: {source}\nGeneration epochs: {plan['generation_epochs']}\n"
          f"Max new tokens: {MAX_NEW_TOKENS}; 4 groups x 48 questions", flush=True)
    state = job / "grouped_state"
    state.mkdir(exist_ok=True)
    gen_epochs = ",".join(map(str, plan["generation_epochs"]))
    fit_epochs = ",".join(map(str, plan["fit_epochs"]))

    def stage(name, script=None, arguments=(), output=None):
        marker = state / (name + ".json")
        if marker.exists():
            record = read_json(marker)
            print(f"SKIP {name} (saved)", flush=True)
            return Path(record["output"]) if record.get("output") else None
        actual = None
        if output is not None:
            actual = Path(output)
            if actual.exists():
                actual = actual.with_name(actual.name + "-retry-" + uuid.uuid4().hex)
        write_json(job / "grouped_progress.json", {"stage": name, "started_at": now()})
        print(f"START {name}", flush=True)
        if script:
            command = [sys.executable, "-B", "-u", code / script, *arguments]
            if actual is not None:
                command += ["--out", actual]
            invoke(command, code)
        else:
            merge_groups(job, plan)
        write_json(marker, {"completed_at": now(), "output": str(actual) if actual else None})
        print(f"DONE {name}", flush=True)
        return actual

    for index, group in enumerate(plan["groups"], 1):
        name = group["name"]
        directory = job / "groups" / name
        print(f"GROUP {index}/4: 32 MATH500 + 16 GPQA; epochs {gen_epochs}", flush=True)
        stage(name + "-generation", "generate.py", ["--run", run, "--epochs", gen_epochs,
              "--sample-ids", directory / "sample_ids.json", "--progress-label", f"group={index}/4",
              "--config", code / "configs" / "generation.json", "--out", directory / "generation"])
        grades = stage(name + "-grading", "grade.py", ["--responses", directory / "generation"], directory / "grades")
        analysis = stage(name + "-analysis", "analyze_run.py", ["--run", run, "--epochs", fit_epochs,
              "--generation-epochs", gen_epochs, "--teacher-forcing", source, "--grades", grades,
              "--sample-ids", directory / "sample_ids.json"], directory / "analysis")
        write_json(directory / "complete.json", {"completed_at": now(), "questions": 48,
                   "epochs": plan["generation_epochs"], "grades": str(grades), "analysis": str(analysis)})
        print(f"GROUP {index}/4 SAVED: {analysis / 'report.html'}", flush=True)
    stage("merge-full-panel")
    grades = stage("full-grading", "grade.py", ["--responses", job / "generation"], job / "grades")
    analysis = stage("full-analysis", "analyze_run.py", ["--run", run, "--epochs", fit_epochs,
          "--generation-epochs", gen_epochs, "--teacher-forcing", source, "--grades", grades], job / "analysis")
    write_json(job / "grouped_complete.json", {"completed_at": now(), "questions": 192,
               "epochs": plan["generation_epochs"], "grades": str(grades), "analysis": str(analysis)})
    write_json(job / "grouped_progress.json", {"stage": "complete", "finished_at": now()})
    print(f"ALL DONE. Report: {analysis / 'report.html'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", nargs="?", default="runs/random-lora-s42")
    p.add_argument("job_name", nargs="?", default="grouped-8192-v1")
    p.add_argument("--teacher-forcing", help="Reuse complete fitting results; auto-detect by default")
    p.add_argument("--reset-generation", action="store_true", help="ONCE: replace this job's generation/reports with backups; preserve fitting results")
    p.add_argument("--worker", help=argparse.SUPPRESS)
    args = p.parse_args()
    configure_runtime()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if args.worker:
        execute(Path(args.worker).resolve())
        return
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.job_name):
        p.error("Invalid job name")
    run = artifact_path(args.run)
    job = (run / "postprocess" / args.job_name).resolve()
    if job.parent != (run / "postprocess").resolve():
        p.error("Job path escapes run/postprocess")
    with job_lock(job):
        print(f"PID={os.getpid()}", flush=True)
        prepare_job(run, job, args)
        invoke([sys.executable, "-B", "-u", job / "grouped_code" / "grouped_evaluation.py", "--worker", job], job)


if __name__ == "__main__":
    main()
