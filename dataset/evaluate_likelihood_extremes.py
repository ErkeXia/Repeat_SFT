"""SERVER: high then low, two fixed four-group panels per run, resumable vLLM evaluation."""
from __future__ import annotations

import argparse
import csv
from html import escape
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
from recipe.data import OverlapIndex
from recipe.grouped import generation_epochs, partition_panel
from recipe.lcb_docker import DEFAULT_IMAGE, DockerJudge
from recipe.paths import artifact_path, configure_runtime
from recipe.pipeline import Pipeline
from recipe.replay import evaluation_epochs
from recipe.resume import verified_code, verify_run_inputs
from recipe.storage import GIB, require_free_space
from recipe.transfer_eval import load_panel, partition
from dataset.likelihood_training import tree_hashes, verify_bundle
import transfer_evaluation

DEFAULT_EPOCHS = [16, 14, 12, 10, 8, 4, 2, 1]
GROUPS = ("high", "low")
PANELS = (("math-gpqa", "math", 192, 48), ("bbh-livecodebench", "transfer", 200, 50))


def overlap_checks(training, math_panel, transfer_panel):
    audits = {}
    for group, rows in training.items():
        index = OverlapIndex(rows)
        matches = []
        for q in math_panel + transfer_panel:
            match = index.match(q)
            if match:
                matches.append({"evaluation_sample_id": q["sample_id"], "task": q["task"],
                                "question": q["question"], **match})
        audits[group] = {"train_samples": len(rows), "checked_math_gpqa": len(math_panel),
                         "checked_bbh_livecodebench": len(transfer_panel), "matches": matches,
                         "methods": ["NFKC/whitespace exact", "char7 Jaccard >= 0.9"],
                         "semantic_overlap_not_ruled_out": True}
    return audits


def freeze_transfer(source, destination, training_sha, audit):
    """Keep every original question/test byte. Rebind ONLY after actual overlap checks."""
    if audit["matches"]:
        raise ValueError("Cannot bind a transfer panel with training overlaps")
    panel, manifest = load_panel(source)
    partition(panel)
    destination.mkdir(parents=True, exist_ok=False)
    names = ["panel.jsonl"] + [q["reference_file"] for q in panel]
    for name in names:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    original_plan = read_json(source / "plan.json")
    write_json(destination / "plan.json", {**original_plan,
               "previous_training_sha256": original_plan["training_sha256"], "training_sha256": training_sha,
               "source_manifest_sha256": file_hash(source / "manifest.json"),
               "source_panel_sha256": file_hash(source / "panel.jsonl"), "overlap_check": audit,
               "rebind_policy": "same selected questions, order, prompts and reference files; no resampling"})
    write_json(destination / "manifest.json", {"format_version": "transfer-evaluation-panel-v1",
               "sources": manifest.get("sources"), "source_manifest_sha256": file_hash(source / "manifest.json"),
               "files": tree_hashes(destination)})
    load_panel(destination, training_sha)


def verify(job):
    suite = job / "suite"
    if read_json(suite / "complete.json")["plan_sha256"] != file_hash(suite / "plan.json"):
        raise ValueError("Four-benchmark suite plan changed")
    plan = read_json(suite / "plan.json")
    verified_code(suite / "code", plan["code_sha256"])
    for name, expected in plan["files"].items():
        path = (suite / name).resolve()
        if not path.is_relative_to(suite.resolve()) or file_hash(path) != expected:
            raise ValueError(f"Frozen suite input/code changed: {name}")
    for group, saved in plan["runs"].items():
        run = Path(saved["path"])
        metadata = verify_run_inputs(run)
        if file_hash(run / "run.json") != saved["run_sha256"]:
            raise ValueError(f"Run identity changed: {group}")
        for epoch, expected in saved["checkpoints"].items():
            ckpt = checkpoint_path(run, int(epoch))
            if file_hash(ckpt / "complete.json") != expected:
                raise ValueError(f"Checkpoint changed: {group}/epoch-{epoch}")
            adapter_spec(ckpt, finetuning=metadata["config"]["finetuning"], base_model=metadata["model"])
    return plan


def prepare(experiment, panel_source, job, args):
    if (job / "suite").exists():
        plan = verify(job)
        if (plan["experiment"] != str(experiment) or plan["panel_source"] != str(panel_source)
                or plan["job_name"] != args.job_name
                or args.epochs and generation_epochs(args.epochs) != plan["epochs"]
                or args.image and args.image != plan["image_requested"]
                or args.judge_config and read_json(args.judge_config) != read_json(job / "suite/judge.json")):
            raise ValueError("Resume inputs differ; use the original command or a new --job-name")
        return plan
    if any(p.name != "pipeline.lock" and not p.name.startswith((".suite-partial-", "overlap-rejections-"))
           for p in job.iterdir()):
        raise ValueError("Nonempty output is not this suite; choose a new job name")
    training_plan = verify_bundle(experiment)
    epochs = generation_epochs(args.epochs or ",".join(map(str, DEFAULT_EPOCHS)))
    runs, training, metadata = {}, {}, {}
    for group in GROUPS:
        run = experiment / group
        info = verify_run_inputs(run)
        if (info.get("replay") or info["config"] != training_plan["config"] or info["model"] != training_plan["model"]
                or info["config"]["expected_train_size"] != 100 or info["config"]["epochs"] != 16
                or info["config"]["finetuning"]["base_quantization"] != "none"
                or info["data_files"]["train.jsonl"] != file_hash(experiment / "bundle/prepared" / group / "train.jsonl")):
            raise ValueError(f"Expected the frozen high/low BF16 LoRA experiment: {run}")
        if not set(epochs) <= set(evaluation_epochs(info)):
            raise ValueError(f"Requested epochs were not scheduled for {group}: {epochs}")
        markers = {}
        for epoch in epochs:
            ckpt = checkpoint_path(run, epoch)
            marker = read_json(ckpt / "complete.json")
            if marker["config_sha256"] != info["config_sha256"] or marker["step"] != epoch * 100:
                raise ValueError(f"Checkpoint identity mismatch: {group}, epoch={epoch}")
            adapter_spec(ckpt, finetuning=info["config"]["finetuning"], base_model=info["model"])
            markers[str(epoch)] = file_hash(ckpt / "complete.json")
        training[group] = list(iter_jsonl(run / "data/train.jsonl"))
        if len(training[group]) != 100:
            raise ValueError(f"Expected 100 training examples in {group}")
        metadata[group] = info
        runs[group] = {"path": str(run), "run_sha256": file_hash(run / "run.json"), "checkpoints": markers}
    if (metadata["high"]["data_files"]["panel.jsonl"] != metadata["low"]["data_files"]["panel.jsonl"]
            or metadata["high"]["tokenizer_files"] != metadata["low"]["tokenizer_files"]):
        raise ValueError("High and low must use exactly the same frozen math/GPQA panel and tokenizer")
    math_panel = list(iter_jsonl(experiment / "high/data/panel.jsonl"))
    partition_panel(math_panel)
    transfer_panel, _ = load_panel(panel_source)  # Old training hash is deliberately not reused.
    partition(transfer_panel)
    audits = overlap_checks(training, math_panel, transfer_panel)
    for group in GROUPS:
        audits[group]["training_sha256"] = metadata[group]["data_files"]["train.jsonl"]
    if any(a["matches"] for a in audits.values()):
        report = job / ("overlap-rejections-" + uuid.uuid4().hex + ".json")
        write_json(report, audits)
        raise ValueError(f"Training/evaluation overlap; stopped without replacing questions. Inspect {report}")
    import math_verify  # Validate the mathematical grader before any GPU work.
    image = args.image or DEFAULT_IMAGE
    judge_config = read_json(args.judge_config or ROOT / "configs/lcb_grading.json")
    judge = DockerJudge(image, judge_config)
    preflight = judge.preflight()
    selected_bytes = sum((panel_source / q["reference_file"]).stat().st_size for q in transfer_panel)
    require_free_space(job, 8 * selected_bytes + 4 * GIB)
    staged = job / (".suite-partial-" + uuid.uuid4().hex)
    staged.mkdir()
    code_sha = snapshot_code(staged / "code")
    for name in ("__init__.py", "evaluate_likelihood_extremes.py", "likelihood_training.py"):
        # These files are not included by the legacy root snapshot enumerator.
        target = staged / "code/dataset" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT / "dataset" / name, target)
    # likelihood_training imports this module; all launcher dependencies are frozen.
    shutil.copy2(ROOT / "dataset/likelihood_utils.py", staged / "code/dataset/likelihood_utils.py")
    write_json(staged / "judge.json", judge_config)
    write_json(staged / "overlap_checks.json", audits)
    for group in GROUPS:
        freeze_transfer(panel_source, staged / "panels" / group, metadata[group]["data_files"]["train.jsonl"], audits[group])
    plan = {"format_version": "likelihood-two-runs-four-benchmarks-v1", "created_at": now(),
            "experiment": str(experiment), "training_plan_identity": training_plan["identity"],
            "panel_source": str(panel_source), "source_panel_manifest_sha256": file_hash(panel_source / "manifest.json"),
            "job_name": args.job_name, "runs": runs, "epochs": epochs, "code_sha256": code_sha,
            "image_requested": image, "image": judge.image, "preflight": preflight,
            "routes": [{"group": group, "panel": panel, "mode": mode, "size": size, "group_size": group_size,
                        "path": str(experiment / group / "postprocess" / (args.job_name + "-" + panel))}
                       for group in GROUPS for panel, mode, size, group_size in PANELS],
            "questions_per_run": 392, "responses": 2 * 392 * len(epochs),
            "execution_order": "high math/GPQA, high BBH/LCB, low math/GPQA, low BBH/LCB; group then descending epoch",
            "files": tree_hashes(staged)}
    write_json(staged / "plan.json", plan)
    write_json(staged / "complete.json", {"plan_sha256": file_hash(staged / "plan.json")})
    staged.rename(job / "suite")
    return plan


def prepare_panels(job, plan, stop_check=lambda: None):
    prepared = []
    for route in plan["routes"]:
        stop_check()
        run, child = Path(plan["runs"][route["group"]]["path"]), Path(route["path"])
        child.mkdir(parents=True, exist_ok=True)
        math_mode = route["mode"] == "math"
        source = run / "data" if math_mode else job / "suite/panels" / route["group"]
        args = SimpleNamespace(mode=route["mode"], epochs=",".join(map(str, plan["epochs"])),
                               image=None if math_mode else plan["image"]["id"],
                               judge_config=None if math_mode else str(job / "suite/judge.json"))
        # Existing child setups retain their frozen image; avoid unnecessary Docker
        # preflight on a fully completed rerun, while verifying the pinned identity.
        if (child / "setup").exists():
            args.image = args.judge_config = None
        p = transfer_evaluation.prepare(run, source, child, args)
        if (p["generation_epochs"] != plan["epochs"] or
                not math_mode and p["image"] != plan["image"]):
            raise ValueError("Child checkpoint list or execution image differs from the suite")
        prepared.append(p)
    # Both runs must compare identical questions and generation/engine settings.
    for a, b in ((0, 2), (1, 3)):
        for key in ("panel_sha256", "panel_ids", "groups", "settings", "generation_backend"):
            if prepared[a][key] != prepared[b][key]:
                raise ValueError(f"High/low evaluation protocols differ: {key}")
        left, right = (Path(plan["routes"][i]["path"]) / "setup/vllm.json" for i in (a, b))
        if read_json(left) != read_json(right):
            raise ValueError("High/low vLLM engine configurations differ")
    return prepared


def overview(job, plan):
    body = '<!doctype html><meta charset="utf-8"><title>High / Low 四任务评测</title>'
    body += '<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:20px}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}</style>'
    body += '<h1>High / Low 四任务评测</h1><p>节点：' + escape(str(plan["epochs"]))
    body += '；每题一次采样，并发 8，上限 8192 新 token。首次答对仅指已观测节点。</p><ul>'
    combined = []
    for route in plan["routes"]:
        child = Path(route["path"])
        relative = os.path.relpath(child, job).replace(os.sep, "/")
        done = (child / "complete.json").exists()
        body += f'<li>{escape(route["group"])} / {escape(route["panel"])}：'
        body += f'<a href="{escape(relative)}/report/report.html">完整报告</a>' if done else '尚未全部完成'
        body += '<ul>'
        for i in range(1, 5):
            group = child / "groups" / f"group-{i:03d}"
            if (group / "complete.json").exists():
                body += f'<li><a href="{escape(relative)}/groups/group-{i:03d}/report/report.html">第 {i} 组报告</a></li>'
        body += '</ul></li>'
        if done:
            with (child / "report/curves.csv").open(encoding="utf-8", newline="") as stream:
                combined.extend({"training_group": route["group"], "panel": route["panel"], **row}
                                for row in csv.DictReader(stream))
    body += '</ul><p><a href="curves.csv">已完成面板的 high/low 汇总 CSV</a>；原始回答、逐题判分和各组报告保存在上述面板目录。</p>'
    if combined:
        body += '<table><tr><th>训练组</th><th>Epoch</th><th>任务</th><th>题数</th><th>正确率</th><th>平均长度</th><th>正常结束率</th><th>截断率</th><th>提取率</th></tr>'
        for row in combined:
            if row["subset"] == "all":
                values = [row[k] for k in ("training_group", "epoch", "task", "n", "accuracy", "mean_generated_tokens",
                                          "eos_end_rate", "length_truncated_rate", "extractable_rate")]
                body += '<tr>' + ''.join(f'<td>{escape(str(v))}</td>' for v in values) + '</tr>'
        body += '</table>'
    (job / "report.html").write_text(body, encoding="utf-8")
    if combined:
        with (job / "curves.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(combined[0]))
            writer.writeheader()
            writer.writerows(combined)
    write_json(job / "summary.json", {"epochs": plan["epochs"], "expected_responses": plan["responses"],
               "completed_panels": [f"{r['group']}/{r['panel']}" for r in plan["routes"] if (Path(r["path"]) / "complete.json").exists()],
               "curves": combined})


def execute(job, worker):
    plan = verify(job)
    prepare_panels(job, plan, worker.check_stop)  # All four setups checked before any GPU generation.
    overview(job, plan)
    for i, route in enumerate(plan["routes"], 1):
        worker.check_stop()
        child = Path(route["path"])
        write_json(job / "progress.json", {"stage": "evaluation", "panel_index": i, **route, "epochs": plan["epochs"]})
        print(f"PANEL {i}/4: {route['group']} / {route['panel']}; four groups of {route['group_size']}; epochs={plan['epochs']}", flush=True)
        code = child / "setup/code"
        worker.invoke([sys.executable, "-B", "-u", code / "transfer_evaluation.py", "--worker", child], code)
        if not (child / "complete.json").is_file():
            raise RuntimeError(f"Panel did not complete: {child}")
        overview(job, plan)
    write_json(job / "complete.json", {"completed_at": now(), "responses": plan["responses"], "epochs": plan["epochs"],
               "suite_plan_sha256": file_hash(job / "suite/plan.json"),
               "panel_completions": {r["path"]: file_hash(Path(r["path"]) / "complete.json") for r in plan["routes"]}})
    write_json(job / "progress.json", {"stage": "complete", "panels_complete": 4})
    print(f"HIGH + LOW DONE: {job / 'report.html'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", default="runs/likelihood-100x16-s42")
    p.add_argument("--panel-dir", default="artifacts/transfer-s2026")
    p.add_argument("--job-name", default="four-benchmarks-vllm-v1")
    p.add_argument("--epochs", help="Default 16,14,12,10,8,4,2,1; add 0 explicitly for Base control")
    p.add_argument("--image")
    p.add_argument("--judge-config")
    p.add_argument("--prepare-only", action="store_true", help="Check checkpoints/panels/Docker and freeze all four jobs; no GPU")
    p.add_argument("--worker", help=argparse.SUPPRESS)
    args = p.parse_args()
    configure_runtime()
    if args.worker:
        job = Path(args.worker).resolve()
        with Pipeline(job, inherited=True) as worker:
            if args.prepare_only:
                plan = verify(job)
                prepare_panels(job, plan, worker.check_stop)
                overview(job, plan)
                print("All four panel jobs prepared; repeat without --prepare-only to generate", flush=True)
            else:
                try:
                    execute(job, worker)
                except BaseException as exc:
                    progress = read_json(job / "progress.json") if (job / "progress.json").exists() else {}
                    progress.update(stage="paused" if isinstance(exc, (SystemExit, KeyboardInterrupt)) else "failed", error=str(exc))
                    write_json(job / "progress.json", progress)
                    raise
        return
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.job_name):
        p.error("Invalid job name")
    experiment, panel_source = artifact_path(args.experiment), artifact_path(args.panel_dir)
    job = (experiment / "postprocess" / args.job_name).resolve()
    if job.parent != (experiment / "postprocess").resolve():
        p.error("Job path escapes experiment/postprocess")
    with Pipeline(job) as worker:
        prepare(experiment, panel_source, job, args)
        worker.check_stop()
        code = job / "suite/code"
        worker.invoke([sys.executable, "-B", "-u", code / "dataset/evaluate_likelihood_extremes.py", "--worker", job,
                       *(["--prepare-only"] if args.prepare_only else [])], code)


if __name__ == "__main__":
    main()
