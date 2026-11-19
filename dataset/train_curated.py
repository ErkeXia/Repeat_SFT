"""One command: Base likelihood, then ten edited answers x 128 epochs of fresh LoRA."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recipe.adapters import adapter_spec
from recipe.common import append_jsonl, digest, file_hash, iter_jsonl, now, read_json, recover_jsonl, write_json
from recipe.paths import artifact_path, configure_runtime, storage_root
from recipe.pipeline import Pipeline
from dataset.curated_training import DEFAULT_CURATION, EPOCHS, SIZE, prepare, verify_bundle
from dataset.likelihood_utils import load_scores
from dataset.train_likelihood_extremes import check_run, command_for


def scoring_complete(out, plan):
    scores = out / "base_likelihood"
    if not (scores / "complete.json").exists():
        return False
    job, complete = read_json(scores / "job.json"), read_json(scores / "complete.json")
    if (job["identity"] != digest(job["spec"]) or job["spec"]["bundle_identity"] != plan["identity"]
            or job["spec"]["model"] != plan["model"] or job["spec"]["scoring"] != plan["scoring"]
            or complete["job_identity"] != job["identity"] or complete["samples"] != SIZE
            or complete["summary_sha256"] != file_hash(scores / "summary.json")
            or complete["records"] != {p.name: file_hash(p) for p in (scores / "records").glob("*.json")}):
        raise ValueError("Base score identity/completion artifacts changed")
    index = list(iter_jsonl(out / "bundle/index.jsonl"))
    if len(load_scores(scores, job, index)) != SIZE:
        raise ValueError("All ten Base likelihoods are required before training")
    return True


def training_complete(out, plan):
    run = out / "train"
    if not (run / "run.json").exists():
        return False
    meta = check_run(out, "train", plan)
    if not (run / "checkpoints/epoch-128/complete.json").exists():
        return False
    for epoch in EPOCHS:
        path = run / "checkpoints" / f"epoch-{epoch:03d}"
        marker = read_json(path / "complete.json")
        if (marker["step"] != epoch * SIZE or marker["config_sha256"] != meta["config_sha256"]
                or not marker["has_training_state"] or not (path / "training_state.pt").is_file()):
            raise ValueError(f"Missing/wrong complete checkpoint: {path}")
        adapter_spec(path, finetuning=plan["config"]["finetuning"], base_model=plan["model"])
    return True


def execute(out, plan, pipeline, score_only=False, stop_after=None):
    recover_jsonl(out / "commands.jsonl")
    code = out / "bundle/code"

    def stage(name, command):
        write_json(out / "progress.json", {"status": "running", "stage": name, "job_identity": plan["identity"]})
        append_jsonl(out / "commands.jsonl", {"timestamp": now(), "job_identity": plan["identity"],
                     "group": "train" if name == "training" else None, "stage": name, "command": list(map(str, command))})
        pipeline.invoke(command, code)

    pipeline.check_stop()
    if not scoring_complete(out, plan):
        python = plan["request"]["vllm_python"]
        if not Path(python).is_file():
            raise FileNotFoundError(f"Missing vLLM interpreter: {python}")
        print("STAGE 1/2: unadapted BF16 Base likelihood for ten edited answers (vLLM, concurrency=8)", flush=True)
        stage("base_likelihood", [python, "-B", "-u", code / "dataset/score_curated.py",
                                 "--experiment", out, "--inherited-lock"])
        if not scoring_complete(out, plan):
            raise RuntimeError("Base scoring did not finish; training has not started")
    else:
        print("SKIP BASE SCORE: ten committed scores verified", flush=True)
    pipeline.check_stop()
    if score_only:
        write_json(out / "progress.json", {"status": "scored", "stage": "base_likelihood",
                                           "job_identity": plan["identity"]})
        return False
    if not training_complete(out, plan):
        command = command_for(out, "train", plan)
        if stop_after is not None:
            command += ["--stop-after", str(stop_after)]
        print("STAGE 2/2: fresh Base LoRA; 10 samples x 128 epochs = 1280 updates", flush=True)
        stage("training", command)
        if not training_complete(out, plan):
            write_json(out / "progress.json", {"status": "paused", "stage": "training",
                                               "job_identity": plan["identity"]})
            print("Paused after saving recovery; repeat the launcher (without --stop-after) to resume", flush=True)
            return False
    else:
        print("SKIP TRAIN: all nine requested milestones verified", flush=True)
    write_json(out / "complete.json", {"job_identity": plan["identity"], "epochs": EPOCHS,
               "updates": SIZE * 128, "score_complete_sha256": file_hash(out / "base_likelihood/complete.json"),
               "checkpoint_markers": {str(e): file_hash(out / "train/checkpoints" / f"epoch-{e:03d}/complete.json")
                                      for e in EPOCHS}})
    write_json(out / "progress.json", {"status": "complete", "stage": "training", "job_identity": plan["identity"]})
    print(f"Complete: {out / 'train'}", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curation", default=str(DEFAULT_CURATION), help="Source text directory in this Git checkout")
    parser.add_argument("--data", default="artifacts/data-s42", help="Original prepared tokenizer, Base revision and eval splits")
    parser.add_argument("--config", default=str(ROOT / "configs/baseline.json"))
    parser.add_argument("--out", default="runs/curated-10x128-s42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vllm-python", help="Default: VLLM_PYTHON_BIN or /data/repeat_sft/venvs/vllm-0.11.0/bin/python")
    parser.add_argument("--prepare-only", action="store_true", help="Freeze/inspect ten samples; do not load weights")
    parser.add_argument("--score-only", action="store_true", help="Finish Base likelihood, then pause before training")
    parser.add_argument("--stop-after", type=int, help="Pause training after this absolute update, e.g. 20")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stop_after is not None and not 1 <= args.stop_after <= SIZE * 128:
        parser.error("--stop-after must be between 1 and 1280")
    if os.name != "posix":
        raise RuntimeError("Run this experiment on the Linux VM; offline unit tests require no model")
    configure_runtime()
    out = Path(args.worker).resolve() if args.worker else artifact_path(args.out)
    with Pipeline(out, inherited=bool(args.worker)) as pipeline:
        try:
            if args.worker:
                execute(out, verify_bundle(out), pipeline, args.score_only, args.stop_after)
                return
            python = args.vllm_python or os.environ.get("VLLM_PYTHON_BIN") or str(storage_root() / "venvs/vllm-0.11.0/bin/python")
            # Do NOT resolve the interpreter symlink: that would leave its virtualenv.
            request = {"curation": str(Path(args.curation).resolve()), "data": str(artifact_path(args.data)),
                       "config": str(Path(args.config).resolve()), "seed": args.seed,
                       "vllm_python": str(Path(python).expanduser().absolute())}
            if (out / "bundle").exists():
                plan = verify_bundle(out)
                if request != plan["request"]:
                    raise ValueError("Different inputs/seed/interpreter paths; repeat the original command or use a new --out")
                print("Using frozen experiment inputs and code; later edits to source texts do not change this run", flush=True)
            else:
                if not args.prepare_only and not Path(request["vllm_python"]).is_file():
                    raise FileNotFoundError(f"Missing vLLM interpreter: {request['vllm_python']}")
                plan = prepare(out, request, pipeline.check_stop)
            print(f"Experiment: {out}\n10 samples; 128 epochs; 1280 updates; checkpoints={EPOCHS}\n"
                  f"LR={plan['config']['learning_rate']}; warmup={plan['config']['warmup_steps']} updates; "
                  f"LoRA rank={plan['config']['finetuning']['rank']}", flush=True)
            pipeline.check_stop()
            if args.prepare_only:
                print(f"Ready: {out / 'bundle/prepared/train/inspection.jsonl'}", flush=True)
                return
            code = out / "bundle/code"
            command = [sys.executable, "-B", "-u", code / "dataset/train_curated.py", "--worker", out]
            if args.score_only:
                command.append("--score-only")
            if args.stop_after is not None:
                command += ["--stop-after", str(args.stop_after)]
            pipeline.invoke(command, code)
        except BaseException as exc:
            progress = read_json(out / "progress.json") if (out / "progress.json").exists() else {}
            progress.update(status="paused" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                            stopped=now(), error=str(exc))
            write_json(out / "progress.json", progress)
            raise


if __name__ == "__main__":
    main()
