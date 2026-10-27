"""Single-GPU, full-parameter BF16 SFT: one example and one update per step."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import signal
import uuid

from recipe.common import (append_jsonl, capture_rng, digest, environment, epoch_order, file_hash,
                           fresh_dir, iter_jsonl, lr_for_update, now, read_json, restore_rng,
                           seed_everything, snapshot_code, validate_prepared, write_json, write_jsonl)
from recipe.checkpoints import latest_recovery, prune_recovery, save_checkpoint
from recipe.encoding import encode_example, inspect_example, pad_batch
from recipe.modeling import load_model, training_loss
from recipe.paths import artifact_path, configure_runtime
from recipe.progress import TrainingProgress


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", help="Prepared dataset directory (required for a fresh run)")
    p.add_argument("--subset", help="Selected training JSONL; default DATA/train.jsonl")
    p.add_argument("--config", default="configs/baseline.json")
    p.add_argument("--run", required=True)
    p.add_argument("--resume", nargs="?", const="latest", help="latest or checkpoint path")
    p.add_argument("--seed", type=int, help="Override training seed only for a fresh run")
    p.add_argument("--stop-after", type=int, help="Pause and save after this absolute update number")
    p.add_argument("--simulate-crash-at", type=int, help=argparse.SUPPRESS)
    args = p.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use one process/GPU: distributed execution would change batch/update semantics")
    configure_runtime()
    import torch
    import bitsandbytes as bnb
    from transformers import AutoTokenizer
    run = artifact_path(args.run)
    print(f"Run directory: {run}", flush=True)
    attempt = now().replace(":", "-") + "-" + uuid.uuid4().hex[:8]

    if args.resume:
        if args.seed is not None or args.subset or args.data:
            p.error("Resume uses frozen run data/config; do not pass --seed, --subset or --data")
        metadata = read_json(run / "run.json")
        config = metadata["config"]
        for name, expected in metadata["data_files"].items():
            if file_hash(run / "data" / name) != expected:
                raise ValueError(f"Run input modified: {name}")
        source = latest_recovery(run) if args.resume == "latest" else artifact_path(args.resume)
        if not source.is_relative_to(run):
            raise ValueError("Recovery checkpoint must belong to this run")
        if not (source / "complete.json").exists():
            raise ValueError("Cannot resume an incomplete checkpoint")
        model_revision = None
    else:
        if not args.data:
            p.error("--data is required for a fresh run")
        prepared = artifact_path(args.data)
        if args.subset:
            args.subset = artifact_path(args.subset)
        manifest = validate_prepared(prepared)
        config = read_json(args.config)
        if args.seed is not None:
            config["seed"] = args.seed
        if config["epochs"] < 1 or config["recovery_every_steps"] < 1 or config["keep_recovery"] < 1:
            raise ValueError("Epochs and recovery settings must be positive")
        if 0 not in config["checkpoint_epochs"] or config["epochs"] not in config["checkpoint_epochs"]:
            raise ValueError("Milestones must include epoch 0 and the final epoch")
        run = fresh_dir(run)
        (run / "data").mkdir()
        for name in ("holdout.jsonl", "panel.jsonl", "overlap_references.jsonl", "manifest.json", "chat_template.jinja"):
            shutil.copy2(prepared / name, run / "data" / name)
        shutil.copy2(args.subset or prepared / "train.jsonl", run / "data" / "train.jsonl")
        if args.subset and (Path(args.subset).parent / "selection.json").exists():
            shutil.copy2(Path(args.subset).parent / "selection.json", run / "data" / "selection.json")
        shutil.copytree(prepared / "tokenizer", run / "tokenizer")
        data_files = {x.name: file_hash(x) for x in (run / "data").iterdir()}
        metadata = {"config": config, "prepared_manifest_sha256": file_hash(prepared / "manifest.json"),
                    "data_files": data_files, "model": manifest["model"],
                    "max_length": manifest["arguments"]["max_length"],
                    "subset_source": str(Path(args.subset).resolve()) if args.subset else str(prepared / "train.jsonl"),
                    "semantics": {"batch_size": 1, "gradient_accumulation_steps": 1, "dtype": "bfloat16",
                                  "full_parameter": True, "optimizer": "bitsandbytes.AdamW8bit",
                                  "loss": "mean over assistant next-token targets including im_end"}}
        source, model_revision = metadata["model"]["id"], metadata["model"]["revision"]

    train_rows = list(iter_jsonl(run / "data" / "train.jsonl"))
    if len(train_rows) != config["expected_train_size"]:
        raise ValueError(f"Expected {config['expected_train_size']} examples, got {len(train_rows)}")
    if len({r["question_id"] for r in train_rows}) != len(train_rows):
        raise ValueError("Duplicate training questions")
    from recipe.data import OverlapIndex
    forbidden = list(iter_jsonl(run / "data" / "holdout.jsonl")) + list(iter_jsonl(run / "data" / "overlap_references.jsonl"))
    overlap = OverlapIndex(forbidden)
    for row in train_rows:
        if overlap.match(row):
            raise ValueError(f"Train/evaluation overlap: {row['sample_id']}")
    tokenizer = AutoTokenizer.from_pretrained(run / "tokenizer", local_files_only=True)
    encoded = [encode_example(tokenizer, r, metadata["max_length"]) for r in train_rows]
    if any(e is None for e in encoded):
        raise ValueError("Subset contains an over-length sample; do not truncate")
    if not args.resume:
        write_jsonl(run / "data" / "inspection.jsonl", [inspect_example(tokenizer, r, e) for r, e in zip(train_rows[:3], encoded[:3])])
        metadata["data_files"]["inspection.jsonl"] = file_hash(run / "data" / "inspection.jsonl")
        metadata["tokenizer_files"] = {x.name: file_hash(x) for x in (run / "tokenizer").iterdir() if x.is_file()}
    for name, expected in metadata["tokenizer_files"].items():
        if file_hash(run / "tokenizer" / name) != expected:
            raise ValueError("Frozen tokenizer changed")
    identity = digest({k: metadata[k] for k in ("config", "model", "data_files", "tokenizer_files", "semantics")})
    code_hash = snapshot_code(run / "attempts" / attempt / "code")
    if args.resume and code_hash != metadata["code_sha256"]:
        raise ValueError("Training code differs from the original run. Use the run's code snapshot to resume.")
    metadata.update(config_sha256=identity, code_sha256=code_hash)
    if not args.resume:
        write_json(run / "run.json", metadata)
    write_json(run / "attempts" / attempt / "environment.json", environment())
    seed_everything(config["seed"], config["deterministic"])
    model = load_model(source, revision=model_revision, attention=config["attention_implementation"], training=True)
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=config["learning_rate"],
                                  betas=tuple(config["betas"]), eps=config["adam_epsilon"],
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda completed: lr_for_update(completed + 1, 1.0, config["warmup_steps"]))
    size, step = len(train_rows), 0
    if args.resume:
        # Only load trusted self-produced pickle state, never an external .pt.
        state = torch.load(source / "training_state.pt", map_location="cpu", weights_only=False)
        if state["config_sha256"] != identity:
            raise ValueError("Recovery config/data fingerprint mismatch")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = state["step"]
        expected_cursor = {"epoch_index": step // size, "position": step % size,
                           "order": epoch_order(size, config["seed"], step // size)}
        if state["cursor"] != expected_cursor:
            raise ValueError("Recovery shuffle/data cursor mismatch")
        restore_rng(state["rng"])
        del state
    else:
        save_checkpoint(run / "checkpoints" / "epoch-000", model, tokenizer, 0, identity,
                        optimizer, scheduler, {"epoch_index": 0, "position": 0,
                                               "order": epoch_order(size, config["seed"], 0)})
    total = size * config["epochs"]
    if args.stop_after is not None and args.stop_after <= step:
        raise ValueError("--stop-after must be greater than the restored step")
    stop_requested = False

    def on_signal(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"Signal {signum}: finishing this update and saving recovery", flush=True)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    log = run / "logs" / (attempt + ".jsonl")
    append_jsonl(run / "attempts.jsonl", {"attempt": attempt, "start_step": step,
                                         "restored_from": str(source) if args.resume else None, "timestamp": now()})
    write_json(run / "attempts" / attempt / "hardware.json", {
        "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
        "bf16_parameters": sum(p.numel() for p in model.parameters() if p.dtype == torch.bfloat16),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})

    def save_progress(path):
        return save_checkpoint(path, model, tokenizer, step, identity, optimizer, scheduler,
                               {"epoch_index": step // size, "position": step % size,
                                "order": epoch_order(size, config["seed"], step // size)})

    with TrainingProgress(run, attempt, size, total, step) as progress:
        while step < total:
            epoch, position = divmod(step, size)
            index = epoch_order(size, config["seed"], epoch)[position]
            row, example = train_rows[index], encoded[index]
            batch = {k: v.to("cuda") for k, v in pad_batch([example], tokenizer.pad_token_id).items()}
            optimizer.zero_grad(set_to_none=True)
            lr = optimizer.param_groups[0]["lr"]
            expected_lr = lr_for_update(step + 1, config["learning_rate"], config["warmup_steps"])
            if abs(lr - expected_lr) > 1e-14:
                raise ValueError(f"Scheduler mismatch: {lr} != {expected_lr}")
            loss = training_loss(model, batch, config["logit_chunk_size"])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite loss before update {step + 1}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            record = {"step": step, "epoch": epoch + 1, "epoch_fraction": step / size,
                      "position_in_epoch": position, "sample_id": row["sample_id"], "question_id": row["question_id"],
                      "loss": loss.item(), "learning_rate": lr, "grad_norm_before_clip": grad_norm.item(),
                      "n_tokens": len(example["input_ids"]), "n_supervised": example["supervised_tokens"],
                      "attempt": attempt, "timestamp": now()}
            append_jsonl(log, record)
            progress.write(f"step={step}/{total} epoch={step / size:.4f} loss={record['loss']:.6f} lr={lr:.3g}")
            del loss, batch
            # Deliberate hard process exit lets smoke_test verify replay from a previous
            # checkpoint, including already-logged but uncommitted updates.
            if args.simulate_crash_at == step:
                os._exit(86)
            if step % size == 0 and step // size in config["checkpoint_epochs"]:
                save_progress(run / "checkpoints" / f"epoch-{step // size:03d}")
            pause = stop_requested or (args.stop_after is not None and step >= args.stop_after)
            if (step % config["recovery_every_steps"] == 0 or pause) and not (
                step % size == 0 and step // size in config["checkpoint_epochs"]
            ):
                save_progress(run / "recovery" / f"step-{step:08d}")
                prune_recovery(run, config["keep_recovery"])
            timing = progress.update(step, paused=pause)
            write_json(run / "progress.json", {"step": step, "total_steps": total, "epoch": step / size,
                                                "status": "complete" if step == total else "paused" if pause else "running",
                                                "timing": timing})
            if pause:
                break


if __name__ == "__main__":
    main()
