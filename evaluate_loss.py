"""SERVER ONLY: per-sample NLL and teacher-forced token accuracy at milestones."""
import argparse
import gc

from recipe.common import (checkpoint_path, environment, file_hash, iter_jsonl, read_json,
                           seed_everything, snapshot_code, write_json)
from recipe.evaluation import bind_output, evaluate_rows
from recipe.modeling import load_model
from recipe.paths import artifact_path, configure_runtime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--epochs", default="0,1,2,4,8,16,32,64")
    p.add_argument("--out", help="Defaults to RUN/evaluations/teacher_forcing")
    p.add_argument("--chunk-size", type=int, help="Defaults to the run's logit_chunk_size")
    p.add_argument("--limit-per-split", type=int, help="Smoke only; formal evaluation uses every example")
    args = p.parse_args()
    configure_runtime()
    import torch
    from transformers import AutoTokenizer
    run = artifact_path(args.run)
    metadata = read_json(run / "run.json")
    if args.chunk_size is None:
        args.chunk_size = metadata["config"].get("logit_chunk_size", 128)
    tokenizer = AutoTokenizer.from_pretrained(run / "tokenizer", local_files_only=True)
    base = artifact_path(args.out) if args.out else run / "evaluations" / "teacher_forcing"
    seed_everything(metadata["config"]["seed"])
    if args.chunk_size < 1 or (args.limit_per_split is not None and args.limit_per_split < 1):
        raise ValueError("Chunk size and sample limits must be positive")
    rows_by_split = {s: list(iter_jsonl(run / "data" / f"{s}.jsonl")) for s in ("train", "holdout")}
    if args.limit_per_split:
        rows_by_split = {s: rows[:args.limit_per_split] for s, rows in rows_by_split.items()}
    for epoch in [int(x) for x in args.epochs.split(",")]:
        ckpt = checkpoint_path(run, epoch)
        out = base / f"epoch-{epoch:03d}"
        bind_output(out, {"run_sha256": file_hash(run / "run.json"), "epoch": epoch,
                          "checkpoint": read_json(ckpt / "complete.json"), "chunk_size": args.chunk_size,
                          "selected_ids": {s: [r["sample_id"] for r in rows] for s, rows in rows_by_split.items()},
                          "data_sha256": {s: file_hash(run / "data" / f"{s}.jsonl") for s in ("train", "holdout")}})
        if not (out / "code").exists():
            snapshot_code(out / "code")
            write_json(out / "environment.json", environment())
        model = load_model(ckpt, attention=metadata["config"]["attention_implementation"],
                           finetuning=metadata["config"].get("finetuning"),
                           base_model=metadata["model"] if "finetuning" in metadata["config"] else None)
        for split in ("train", "holdout"):
            evaluate_rows(model, tokenizer, rows_by_split[split],
                          out / split, metadata["max_length"], args.chunk_size)
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
