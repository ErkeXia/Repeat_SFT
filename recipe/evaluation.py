"""Resumable per-example teacher-forcing evaluation."""
from __future__ import annotations

from pathlib import Path

from .common import append_jsonl, code_fingerprint, digest, read_json, recover_jsonl, write_json
from .encoding import encode_example, pad_batch
from .modeling import teacher_forcing


def bind_output(out, identity):
    identity = {**identity, "code_sha256": code_fingerprint()}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    metadata = out / "identity.json"
    if metadata.exists():
        if read_json(metadata) != identity:
            raise ValueError(f"Output belongs to a different input/config/code revision: {out}")
    else:
        if list(out.iterdir()):
            raise ValueError(f"Nonempty output without identity: {out}")
        write_json(metadata, identity)


def evaluate_rows(model, tokenizer, rows, out, max_length=10240, chunk_size=128, save_tokens=True):
    import numpy as np
    import torch
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if save_tokens:
        (out / "tokens").mkdir(exist_ok=True)
    log = out / "samples.jsonl"
    previous = recover_jsonl(log)
    done = {r["sample_id"] for r in previous}
    wanted = {r["sample_id"] for r in rows}
    if len(done) != len(previous) or not done.issubset(wanted):
        raise ValueError("Duplicate or unexpected sample in score file")
    model.eval()
    for row in rows:
        if row["sample_id"] in done:
            continue
        encoded = encode_example(tokenizer, row, max_length)
        if encoded is None:
            raise ValueError("Evaluation sample exceeds prepared length")
        batch = {k: v.to(next(model.parameters()).device) for k, v in pad_batch([encoded], tokenizer.pad_token_id).items()}
        result = teacher_forcing(model, batch, chunk_size)
        metrics = {k: result[k] for k in ("mean_loss", "nll_sum", "token_accuracy", "correct_tokens", "n_tokens")}
        if save_tokens:
            name = "tokens/" + row["sample_id"] + ".npz"
            np.savez_compressed(out / name, **{k: result[k] for k in ("positions", "nll", "predicted_ids", "target_ids")})
            metrics["token_file"] = name
        end_index = list(result["positions"]).index(encoded["assistant_end"])
        metrics["end_token_loss"] = float(result["nll"][end_index])
        metrics["end_token_correct"] = bool(result["predicted_ids"][end_index] == result["target_ids"][end_index])
        record = {"sample_id": row["sample_id"], "question_id": row["question_id"], "source": row["source"], **metrics}
        append_jsonl(log, record)
        previous.append(record)
        print(f"teacher-forcing {len(previous)}/{len(rows)} loss={metrics['mean_loss']:.6f}", flush=True)
    count = sum(r["n_tokens"] for r in previous)
    summary = {"samples": len(previous), "supervised_tokens": count,
               "sample_mean_loss": sum(r["mean_loss"] for r in previous) / len(previous),
               "token_weighted_loss": sum(r["nll_sum"] for r in previous) / count,
               "sample_mean_token_accuracy": sum(r["token_accuracy"] for r in previous) / len(previous),
               "token_weighted_accuracy": sum(r["correct_tokens"] for r in previous) / count,
               "end_token_accuracy": sum(r["end_token_correct"] for r in previous) / len(previous)}
    write_json(out / "summary.json", summary)
    return summary
