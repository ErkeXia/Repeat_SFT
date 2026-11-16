"""Full-corpus likelihood artifacts; GPU-free helpers shared by scoring and reports."""
from __future__ import annotations

import math
import os
from pathlib import Path
import shutil
import uuid

from recipe.common import digest, file_hash, read_json, write_json
from recipe.data import adapt_sft
from recipe.encoding import encode_example

DATASET = "jasonrqh/Math-CoT-20k"
DATA_REVISION = "1435fb21d4fecc8ad4966a26f22a874cf2b527f1"
DATA_FILE = "Math-CoT-20k.parquet"
DATA_SHA256 = "3166f342ec838ebe0979be43923abb33fa1c80f33043aa6acc04e5d82161142a"
DATA_ROWS = 20480
PROTOCOL = "math-cot-full-base-likelihood-v1"


def copy_atomic(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp-" + uuid.uuid4().hex)
    shutil.copyfile(source, temp)
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temp, target)


def iter_rows(path, limit=None):
    import pyarrow.parquet as pq
    index = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
        for original in batch.to_pylist():
            if limit is not None and index >= limit:
                return
            yield adapt_sft(original, index)
            index += 1


def encode_full(tokenizer, row, context_length):
    # vLLM needs room for one discarded generated token. Never apply the SFT
    # 10,240-token eligibility filter or the free-generation 8,192-token cap.
    encoded = encode_example(tokenizer, row, max_length=context_length - 1)
    if encoded is None:
        raise ValueError(f"row_index={row['row_index']} exceeds context {context_length}-1; "
                         "no sample was truncated or silently excluded")
    if encoded["prompt_length"] < 1 or encoded["supervised_tokens"] < 1:
        raise ValueError("Invalid assistant supervision span")
    return encoded


def describe(row, encoded):
    return {"row_index": row["row_index"], "sample_id": row["sample_id"],
            "question_id": row["question_id"], "source": row["source"],
            "publisher_response_length": row["original_metadata"].get("response_length"),
            "total_tokens": len(encoded["input_ids"]), "prompt_tokens": encoded["prompt_length"],
            "assistant_end": encoded["assistant_end"], "n_tokens": encoded["supervised_tokens"],
            "input_sha256": digest(encoded["input_ids"])}


def extract_logprobs(output, encoded):
    """vLLM prompt_logprobs[i] predicts input_ids[i] from ids[:i], NOT i+1."""
    ids = encoded["input_ids"]
    if not output.finished or list(output.prompt_token_ids) != ids:
        raise ValueError("vLLM returned an unfinished request or different input tokens")
    values = output.prompt_logprobs
    if values is None or len(values) != len(ids):
        raise ValueError("Missing prompt logprobs; prefix caching must be disabled")
    result = []
    for i in range(encoded["prompt_length"], encoded["assistant_end"] + 1):
        if values[i] is None or ids[i] not in values[i]:
            raise ValueError(f"Missing reference-token logprob at position {i}")
        value = float(values[i][ids[i]].logprob)
        if not math.isfinite(value) or value > 0:
            raise ValueError(f"Invalid reference-token logprob at position {i}: {value}")
        result.append(value)
    return result


def metrics(logprobs):
    if not len(logprobs) or any(not math.isfinite(float(x)) or x > 0 for x in logprobs):
        raise ValueError("Expected nonempty, finite, nonpositive log probabilities")
    total = math.fsum(float(x) for x in logprobs)
    mean = total / len(logprobs)
    return {"sum_logprob": total, "mean_logprob": mean, "mean_nll": -mean,
            "normalized_likelihood": math.exp(mean),
            "sequence_log10_likelihood": total / math.log(10),
            "perplexity": math.exp(-mean) if -mean < 709 else None,
            "end_token_logprob": float(logprobs[-1])}


def save_score(out, identity, description, encoded, logprobs):
    """Token payload first, JSON commit last. An uncommitted payload is retryable."""
    import numpy as np
    out = Path(out)
    name = f"row-{description['row_index']:05d}"
    record = out / "records" / (name + ".json")
    if record.exists():
        raise FileExistsError(f"Refusing to overwrite a committed score: {record}")
    payload = out / "tokens" / (name + ".npz")
    payload.parent.mkdir(parents=True, exist_ok=True)
    temp = payload.with_name(payload.name + ".tmp-" + uuid.uuid4().hex)
    # Use the same float64 values in persisted tokens and aggregate calculations.
    logprobs = np.asarray(logprobs, dtype=np.float64)
    if logprobs.shape != (description["n_tokens"],):
        raise ValueError("Wrong number of assistant log probabilities")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, input_ids=np.asarray(encoded["input_ids"], dtype=np.int32),
                            assistant_start=np.int64(encoded["prompt_length"]),
                            assistant_end=np.int64(encoded["assistant_end"]), logprobs=logprobs)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, payload)
    result = {**description, **metrics(logprobs), "job_identity": identity,
              "token_file": payload.relative_to(out).as_posix(), "token_sha256": file_hash(payload)}
    write_json(record, result)
    return result


def load_scores(out, job, index, verify_tokens=True):
    """Original row is the unique key, even if future data have duplicate text."""
    import numpy as np
    out = Path(out)
    expected = {x["row_index"]: x for x in index}
    if len(expected) != len(index):
        raise ValueError("Duplicate original row indices in prepared index")
    scores = {}
    for path in sorted((out / "records").glob("*.json")):
        score = read_json(path)
        row = score["row_index"]
        if row not in expected or row in scores or path.name != f"row-{row:05d}.json":
            raise ValueError(f"Unexpected/duplicate result: {path}")
        if score["job_identity"] != job["identity"]:
            raise ValueError(f"Result belongs to a different experiment: {path}")
        for key, value in expected[row].items():
            if score.get(key) != value:
                raise ValueError(f"Result/index mismatch: row={row}, field={key}")
        if score["token_file"] != f"tokens/row-{row:05d}.npz":
            raise ValueError("Unexpected token artifact path")
        if verify_tokens:
            payload = out / score["token_file"]
            if file_hash(payload) != score["token_sha256"]:
                raise ValueError(f"Damaged token payload: {payload}; restore it before resuming")
            with np.load(payload, allow_pickle=False) as tokens:
                if (digest(tokens["input_ids"].tolist()) != score["input_sha256"]
                        or int(tokens["assistant_start"]) != score["prompt_tokens"]
                        or int(tokens["assistant_end"]) != score["assistant_end"]
                        or tokens["logprobs"].shape != (score["n_tokens"],)):
                    raise ValueError(f"Token alignment mismatch: row={row}")
                for key, value in metrics(tokens["logprobs"]).items():
                    if score[key] != value:
                        raise ValueError(f"Metric/payload mismatch: row={row}, field={key}")
        scores[row] = score
    return scores
