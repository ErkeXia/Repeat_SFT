"""Selection and immutable inputs for two independent likelihood-extreme LoRA runs."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import os
from pathlib import Path
import shutil
import uuid

from recipe.adapters import validate_finetuning
from recipe.common import (ROOT, digest, environment, file_hash, iter_jsonl, now, read_json,
                           snapshot_code, validate_prepared, write_json, write_jsonl)
from recipe.data import OverlapIndex
from recipe.encoding import encode_example, inspect_example
from recipe.storage import GIB, require_free_space
from dataset.likelihood_utils import DATA_ROWS, DATA_SHA256, describe, load_scores

EPOCHS = [0, 1, 2, 4, 8, 10, 12, 14, 16]
GROUPS = ("high", "low")
SIZE = 100


def tree_hashes(path):
    return {p.relative_to(path).as_posix(): file_hash(p)
            for p in sorted(Path(path).rglob("*")) if p.is_file()}


def training_config(original, seed):
    config = deepcopy(original)
    validate_finetuning(config["finetuning"])
    if config["finetuning"]["base_quantization"] != "none":
        raise ValueError("These likelihood scores use BF16 Base; use BF16 LoRA, not NF4")
    config.update(seed=seed, expected_train_size=SIZE, epochs=16,
                  checkpoint_epochs=EPOCHS, recovery_every_steps=100, keep_recovery=2)
    return config


def storage_estimate(model_config, config):
    """No weights loaded: count LoRA A/B matrices from the frozen Qwen config."""
    if model_config["model_type"] != "qwen3":
        raise ValueError("Storage estimator expects dense Qwen3")
    h, m = model_config["hidden_size"], model_config["intermediate_size"]
    q = model_config["num_attention_heads"] * model_config["head_dim"]
    kv = model_config["num_key_value_heads"] * model_config["head_dim"]
    dimensions = {"q_proj": h + q, "k_proj": h + kv, "v_proj": h + kv, "o_proj": q + h,
                  "gate_proj": h + m, "up_proj": h + m, "down_proj": m + h}
    ft = config["finetuning"]
    count = model_config["num_hidden_layers"] * ft["rank"] * sum(dimensions[t] for t in ft["target_modules"])
    retained = len(GROUPS) * (len(EPOCHS) + config["keep_recovery"])
    typical = count * 6 + 32 * 1024**2  # FP32 adapters + two 8-bit moments + metadata/tokenizer.
    upper = int((count * 12 + 64 * 1024**2) * 1.05)  # Conservative FP32 moment fallback.
    return {"lora_parameters": count, "adapter_weight_bytes": count * 4,
            "retained_checkpoint_count": retained, "typical_checkpoint_bytes": typical,
            "typical_retained_bytes": typical * retained,
            "required_bytes": (retained + 1) * upper + 2 * GIB,
            "reserve_bytes": 8 * GIB,
            "note": "Includes epoch 0, optimizer/RNG/cursor, two rolling recoveries per run, "
                    "one temporary checkpoint, metadata/code and 8 GiB reserve. Base cache is shared."}


def select_extremes(pool, scores, size=SIZE):
    if len(pool) < size * 2:
        raise ValueError("Not enough eligible examples for two disjoint groups")
    if len({r["question_id"] for r in pool}) != len(pool):
        raise ValueError("Prepared pool contains duplicate questions")
    by_sample = {}
    for score in scores.values():
        sid = score["sample_id"]
        if sid in by_sample:
            raise ValueError("Ambiguous duplicate sample IDs in full-corpus likelihood results")
        by_sample[sid] = score
    pairs = []
    for row in pool:
        score = by_sample.get(row["sample_id"])
        if score is None or any(row[k] != score[k] for k in ("question_id", "row_index", "source")):
            raise ValueError(f"Eligible sample not matched to the frozen scored corpus: {row['sample_id']}")
        pairs.append((row, score))
    pairs.sort(key=lambda pair: (-pair[1]["mean_logprob"], pair[1]["row_index"]))
    selected = []
    for rank, (row, score) in enumerate(pairs, 1):
        selected.append({**row, "initial_likelihood": {**score, "eligible_rank": rank}})
    return {"high": selected[:size], "low": selected[-size:]}


def group_summary(rows):
    return {"size": len(rows), "sources": dict(Counter(r["source"] for r in rows)),
            "mean_initial_nll": sum(r["initial_likelihood"]["mean_nll"] for r in rows) / len(rows),
            "mean_normalized_likelihood": sum(r["initial_likelihood"]["normalized_likelihood"] for r in rows) / len(rows),
            "mean_supervised_tokens": sum(r["initial_likelihood"]["n_tokens"] for r in rows) / len(rows),
            "max_total_tokens": max(r["initial_likelihood"]["total_tokens"] for r in rows),
            "original_row_indices": [r["row_index"] for r in rows],
            "sample_ids": [r["sample_id"] for r in rows]}


def plan_experiment(scores_dir, data_dir, config_path, seed):
    print("Validating the complete likelihood corpus and original prepared pool...", flush=True)
    job = read_json(scores_dir / "job.json")
    spec = job["spec"]
    if (job["identity"] != digest(spec) or spec["selected_rows"] != DATA_ROWS
            or spec["dataset"]["sha256"] != DATA_SHA256 or spec["dtype"] != "bfloat16"
            or spec["adapter"] is not None):
        raise ValueError("Require full-corpus, unadapted BF16 Base likelihood results (not smoke)")
    marker = read_json(scores_dir / "prepared.json")
    if (marker["job_identity"] != job["identity"]
            or marker["index_sha256"] != file_hash(scores_dir / "index.jsonl")
            or file_hash(scores_dir / "source.parquet") != DATA_SHA256
            or tree_hashes(scores_dir / "tokenizer") != spec["tokenizer_files"]
            or file_hash(scores_dir / "base_config.json") != spec["base_config_sha256"]):
        raise ValueError("Likelihood inputs/encoding index changed")
    index = list(iter_jsonl(scores_dir / "index.jsonl"))
    if [r["row_index"] for r in index] != list(range(DATA_ROWS)):
        raise ValueError("Likelihood index is not the complete original corpus")
    scores = load_scores(scores_dir, job, index)
    if len(scores) != DATA_ROWS:
        raise ValueError(f"Likelihood scoring unfinished: {len(scores)}/{DATA_ROWS}; finish it first")
    prepared = validate_prepared(data_dir)
    if prepared["model"] != spec["model"]:
        raise ValueError("Prepared data and likelihood results use different Base revisions")
    for name in ("pool.jsonl", "holdout.jsonl", "panel.jsonl", "overlap_references.jsonl", "chat_template.jinja"):
        if name not in prepared["files"]:
            raise ValueError(f"Original prepare_data manifest must bind {name}")
    pool = list(iter_jsonl(data_dir / "pool.jsonl"))
    groups = select_extremes(pool, scores)
    config = training_config(read_json(config_path), seed)
    model_config = read_json(scores_dir / "base_config.json")
    max_length = prepared["arguments"]["max_length"]
    if max_length != 10240:
        raise ValueError("This experiment retains the original 10,240-token training eligibility rule")
    forbidden = list(iter_jsonl(data_dir / "holdout.jsonl")) + list(iter_jsonl(data_dir / "overlap_references.jsonl"))
    forbidden += list(iter_jsonl(data_dir / "panel.jsonl"))
    overlap = OverlapIndex(forbidden)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(data_dir / "tokenizer", local_files_only=True)
    inspections = {}
    for name, rows in groups.items():
        inspections[name] = []
        for row in rows:
            if overlap.match(row):
                raise ValueError(f"Selected {name} sample overlaps holdout/evaluation: {row['sample_id']}")
            if row.get("verification", {}).get("correct") is not True:
                raise ValueError("Selected sample lacks the original preparation's answer verification")
            encoded = encode_example(tokenizer, row, max_length)
            if encoded is None:
                raise ValueError("Selected sample exceeds original length limit; no truncation")
            score = row["initial_likelihood"]
            if any(score[k] != v for k, v in describe(row, encoded).items()):
                raise ValueError("Training template/text/tokens differ from likelihood scoring")
            if len(inspections[name]) < 3:
                inspections[name].append(inspect_example(tokenizer, row, encoded))
    request = {"scores": str(scores_dir), "data": str(data_dir), "config": str(config_path), "seed": seed}
    plan = {"protocol": "likelihood-extremes-lora-100x16-v1", "created": now(), "request": request,
            "scoring_identity": job["identity"], "scoring_manifest_sha256": file_hash(scores_dir / "job.json"),
            "prepared_manifest_sha256": file_hash(data_dir / "manifest.json"), "model": spec["model"],
            "selection_scope": "original prepared pool only; mean_logprob descending, row_index tie",
            "eligible_pool_size": len(pool), "config": config, "groups": {k: group_summary(v) for k, v in groups.items()},
            "storage": storage_estimate(model_config, config), "updates_per_group": SIZE * 16,
            "group_order": list(GROUPS), "max_length": max_length}
    return plan, groups, inspections, prepared


def initialize_bundle(out, plan, groups, inspections, prepared):
    """Publish everything needed by both runs together, before starting either one."""
    destination = out / "bundle"
    if destination.exists():
        raise FileExistsError(destination)
    require_free_space(out, plan["storage"]["required_bytes"], plan["storage"]["reserve_bytes"])
    stage = out / (".initializing-" + uuid.uuid4().hex)
    stage.mkdir(parents=True)
    data = Path(plan["request"]["data"])
    snapshot_code(stage / "code")
    # Existing trainer snapshots ignore dataset/. Keep its fingerprint unchanged
    # and explicitly freeze the new launcher/selection modules alongside it.
    for name in ("__init__.py", "likelihood_training.py", "train_likelihood_extremes.py", "likelihood_utils.py",
                 "LIKELIHOOD_TRAINING.md"):
        target = stage / "code" / "dataset" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT / "dataset" / name, target)
    write_json(stage / "config.json", plan["config"])
    write_json(stage / "environment.json", environment())
    for name, rows in groups.items():
        target = stage / "prepared" / name
        target.mkdir(parents=True)
        for filename in ("holdout.jsonl", "panel.jsonl", "overlap_references.jsonl", "chat_template.jinja"):
            shutil.copy2(data / filename, target / filename)
        shutil.copytree(data / "tokenizer", target / "tokenizer")
        write_jsonl(target / "train.jsonl", rows)
        write_jsonl(target / "inspection.jsonl", inspections[name])
        write_json(target / "likelihood_selection.json", {"group": name, "summary": plan["groups"][name],
                   "scoring_identity": plan["scoring_identity"], "scope": plan["selection_scope"]})
        manifest = {"format_version": 1, "arguments": {**prepared["arguments"], "train_size": SIZE},
                    "model": prepared["model"], "sources": prepared.get("sources", {}),
                    "parent_manifest_sha256": plan["prepared_manifest_sha256"], "selection": plan["groups"][name],
                    "stats": {**prepared.get("stats", {}), "train": SIZE}, "files": tree_hashes(target)}
        write_json(target / "manifest.json", manifest)
    # Bind full selected text, config and executable code; resume needs no source artifacts.
    saved = {**plan, "files": tree_hashes(stage)}
    saved["identity"] = digest(saved)
    write_json(stage / "plan.json", saved)
    os.replace(stage, destination)
    return saved


def verify_bundle(out):
    bundle = out / "bundle"
    plan = read_json(bundle / "plan.json")
    if plan["identity"] != digest({k: v for k, v in plan.items() if k != "identity"}):
        raise ValueError("Experiment plan changed")
    actual = tree_hashes(bundle)
    actual.pop("plan.json")
    if actual != plan["files"]:
        raise ValueError("Frozen selected text, code, config or tokenizer changed")
    return plan
