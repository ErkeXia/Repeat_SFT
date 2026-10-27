"""SERVER ONLY: freeze revisions, tokenizer, eligible pool, splits, and eval panel."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
import random

from recipe.common import (append_jsonl, environment, file_hash, fresh_dir, iter_jsonl,
                           snapshot_code, write_json, write_jsonl)
from recipe.data import OverlapIndex, adapt_gpqa, adapt_math, adapt_sft, unique_questions
from recipe.encoding import encode_example, inspect_example
from recipe.grading import grade_one
from recipe.paths import artifact_path, configure_runtime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    p.add_argument("--model-revision", default="main")
    p.add_argument("--sft-revision", default="main")
    p.add_argument("--math-revision", default="main")
    p.add_argument("--gpqa-revision", default="main")
    p.add_argument("--gpqa-config", default="gpqa_diamond")
    p.add_argument("--sft-jsonl", help="Optional local copy using the original schema")
    p.add_argument("--math-jsonl", help="Optional local MATH500 file using original schema")
    p.add_argument("--gpqa-csv", help="Optional already-authorized GPQA CSV")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--panel-seed", type=int, default=2026)
    p.add_argument("--train-size", type=int, default=400)
    p.add_argument("--holdout-size", type=int, default=200)
    p.add_argument("--math-size", type=int, default=128)
    p.add_argument("--gpqa-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=10240)
    p.add_argument("--inspect", type=int, default=3)
    args = p.parse_args()
    if min(args.train_size, args.holdout_size, args.math_size, args.gpqa_size, args.max_length) < 1:
        p.error("Sizes must be positive")
    configure_runtime()
    # Heavy dependencies are imported only after argparse, so --help is offline.
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    import math_verify  # Fail before scanning if the declared verifier is missing.

    out = fresh_dir(artifact_path(args.out))
    print(f"Prepared data directory: {out}", flush=True)
    api = HfApi()
    model_sha = api.model_info(args.model, revision=args.model_revision).sha
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=model_sha, trust_remote_code=False)
    tokenizer.save_pretrained(out / "tokenizer")
    (out / "chat_template.jinja").write_text(tokenizer.chat_template, encoding="utf-8")
    provenance = {"model": {"id": args.model, "revision": model_sha}, "sources": {}}

    def dataset(repo, revision, split, config=None, local=None, csv_file=False):
        if local:
            provenance["sources"][repo] = {"local_file": str(Path(local).resolve()), "sha256": file_hash(local),
                                            "config": config, "split": split}
            if csv_file:
                with open(local, encoding="utf-8-sig", newline="") as f:
                    return list(csv.DictReader(f))
            return list(iter_jsonl(local))
        sha = api.dataset_info(repo, revision=revision).sha
        provenance["sources"][repo] = {"revision": sha, "config": config, "split": split}
        return load_dataset(repo, name=config, revision=sha, split=split, streaming=True)

    math = unique_questions([adapt_math(r, i) for i, r in enumerate(dataset(
        "HuggingFaceH4/MATH-500", args.math_revision, "test", local=args.math_jsonl))])
    gpqa = unique_questions([adapt_gpqa(r, i, args.panel_seed) for i, r in enumerate(dataset(
        "Idavidrein/gpqa", args.gpqa_revision, "train", args.gpqa_config, args.gpqa_csv, True))])
    rng = random.Random(args.panel_seed)
    panel = rng.sample(math, args.math_size) + rng.sample(gpqa, args.gpqa_size)
    write_jsonl(out / "panel.jsonl", panel)
    # Decontaminate against the whole loaded benchmark, not just the small panel.
    references = math + gpqa
    write_jsonl(out / "overlap_references.jsonl", references)
    benchmark_index = OverlapIndex(references)
    stats, pool, seen = Counter(), [], set()
    for index, raw in enumerate(dataset("jasonrqh/Math-CoT-20k", args.sft_revision, "train", local=args.sft_jsonl)):
        stats["input_rows"] += 1
        row = adapt_sft(raw, index)
        reason, detail = None, None
        if row["question_id"] in seen:
            reason = "duplicate_question"
        else:
            detail = benchmark_index.match(row)
            if detail:
                reason = "benchmark_overlap"
        if reason is None:
            encoded = encode_example(tokenizer, row, args.max_length)
            if encoded is None:
                reason = "over_length"
            else:
                verified = grade_one(row["response"], row["answer"], "math")
                if verified["correct"] is not True:
                    reason, detail = "answer_verification_failed", verified
        if reason:
            stats[reason] += 1
            append_jsonl(out / "rejections.jsonl", {"row_index": index, "sample_id": row["sample_id"],
                                                    "question_id": row["question_id"], "question": row["question"],
                                                    "reason": reason, "detail": detail})
            continue
        seen.add(row["question_id"])
        row.update(n_tokens=len(encoded["input_ids"]), n_supervised=encoded["supervised_tokens"],
                   verification=verified)
        pool.append(row)
        stats["eligible"] += 1
        if index % 500 == 0:
            print(dict(stats), flush=True)
    split_rng = random.Random(args.seed)
    shuffled = list(pool)
    split_rng.shuffle(shuffled)
    holdout = shuffled[:args.holdout_size]
    if len(holdout) != args.holdout_size:
        raise ValueError("Insufficient eligible holdout questions")
    holdout_ids = {r["question_id"] for r in holdout}
    holdout_index = OverlapIndex(holdout)
    candidates = []
    for row in shuffled[args.holdout_size:]:
        match = holdout_index.match(row)
        if match:
            append_jsonl(out / "holdout_overlap.jsonl", {"sample_id": row["sample_id"], **match})
            stats["holdout_near_overlap"] += 1
        else:
            candidates.append(row)
    train = split_rng.sample(candidates, args.train_size)
    assert not holdout_ids.intersection(r["question_id"] for r in candidates)
    write_jsonl(out / "pool.jsonl", candidates)
    write_jsonl(out / "holdout.jsonl", holdout)
    write_jsonl(out / "train.jsonl", train)
    inspections = [inspect_example(tokenizer, row, encode_example(tokenizer, row, args.max_length))
                   for row in train[:args.inspect]]
    write_jsonl(out / "inspection.jsonl", inspections)
    for example in inspections:
        print("\nSAMPLE", example["sample_id"], "\nACTUAL INPUT:\n", example["rendered_text"],
              "\nSUPERVISED ONLY:\n", example["supervised_text"], "\nTAIL:", example["tail_tokens"], flush=True)
    stats.update(train=len(train), holdout=len(holdout), pool=len(candidates), panel=len(panel))
    write_json(out / "overlap_report.json", {"counts": dict(stats), "reference_questions": len(references),
                                            "methods": ["NFKC + whitespace exact", "char7 Jaccard >= 0.9"],
                                            "semantic_overlap_not_ruled_out": True})
    code_hash = snapshot_code(out / "code")
    files = {p.relative_to(out).as_posix(): file_hash(p) for p in out.iterdir() if p.is_file()}
    files.update({p.relative_to(out).as_posix(): file_hash(p) for p in (out / "tokenizer").iterdir() if p.is_file()})
    write_json(out / "manifest.json", {"format_version": 1, "arguments": vars(args), **provenance,
                                       "stats": dict(stats), "files": files, "code_sha256": code_hash,
                                       "environment": environment(),
                                       "correctness_policy": "publisher verified + independent final-answer equivalence"})
    print(f"Prepared {out}; eligible={len(candidates)}, train={len(train)}, holdout={len(holdout)}")


if __name__ == "__main__":
    main()
