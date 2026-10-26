"""Build source/length-matched low/mid/high initial-loss subsets without a GPU."""
import argparse
import math
from pathlib import Path
import statistics

from recipe.common import file_hash, fresh_dir, iter_jsonl, read_json, snapshot_code, validate_prepared, write_json, write_jsonl
from recipe.data import matched_loss_groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--scores", required=True, help="score_pool.py output directory")
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--length-bin", type=int, default=512)
    args = p.parse_args()
    if args.length_bin < 1 or args.size < 1:
        p.error("size and length-bin must be positive")
    data, scores = Path(args.data), Path(args.scores)
    manifest = validate_prepared(data)
    identity = read_json(scores / "identity.json")
    if (identity["pool_sha256"] != file_hash(data / "pool.jsonl") or identity["model"] != manifest["model"]
            or identity["prepared_sha256"] != file_hash(data / "manifest.json")):
        raise ValueError("Scores are not from this dataset / original Base revision")
    rows = list(iter_jsonl(data / "pool.jsonl"))
    score_rows = list(iter_jsonl(scores / "samples.jsonl"))
    values = {r["sample_id"]: r["mean_loss"] for r in score_rows}
    if len(values) != len(score_rows) or set(values) != {r["sample_id"] for r in rows}:
        raise ValueError("Need exactly one initial-loss score for every pool sample")
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError("Nonfinite initial loss")
    groups, matching = matched_loss_groups(rows, values, args.size, args.seed, args.length_bin)
    out = fresh_dir(args.out)
    stats = {}
    for name, selected in groups.items():
        write_jsonl(out / f"{name}.jsonl", [{**r, "base_mean_loss": values[r["sample_id"]]} for r in selected])
        stats[name] = {"count": len(selected), "mean_loss": statistics.mean(values[r["sample_id"]] for r in selected),
                       "mean_supervised_tokens": statistics.mean(r["n_supervised"] for r in selected),
                       "median_supervised_tokens": statistics.median(r["n_supervised"] for r in selected),
                       "min_supervised_tokens": min(r["n_supervised"] for r in selected),
                       "max_supervised_tokens": max(r["n_supervised"] for r in selected)}
    write_json(out / "selection.json", {"arguments": vars(args), "score_identity": identity, "matching": matching,
                                        "groups": stats, "scores_sha256": file_hash(scores / "samples.jsonl"),
                                        "files": {n: file_hash(out / f"{n}.jsonl") for n in groups},
                                        "code_sha256": snapshot_code(out / "code")})
    print(stats)


if __name__ == "__main__":
    main()
