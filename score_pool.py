"""SERVER ONLY: score the eligible, holdout-free pool using the original Base."""
import argparse

from recipe.common import environment, file_hash, iter_jsonl, seed_everything, snapshot_code, validate_prepared, write_json
from recipe.evaluation import bind_output, evaluate_rows
from recipe.modeling import load_model
from recipe.paths import artifact_path, configure_runtime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--chunk-size", type=int, default=128)
    args = p.parse_args()
    configure_runtime()
    from transformers import AutoTokenizer
    data, out = artifact_path(args.data), artifact_path(args.out)
    manifest = validate_prepared(data)
    identity = {"prepared_sha256": file_hash(data / "manifest.json"), "pool_sha256": file_hash(data / "pool.jsonl"),
                "model": manifest["model"], "chunk_size": args.chunk_size, "kind": "original_base_pool"}
    bind_output(out, identity)
    if not (out / "code").exists():
        snapshot_code(out / "code")
        write_json(out / "environment.json", environment())
    seed_everything(0)
    tokenizer = AutoTokenizer.from_pretrained(data / "tokenizer", local_files_only=True)
    model = load_model(manifest["model"]["id"], revision=manifest["model"]["revision"])
    evaluate_rows(model, tokenizer, list(iter_jsonl(data / "pool.jsonl")), out,
                  manifest["arguments"]["max_length"], args.chunk_size, save_tokens=False)


if __name__ == "__main__":
    main()
