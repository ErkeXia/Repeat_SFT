"""SERVER ONLY: score the eligible, holdout-free pool using the original Base."""
import argparse

from recipe.common import environment, file_hash, iter_jsonl, read_json, seed_everything, snapshot_code, validate_prepared, write_json
from recipe.adapters import validate_finetuning
from recipe.evaluation import bind_output, evaluate_rows
from recipe.modeling import load_model
from recipe.paths import artifact_path, configure_runtime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="configs/baseline.json", help="Same Base quantization as subsequent LoRA runs")
    p.add_argument("--chunk-size", type=int, help="Defaults to config logit_chunk_size")
    args = p.parse_args()
    configure_runtime()
    from transformers import AutoTokenizer
    data, out = artifact_path(args.data), artifact_path(args.out)
    manifest = validate_prepared(data)
    config = read_json(args.config)
    finetuning = validate_finetuning(config.get("finetuning"))
    args.chunk_size = config["logit_chunk_size"] if args.chunk_size is None else args.chunk_size
    if args.chunk_size < 1:
        p.error("--chunk-size must be positive")
    identity = {"prepared_sha256": file_hash(data / "manifest.json"), "pool_sha256": file_hash(data / "pool.jsonl"),
                "model": manifest["model"], "chunk_size": args.chunk_size, "kind": "original_base_pool",
                "finetuning": finetuning, "attention_implementation": config["attention_implementation"]}
    bind_output(out, identity)
    if not (out / "code").exists():
        snapshot_code(out / "code")
        write_json(out / "environment.json", environment())
    seed_everything(0)
    tokenizer = AutoTokenizer.from_pretrained(data / "tokenizer", local_files_only=True)
    model = load_model(manifest["model"]["id"], revision=manifest["model"]["revision"],
                       attention=config["attention_implementation"], finetuning=finetuning)
    evaluate_rows(model, tokenizer, list(iter_jsonl(data / "pool.jsonl")), out,
                  manifest["arguments"]["max_length"], args.chunk_size, save_tokens=False)


if __name__ == "__main__":
    main()
