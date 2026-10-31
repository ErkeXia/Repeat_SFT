"""SERVER ONLY: checkpoint-by-checkpoint fixed-panel generation; no grading."""
from __future__ import annotations

import argparse
import gc
import time

from recipe.common import (append_jsonl, checkpoint_path, environment, file_hash, iter_jsonl, read_json,
                           recover_jsonl, seed_everything, snapshot_code, stable_seed, write_json)
from recipe.evaluation import bind_output
from recipe.modeling import load_model
from recipe.grouped import select_panel
from recipe.paths import artifact_path, configure_runtime


def stop_details(ids, eos_id, chat_end_id, limit):
    last = ids[-1] if ids else None
    normal = last == chat_end_id
    eos = last == eos_id
    reached = len(ids) >= limit
    return {"stop_reason": "chat_end" if normal else "base_eos" if eos else "length" if reached else "other",
            "normal_chat_end": normal, "ended_with_eos": normal or eos,
            "reached_length_limit": reached, "length_truncated": reached and not (normal or eos),
            "last_token_id": last}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--epochs", default="64,32,16,8,4,2,1")
    p.add_argument("--config", default="configs/generation.json")
    p.add_argument("--out", help="Defaults to RUN/evaluations/generation")
    p.add_argument("--limit-per-task", type=int, help="Smoke test only; records a distinct evaluation identity")
    p.add_argument("--sample-ids", help="JSON list of fixed panel IDs, in generation order")
    p.add_argument("--progress-label", default="", help="Label printed with per-question progress")
    args = p.parse_args()
    configure_runtime()
    import torch
    from transformers import AutoTokenizer, GenerationConfig
    run = artifact_path(args.run)
    metadata = read_json(run / "run.json")
    settings = read_json(args.config)
    if set(settings) != {"seed", "do_sample", "temperature", "top_p", "top_k", "max_new_tokens", "repetition_penalty"}:
        raise ValueError("Unexpected/missing generation settings")
    if settings["max_new_tokens"] < 1:
        raise ValueError("max_new_tokens must be positive")
    tokenizer = AutoTokenizer.from_pretrained(run / "tokenizer", local_files_only=True)
    panel = list(iter_jsonl(run / "data" / "panel.jsonl"))
    if args.sample_ids:
        if args.limit_per_task is not None:
            p.error("--sample-ids and --limit-per-task cannot be combined")
        panel = select_panel(panel, read_json(artifact_path(args.sample_ids)))
    if args.limit_per_task is not None:
        if args.limit_per_task < 1:
            raise ValueError("limit-per-task must be positive")
        panel = [r for task in sorted({r["task"] for r in panel})
                 for r in [x for x in panel if x["task"] == task][:args.limit_per_task]]
    base = artifact_path(args.out) if args.out else run / "evaluations" / "generation"
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_id = tokenizer.eos_token_id
    generation_config = GenerationConfig(
        **{k: v for k, v in settings.items() if k != "seed"},
        eos_token_id=list(dict.fromkeys([end_id, eos_id])), pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id, num_beams=1, num_return_sequences=1,
        use_cache=True, min_new_tokens=0, forced_eos_token_id=None,
    )
    prompts = {r["sample_id"]: tokenizer.apply_chat_template(r["messages"], tokenize=True, add_generation_prompt=True)
               for r in panel}
    for epoch in [int(x) for x in args.epochs.split(",")]:
        ckpt = checkpoint_path(run, epoch)
        out = base / f"epoch-{epoch:03d}"
        bind_output(out, {"run_sha256": file_hash(run / "run.json"), "epoch": epoch,
                          "checkpoint": read_json(ckpt / "complete.json"), "settings": settings,
                          "generation_config": generation_config.to_dict(),
                          "panel_sha256": file_hash(run / "data" / "panel.jsonl"),
                          "selected_ids": [r["sample_id"] for r in panel], "seed_policy": "seed + question identity, same at every checkpoint"})
        if not (out / "code").exists():
            snapshot_code(out / "code")
            write_json(out / "environment.json", environment())
        log = out / "responses.jsonl"
        previous = recover_jsonl(log)
        done = {r["sample_id"] for r in previous}
        if len(done) != len(previous) or not done.issubset(prompts):
            raise ValueError("Duplicate or unexpected generated sample")
        if len(done) == len(panel):
            write_json(out / "complete.json", {"samples": len(panel), "responses_sha256": file_hash(log)})
            continue
        seed_everything(settings["seed"])
        model = load_model(ckpt, attention=metadata["config"]["attention_implementation"],
                           finetuning=metadata["config"].get("finetuning"),
                           base_model=metadata["model"] if "finetuning" in metadata["config"] else None)
        model.config.use_cache = True
        context_limit = model.config.max_position_embeddings
        if any(len(v) + settings["max_new_tokens"] > context_limit for v in prompts.values()):
            raise ValueError(f"Prompt + {settings['max_new_tokens']}/new-token budget exceeds model context {context_limit}. "
                             "Do not silently truncate: choose a consistent documented configuration for ALL checkpoints.")
        for row in panel:
            if row["sample_id"] in done:
                continue
            seed = stable_seed(settings["seed"], row["sample_id"])
            seed_everything(seed)
            prompt = prompts[row["sample_id"]]
            inputs = torch.tensor([prompt], dtype=torch.long, device="cuda")
            start = time.monotonic()
            try:
                with torch.inference_mode():
                    output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                            generation_config=generation_config, logits_to_keep=1)
                ids = output[0, len(prompt):].tolist()
            except Exception as exc:
                append_jsonl(out / "errors.jsonl", {"sample_id": row["sample_id"], "error": str(exc),
                                                     "seed": seed, "epoch": epoch})
                raise  # Never drop a difficult/OOM question from the denominator.
            record = {"sample_id": row["sample_id"], "question_id": row["question_id"], "task": row["task"],
                      "epoch": epoch, "seed": seed, "messages": row["messages"], "gold_answer": row["gold_answer"],
                      "prompt_token_ids": prompt, "generated_token_ids": ids,
                      "raw_response": tokenizer.decode(ids, skip_special_tokens=False),
                      "response_text": tokenizer.decode(ids, skip_special_tokens=True),
                      "generated_tokens": len(ids), "prompt_tokens": len(prompt),
                      "generation_seconds": time.monotonic() - start,
                      **stop_details(ids, eos_id, end_id, settings["max_new_tokens"])}
            append_jsonl(log, record)
            done.add(row["sample_id"])
            label = args.progress_label + " " if args.progress_label else ""
            print(f"{label}epoch={epoch} {len(done)}/{len(panel)} tokens={len(ids)} stop={record['stop_reason']}", flush=True)
            del output, inputs
        write_json(out / "complete.json", {"samples": len(panel), "responses_sha256": file_hash(log)})
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
