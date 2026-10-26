"""Use the saved tokenizer's real template; never truncate a response."""
from __future__ import annotations

import re
import unicodedata

IGNORE_INDEX = -100


def normalize_question(text):
    # Preserve operators, numbers, and letter case: a and A can mean different things.
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def question_id(text):
    from .common import digest
    return digest(normalize_question(text))


def encode_example(tokenizer, row, max_length=10240):
    messages = row["messages"]
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("Expected a prompt ending in one user turn")
    response = row["response"]
    if not response.strip():
        raise ValueError("Empty response")
    # Prevent embedded role delimiters from silently modifying supervision boundaries.
    for text in [response, *(m["content"] for m in messages)]:
        if any(s in text for s in ("<|im_start|>", "<|im_end|>", "<|endoftext|>")):
            raise ValueError("Reserved chat delimiter found in dataset text")
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": response}],
        tokenize=False, add_generation_prompt=False,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    ids = tokenizer.encode(full_text, add_special_tokens=False)
    if ids[:len(prompt_ids)] != prompt_ids or not full_text.startswith(prompt_text):
        raise ValueError("Chat template prompt is not an exact token prefix; inspect before training")
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if tokenizer.decode([end_id], skip_special_tokens=False) != "<|im_end|>":
        raise ValueError("Expected Qwen chat end token is missing")
    end_positions = [i for i in range(len(prompt_ids), len(ids)) if ids[i] == end_id]
    if len(end_positions) != 1:
        raise ValueError("Expected exactly one assistant end token")
    assistant_end = end_positions[0]
    # Qwen may normalize newlines around <think>. Compare meaningful content,
    # including both reasoning and final answer, so template changes fail loudly.
    assistant_text = full_text[len(prompt_text):].split("<|im_end|>")[0]
    content_key = lambda s: re.sub(r"\s+", "", s.replace("<think>", "").replace("</think>", ""))
    if content_key(response) != content_key(assistant_text):
        raise ValueError("Template altered or dropped solution content")
    if len(ids) > max_length:
        return None
    labels = [IGNORE_INDEX] * len(ids)
    labels[len(prompt_ids):assistant_end + 1] = ids[len(prompt_ids):assistant_end + 1]
    assert labels[assistant_end] == end_id
    assert all(v == IGNORE_INDEX for v in labels[:len(prompt_ids)])
    return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids),
            "prompt_length": len(prompt_ids), "assistant_end": assistant_end,
            "supervised_tokens": assistant_end + 1 - len(prompt_ids),
            "rendered_text": full_text, "supervised_text": tokenizer.decode(
                ids[len(prompt_ids):assistant_end + 1], skip_special_tokens=False)}


def pad_batch(examples, pad_token_id):
    """Right padding is excluded by position, even when PAD equals EOS."""
    import torch
    length = max(len(x["input_ids"]) for x in examples)
    result = {}
    for key, pad in (("input_ids", pad_token_id), ("labels", IGNORE_INDEX), ("attention_mask", 0)):
        result[key] = torch.tensor([x[key] + [pad] * (length - len(x[key])) for x in examples])
    return result


def inspect_example(tokenizer, row, encoded):
    return {"sample_id": row["sample_id"], "question": row["question"],
            "original_response": row["response"], "rendered_text": encoded["rendered_text"],
            "supervised_text": encoded["supervised_text"],
            "masked_prompt_text": tokenizer.decode(encoded["input_ids"][:encoded["prompt_length"]], skip_special_tokens=False),
            "prompt_length": encoded["prompt_length"], "total_tokens": len(encoded["input_ids"]),
            "supervised_tokens": encoded["supervised_tokens"], "assistant_end": encoded["assistant_end"],
            "tail_tokens": [{"position": i, "id": encoded["input_ids"][i], "label": encoded["labels"][i],
                             "text": tokenizer.decode([encoded["input_ids"][i]], skip_special_tokens=False)}
                            for i in range(max(0, len(encoded["input_ids"]) - 10), len(encoded["input_ids"]))]}
