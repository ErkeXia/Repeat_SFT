"""LoRA protocol validation and adapter-only provenance (no GPU imports)."""
from __future__ import annotations

import math
from pathlib import Path

from .common import file_hash, read_json

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
SPEC_NAME = "recipe_model.json"


def validate_finetuning(settings):
    required = {"method", "rank", "alpha", "dropout", "target_modules", "base_quantization"}
    if not isinstance(settings, dict) or set(settings) != required or settings["method"] != "lora":
        raise ValueError("New training requires an explicit LoRA finetuning configuration; use configs/baseline.json. "
                         "Resume legacy full-parameter runs with their original frozen code.")
    if isinstance(settings["rank"], bool) or not isinstance(settings["rank"], int) or settings["rank"] < 1:
        raise ValueError("LoRA rank must be a positive integer")
    if not math.isfinite(settings["alpha"]) or settings["alpha"] <= 0:
        raise ValueError("LoRA alpha must be positive")
    if not math.isfinite(settings["dropout"]) or not 0 <= settings["dropout"] < 1:
        raise ValueError("LoRA dropout must be in [0, 1)")
    targets = settings["target_modules"]
    if not isinstance(targets, list) or not targets or len(set(targets)) != len(targets) or not set(targets) <= set(TARGETS):
        raise ValueError("LoRA targets must be unique Qwen attention/MLP projections; embeddings/head stay frozen")
    if settings["base_quantization"] not in ("nf4", "none"):
        raise ValueError("base_quantization must be nf4 (4-bit QLoRA) or none (BF16 LoRA)")
    return settings


def adapter_spec(path, *, finetuning=None, base_model=None):
    """Only accept self-produced, completed adapters with a pinned Base and hashes."""
    path = Path(path)
    spec = read_json(path / SPEC_NAME)
    if spec.get("format_version") != 1 or spec.get("compute_dtype") != "bfloat16":
        raise ValueError("Unsupported adapter model specification")
    validate_finetuning(spec["finetuning"])
    if not spec["base_model"].get("id") or not spec["base_model"].get("revision"):
        raise ValueError("Adapter must record its Base model and immutable revision")
    if finetuning is not None and spec["finetuning"] != finetuning:
        raise ValueError("Checkpoint LoRA/quantization configuration differs from run")
    if base_model is not None and spec["base_model"] != base_model:
        raise ValueError("Checkpoint Base model/revision differs from run")
    marker = read_json(path / "complete.json")
    files = marker.get("model_files", {})
    if not {SPEC_NAME, "adapter_config.json", "adapter_model.safetensors"} <= set(files):
        raise ValueError("Adapter checkpoint is missing its integrity manifest")
    for name, expected in files.items():
        file = (path / name).resolve()
        if file.parent != path.resolve() or file_hash(file) != expected:
            raise ValueError(f"Adapter checkpoint changed: {name}")
    saved = read_json(path / "adapter_config.json")
    cfg = spec["finetuning"]
    if (saved.get("peft_type") != "LORA" or saved.get("task_type") != "CAUSAL_LM"
            or saved.get("r") != cfg["rank"] or saved.get("lora_alpha") != cfg["alpha"]
            or saved.get("lora_dropout") != cfg["dropout"]
            or set(saved.get("target_modules", [])) != set(cfg["target_modules"])
            or saved.get("bias") != "none" or saved.get("modules_to_save")
            or saved.get("base_model_name_or_path") != spec["base_model"]["id"]
            or saved.get("revision") != spec["base_model"]["revision"]):
        raise ValueError("PEFT adapter configuration differs from recorded protocol")
    return spec


def trainable_parameters(model):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters")
    return params


def assert_adapter_training(model):
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    if not names or any("lora_A." not in n and "lora_B." not in n for n in names):
        raise ValueError("Only LoRA A/B parameters may be trainable; Base, embeddings and head must stay frozen")
    return {"trainable_parameters": sum(p.numel() for p in trainable_parameters(model)),
            "trainable_parameter_names": names, "full_parameter": False}


def checkpoint_state_dict(model):
    if getattr(model, "peft_config", None):
        from peft import get_peft_model_state_dict
        return get_peft_model_state_dict(model, save_embedding_layers=False)
    return model.state_dict()
