"""Frozen Qwen Base + LoRA; bounded exact loss shared by training and evaluation."""
from __future__ import annotations

from pathlib import Path
from .adapters import adapter_spec, assert_adapter_training, validate_finetuning


def causal_model(model):
    # PEFT wraps QwenForCausalLM. Access the underlying decoder WITH its injected
    # LoRA projections; model.model on a PEFT wrapper is not the bare decoder.
    return model.get_base_model() if getattr(model, "peft_config", None) else model


def prepare_base(model, finetuning):
    """Use identical frozen dtypes for training, Base scoring and adapter inference."""
    import torch
    from peft import prepare_model_for_kbit_training
    model.requires_grad_(False)
    if finetuning["base_quantization"] == "nf4":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        # PEFT promotes all unquantized parameters to FP32. Qwen's large frozen
        # embedding/head would waste ~1.5 GiB; keep these BF16 in EVERY phase.
        model.get_input_embeddings().to(dtype=torch.bfloat16)
        model.get_output_embeddings().to(dtype=torch.bfloat16)
    return model


def attach_adapter(model, settings, base_model, adapter_path=None, training=True):
    from peft import LoraConfig, PeftModel, get_peft_model
    if adapter_path is None:
        config = LoraConfig(task_type="CAUSAL_LM", r=settings["rank"], lora_alpha=settings["alpha"],
                            lora_dropout=settings["dropout"], target_modules=settings["target_modules"],
                            bias="none", init_lora_weights=True, revision=base_model["revision"])
        model = get_peft_model(model, config, autocast_adapter_dtype=True)
    else:
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=training,
                                         autocast_adapter_dtype=True, local_files_only=True,
                                         torch_device=str(next(model.parameters()).device))
    # Explicitly pin this; never let PEFT resolve a mutable 'main' on reload.
    model.peft_config["default"].base_model_name_or_path = base_model["id"]
    model.peft_config["default"].revision = base_model["revision"]
    model._recipe_model_spec = {"format_version": 1, "base_model": dict(base_model),
                                "finetuning": settings, "compute_dtype": "bfloat16"}
    if training:
        assert_adapter_training(model)
    return model


def load_model(path, revision=None, attention="sdpa", training=False, finetuning=None, base_model=None):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This experiment requires a CUDA GPU with BF16 support")
    adapter_path = Path(path) if (Path(path) / "adapter_config.json").is_file() else None
    if adapter_path is not None:
        spec = adapter_spec(adapter_path, finetuning=finetuning, base_model=base_model)
        finetuning, base_model = spec["finetuning"], spec["base_model"]
        path, revision = base_model["id"], base_model["revision"]
    elif base_model is not None and (str(path) != base_model["id"] or revision != base_model["revision"]):
        raise ValueError("Expected the pinned Base or an adapter checkpoint, not legacy full weights")
    if training or finetuning is not None:
        validate_finetuning(finetuning)
        if not revision:
            raise ValueError("LoRA experiments require a pinned Base model revision")
    options = {}
    if finetuning and finetuning["base_quantization"] == "nf4":
        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    # Direct single-GPU placement also avoids first materializing BF16 weights on
    # the GPU before quantization. Never use device_map='auto' to hide CPU offload.
    model = AutoModelForCausalLM.from_pretrained(str(path), revision=revision, torch_dtype=torch.bfloat16,
        attn_implementation=attention, trust_remote_code=False, device_map={"": 0}, **options)
    if model.config.model_type != "qwen3":
        raise ValueError("The memory-bounded loss implementation expects dense Qwen3")
    if finetuning:
        model = prepare_base(model, finetuning)
    if training or adapter_path is not None:
        model = attach_adapter(model, finetuning, base_model or {"id": str(path), "revision": revision},
                               adapter_path, training)
    if training:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
    else:
        model.requires_grad_(False)
        model.eval()
    return model


def supervised_hidden(model, batch):
    import torch
    model = causal_model(model)
    hidden = model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                         use_cache=False, return_dict=True).last_hidden_state
    labels = batch["labels"][:, 1:]
    valid = labels.ne(-100)
    if not bool(valid.any()):
        raise ValueError("No supervised next-token targets")
    return hidden[:, :-1][valid], labels[valid], torch.nonzero(valid, as_tuple=False)[:, 1] + 1


def training_loss(model, batch, chunk_size=128):
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    hidden, targets, _ = supervised_hidden(model, batch)
    head = causal_model(model).lm_head

    def chunk_loss(h, t):
        return F.cross_entropy(head(h).float(), t, reduction="sum")

    # Recompute LM-head chunks during backward instead of retaining a [10k,152k]
    # FP32 logits tensor. All assistant tokens have exactly equal loss weight.
    losses = [checkpoint(chunk_loss, h, t, use_reentrant=False)
              for h, t in zip(hidden.split(chunk_size), targets.split(chunk_size))]
    return sum(losses) / targets.numel()


def teacher_forcing(model, batch, chunk_size=128):
    import torch
    import torch.nn.functional as F
    with torch.inference_mode():
        hidden, targets, positions = supervised_hidden(model, batch)
        head = causal_model(model).lm_head
        losses, predicted = [], []
        for h, t in zip(hidden.split(chunk_size), targets.split(chunk_size)):
            logits = head(h).float()
            losses.append(F.cross_entropy(logits, t, reduction="none").cpu())
            predicted.append(logits.argmax(-1).cpu())
        nll, predictions, gold = torch.cat(losses), torch.cat(predicted), targets.cpu()
        return {"mean_loss": nll.mean().item(), "nll_sum": nll.sum().item(),
                "token_accuracy": predictions.eq(gold).float().mean().item(),
                "correct_tokens": predictions.eq(gold).sum().item(), "n_tokens": len(gold),
                "positions": positions.cpu().numpy(), "nll": nll.numpy(),
                "predicted_ids": predictions.numpy(), "target_ids": gold.numpy()}
