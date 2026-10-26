"""BF16 Qwen loading and memory-bounded, exact teacher-forced cross entropy."""
from __future__ import annotations


def load_model(path, revision=None, attention="sdpa", training=False):
    import torch
    from transformers import AutoModelForCausalLM
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This experiment requires a CUDA GPU with BF16 support")
    model = AutoModelForCausalLM.from_pretrained(
        str(path), revision=revision, torch_dtype=torch.bfloat16,
        attn_implementation=attention, trust_remote_code=False,
    ).to("cuda")
    if model.config.model_type != "qwen3":
        raise ValueError("The memory-bounded loss implementation expects dense Qwen3")
    if training:
        model.requires_grad_(True)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
    else:
        model.eval()
    return model


def supervised_hidden(model, batch):
    import torch
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

    def chunk_loss(h, t):
        return F.cross_entropy(model.lm_head(h).float(), t, reduction="sum")

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
        losses, predicted = [], []
        for h, t in zip(hidden.split(chunk_size), targets.split(chunk_size)):
            logits = model.lm_head(h).float()
            losses.append(F.cross_entropy(logits, t, reduction="none").cpu())
            predicted.append(logits.argmax(-1).cpu())
        nll, predictions, gold = torch.cat(losses), torch.cat(predicted), targets.cpu()
        return {"mean_loss": nll.mean().item(), "nll_sum": nll.sum().item(),
                "token_accuracy": predictions.eq(gold).float().mean().item(),
                "correct_tokens": predictions.eq(gold).sum().item(), "n_tokens": len(gold),
                "positions": positions.cpu().numpy(), "nll": nll.numpy(),
                "predicted_ids": predictions.numpy(), "target_ids": gold.numpy()}
