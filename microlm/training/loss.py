import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    logsumexp = torch.log(torch.exp(shifted).sum(dim=-1))
    target_logits = shifted.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (logsumexp - target_logits).mean()


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.shape[:-1] != targets.shape:
        raise ValueError("logits and targets must agree on batch/time dimensions")
    if targets.shape != loss_mask.shape:
        raise ValueError("targets and loss_mask must have the same shape")

    shifted = logits - logits.max(dim=-1, keepdim=True).values
    logsumexp = torch.log(torch.exp(shifted).sum(dim=-1))
    safe_targets = targets.clamp_min(0)
    target_logits = shifted.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    losses = logsumexp - target_logits

    mask = loss_mask.to(losses.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (losses * mask).sum() / denom
