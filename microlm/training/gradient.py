from collections.abc import Iterable

import torch


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return

    total_sq_norm = 0.0
    for p in params_with_grad:
        grad_norm = torch.norm(p.grad.detach(), p=2)
        total_sq_norm += grad_norm.item() ** 2
    total_l2_norm = total_sq_norm**0.5

    eps = 1e-6
    if total_l2_norm > max_l2_norm:
        scale = max_l2_norm / (total_l2_norm + eps)
        for p in params_with_grad:
            p.grad.mul_(scale)
