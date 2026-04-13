from __future__ import annotations

from collections import OrderedDict

import torch

def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str
):
    ##1.将需要保存的必要信息存入checkpoint中
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }

    ##2.保存
    torch.save(checkpoint, out)

def load_checkpoint(
        src:str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer
)->int:
    ##1.加载checkpoint
    checkpoint = torch.load(src, map_location = 'cpu')

    ##2.恢复model和optimizer
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    ##返回迭代步数
    return checkpoint['iteration']


def _normalize_state_dict(
        state_dict: dict[str, torch.Tensor] | OrderedDict[str, torch.Tensor]
) -> OrderedDict[str, torch.Tensor]:
    normalized: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]
        normalized[key] = value
    return normalized


def load_model_state(
        src: str,
        model: torch.nn.Module
) -> OrderedDict[str, torch.Tensor]:
    checkpoint = torch.load(src, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, (dict, OrderedDict)):
        raise TypeError(f"Unsupported checkpoint format at {src}")
    normalized = _normalize_state_dict(state_dict)
    model.load_state_dict(normalized)
    return normalized
