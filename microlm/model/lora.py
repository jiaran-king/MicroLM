"""LoRA (Low-Rank Adaptation) for MicroLM.

Wraps existing Linear layers with trainable low-rank matrices A and B:
    output = W·x + (α/r)·B·A·x

Original weights are frozen; only A and B receive gradients.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn

from .transformer import Linear


class LoRALinear(nn.Module):
    """Drop-in replacement for Linear that adds a low-rank adapter."""

    def __init__(
        self,
        original: Linear,
        r: int = 8,
        alpha: float = 16.0,
    ) -> None:
        super().__init__()
        self.original = original
        self.original.weight.requires_grad_(False)

        out_features, in_features = original.weight.shape
        self.r = r
        self.scaling = alpha / r
        device = original.weight.device

        # A: (r, in_features),  B: (out_features, r)
        self.lora_A = nn.Parameter(torch.empty(r, in_features, device=device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self._merged = False

    # ---- forward --------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_out = torch.einsum("... i, o i -> ... o", x, self.original.weight)
        if self._merged:
            return original_out
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return original_out + lora_out

    # ---- merge / unmerge ------------------------------------------------

    @torch.no_grad()
    def merge(self) -> None:
        """Fold LoRA weights into the original weight (for inference)."""
        if self._merged:
            return
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.original.weight.add_(delta)
        self._merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """Undo merge (restore original weight)."""
        if not self._merged:
            return
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.original.weight.sub_(delta)
        self._merged = False

    @property
    def merged(self) -> bool:
        return self._merged


# ---- apply LoRA to a TransformerLM ----------------------------------------

_DEFAULT_TARGETS = {"q_proj", "k_proj", "v_proj", "output_proj"}


def _replace_module(root: nn.Module, full_name: str, new_module: nn.Module) -> None:
    parts = full_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def apply_lora_to_model(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    target_names: Iterable[str] | None = None,
) -> None:
    """Replace matching Linear layers in *model* with LoRALinear.

    All original parameters are frozen; only LoRA A/B matrices are trainable.
    """
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad_(False)

    if target_names is None:
        target_names = _DEFAULT_TARGETS
    target_set = set(target_names)

    # Collect matches first (can't mutate during iteration)
    replacements: list[tuple[str, LoRALinear]] = []
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in target_set and isinstance(module, Linear):
            replacements.append((name, LoRALinear(module, r=r, alpha=alpha)))

    for name, lora_layer in replacements:
        _replace_module(model, name, lora_layer)


def get_lora_params(model: nn.Module) -> list[nn.Parameter]:
    """Return all LoRA parameters (A and B matrices)."""
    params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.append(module.lora_A)
            params.append(module.lora_B)
    return params


def get_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return a state dict containing only LoRA weights."""
    sd: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            sd[f"{name}.lora_A"] = module.lora_A.data.cpu()
            sd[f"{name}.lora_B"] = module.lora_B.data.cpu()
    return sd


def load_lora_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Load LoRA weights from a saved state dict."""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            a_key = f"{name}.lora_A"
            b_key = f"{name}.lora_B"
            if a_key in state_dict:
                module.lora_A.data.copy_(state_dict[a_key])
            if b_key in state_dict:
                module.lora_B.data.copy_(state_dict[b_key])


def merge_lora(model: nn.Module) -> None:
    """Merge all LoRALinear layers in the model (inference mode)."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()


def unmerge_lora(model: nn.Module) -> None:
    """Unmerge all LoRALinear layers in the model."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()


def print_trainable_params(model: nn.Module) -> None:
    """Print a summary of total vs trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable / total * 100 if total > 0 else 0
    print(f"Total params: {total:,} | Trainable (LoRA): {trainable:,} ({ratio:.2f}%)")
