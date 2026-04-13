from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from microlm.model import Linear, RMSNorm, TransformerLM, scaled_dot_product_attention, silu


def test_linear_matches_manual_einsum() -> None:
    layer = Linear(3, 2)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]]))

    x = torch.tensor([[0.1, 0.2, 0.3], [1.0, -1.0, 2.0]])
    expected = torch.einsum("... i, o i -> ... o", x, layer.weight)

    actual = layer(x)

    torch.testing.assert_close(actual, expected)


def test_rmsnorm_matches_manual_formula() -> None:
    layer = RMSNorm(4, eps=1e-5)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([1.0, 1.5, 0.5, 2.0]))

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    expected = (x / rms) * layer.weight

    actual = layer(x)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_scaled_dot_product_attention_matches_manual_masked_softmax() -> None:
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
    k = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
    v = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
    mask = torch.tensor([[[True, False], [True, True]]])

    scores = torch.einsum("... q d, ... k d -> ... q k", q, k) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    expected = torch.softmax(scores, dim=-1) @ v

    actual = scaled_dot_product_attention(q, k, v, mask)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_silu_matches_pytorch() -> None:
    x = torch.tensor([[0.2, -1.0, 3.0], [1.5, 0.0, -0.3]])

    torch.testing.assert_close(silu(x), F.silu(x), atol=1e-6, rtol=1e-6)


def test_transformer_lm_forward_shape_and_context_guard() -> None:
    model = TransformerLM(
        vocab_size=64,
        context_length=8,
        d_model=16,
        num_layers=2,
        num_heads=4,
        d_ff=32,
        rope_theta=10000.0,
    )

    token_ids = torch.randint(0, 64, (2, 8))
    logits = model(token_ids)

    assert logits.shape == (2, 8, 64)

    too_long = torch.randint(0, 64, (1, 9))
    try:
        model(too_long)
    except ValueError as exc:
        assert "context length" in str(exc)
    else:
        raise AssertionError("Expected TransformerLM to reject sequences longer than context_length")

