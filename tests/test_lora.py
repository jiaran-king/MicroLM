"""Tests for microlm.model.lora — LoRA module."""

import torch
import pytest

from microlm.model.transformer import Linear, TransformerLM
from microlm.model.lora import (
    LoRALinear,
    apply_lora_to_model,
    get_lora_params,
    get_lora_state_dict,
    load_lora_state_dict,
    merge_lora,
    unmerge_lora,
    print_trainable_params,
)


def _tiny_model(**overrides):
    defaults = dict(
        vocab_size=128,
        context_length=32,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=128,
        rope_theta=10000.0,
    )
    defaults.update(overrides)
    return TransformerLM(**defaults)


# ---- LoRALinear unit tests -------------------------------------------------


class TestLoRALinear:
    def test_output_shape(self):
        linear = Linear(64, 32)
        lora = LoRALinear(linear, r=4, alpha=8.0)
        x = torch.randn(2, 10, 64)
        out = lora(x)
        assert out.shape == (2, 10, 32)

    def test_original_weight_frozen(self):
        linear = Linear(64, 32)
        lora = LoRALinear(linear, r=4, alpha=8.0)
        assert not lora.original.weight.requires_grad

    def test_lora_params_have_grad(self):
        linear = Linear(64, 32)
        lora = LoRALinear(linear, r=4, alpha=8.0)
        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad

    def test_B_initialised_zero(self):
        """LoRA starts as identity (zero B), so output == original."""
        linear = Linear(64, 32)
        lora = LoRALinear(linear, r=4, alpha=8.0)
        x = torch.randn(2, 10, 64)
        original_out = torch.einsum("... i, o i -> ... o", x, linear.weight)
        lora_out = lora(x)
        torch.testing.assert_close(lora_out, original_out)

    def test_merge_unmerge_roundtrip(self):
        linear = Linear(64, 32)
        lora = LoRALinear(linear, r=4, alpha=8.0)
        x = torch.randn(2, 10, 64)

        out_before = lora(x)
        lora.merge()
        out_merged = lora(x)
        torch.testing.assert_close(out_before, out_merged, atol=1e-5, rtol=1e-5)

        lora.unmerge()
        out_unmerged = lora(x)
        torch.testing.assert_close(out_unmerged, out_before, atol=1e-5, rtol=1e-5)

    def test_merge_flag(self):
        linear = Linear(64, 32)
        lora = LoRALinear(linear, r=4, alpha=8.0)
        assert not lora.merged
        lora.merge()
        assert lora.merged
        lora.unmerge()
        assert not lora.merged


# ---- apply_lora_to_model tests ---------------------------------------------


class TestApplyLora:
    def test_default_targets_replaced(self):
        model = _tiny_model()
        apply_lora_to_model(model, r=4, alpha=8.0)
        for layer in model.layers:
            assert isinstance(layer.attn.q_proj, LoRALinear)
            assert isinstance(layer.attn.k_proj, LoRALinear)
            assert isinstance(layer.attn.v_proj, LoRALinear)
            assert isinstance(layer.attn.output_proj, LoRALinear)

    def test_custom_targets(self):
        model = _tiny_model()
        apply_lora_to_model(model, r=4, alpha=8.0, target_names=["w1", "w2", "w3"])
        for layer in model.layers:
            assert isinstance(layer.ffn.w1, LoRALinear)
            assert isinstance(layer.ffn.w2, LoRALinear)
            assert isinstance(layer.ffn.w3, LoRALinear)
            # attention projections should NOT be replaced
            assert not isinstance(layer.attn.q_proj, LoRALinear)

    def test_forward_runs_after_lora(self):
        model = _tiny_model()
        apply_lora_to_model(model, r=4, alpha=8.0)
        ids = torch.randint(0, 128, (1, 16))
        logits = model(ids)
        assert logits.shape == (1, 16, 128)

    def test_only_lora_params_trainable(self):
        model = _tiny_model()
        apply_lora_to_model(model, r=4, alpha=8.0)
        params = get_lora_params(model)
        assert len(params) > 0
        lora_ids = {id(p) for p in params}
        for p in model.parameters():
            if p.requires_grad:
                assert id(p) in lora_ids

    def test_backward_updates_only_lora(self):
        model = _tiny_model()
        apply_lora_to_model(model, r=4, alpha=8.0)
        ids = torch.randint(0, 128, (1, 16))
        logits = model(ids)
        loss = logits.sum()
        loss.backward()

        lora_ids = {id(p) for p in get_lora_params(model)}
        for p in model.parameters():
            if id(p) in lora_ids:
                assert p.grad is not None
            else:
                assert p.grad is None


# ---- state dict save / load ------------------------------------------------


class TestLoRAStateDict:
    def test_save_load_roundtrip(self):
        ids = torch.randint(0, 128, (1, 16))

        torch.manual_seed(42)
        model = _tiny_model()
        apply_lora_to_model(model, r=4, alpha=8.0)

        # Train one step
        logits = model(ids)
        loss = logits.sum()
        loss.backward()
        for p in get_lora_params(model):
            p.data -= 0.01 * p.grad

        out_trained = model(ids)
        sd = get_lora_state_dict(model)

        # Rebuild with same seed so original weights match
        torch.manual_seed(42)
        model2 = _tiny_model()
        apply_lora_to_model(model2, r=4, alpha=8.0)
        load_lora_state_dict(model2, sd)
        out_loaded = model2(ids)

        torch.testing.assert_close(out_trained, out_loaded)


# ---- merge / unmerge on full model ----------------------------------------


class TestModelMerge:
    def test_merge_preserves_output(self):
        model = _tiny_model()
        apply_lora_to_model(model, r=4, alpha=8.0)
        ids = torch.randint(0, 128, (1, 16))

        out_before = model(ids).detach().clone()
        merge_lora(model)
        out_after = model(ids).detach().clone()
        torch.testing.assert_close(out_before, out_after, atol=1e-5, rtol=1e-5)

        unmerge_lora(model)
        out_unmerged = model(ids).detach().clone()
        torch.testing.assert_close(out_unmerged, out_before, atol=1e-5, rtol=1e-5)
