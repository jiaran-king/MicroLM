from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from microlm.model import TransformerLM
from microlm.training import AdamW, get_batch, learning_rate_schedule, load_checkpoint, save_checkpoint


def test_learning_rate_schedule_matches_reference_values() -> None:
    actual = [
        learning_rate_schedule(t=it, alpha_max=1.0, alpha_min=0.1, Tw=7, Tc=21)
        for it in range(25)
    ]
    expected = [
        0.0,
        0.14285714285714285,
        0.2857142857142857,
        0.42857142857142855,
        0.5714285714285714,
        0.7142857142857143,
        0.8571428571428571,
        1.0,
        0.9887175604818206,
        0.9554359905560885,
        0.9018241671106134,
        0.8305704108364301,
        0.7452476826029011,
        0.6501344202803414,
        0.55,
        0.44986557971965857,
        0.3547523173970989,
        0.26942958916356996,
        0.19817583288938662,
        0.14456400944391146,
        0.11128243951817937,
        0.1,
        0.1,
        0.1,
        0.1,
    ]

    np.testing.assert_allclose(np.array(actual), np.array(expected))


def test_adamw_single_step_matches_manual_update() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    parameter.grad = torch.tensor([0.1, -0.2], dtype=torch.float32)
    optimizer = AdamW([parameter], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    grad = parameter.grad.clone()
    initial = parameter.detach().clone()
    optimizer.step()

    exp_avg = (1 - 0.9) * grad
    exp_avg_sq = (1 - 0.999) * grad.pow(2)
    adjusted_lr = 1e-2 * ((1 - 0.999**1) ** 0.5) / (1 - 0.9**1)
    expected = initial.addcdiv(exp_avg, exp_avg_sq.sqrt().add(1e-8), value=-adjusted_lr)
    expected = expected + (-1e-2 * 0.01) * expected

    torch.testing.assert_close(parameter.detach(), expected, atol=1e-7, rtol=1e-6)


def test_get_batch_returns_shifted_targets_for_sequential_data() -> None:
    dataset = np.arange(100, dtype=np.uint16)
    x, y = get_batch(dataset=dataset, batch_size=4, context_length=8, device="cpu")

    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    torch.testing.assert_close(y, x + 1)


def test_checkpoint_roundtrip_restores_iteration_and_weights(tmp_path: Path) -> None:
    model = TransformerLM(
        vocab_size=32,
        context_length=8,
        d_model=16,
        num_layers=1,
        num_heads=4,
        d_ff=32,
        rope_theta=10000.0,
    )
    optimizer = AdamW(model.parameters(), lr=1e-3)

    inputs = torch.randint(0, 32, (2, 8))
    loss = model(inputs).sum()
    loss.backward()
    optimizer.step()

    ckpt_path = tmp_path / "model.pt"
    save_checkpoint(model, optimizer, iteration=17, out=str(ckpt_path))

    restored_model = TransformerLM(
        vocab_size=32,
        context_length=8,
        d_model=16,
        num_layers=1,
        num_heads=4,
        d_ff=32,
        rope_theta=10000.0,
    )
    restored_optimizer = AdamW(restored_model.parameters(), lr=1e-3)
    iteration = load_checkpoint(str(ckpt_path), restored_model, restored_optimizer)

    assert iteration == 17

    for saved_param, restored_param in zip(model.parameters(), restored_model.parameters(), strict=True):
        torch.testing.assert_close(saved_param, restored_param)


def test_pretrain_config_contains_required_sections() -> None:
    config_path = Path("configs/pretrain_baseline.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert set(config) >= {"model", "optimizer", "training", "data", "logging"}
    assert config["model"]["context_length"] == 512
    assert config["model"]["num_layers"] == 8
    assert config["training"]["seed"] == 42
