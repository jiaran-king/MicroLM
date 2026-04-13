import argparse
import json
import os

import torch
import numpy as np
import wandb

from microlm.model import TransformerLM
from microlm.training import AdamW
from microlm.training import gradient_clipping as clip_gradient_norm
from microlm.training import learning_rate_schedule as get_lr_cosine_schedule
from microlm.training import get_batch
from microlm.training import save_checkpoint, load_checkpoint
from microlm.training import cross_entropy

def load_config_defaults(config_path: str | None) -> dict[str, object]:
    if config_path is None:
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model = config.get("model", {})
    optimizer = config.get("optimizer", {})
    training = config.get("training", {})
    data = config.get("data", {})
    logging = config.get("logging", {})

    defaults: dict[str, object] = {
        "batch_size": training.get("batch_size"),
        "context_length": model.get("context_length"),
        "d_model": model.get("d_model"),
        "num_heads": model.get("num_heads"),
        "num_layers": model.get("num_layers"),
        "d_ff": model.get("d_ff"),
        "vocab_size": model.get("vocab_size"),
        "norm_mode": model.get("norm_mode"),
        "ffn_type": model.get("ffn_type"),
        "lr": optimizer.get("lr"),
        "max_iters": training.get("max_iters"),
        "warmup_iters": optimizer.get("warmup_iters"),
        "min_lr": optimizer.get("min_lr"),
        "max_norm": optimizer.get("max_norm"),
        "weight_decay": optimizer.get("weight_decay"),
        "train_data_path": data.get("train_data_path"),
        "valid_data_path": data.get("valid_data_path"),
        "out_dir": training.get("out_dir"),
        "device": training.get("device"),
        "run_name": logging.get("run_name"),
        "wandb_project": logging.get("wandb_project"),
        "wandb_mode": logging.get("mode"),
        "seed": training.get("seed"),
        "rope_theta": model.get("rope_theta"),
    }

    if model.get("use_rms_norm") is False:
        defaults["no_rms_norm"] = True
    if model.get("rope_theta") is None:
        defaults["no_rope"] = True

    return {key: value for key, value in defaults.items() if value is not None}


def build_parser(defaults: dict[str, object]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=defaults.get("config"))

    parser.add_argument("--batch_size", type=int, default=defaults.get("batch_size", 32))
    parser.add_argument("--context_length", type=int, default=defaults.get("context_length", 256))
    parser.add_argument("--d_model", type=int, default=defaults.get("d_model", 512))
    parser.add_argument("--num_heads", type=int, default=defaults.get("num_heads", 8))
    parser.add_argument("--num_layers", type=int, default=defaults.get("num_layers", 4))
    parser.add_argument("--d_ff", type=int, default=defaults.get("d_ff", 2048))
    parser.add_argument("--vocab_size", type=int, default=defaults.get("vocab_size", 10000))

    parser.add_argument(
        "--no_rms_norm",
        action="store_true",
        default=defaults.get("no_rms_norm", False),
        help="Disable RMSNorm completely",
    )
    parser.add_argument(
        "--norm_mode",
        type=str,
        default=defaults.get("norm_mode", "pre"),
        choices=["pre", "post"],
        help="Normalization placement",
    )
    parser.add_argument(
        "--no_rope",
        action="store_true",
        default=defaults.get("no_rope", False),
        help="Disable Rotary Positional Embeddings",
    )
    parser.add_argument(
        "--ffn_type",
        type=str,
        default=defaults.get("ffn_type", "swiglu"),
        choices=["swiglu", "silu"],
        help="Type of Feed_Forward Network",
    )
    parser.add_argument("--rope_theta", type=float, default=defaults.get("rope_theta", 10000.0))

    parser.add_argument("--lr", type=float, default=defaults.get("lr", 6e-4))
    parser.add_argument("--max_iters", type=int, default=defaults.get("max_iters", 10000))
    parser.add_argument("--warmup_iters", type=int, default=defaults.get("warmup_iters", 1000))
    parser.add_argument("--min_lr", type=float, default=defaults.get("min_lr", 6e-5))
    parser.add_argument("--max_norm", type=float, default=defaults.get("max_norm", 1.0))
    parser.add_argument("--weight_decay", type=float, default=defaults.get("weight_decay", 0.1))

    parser.add_argument("--train_data_path", type=str, required="train_data_path" not in defaults, default=defaults.get("train_data_path"))
    parser.add_argument("--valid_data_path", type=str, required="valid_data_path" not in defaults, default=defaults.get("valid_data_path"))
    parser.add_argument("--out_dir", type=str, default=defaults.get("out_dir", "out"))
    parser.add_argument("--device", type=str, default=defaults.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))

    parser.add_argument("--run_name", type=str, default=defaults.get("run_name"), help="WandB run name")
    parser.add_argument("--wandb_project", type=str, default=defaults.get("wandb_project", "micro-lm"))
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=defaults.get("wandb_mode", "online"),
        choices=["online", "offline", "disabled"],
    )
    return parser


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, remaining = config_parser.parse_known_args()

    defaults = load_config_defaults(config_args.config)
    defaults["config"] = config_args.config
    parser = build_parser(defaults)
    return parser.parse_args(remaining)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_token_data(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path, mmap_mode="r")
    return np.memmap(path, dtype=np.uint16, mode="r")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    if not os.path.exists(args.train_data_path):
        raise FileNotFoundError(f"Training data not found at {args.train_data_path}")
    if not os.path.exists(args.valid_data_path):
        raise FileNotFoundError(f"Validation data not found at {args.valid_data_path}")

    train_data = load_token_data(args.train_data_path)
    val_data = load_token_data(args.valid_data_path)

    print(f"训练集大小： {len(train_data)} tokens")
    print(f"验证集大小 {len(val_data)} tokens")

    actual_rope_theta = None if args.no_rope else args.rope_theta
    use_rms_norm = not args.no_rms_norm

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=actual_rope_theta,
        use_rms_norm=use_rms_norm,
        norm_mode=args.norm_mode,
        ffn_type=args.ffn_type,
        device=args.device,
    ).to(args.device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {num_params:,}")
    print(f"Model Config: Norm={args.norm_mode}, UseNorm={use_rms_norm}, FFN={args.ffn_type}, RoPE={not args.no_rope}")

    model_config = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "rope_theta": actual_rope_theta if actual_rope_theta is not None else 10000.0,
        "use_rms_norm": use_rms_norm,
        "norm_mode": args.norm_mode,
        "ffn_type": args.ffn_type,
    }
    with open(os.path.join(args.out_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    resolved_config = vars(args).copy()
    with open(os.path.join(args.out_dir, "resolved_train_config.json"), "w", encoding="utf-8") as f:
        json.dump(resolved_config, f, indent=2)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_iter = 0
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    if os.path.exists(ckpt_path):
        start_iter = load_checkpoint(ckpt_path, model, optimizer)
        print(f"Resuming from iteration {start_iter}")

    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        mode=args.wandb_mode,
        config=resolved_config,
    )

    for it in range(start_iter, args.max_iters):
        lr = get_lr_cosine_schedule(it, args.lr, args.min_lr, args.warmup_iters, args.max_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        model.train()
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        logits = model(x)
        loss = cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        clip_gradient_norm(model.parameters(), args.max_norm)
        optimizer.step()

        if it % 100 == 0 or it == args.max_iters - 1:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(val_data, args.batch_size, args.context_length, args.device)
                v_logits = model(vx)
                v_loss = cross_entropy(v_logits, vy)
                print(f"Iter {it}: train_loss {loss.item():.4f}, val_loss {v_loss.item():.4f}, lr {lr:.2e}")
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "val/loss": v_loss.item(),
                        "lr": lr,
                        "iter": it + 1,
                    }
                )

        if it % 1000 == 0 and it > 0:
            save_checkpoint(model, optimizer, it, ckpt_path)

    save_checkpoint(model, optimizer, args.max_iters, os.path.join(args.out_dir, "ckpt_final.pt"))
    wandb.finish()

if __name__ == "__main__":
    main()
