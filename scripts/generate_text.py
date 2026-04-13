from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch

from microlm.model import TransformerLM
from microlm.inference import resolve_generation_prompt
from microlm.tokenizer import BPETokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text from a trained MicroLM checkpoint."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        required=True,
        help="Path to a model checkpoint or raw state_dict file.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        help="Optional JSON config file with model hyperparameters.",
    )
    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=Path("output/tinystories_bpe_10k/vocab.json"),
    )
    parser.add_argument(
        "--merges-path",
        type=Path,
        default=Path("output/tinystories_bpe_10k/merge.txt"),
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Prompt string used for generation.",
    )
    parser.add_argument(
        "--conversations-json",
        type=str,
        default=None,
        help="JSON string containing a chat-style conversations list.",
    )
    parser.add_argument(
        "--conversations-path",
        type=Path,
        default=None,
        help="Path to a JSON file containing a chat-style conversations list.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of newly generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature. Use 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling threshold.",
    )
    parser.add_argument(
        "--special-token",
        action="append",
        dest="special_tokens",
        default=None,
        help="Special token reserved by the tokenizer. May be passed multiple times.",
    )
    parser.add_argument(
        "--eos-token",
        type=str,
        default=None,
        help="Optional special token string that stops generation when produced.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on. Use 'auto' to prefer CUDA and fall back to CPU.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
        help="Model parameter dtype used at inference time.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling.",
    )
    parser.add_argument(
        "--show-token-ids",
        action="store_true",
        help="Print prompt and generated token ids alongside decoded text.",
    )
    parser.add_argument(
        "--print-new-text-only",
        action="store_true",
        help="Print only the newly generated suffix instead of the full decoded sequence.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        help="Override context length when no config file is provided.",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        help="Override d_model when no config file is provided.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        help="Override num_layers when no config file is provided.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        help="Override num_heads when no config file is provided.",
    )
    parser.add_argument(
        "--d-ff",
        type=int,
        help="Override d_ff when no config file is provided.",
    )
    parser.add_argument(
        "--rope-theta",
        type=float,
        default=10000.0,
        help="Override RoPE theta when no config file is provided.",
    )
    return parser.parse_args()


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[dtype_name]


def resolve_model_dtype(dtype_name: str, device: str) -> torch.dtype:
    dtype = get_torch_dtype(dtype_name)
    if dtype == torch.float16 and device == "cpu":
        return torch.float32
    if dtype == torch.bfloat16 and device == "cpu" and not torch.backends.mkldnn.is_available():
        return torch.float32
    return dtype


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg

    if not torch.cuda.is_available():
        return "cpu"

    try:
        torch.empty(1, device="cuda")
        return "cuda"
    except Exception:
        return "cpu"


def load_model_config(args: argparse.Namespace, vocab_size: int) -> dict[str, int | float]:
    if args.config_path is not None:
        with args.config_path.open("r", encoding="utf-8") as f:
            raw_config = json.load(f)
        return {
            "vocab_size": int(raw_config.get("vocab_size", vocab_size)),
            "context_length": int(raw_config["context_length"]),
            "d_model": int(raw_config["d_model"]),
            "num_layers": int(raw_config["num_layers"]),
            "num_heads": int(raw_config["num_heads"]),
            "d_ff": int(raw_config["d_ff"]),
            "rope_theta": float(raw_config.get("rope_theta", 10000.0)),
        }

    required_fields = {
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
    }
    missing = [name for name, value in required_fields.items() if value is None]
    if missing:
        raise ValueError(
            "Missing model hyperparameters. Provide --config-path or all of: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )

    return {
        "vocab_size": vocab_size,
        "context_length": int(args.context_length),
        "d_model": int(args.d_model),
        "num_layers": int(args.num_layers),
        "num_heads": int(args.num_heads),
        "d_ff": int(args.d_ff),
        "rope_theta": float(args.rope_theta),
    }


def normalize_state_dict_keys(state_dict: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    normalized = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        normalized[key] = value
    return normalized


def load_state_dict(checkpoint_path: Path, device: str) -> OrderedDict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, (dict, OrderedDict)):
        raise TypeError(f"Unsupported checkpoint format at {checkpoint_path}")
    return normalize_state_dict_keys(OrderedDict(state_dict))


def sample_greedy_or_temperature(
    model: TransformerLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    if temperature == 0.0:
        model.eval()
        generated = prompt_ids.clone()
        for _ in range(max_new_tokens):
            idx_cond = generated[:, -model.context_length :]
            logits = model(idx_cond)[:, -1, :]
            if top_p < 1.0:
                logits = model._top_p_filter(logits, top_p)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        return generated

    return model.generate(
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        temperature=temperature,
        top_p=top_p,
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    special_tokens = args.special_tokens or ["<|endoftext|>"]
    tokenizer = BPETokenizer.from_files(
        str(args.vocab_path),
        str(args.merges_path),
        special_tokens=special_tokens,
    )

    config = load_model_config(args, vocab_size=len(tokenizer.id_to_vocab))
    dtype = resolve_model_dtype(args.dtype, device)

    model = TransformerLM(
        vocab_size=int(config["vocab_size"]),
        context_length=int(config["context_length"]),
        d_model=int(config["d_model"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        d_ff=int(config["d_ff"]),
        rope_theta=float(config["rope_theta"]),
        device=device,
        dtype=dtype,
    ).to(device)

    state_dict = load_state_dict(args.checkpoint_path, device)
    model.load_state_dict(state_dict)
    model.eval()

    eos_token_id = None
    if args.eos_token is not None:
        eos_token_bytes = args.eos_token.encode("utf-8")
        if eos_token_bytes not in tokenizer.vocab_to_id:
            raise ValueError(f"EOS token {args.eos_token!r} is not in the tokenizer vocab")
        eos_token_id = tokenizer.vocab_to_id[eos_token_bytes]

    generation_prompt = resolve_generation_prompt(
        prompt=args.prompt,
        conversations_json=args.conversations_json,
        conversations_path=args.conversations_path,
    )

    prompt_token_ids = tokenizer.encode(generation_prompt)
    if not prompt_token_ids:
        raise ValueError("Prompt encodes to an empty token sequence. Provide a non-empty prompt.")

    prompt_tensor = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        generated = sample_greedy_or_temperature(
            model=model,
            prompt_ids=prompt_tensor,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos_token_id,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    full_ids = generated[0].tolist()
    new_ids = full_ids[len(prompt_token_ids) :]
    full_text = tokenizer.decode(full_ids)
    new_text = tokenizer.decode(new_ids)

    if args.show_token_ids:
        print(f"prompt_token_ids={prompt_token_ids}")
        print(f"generated_token_ids={new_ids}")

    if args.print_new_text_only:
        print(new_text)
    else:
        print(full_text)


if __name__ == "__main__":
    main()
