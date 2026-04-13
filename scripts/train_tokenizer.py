from __future__ import annotations

import argparse
import json
from pathlib import Path

from microlm.tokenizer import save_tokenizer_files, train_bpe


def load_config_defaults(config_path: str | None) -> dict[str, object]:
    if config_path is None:
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    defaults: dict[str, object] = {
        "input_path": config.get("input_path"),
        "vocab_size": config.get("vocab_size"),
        "special_tokens": config.get("special_tokens"),
        "output_dir": config.get("output_dir"),
    }
    return {key: value for key, value in defaults.items() if value is not None}


def build_parser(defaults: dict[str, object]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer for MicroLM.")
    parser.add_argument("--config", type=str, default=defaults.get("config"))
    parser.add_argument(
        "--input-path",
        type=Path,
        required="input_path" not in defaults,
        default=defaults.get("input_path"),
        help="UTF-8 text corpus used for tokenizer training.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        required="vocab_size" not in defaults,
        default=defaults.get("vocab_size"),
        help="Target vocabulary size including bytes and special tokens.",
    )
    parser.add_argument(
        "--special-token",
        action="append",
        dest="special_tokens",
        default=defaults.get("special_tokens"),
        help="Special token reserved in the vocabulary. May be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required="output_dir" not in defaults,
        default=defaults.get("output_dir"),
        help="Directory where vocab.json and merge.txt will be written.",
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


def main() -> None:
    args = parse_args()
    special_tokens = args.special_tokens or ["<|endoftext|>"]
    vocab, merges = train_bpe(
        input_path=str(args.input_path),
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
    )
    save_tokenizer_files(vocab=vocab, merges=merges, out_dir=str(args.output_dir))
    print(f"saved tokenizer files to {args.output_dir}")


if __name__ == "__main__":
    main()
