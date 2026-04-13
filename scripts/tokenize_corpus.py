from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path

import numpy as np

from microlm.tokenizer import BPETokenizer


def load_config_defaults(config_path: str | None) -> dict[str, object]:
    if config_path is None:
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    tokenizer = config.get("tokenizer", {})
    data = config.get("data", {})
    output = config.get("output", {})

    defaults: dict[str, object] = {
        "vocab_path": tokenizer.get("vocab_path"),
        "merges_path": tokenizer.get("merges_path"),
        "special_tokens": tokenizer.get("special_tokens"),
        "train_path": data.get("train_path"),
        "valid_path": data.get("valid_path"),
        "output_dir": output.get("output_dir"),
        "read_chunk_bytes": output.get("read_chunk_bytes"),
        "token_batch_size": output.get("token_batch_size"),
    }
    return {key: value for key, value in defaults.items() if value is not None}


def build_parser(defaults: dict[str, object]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Encode text train/valid splits into uint16 token ID arrays."
    )
    parser.add_argument("--config", type=str, default=defaults.get("config"))
    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=defaults.get("vocab_path", Path("output/tinystories_bpe_10k/vocab.json")),
    )
    parser.add_argument(
        "--merges-path",
        type=Path,
        default=defaults.get("merges_path", Path("output/tinystories_bpe_10k/merge.txt")),
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=defaults.get("train_path", Path("data/TinyStoriesV2-GPT4-train.txt")),
    )
    parser.add_argument(
        "--valid-path",
        type=Path,
        default=defaults.get("valid_path", Path("data/TinyStoriesV2-GPT4-valid.txt")),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=defaults.get("output_dir", Path("output/tinystories_tokenized")),
    )
    parser.add_argument(
        "--special-token",
        action="append",
        dest="special_tokens",
        default=defaults.get("special_tokens"),
        help="Special token to reserve while loading the tokenizer. May be passed multiple times.",
    )
    parser.add_argument(
        "--read-chunk-bytes",
        type=int,
        default=defaults.get("read_chunk_bytes", 4 * 1024 * 1024),
        help="Number of UTF-8 text bytes to read from disk at a time.",
    )
    parser.add_argument(
        "--token-batch-size",
        type=int,
        default=defaults.get("token_batch_size", 1_000_000),
        help="Number of token IDs to materialize at once while counting/writing.",
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


def iter_file_chunks(path: Path, chunk_size: int):
    with path.open("r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                return
            yield chunk


def iter_token_batches(
    tokenizer: BPETokenizer,
    path: Path,
    read_chunk_bytes: int,
    token_batch_size: int,
):
    token_iter = tokenizer.encode_iterable(iter_file_chunks(path, read_chunk_bytes))
    while True:
        batch = np.fromiter(islice(token_iter, token_batch_size), dtype=np.uint16)
        if batch.size == 0:
            return
        yield batch


def count_tokens(
    tokenizer: BPETokenizer,
    path: Path,
    read_chunk_bytes: int,
    token_batch_size: int,
) -> tuple[int, int]:
    total_tokens = 0
    max_token_id = -1
    for batch in iter_token_batches(tokenizer, path, read_chunk_bytes, token_batch_size):
        total_tokens += int(batch.size)
        batch_max = int(batch.max())
        if batch_max > max_token_id:
            max_token_id = batch_max
    return total_tokens, max_token_id


def write_tokens(
    tokenizer: BPETokenizer,
    path: Path,
    out_path: Path,
    total_tokens: int,
    read_chunk_bytes: int,
    token_batch_size: int,
) -> None:
    array = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.uint16,
        shape=(total_tokens,),
    )
    offset = 0
    for batch in iter_token_batches(tokenizer, path, read_chunk_bytes, token_batch_size):
        next_offset = offset + batch.size
        array[offset:next_offset] = batch
        offset = next_offset
    array.flush()


def main() -> None:
    args = parse_args()
    if args.special_tokens is None:
        args.special_tokens = ["<|endoftext|>"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = BPETokenizer.from_files(
        str(args.vocab_path),
        str(args.merges_path),
        special_tokens=args.special_tokens,
    )

    vocab_size = len(tokenizer.id_to_vocab)
    if vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError(f"Tokenizer vocab size {vocab_size} does not fit in uint16 IDs")

    datasets = {
        "train": args.train_path,
        "valid": args.valid_path,
    }
    metadata: dict[str, object] = {
        "dtype": "uint16",
        "tokenizer_vocab_path": str(args.vocab_path),
        "tokenizer_merges_path": str(args.merges_path),
        "special_tokens": args.special_tokens,
        "vocab_size": vocab_size,
        "datasets": {},
    }

    for split, path in datasets.items():
        out_path = args.output_dir / f"{split}_ids.npy"
        print(f"[{split}] counting tokens in {path} ...")
        total_tokens, max_token_id = count_tokens(
            tokenizer,
            path,
            args.read_chunk_bytes,
            args.token_batch_size,
        )
        print(f"[{split}] writing {total_tokens} tokens to {out_path} ...")
        write_tokens(
            tokenizer,
            path,
            out_path,
            total_tokens,
            args.read_chunk_bytes,
            args.token_batch_size,
        )
        metadata["datasets"][split] = {
            "source_path": str(path),
            "token_ids_path": str(out_path),
            "num_tokens": total_tokens,
            "max_token_id": max_token_id,
        }

    metadata_path = args.output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
