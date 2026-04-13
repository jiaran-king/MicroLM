from __future__ import annotations

from pathlib import Path

from microlm.tokenizer import BPETokenizer, save_tokenizer_files, train_bpe


def build_tokenizer(tmp_path: Path) -> BPETokenizer:
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(
        "hello world\nhello tokenizer\nhello<|endoftext|>world\n",
        encoding="utf-8",
    )
    vocab, merges = train_bpe(
        input_path=str(corpus_path),
        vocab_size=270,
        special_tokens=["<|endoftext|>"],
    )

    tokenizer_dir = tmp_path / "tokenizer"
    save_tokenizer_files(vocab, merges, str(tokenizer_dir))
    return BPETokenizer.from_files(
        str(tokenizer_dir / "vocab.json"),
        str(tokenizer_dir / "merge.txt"),
        special_tokens=["<|endoftext|>"],
    )


def test_tokenizer_roundtrip_with_special_token(tmp_path: Path) -> None:
    tokenizer = build_tokenizer(tmp_path)
    text = "hello<|endoftext|>world"

    ids = tokenizer.encode(text)

    assert tokenizer.decode(ids) == text
    assert tokenizer.vocab_to_id[b"<|endoftext|>"] in ids


def test_encode_iterable_matches_full_encode(tmp_path: Path) -> None:
    tokenizer = build_tokenizer(tmp_path)
    text = "hello world\nhello tokenizer\n"
    chunks = ["hel", "lo w", "orld\nhe", "llo tok", "enizer\n"]

    full_ids = tokenizer.encode(text)
    streamed_ids = list(tokenizer.encode_iterable(chunks))

    assert streamed_ids == full_ids

