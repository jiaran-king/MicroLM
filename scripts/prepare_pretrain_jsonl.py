from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import codecs
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert line-delimited {'text': ...} corpora into train/valid text files with enhanced cleaning."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to the source JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/pretrain"),
        help="Directory for the generated train/valid text files.",
    )
    parser.add_argument(
        "--text-key",
        type=str,
        default="text",
        help="JSON key that contains the raw document text.",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.01,
        help="Fraction of documents routed to validation via a deterministic text hash.",
    )
    parser.add_argument(
        "--document-separator",
        type=str,
        default="###",
        help="Special token inserted between documents in train/valid outputs.",
    )
    parser.add_argument(
        "--replace-literal",
        action="append",
        default=[],
        help="Literal replacement rule in old=new form. Supports escape sequences like \\n on the right-hand side.",
    )
    # --- new cleaning arguments ---
    parser.add_argument(
        "--min-length",
        type=int,
        default=50,
        help="Minimum document length in characters after cleaning. Shorter docs are dropped. 0 to disable.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=0,
        help="Maximum document length in characters. 0 means no limit.",
    )
    parser.add_argument(
        "--max-length-action",
        choices=["drop", "truncate"],
        default="drop",
        help="What to do with documents exceeding --max-length: drop or truncate.",
    )
    parser.add_argument(
        "--clean-html",
        action="store_true",
        default=False,
        help="Remove HTML-style tags (<...>) from text.",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        default=False,
        help="Skip exact deduplication.",
    )
    return parser


def should_use_valid_split(text: str, valid_ratio: float) -> bool:
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return bucket / 2**64 < valid_ratio


def parse_replacement_rules(raw_rules: list[str]) -> list[tuple[str, str]]:
    rules: list[tuple[str, str]] = []
    for raw_rule in raw_rules:
        if "=" not in raw_rule:
            raise ValueError(f"Invalid --replace-literal rule {raw_rule!r}; expected old=new")
        old, new = raw_rule.split("=", 1)
        new = codecs.decode(new, "unicode_escape")
        rules.append((old, new))
    return rules


# ---------- cleaning helpers ----------

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def clean_control_chars(text: str) -> tuple[str, bool]:
    """Remove control characters except \\n and \\t. Returns (cleaned, was_modified)."""
    cleaned = _CONTROL_CHAR_RE.sub("", text)
    return cleaned, len(cleaned) != len(text)


def compress_whitespace(text: str) -> tuple[str, bool]:
    """Compress runs of spaces/tabs to single space; compress 3+ newlines to 2."""
    cleaned = _MULTI_SPACE_RE.sub(" ", text)
    cleaned = _MULTI_NEWLINE_RE.sub("\n\n", cleaned)
    return cleaned, cleaned != text


def clean_html_tags(text: str) -> tuple[str, bool]:
    """Remove HTML-style tags. Returns (cleaned, was_modified)."""
    cleaned = _HTML_TAG_RE.sub("", text)
    return cleaned, len(cleaned) != len(text)


def compute_length_stats(lengths: list[int]) -> dict:
    """Compute descriptive statistics for document lengths."""
    if not lengths:
        return {"count": 0}
    sorted_lengths = sorted(lengths)
    n = len(sorted_lengths)
    total = sum(sorted_lengths)
    mean = total / n
    if n % 2 == 0:
        median = (sorted_lengths[n // 2 - 1] + sorted_lengths[n // 2]) / 2
    else:
        median = sorted_lengths[n // 2]

    def percentile(p: float) -> int:
        k = (n - 1) * p / 100
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_lengths[int(k)]
        return int(sorted_lengths[f] * (c - k) + sorted_lengths[c] * (k - f))

    return {
        "count": n,
        "min": sorted_lengths[0],
        "max": sorted_lengths[-1],
        "mean": round(mean, 1),
        "median": int(median),
        "p25": percentile(25),
        "p75": percentile(75),
        "p95": percentile(95),
        "p99": percentile(99),
        "total_chars": total,
    }


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.valid_ratio < 1.0:
        raise ValueError("--valid-ratio must be between 0 (inclusive) and 1 (exclusive)")
    replacement_rules = parse_replacement_rules(args.replace_literal)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.txt"
    valid_path = args.output_dir / "valid.txt"
    tokenizer_corpus_path = args.output_dir / "tokenizer_corpus.txt"
    metadata_path = args.output_dir / "metadata.json"

    # counters
    total_raw = 0
    skipped_empty = 0
    skipped_short = 0
    skipped_long = 0
    cleaned_control = 0
    cleaned_html_count = 0
    compressed_ws = 0
    duplicates_removed = 0

    train_docs = 0
    valid_docs = 0
    train_chars = 0
    valid_chars = 0

    all_lengths: list[int] = []
    seen_hashes: set[str] = set() if not args.no_dedup else set()

    with (
        args.input_path.open("r", encoding="utf-8") as src,
        train_path.open("w", encoding="utf-8") as train_f,
        valid_path.open("w", encoding="utf-8") as valid_f,
        tokenizer_corpus_path.open("w", encoding="utf-8") as tokenizer_f,
    ):
        for line_number, raw_line in enumerate(src, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = record.get(args.text_key)
            if not isinstance(text, str):
                continue

            total_raw += 1

            # 1. strip
            text = text.strip()
            if not text:
                skipped_empty += 1
                continue

            # 2. literal replacement
            for old, new in replacement_rules:
                text = text.replace(old, new)

            # 3. control character cleaning
            text, ctrl_modified = clean_control_chars(text)
            if ctrl_modified:
                cleaned_control += 1

            # 4. HTML tag cleaning
            if args.clean_html:
                text, html_modified = clean_html_tags(text)
                if html_modified:
                    cleaned_html_count += 1

            # 5. whitespace compression
            text, ws_modified = compress_whitespace(text)
            if ws_modified:
                compressed_ws += 1

            # 6. final strip
            text = text.strip()
            if not text:
                skipped_empty += 1
                continue

            # 7. length filtering
            text_len = len(text)
            if args.min_length > 0 and text_len < args.min_length:
                skipped_short += 1
                continue
            if args.max_length > 0 and text_len > args.max_length:
                if args.max_length_action == "drop":
                    skipped_long += 1
                    continue
                else:  # truncate
                    text = text[:args.max_length]
                    text_len = args.max_length

            # 8. deduplication
            if not args.no_dedup:
                doc_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if doc_hash in seen_hashes:
                    duplicates_removed += 1
                    continue
                seen_hashes.add(doc_hash)

            all_lengths.append(text_len)

            tokenizer_f.write(text)
            tokenizer_f.write("\n")

            target = valid_f if should_use_valid_split(text, args.valid_ratio) else train_f
            target.write(text)
            target.write("\n")
            target.write(args.document_separator)
            target.write("\n")

            if target is train_f:
                train_docs += 1
                train_chars += text_len
            else:
                valid_docs += 1
                valid_chars += text_len

    total_kept = train_docs + valid_docs

    filter_stats = {
        "total_raw_documents": total_raw,
        "skipped_empty": skipped_empty,
        "skipped_short": skipped_short,
        "skipped_long": skipped_long,
        "cleaned_html": cleaned_html_count,
        "cleaned_control_chars": cleaned_control,
        "compressed_whitespace": compressed_ws,
        "duplicates_removed": duplicates_removed,
        "total_kept": total_kept,
        "filter_rate": f"{(1 - total_kept / total_raw) * 100:.2f}%" if total_raw > 0 else "0.00%",
    }

    metadata = {
        "source_path": str(args.input_path),
        "text_key": args.text_key,
        "document_separator": args.document_separator,
        "valid_ratio": args.valid_ratio,
        "replacement_rules": [
            {"old": old, "new": new} for old, new in replacement_rules
        ],
        "cleaning_config": {
            "min_length": args.min_length,
            "max_length": args.max_length,
            "max_length_action": args.max_length_action,
            "clean_html": args.clean_html,
            "dedup_enabled": not args.no_dedup,
        },
        "filter_stats": filter_stats,
        "length_stats": compute_length_stats(all_lengths),
        "train": {
            "path": str(train_path),
            "documents": train_docs,
            "characters": train_chars,
            "avg_doc_length": round(train_chars / train_docs, 1) if train_docs > 0 else 0,
        },
        "valid": {
            "path": str(valid_path),
            "documents": valid_docs,
            "characters": valid_chars,
            "avg_doc_length": round(valid_chars / valid_docs, 1) if valid_docs > 0 else 0,
        },
        "tokenizer_corpus": {
            "path": str(tokenizer_corpus_path),
            "documents": total_kept,
        },
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"wrote train split to {train_path}")
    print(f"wrote valid split to {valid_path}")
    print(f"wrote tokenizer corpus to {tokenizer_corpus_path}")
    print(f"saved metadata to {metadata_path}")
    print(
        "documents: "
        f"raw={total_raw}, kept={total_kept}, "
        f"empty={skipped_empty}, short={skipped_short}, "
        f"long={skipped_long}, dupes={duplicates_removed}"
    )


if __name__ == "__main__":
    main()
