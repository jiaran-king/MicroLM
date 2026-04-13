from __future__ import annotations

import json
from pathlib import Path

from microlm.tokenizer import BPETokenizer, save_tokenizer_files, train_bpe
from microlm.training import SFTDataset, build_generation_prompt, render_chat_prompt


def build_tokenizer(tmp_path: Path) -> BPETokenizer:
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(
        "<|system|>\n<|user|>\n<|assistant|>\n<|endoftext|>\n"
        "你好\n世界\n你好，我是一个中文助手。\n",
        encoding="utf-8",
    )
    vocab, merges = train_bpe(
        input_path=str(corpus_path),
        vocab_size=280,
        special_tokens=["<|endoftext|>"],
    )

    tokenizer_dir = tmp_path / "tokenizer"
    save_tokenizer_files(vocab, merges, str(tokenizer_dir))
    return BPETokenizer.from_files(
        str(tokenizer_dir / "vocab.json"),
        str(tokenizer_dir / "merge.txt"),
        special_tokens=["<|endoftext|>"],
    )


def test_render_chat_prompt_and_generation_prompt() -> None:
    conversations = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
    ]

    rendered = render_chat_prompt(conversations, eos_token="<|endoftext|>")
    generation_prompt = build_generation_prompt([{"role": "user", "content": "你好"}])

    assert rendered.startswith("<|user|>\n你好\n<|assistant|>\n你好！\n<|endoftext|>\n")
    assert generation_prompt.endswith("<|assistant|>\n")


def test_sft_dataset_masks_only_assistant_tokens(tmp_path: Path) -> None:
    tokenizer = build_tokenizer(tmp_path)
    jsonl_path = tmp_path / "sft.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "conversations": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好，我是一个中文助手。"},
                ]
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = SFTDataset(
        jsonl_path=str(jsonl_path),
        tokenizer=tokenizer,
        max_length=64,
        eos_token="<|endoftext|>",
    )
    input_ids, labels = dataset[0]
    input_ids_list = input_ids.tolist()
    labels_list = labels.tolist()

    assistant_header_ids = tokenizer.encode("<|assistant|>\n")
    header_index = -1
    for index in range(len(input_ids_list) - len(assistant_header_ids) + 1):
        if input_ids_list[index : index + len(assistant_header_ids)] == assistant_header_ids:
            header_index = index
            break

    assert header_index >= 0
    first_label_index = next(i for i, value in enumerate(labels_list) if value != -100)
    assert first_label_index == header_index + len(assistant_header_ids)
    assert labels_list[first_label_index] == input_ids_list[first_label_index]
    assert all(value == -100 for value in labels_list[:first_label_index])


def test_sft_dataset_supports_tool_turns(tmp_path: Path) -> None:
    tokenizer = build_tokenizer(tmp_path)
    jsonl_path = tmp_path / "sft_tool.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "conversations": [
                    {"role": "user", "content": "查一下天气。"},
                    {"role": "tool", "content": "weather: sunny"},
                    {"role": "assistant", "content": "今天是晴天。"},
                ]
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = SFTDataset(
        jsonl_path=str(jsonl_path),
        tokenizer=tokenizer,
        max_length=64,
        eos_token="<|endoftext|>",
    )
    input_ids, labels = dataset[0]

    assert input_ids.shape == labels.shape == (64,)
    assert any(value != -100 for value in labels.tolist())
