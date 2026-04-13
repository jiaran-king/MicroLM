from __future__ import annotations

import json
from pathlib import Path

import pytest

from microlm.inference import load_conversations_from_json, load_conversations_from_path, resolve_generation_prompt


def test_resolve_generation_prompt_from_conversations_json() -> None:
    raw = json.dumps(
        [
            {"role": "user", "content": "你好"},
        ],
        ensure_ascii=False,
    )

    prompt = resolve_generation_prompt(
        prompt="fallback prompt",
        conversations_json=raw,
        conversations_path=None,
        eos_token="<|endoftext|>",
    )

    assert prompt.startswith("<|user|>\n你好\n<|assistant|>\n")
    assert prompt.endswith("<|assistant|>\n")


def test_load_conversations_from_path(tmp_path: Path) -> None:
    conversations_path = tmp_path / "conversations.json"
    conversations_path.write_text(
        json.dumps(
            [
                {"role": "system", "content": "你是一个助手。"},
                {"role": "user", "content": "介绍一下你自己。"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    conversations = load_conversations_from_path(conversations_path)

    assert conversations == [
        {"role": "system", "content": "你是一个助手。"},
        {"role": "user", "content": "介绍一下你自己。"},
    ]


def test_load_conversations_from_json_rejects_invalid_payload() -> None:
    with pytest.raises(ValueError, match="non-empty list"):
        load_conversations_from_json("[]")

    with pytest.raises(ValueError, match="role/content"):
        load_conversations_from_json(json.dumps([{"role": "user"}], ensure_ascii=False))


def test_resolve_generation_prompt_rejects_multiple_chat_sources(tmp_path: Path) -> None:
    conversations_path = tmp_path / "conversations.json"
    conversations_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="only one"):
        resolve_generation_prompt(
            prompt="fallback prompt",
            conversations_json="[]",
            conversations_path=conversations_path,
            eos_token="<|endoftext|>",
        )
