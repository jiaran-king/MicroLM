from __future__ import annotations

import json
from pathlib import Path

from microlm.training import build_generation_prompt


def _normalize_conversations(raw_conversations: object) -> list[dict[str, str]]:
    if not isinstance(raw_conversations, list) or not raw_conversations:
        raise ValueError("conversations must be a non-empty list")

    conversations: list[dict[str, str]] = []
    for index, message in enumerate(raw_conversations):
        if not isinstance(message, dict):
            raise ValueError(f"conversation turn {index} must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"conversation turn {index} must contain string role/content")
        conversations.append({"role": role, "content": content})
    return conversations


def load_conversations_from_json(raw_json: str) -> list[dict[str, str]]:
    parsed = json.loads(raw_json)
    return _normalize_conversations(parsed)


def load_conversations_from_path(path: str | Path) -> list[dict[str, str]]:
    conversation_path = Path(path)
    return load_conversations_from_json(conversation_path.read_text(encoding="utf-8"))


def resolve_generation_prompt(
    prompt: str | None,
    conversations_json: str | None,
    conversations_path: str | Path | None,
    eos_token: str = "<|endoftext|>",
) -> str:
    if conversations_json is not None and conversations_path is not None:
        raise ValueError("Provide only one of conversations_json or conversations_path")

    if conversations_json is not None:
        conversations = load_conversations_from_json(conversations_json)
        return build_generation_prompt(conversations, eos_token=eos_token)

    if conversations_path is not None:
        conversations = load_conversations_from_path(conversations_path)
        return build_generation_prompt(conversations, eos_token=eos_token)

    if prompt is None or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string")

    return prompt
