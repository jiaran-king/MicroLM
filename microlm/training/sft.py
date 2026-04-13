from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


DEFAULT_CHAT_SYSTEM_PROMPTS = [
    "你是一个知识丰富的AI助手，请尽力给出准确、简洁的回答。",
    "你是一个可靠的中文助手，请根据用户问题给出有帮助的回复。",
    "You are a helpful AI assistant.",
    "You are a knowledgeable and concise assistant.",
]

ROLE_MARKERS = {
    "system": "<|system|>\n",
    "user": "<|user|>\n",
    "assistant": "<|assistant|>\n",
    "tool": "<|tool|>\n",
}


def normalize_conversations(conversations: list[dict[str, str]]) -> list[dict[str, str]]:
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("conversations must be a non-empty list")

    normalized: list[dict[str, str]] = []
    for index, message in enumerate(conversations):
        if not isinstance(message, dict):
            raise ValueError(f"conversation turn {index} must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"conversation turn {index} must contain string role/content")
        role = role.strip().lower()
        if role not in ROLE_MARKERS:
            raise ValueError(f"unsupported conversation role {role!r}")
        content = content.strip()
        if not content:
            continue
        normalized.append({"role": role, "content": content})

    if not normalized:
        raise ValueError("conversation list becomes empty after normalization")

    return normalized


def maybe_add_system_prompt(
    conversations: list[dict[str, str]],
    rng: random.Random,
    system_prompt_ratio: float,
    system_prompts: list[str] | None = None,
) -> list[dict[str, str]]:
    if not conversations:
        return conversations
    if conversations[0]["role"] == "system":
        return conversations
    if system_prompt_ratio <= 0.0:
        return conversations
    prompts = system_prompts or DEFAULT_CHAT_SYSTEM_PROMPTS
    if not prompts:
        return conversations
    if rng.random() >= system_prompt_ratio:
        return conversations
    injected = {"role": "system", "content": rng.choice(prompts)}
    return [injected, *conversations]


def render_chat_prompt(
    conversations: list[dict[str, str]],
    eos_token: str = "<|endoftext|>",
    add_generation_prompt: bool = False,
) -> str:
    parts: list[str] = []
    for message in conversations:
        role = message["role"]
        content = message["content"]
        parts.append(ROLE_MARKERS[role])
        parts.append(content)
        parts.append("\n")
        if role == "assistant":
            parts.append(eos_token)
            parts.append("\n")

    if add_generation_prompt:
        parts.append(ROLE_MARKERS["assistant"])

    return "".join(parts)


def _find_subsequence(sequence: list[int], pattern: list[int], start: int = 0) -> int:
    if not pattern:
        return start
    limit = len(sequence) - len(pattern) + 1
    for index in range(start, max(limit, start)):
        if sequence[index : index + len(pattern)] == pattern:
            return index
    return -1


def build_loss_labels(
    input_ids: list[int],
    tokenizer,
    max_length: int,
    assistant_header_ids: list[int],
    eos_boundary_ids: list[int],
    pad_token_id: int,
) -> list[int]:
    labels = [-100] * len(input_ids)
    index = 0
    while index < len(input_ids):
        header_index = _find_subsequence(input_ids, assistant_header_ids, start=index)
        if header_index < 0:
            break
        start = header_index + len(assistant_header_ids)
        end = _find_subsequence(input_ids, eos_boundary_ids, start=start)
        if end < 0:
            end = len(input_ids)
            boundary = end
        else:
            boundary = min(end + len(eos_boundary_ids), max_length)
        for position in range(start, min(boundary, len(input_ids))):
            if input_ids[position] != pad_token_id:
                labels[position] = input_ids[position]
        index = boundary if end >= 0 else len(input_ids)
    return labels


class SFTDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer,
        max_length: int = 1024,
        system_prompt_ratio: float = 0.0,
        seed: int = 42,
        eos_token: str = "<|endoftext|>",
        system_prompts: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.jsonl_path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt_ratio = system_prompt_ratio
        self.seed = seed
        self.eos_token = eos_token
        self.system_prompts = system_prompts or DEFAULT_CHAT_SYSTEM_PROMPTS
        self.assistant_header_ids = tokenizer.encode(ROLE_MARKERS["assistant"])
        self.eos_boundary_ids = tokenizer.encode(f"{eos_token}\n")
        eos_token_bytes = eos_token.encode("utf-8")
        if eos_token_bytes not in tokenizer.vocab_to_id:
            raise ValueError(f"EOS token {eos_token!r} is not in the tokenizer vocabulary")
        self.pad_token_id = tokenizer.vocab_to_id[eos_token_bytes]
        self._offsets: list[int] = []

        with self.jsonl_path.open("r", encoding="utf-8") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    self._offsets.append(offset)

        if not self._offsets:
            raise ValueError(f"No usable SFT samples found in {self.jsonl_path}")

    def __len__(self) -> int:
        return len(self._offsets)

    def _read_sample(self, index: int) -> dict[str, object]:
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            f.seek(self._offsets[index])
            return json.loads(f.readline())

    def _prepare_conversations(self, sample: dict[str, object], index: int) -> list[dict[str, str]]:
        conversations = sample.get("conversations")
        if not isinstance(conversations, list):
            raise ValueError("SFT sample must contain a conversations list")
        normalized = normalize_conversations(conversations)
        rng = random.Random(self.seed + index)
        return maybe_add_system_prompt(
            normalized,
            rng=rng,
            system_prompt_ratio=self.system_prompt_ratio,
            system_prompts=self.system_prompts,
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self._read_sample(index)
        conversations = self._prepare_conversations(sample, index)
        rendered = render_chat_prompt(conversations, eos_token=self.eos_token, add_generation_prompt=False)
        input_ids = self.tokenizer.encode(rendered)[: self.max_length]
        input_ids += [self.pad_token_id] * (self.max_length - len(input_ids))
        labels = build_loss_labels(
            input_ids=input_ids,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            assistant_header_ids=self.assistant_header_ids,
            eos_boundary_ids=self.eos_boundary_ids,
            pad_token_id=self.pad_token_id,
        )
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def build_generation_prompt(
    conversations: list[dict[str, str]],
    eos_token: str = "<|endoftext|>",
) -> str:
    normalized = normalize_conversations(conversations)
    if normalized[-1]["role"] == "assistant":
        raise ValueError("generation prompt should end with user/system turns, not assistant")
    return render_chat_prompt(normalized, eos_token=eos_token, add_generation_prompt=True)
