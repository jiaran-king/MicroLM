"""MicroLM interactive multi-turn chat REPL.

Supports SFT and LoRA checkpoints, runtime parameter switching,
conversation history with context-length aware truncation, and session logging.

Usage:
    # SFT checkpoint
    python scripts/chat.py \
        --checkpoint-path outputs/sft_baseline/ckpt_final.pt \
        --config-path outputs/sft_baseline/model_config.json \
        --vocab-path outputs/tokenizer_full_clean/vocab.json \
        --merges-path outputs/tokenizer_full_clean/merge.txt \
        --eos-token "</s>"

    # LoRA checkpoint
    python scripts/chat.py \
        --checkpoint-path outputs/pretrain_full_corpus/ckpt_final.pt \
        --lora-path outputs/sft_lora/lora_adaptor.pt \
        --config-path outputs/sft_baseline/model_config.json \
        --vocab-path outputs/tokenizer_full_clean/vocab.json \
        --merges-path outputs/tokenizer_full_clean/merge.txt \
        --eos-token "</s>"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import torch

from microlm.model import TransformerLM, apply_lora_to_model, load_lora_state_dict, merge_lora
from microlm.tokenizer import BPETokenizer
from microlm.training.sft import ROLE_MARKERS


def _remove_surrogates(text: str) -> str:
    """Remove Unicode surrogate characters that crash .encode('utf-8').

    Small LMs can produce token sequences that decode to surrogate code points
    (U+D800–U+DFFF).  Storing these in conversation history and re-encoding on
    the next turn raises ``UnicodeEncodeError``.  Stripping them is safe because
    they carry no semantic meaning.
    """
    # Python's regex module supports \p{Cs} (surrogate code points)
    import regex as re
    return re.sub(r"[\ud800-\udfff]", "", text)


# ---------------------------------------------------------------------------
# Model / tokenizer loading helpers (adapted from generate_text.py)
# ---------------------------------------------------------------------------

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


def load_model_config(config_path: Path, vocab_size: int) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {
        "vocab_size": int(raw.get("vocab_size", vocab_size)),
        "context_length": int(raw["context_length"]),
        "d_model": int(raw["d_model"]),
        "num_layers": int(raw["num_layers"]),
        "num_heads": int(raw["num_heads"]),
        "d_ff": int(raw["d_ff"]),
        "rope_theta": float(raw.get("rope_theta", 10000.0)),
    }


def normalize_state_dict_keys(state_dict: OrderedDict) -> OrderedDict:
    normalized = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]
        normalized[key] = value
    return normalized


def load_state_dict(checkpoint_path: Path, device: str) -> OrderedDict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, (dict, OrderedDict)):
        raise TypeError(f"Unsupported checkpoint format at {checkpoint_path}")
    return normalize_state_dict_keys(OrderedDict(state_dict))


# ---------------------------------------------------------------------------
# Chat session
# ---------------------------------------------------------------------------

class ChatSession:
    """Manages multi-turn conversation state and model interaction."""

    def __init__(
        self,
        model: TransformerLM,
        tokenizer: BPETokenizer,
        eos_token: str,
        context_length: int,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.9,
        system_prompt: str | None = None,
        log_path: str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token = eos_token
        self.context_length = context_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt
        self.conversations: list[dict[str, str]] = []
        self.log_path = log_path
        self._log_file = None

        # Resolve eos_token_id — only use it if the token is within model's vocab range
        eos_bytes = eos_token.encode("utf-8")
        model_vocab_size = model.token_embeddings.weight.shape[0]
        if eos_bytes in tokenizer.vocab_to_id:
            eid = tokenizer.vocab_to_id[eos_bytes]
            self.eos_token_id = eid if eid < model_vocab_size else None
        else:
            self.eos_token_id = None

        if log_path:
            self._log_file = open(log_path, "a", encoding="utf-8")

        if system_prompt:
            self._log_entry({"role": "system", "content": system_prompt})

    # ---- conversation management ------------------------------------------

    def _build_prompt_conversations(self) -> list[dict[str, str]]:
        """Build the conversation list with optional system prompt prepended."""
        convs = []
        if self.system_prompt:
            convs.append({"role": "system", "content": self.system_prompt})
        convs.extend(self.conversations)
        return convs

    def _truncate_conversations(self, prompt_ids: list[int]) -> list[dict[str, str]]:
        """Truncate earliest turns if prompt exceeds context budget.

        Keeps system prompt + most recent turns. Removes complete
        (user, assistant) pairs from the front.
        """
        budget = self.context_length - self.max_new_tokens - 16  # 16 token margin
        if len(prompt_ids) <= budget:
            return self._build_prompt_conversations()

        convs = self._build_prompt_conversations()
        # System prompt is always at index 0 if present
        has_system = convs and convs[0]["role"] == "system"
        system_part = [convs[0]] if has_system else []
        dialogue = convs[1:] if has_system else convs

        # Remove pairs from the front until it fits
        while dialogue and len(prompt_ids) > budget:
            # Remove the first turn (could be user or assistant)
            dialogue.pop(0)
            # Re-encode to check length
            trial = system_part + dialogue
            if dialogue:
                try:
                    rendered = self._render_prompt(trial)
                    prompt_ids = self.tokenizer.encode(rendered)
                except Exception:
                    break
            else:
                break

        return system_part + dialogue

    def _render_prompt(self, convs: list[dict[str, str]]) -> str:
        """Render conversation list into a prompt string for generation.

        Must match the SFT training format (render_chat_prompt in sft.py):
        - Each assistant message is followed by eos_token + "\\n" (only if
          eos_token_id is within the model's vocab range)
        - Final assistant marker is appended to trigger generation
        """
        parts: list[str] = []
        for message in convs:
            role = message["role"]
            content = message["content"]
            parts.append(ROLE_MARKERS[role])
            parts.append(content)
            parts.append("\n")
            if role == "assistant" and self.eos_token_id is not None:
                # Only append EOS if its token ID is valid for this model
                parts.append(self.eos_token)
                parts.append("\n")
        # Append assistant marker to trigger generation
        parts.append(ROLE_MARKERS["assistant"])
        return "".join(parts)

    def chat(self, user_input: str) -> str:
        """Process one user turn and return the assistant's reply."""
        self.conversations.append({"role": "user", "content": user_input})
        self._log_entry({"role": "user", "content": user_input})

        # Build prompt
        convs = self._build_prompt_conversations()
        prompt_text = self._render_prompt(convs)
        prompt_ids = self.tokenizer.encode(prompt_text)

        # Safety: clip token IDs that exceed model vocab size (e.g. EOS token
        # may be at index 6400 while model embedding only has 6400 slots)
        model_vocab_size = self.model.token_embeddings.weight.shape[0]
        prompt_ids = [tid for tid in prompt_ids if tid < model_vocab_size]

        # Truncate if needed
        budget = self.context_length - self.max_new_tokens - 16
        if len(prompt_ids) > budget:
            convs = self._truncate_conversations(prompt_ids)
            prompt_text = self._render_prompt(convs)
            prompt_ids = self.tokenizer.encode(prompt_text)

        if not prompt_ids:
            reply = "[Error: prompt is empty after truncation]"
            self.conversations.append({"role": "assistant", "content": reply})
            return reply

        # Ensure we don't exceed context
        max_prompt_len = self.context_length - self.max_new_tokens
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:]

        prompt_tensor = torch.tensor(
            [prompt_ids], dtype=torch.long, device=next(self.model.parameters()).device
        )

        # Generate
        with torch.no_grad():
            if self.temperature == 0.0:
                # Greedy
                generated = prompt_tensor.clone()
                for _ in range(self.max_new_tokens):
                    idx_cond = generated[:, -self.model.context_length:]
                    logits = self.model(idx_cond)[:, -1, :]
                    if self.top_p < 1.0:
                        logits = self.model._top_p_filter(logits, self.top_p)
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    generated = torch.cat((generated, next_token), dim=1)
                    if self.eos_token_id is not None and (next_token == self.eos_token_id).all():
                        break
                full_ids = generated[0].tolist()
            else:
                output = self.model.generate(
                    prompt_ids=prompt_tensor,
                    max_new_tokens=self.max_new_tokens,
                    eos_token_id=self.eos_token_id,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                full_ids = output[0].tolist()

        new_ids = full_ids[len(prompt_ids):]
        reply = self.tokenizer.decode(new_ids).strip()

        # Sanitize: remove surrogate characters that would crash re-encoding
        reply = _remove_surrogates(reply)

        # Clean up EOS token from reply
        if self.eos_token and reply.endswith(self.eos_token):
            reply = reply[: -len(self.eos_token)].strip()

        self.conversations.append({"role": "assistant", "content": reply})
        self._log_entry({"role": "assistant", "content": reply})

        return reply

    # ---- logging ----------------------------------------------------------

    def _log_entry(self, entry: dict[str, str]) -> None:
        if self._log_file is None:
            return
        entry_with_ts = {
            "role": entry["role"],
            "content": entry["content"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._log_file.write(json.dumps(entry_with_ts, ensure_ascii=False) + "\n")
        self._log_file.flush()

    def save_log(self, path: str | None = None) -> None:
        """Save full conversation to a JSONL file."""
        target = path or self.log_path
        if not target:
            print("[No log path specified. Use /save <path>]")
            return
        with open(target, "w", encoding="utf-8") as f:
            # Write system prompt if present
            if self.system_prompt:
                entry = {
                    "role": "system",
                    "content": self.system_prompt,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            for conv in self.conversations:
                entry = {
                    "role": conv["role"],
                    "content": conv["content"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"[Session saved to {target}]")

    def close(self) -> None:
        if self._log_file:
            self._log_file.close()

    # ---- REPL commands ----------------------------------------------------

    def clear_history(self) -> None:
        self.conversations.clear()
        print("[Conversation history cleared]")

    def show_history(self) -> None:
        if self.system_prompt:
            print(f"  system: {self.system_prompt}")
        for i, conv in enumerate(self.conversations):
            role = conv["role"]
            content = conv["content"]
            # Truncate long content for display
            if len(content) > 100:
                content = content[:100] + "..."
            print(f"  [{role}]: {content}")

    def set_temperature(self, value: str) -> None:
        try:
            self.temperature = float(value)
            print(f"[temperature set to {self.temperature}]")
        except ValueError:
            print(f"[Invalid value: {value}]")

    def set_top_p(self, value: str) -> None:
        try:
            v = float(value)
            if not (0.0 <= v <= 1.0):
                raise ValueError
            self.top_p = v
            print(f"[top_p set to {self.top_p}]")
        except ValueError:
            print(f"[Invalid value: {value}. Must be in [0.0, 1.0]]")

    def set_system_prompt(self, text: str) -> None:
        self.system_prompt = text if text else None
        if self.system_prompt:
            print(f"[system prompt set: {self.system_prompt}]")
        else:
            print("[system prompt cleared]")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_chat_model(
    checkpoint_path: Path,
    config_path: Path,
    vocab_path: Path,
    merges_path: Path,
    eos_token: str,
    lora_path: Path | None = None,
    dtype: str = "float32",
    device: str = "auto",
) -> tuple[TransformerLM, BPETokenizer, dict]:
    """Load model, tokenizer, and config. Returns (model, tokenizer, config)."""
    device = resolve_device(device)
    torch_dtype = resolve_model_dtype(dtype, device)

    special_tokens = [eos_token] if eos_token else []
    tokenizer = BPETokenizer.from_files(
        str(vocab_path),
        str(merges_path),
        special_tokens=special_tokens,
    )

    config = load_model_config(config_path, vocab_size=len(tokenizer.id_to_vocab))

    model = TransformerLM(
        vocab_size=int(config["vocab_size"]),
        context_length=int(config["context_length"]),
        d_model=int(config["d_model"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        d_ff=int(config["d_ff"]),
        rope_theta=float(config["rope_theta"]),
        device=device,
        dtype=torch_dtype,
    ).to(device)

    state_dict = load_state_dict(checkpoint_path, device)
    model.load_state_dict(state_dict)

    # Apply LoRA if specified
    if lora_path is not None:
        apply_lora_to_model(model)
        lora_state = torch.load(lora_path, map_location=device, weights_only=False)
        load_lora_state_dict(model, lora_state)
        merge_lora(model)

    model.eval()
    return model, tokenizer, config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MicroLM interactive multi-turn chat.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
REPL commands:
  /temp <value>       Set sampling temperature
  /topp <value>       Set top-p (nucleus sampling threshold)
  /system <text>      Set or clear system prompt
  /clear              Clear conversation history
  /history            Show current conversation history
  /save [path]        Save session to JSONL file
  /help               Show this help
  /quit               Exit
""",
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--lora-path", type=Path, default=None, help="Optional LoRA adaptor path")
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, required=True)
    parser.add_argument("--merges-path", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--system-prompt", type=str, default=None)
    parser.add_argument("--log", type=str, default=None, help="Path to save session log (JSONL)")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--eos-token", type=str, default="</s>")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


HELP_TEXT = """\
REPL commands:
  /temp <value>       Set sampling temperature
  /topp <value>       Set top-p (nucleus sampling threshold)
  /system <text>      Set or clear system prompt (empty to clear)
  /clear              Clear conversation history
  /history            Show current conversation history
  /save [path]        Save session to JSONL file
  /help               Show this help
  /quit               Exit

Type your message to chat with the model. Press Ctrl+C or /quit to exit.
"""


def repl(session: ChatSession) -> None:
    """Run the interactive read-eval-print loop."""
    print("=" * 60)
    print("  MicroLM Chat  (type /help for commands, /quit to exit)")
    print("=" * 60)
    print(f"  temperature={session.temperature}  top_p={session.top_p}  "
          f"max_new_tokens={session.max_new_tokens}")
    if session.system_prompt:
        print(f"  system: {session.system_prompt}")
    print()

    try:
        while True:
            try:
                user_input = input("你> ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            # Handle REPL commands
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if cmd in ("/quit", "/exit", "/q"):
                    break
                elif cmd == "/help":
                    print(HELP_TEXT)
                elif cmd == "/temp":
                    if arg:
                        session.set_temperature(arg)
                    else:
                        print(f"[current temperature: {session.temperature}]")
                elif cmd == "/topp":
                    if arg:
                        session.set_top_p(arg)
                    else:
                        print(f"[current top_p: {session.top_p}]")
                elif cmd == "/system":
                    session.set_system_prompt(arg)
                elif cmd == "/clear":
                    session.clear_history()
                elif cmd == "/history":
                    session.show_history()
                elif cmd == "/save":
                    session.save_log(arg if arg else None)
                else:
                    print(f"[Unknown command: {cmd}. Type /help for available commands]")
                continue

            # Normal chat
            start = time.time()
            reply = session.chat(user_input)
            elapsed = time.time() - start
            print(f"\nAI> {reply}")
            print(f"    [{elapsed:.1f}s]")
            print()

    except KeyboardInterrupt:
        print()
    finally:
        session.close()
        print("[Session ended]")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    print("Loading model...")
    model, tokenizer, config = load_chat_model(
        checkpoint_path=args.checkpoint_path,
        config_path=args.config_path,
        vocab_path=args.vocab_path,
        merges_path=args.merges_path,
        eos_token=args.eos_token,
        lora_path=args.lora_path,
        dtype=args.dtype,
        device=args.device,
    )
    device = resolve_device(args.device)
    print(f"Model loaded on {device} (context_length={config['context_length']})")
    if args.lora_path:
        print(f"LoRA adaptor loaded from {args.lora_path} (merged)")

    session = ChatSession(
        model=model,
        tokenizer=tokenizer,
        eos_token=args.eos_token,
        context_length=int(config["context_length"]),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        system_prompt=args.system_prompt,
        log_path=args.log,
    )

    repl(session)


if __name__ == "__main__":
    main()
