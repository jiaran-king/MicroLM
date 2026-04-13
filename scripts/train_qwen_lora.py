#!/usr/bin/env python3
"""train_qwen_lora.py — LoRA fine-tuning of Qwen2.5-1.5B-Instruct on InstructIE data.

Usage:
    python scripts/train_qwen_lora.py --config configs/qwen_lora_structured_smoke.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel


# ── Config ──────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Dataset ─────────────────────────────────────────────────────────

class InstructIEDataset(Dataset):
    """Loads 6A chat-style JSONL and builds (input_ids, labels) with loss masking.

    Only assistant tokens contribute to loss; everything else is masked with -100.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        system_prompt: str | None = None,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.samples = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                messages = record["messages"]
                processed = self._process_sample(messages)
                if processed is not None:
                    self.samples.append(processed)

        print(f"  Loaded {len(self.samples)} samples from {data_path}")

    def _process_sample(self, messages: list[dict]) -> dict | None:
        # Build full message list with system prompt
        full_messages = []
        if self.system_prompt:
            full_messages.append({"role": "system", "content": self.system_prompt})
        for msg in messages:
            full_messages.append(msg)

        # Tokenize full conversation
        full_ids = self.tokenizer.apply_chat_template(
            full_messages, tokenize=True, truncation=True, max_length=self.max_length
        )
        if len(full_ids) < 2:
            return None

        # Build prefix (everything before assistant output)
        prefix_messages = full_messages[:-1]  # exclude assistant
        prefix_ids = self.tokenizer.apply_chat_template(
            prefix_messages, tokenize=True, add_generation_prompt=True
        )

        # Verify prefix matches
        if full_ids[:len(prefix_ids)] != prefix_ids:
            # Fallback: just use full_ids, mask everything except last portion
            prefix_len = len(prefix_ids)
        else:
            prefix_len = len(prefix_ids)

        # Build labels: -100 for non-assistant tokens
        labels = [-100] * prefix_len + full_ids[prefix_len:]

        # Truncate to max_length
        input_ids = full_ids[:self.max_length]
        labels = labels[:self.max_length]

        return {"input_ids": input_ids, "labels": labels}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.tensor(s["input_ids"], dtype=torch.long),
            torch.tensor(s["labels"], dtype=torch.long),
        )


def collate_fn(batch):
    """Pad sequences to the same length within a batch."""
    input_ids_list, labels_list = zip(*batch)
    max_len = max(len(ids) for ids in input_ids_list)

    pad_id = 151643  # Qwen's pad token id
    padded_inputs = []
    padded_labels = []
    attention_masks = []

    for ids, labs in zip(input_ids_list, labels_list):
        pad_len = max_len - len(ids)
        padded_inputs.append(torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)]))
        padded_labels.append(torch.cat([labs, torch.full((pad_len,), -100, dtype=torch.long)]))
        mask = torch.cat([torch.ones(len(ids)), torch.zeros(pad_len)])
        attention_masks.append(mask)

    return (
        torch.stack(padded_inputs),
        torch.stack(padded_labels),
        torch.stack(attention_masks).bool(),
    )


# ── Training ────────────────────────────────────────────────────────

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for input_ids, labels, attention_mask in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            loss_mask = (shift_labels != -100).float()

            # Per-token cross entropy
            per_token_loss = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction="none",
            ).reshape(shift_labels.shape)

            n_tokens = loss_mask.sum().item()
            if n_tokens > 0:
                total_loss += (per_token_loss * loss_mask).sum().item()
                total_tokens += n_tokens

    model.train()
    return total_loss / total_tokens if total_tokens > 0 else float("nan")


def print_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable / total * 100 if total > 0 else 0
    print(f"Total params: {total:,} | Trainable (LoRA): {trainable:,} ({ratio:.2f}%)")


def train(cfg: dict):
    set_seed(cfg["training"]["seed"])
    device = cfg["training"]["device"]
    out_dir = Path(cfg["training"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    with open(out_dir / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    # ── Load tokenizer ──────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"], trust_remote_code=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load model ──────────────────────────────────────────────
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        trust_remote_code=True,
        torch_dtype=torch.float16 if cfg["training"].get("fp16") else torch.float32,
        device_map=device,
    )

    # ── Apply LoRA ──────────────────────────────────────────────
    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["targets"],
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    print_trainable_params(model)

    # ── Load data ───────────────────────────────────────────────
    print("Loading data...")
    system_prompt = cfg.get("system_prompt")
    max_length = cfg["training"]["max_length"]

    train_ds = InstructIEDataset(
        cfg["data"]["train_data_path"],
        tokenizer=tokenizer,
        max_length=max_length,
        system_prompt=system_prompt,
        seed=cfg["training"]["seed"],
    )
    valid_ds = InstructIEDataset(
        cfg["data"]["valid_data_path"],
        tokenizer=tokenizer,
        max_length=max_length,
        system_prompt=system_prompt,
        seed=cfg["training"]["seed"],
    )

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    # ── Optimizer ───────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"].get("weight_decay", 0.01),
    )

    # ── Training loop ───────────────────────────────────────────
    grad_accum = cfg["training"].get("gradient_accumulation_steps", 1)
    max_steps = cfg["training"]["max_steps"]
    eval_interval = cfg["training"]["eval_interval"]
    save_interval = cfg["training"]["save_interval"]
    warmup_steps = cfg["training"].get("warmup_steps", 0)
    max_grad_norm = cfg["training"].get("max_grad_norm", 1.0)

    # Linear warmup + constant lr
    def get_lr_multiplier(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_multiplier)

    log_path = out_dir / "train_log.jsonl"
    best_val_loss = float("inf")
    train_iter = iter(train_loader)

    print(f"\nStarting training: {max_steps} steps, grad_accum={grad_accum}")
    print(f"  Train samples: {len(train_ds)}, Valid samples: {len(valid_ds)}")
    print(f"  Effective batch size: {batch_size * grad_accum}")
    print(f"  Device: {device}, FP16: {cfg['training'].get('fp16', False)}")

    model.train()
    t0 = time.time()

    for step in range(max_steps):
        # Accumulate gradients
        optimizer.zero_grad()
        accumulated_loss = 0.0

        for micro_step in range(grad_accum):
            try:
                input_ids, labels, attention_mask = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                input_ids, labels, attention_mask = next(train_iter)

            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum
            loss.backward()
            accumulated_loss += loss.item()

        # Clip & step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        completed_step = step + 1
        train_loss = accumulated_loss

        # Eval
        if completed_step % eval_interval == 0 or completed_step == max_steps:
            val_loss = evaluate(model, valid_loader, device)
            elapsed = time.time() - t0
            print(
                f"Step {completed_step}/{max_steps} | "
                f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
                f"lr {scheduler.get_last_lr()[0]:.2e} | "
                f"elapsed {elapsed:.0f}s"
            )
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "step": completed_step,
                    "train_loss": round(train_loss, 6),
                    "val_loss": round(val_loss, 6),
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_sec": round(elapsed, 1),
                }, ensure_ascii=False) + "\n")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_pretrained(out_dir / "best_adaptor")
                print(f"  -> New best val_loss: {val_loss:.4f}")

        # Save checkpoint
        if completed_step % save_interval == 0 or completed_step == max_steps:
            ckpt_dir = out_dir / f"ckpt_step_{completed_step}"
            model.save_pretrained(ckpt_dir)
            print(f"  -> Saved checkpoint: {ckpt_dir}")

    # Save final
    model.save_pretrained(out_dir / "adaptor_final")
    print(f"\nTraining complete. Final adaptor saved to {out_dir / 'adaptor_final'}")

    # Print summary
    if log_path.exists():
        with open(log_path, "r") as f:
            lines = f.readlines()
        if lines:
            last = json.loads(lines[-1])
            first = json.loads(lines[0])
            print(f"  Steps: {first['step']} -> {last['step']}")
            print(f"  Train loss: {first['train_loss']:.4f} -> {last['train_loss']:.4f}")
            print(f"  Val loss: {first['val_loss']:.4f} -> {last['val_loss']:.4f}")
            print(f"  Best val loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning of Qwen on InstructIE data")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
