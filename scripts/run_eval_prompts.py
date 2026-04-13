"""Run fixed evaluation prompts against multiple checkpoints and save results.

Usage:
    python scripts/run_eval_prompts.py \
        --eval-file eval/prompts_baseline.json \
        --models pretrain=outputs/pretrain_full_corpus/ckpt_final.pt \
                 baseline=outputs/sft_baseline/ckpt_final.pt \
                 lora=outputs/sft_lora/ckpt_final.pt \
        --out-dir results/lora_vs_full_sft
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from microlm.model import TransformerLM
from microlm.model.lora import load_lora_state_dict, merge_lora
from microlm.tokenizer import BPETokenizer
from microlm.training import build_generation_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run eval prompts against multiple checkpoints.")
    parser.add_argument("--eval-file", type=Path, default=Path("eval/prompts_baseline.json"))
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        metavar="NAME=PATH",
        help="Model checkpoints as name=path pairs.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/lora_vs_full_sft"))
    parser.add_argument("--vocab-path", type=Path, default=Path("outputs/tokenizer_full_clean/vocab.json"))
    parser.add_argument("--merges-path", type=Path, default=Path("outputs/tokenizer_full_clean/merge.txt"))
    parser.add_argument("--special-token", action="append", dest="special_tokens", default=None)
    parser.add_argument("--eos-token", type=str, default="</s>")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--lora-adaptor", type=Path, default=None,
                        help="Path to a lora_adaptor.pt to apply to the 'lora' model.")
    return parser.parse_args()


def parse_model_specs(specs: list[str]) -> list[tuple[str, Path]]:
    models = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid model spec '{spec}'. Use NAME=PATH format.")
        name, path = spec.split("=", 1)
        models.append((name, Path(path)))
    return models


def load_model(
    checkpoint_path: Path,
    model_config: dict,
    device: str,
    dtype: torch.dtype,
    lora_adaptor_path: Path | None = None,
) -> TransformerLM:
    import torch.nn as nn

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    cleaned = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}

    # Detect LoRA checkpoint: keys contain "original.weight"
    is_lora_ckpt = any("original.weight" in k for k in cleaned)
    # Remap LoRA checkpoint keys: strip LoRA wrapper to get plain Linear weights
    if is_lora_ckpt:
        remapped = {}
        for k, v in cleaned.items():
            if k.endswith(".original.weight"):
                remapped[k.replace(".original.weight", ".weight")] = v
            elif ".lora_" in k:
                continue  # skip lora A/B, use adaptor file instead
            else:
                remapped[k] = v
        cleaned = remapped

    # Determine actual vocab_size from checkpoint weights
    ckpt_vocab_size = cleaned.get("token_embeddings.weight").shape[0]

    model = TransformerLM(
        vocab_size=ckpt_vocab_size,
        context_length=int(model_config["context_length"]),
        d_model=int(model_config["d_model"]),
        num_layers=int(model_config["num_layers"]),
        num_heads=int(model_config["num_heads"]),
        d_ff=int(model_config["d_ff"]),
        rope_theta=float(model_config.get("rope_theta", 1000000.0)),
        use_rms_norm=True,
        norm_mode="pre",
        ffn_type="swiglu",
        device=device,
        dtype=dtype,
    ).to(device)

    model.load_state_dict(cleaned, strict=True)

    # Resize if tokenizer has more tokens than checkpoint
    tokenizer_vocab = model_config.get("_tokenizer_vocab_size", ckpt_vocab_size)
    if tokenizer_vocab > ckpt_vocab_size:
        d_model = int(model_config["d_model"])
        old_emb = model.token_embeddings.weight.data
        new_emb = torch.zeros(tokenizer_vocab, d_model, device=old_emb.device, dtype=old_emb.dtype)
        new_emb[:old_emb.shape[0]] = old_emb
        model.token_embeddings.weight = nn.Parameter(new_emb)

        old_head = model.lm_head.weight.data
        new_head = torch.zeros(tokenizer_vocab, d_model, device=old_head.device, dtype=old_head.dtype)
        new_head[:old_head.shape[0]] = old_head
        model.lm_head.weight = nn.Parameter(new_head)
        print(f"  Resized vocab: {ckpt_vocab_size} -> {tokenizer_vocab}")

    if lora_adaptor_path is not None and lora_adaptor_path.exists():
        from microlm.model.lora import apply_lora_to_model
        apply_lora_to_model(model, r=8, alpha=16.0)
        lora_sd = torch.load(lora_adaptor_path, map_location=device, weights_only=True)
        load_lora_state_dict(model, lora_sd)
        merge_lora(model)
        print(f"  Loaded and merged LoRA adaptor from {lora_adaptor_path}")

    model.eval()
    return model


def generate(
    model: TransformerLM,
    tokenizer: BPETokenizer,
    prompt_text: str,
    eos_token_id: int | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
) -> tuple[str, list[int]]:
    token_ids = tokenizer.encode(prompt_text)
    prompt_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        if temperature == 0.0:
            generated = prompt_tensor.clone()
            for _ in range(max_new_tokens):
                logits = model(generated[:, -model.context_length:])[:, -1, :]
                if top_p < 1.0:
                    logits = model._top_p_filter(logits, top_p)
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                generated = torch.cat((generated, next_token), dim=1)
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break
            new_ids = generated[0].tolist()[len(token_ids):]
        else:
            out = model.generate(
                prompt_ids=prompt_tensor,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                temperature=temperature,
                top_p=top_p,
            )
            new_ids = out[0].tolist()[len(token_ids):]

    return tokenizer.decode(new_ids), new_ids


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)

    special_tokens = args.special_tokens or [args.eos_token]
    tokenizer = BPETokenizer.from_files(
        str(args.vocab_path),
        str(args.merges_path),
        special_tokens=special_tokens,
    )

    eos_token_id = tokenizer.vocab_to_id.get(args.eos_token.encode("utf-8"))

    with args.eval_file.open("r", encoding="utf-8") as f:
        eval_data = json.load(f)

    gen_params = eval_data["generation_params"]
    prompts = eval_data["prompts"]
    temperature = gen_params["temperature"]
    top_p = gen_params["top_p"]
    max_new_tokens = gen_params["max_new_tokens"]

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    model_specs = parse_model_specs(args.models)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for model_name, checkpoint_path in model_specs:
        if not checkpoint_path.exists():
            print(f"SKIP {model_name}: checkpoint not found at {checkpoint_path}")
            continue

        print(f"\n=== Evaluating: {model_name} ({checkpoint_path}) ===")

        # Try loading model_config.json from the same directory
        config_path = checkpoint_path.parent / "model_config.json"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                model_config = json.load(f)
            # Pass tokenizer vocab size for potential resize
            model_config["_tokenizer_vocab_size"] = len(tokenizer.id_to_vocab)
        else:
            raise FileNotFoundError(f"No model_config.json found next to {checkpoint_path}")

        lora_path = args.lora_adaptor if model_name == "lora" else None
        model = load_model(checkpoint_path, model_config, args.device, dtype, lora_path)

        results = []
        total_time = 0.0
        for prompt_item in prompts:
            prompt_id = prompt_item["id"]
            category = prompt_item["category"]

            # Build generation prompt
            if "conversations" in prompt_item:
                prompt_text = build_generation_prompt(
                    prompt_item["conversations"], eos_token=args.eos_token,
                )
            else:
                prompt_text = prompt_item["prompt_text"]

            torch.manual_seed(gen_params["seed"])
            t0 = time.time()
            output_text, output_ids = generate(
                model, tokenizer, prompt_text, eos_token_id,
                max_new_tokens, temperature, top_p, args.device,
            )
            elapsed = time.time() - t0
            total_time += elapsed

            result = {
                "id": prompt_id,
                "category": category,
                "input": prompt_item.get("conversations") or prompt_item.get("prompt_text"),
                "output": output_text,
                "output_tokens": len(output_ids),
                "latency_s": round(elapsed, 3),
            }
            results.append(result)
            print(f"  [{category}] {prompt_id}: {output_text[:80]}{'...' if len(output_text) > 80 else ''}")

        avg_latency = total_time / len(results) if results else 0
        print(f"  Total: {total_time:.1f}s, Avg: {avg_latency:.2f}s/prompt")

        all_results[model_name] = {
            "checkpoint": str(checkpoint_path),
            "generation_params": gen_params,
            "results": results,
            "total_time_s": round(total_time, 2),
            "avg_latency_s": round(avg_latency, 3),
        }

        # Free GPU memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save results
    out_file = args.out_dir / "eval_results.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
