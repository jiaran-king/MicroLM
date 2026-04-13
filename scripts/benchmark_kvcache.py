"""Benchmark KV Cache vs no-cache generation for MicroLM.

Usage:
    python scripts/benchmark_kvcache.py \
        --checkpoint outputs/sft_baseline/ckpt_final.pt \
        --out-dir results/

Outputs:
    results/kvcache_benchmark.csv   — raw benchmark data
    results/kvcache_benchmark.json  — structured results with metadata
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from microlm.model import TransformerLM
from microlm.model.transformer import KVCache
from microlm.tokenizer import BPETokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark KV Cache for MicroLM")
    p.add_argument("--checkpoint", type=Path, default=Path("outputs/sft_baseline/ckpt_final.pt"))
    p.add_argument("--vocab-path", type=Path, default=Path("outputs/tokenizer_full_clean/vocab.json"))
    p.add_argument("--merges-path", type=Path, default=Path("outputs/tokenizer_full_clean/merge.txt"))
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    p.add_argument("--warmup-runs", type=int, default=2)
    p.add_argument("--bench-runs", type=int, default=5)
    p.add_argument("--eos-token", type=str, default="</s>")
    return p.parse_args()


# ─── Model loading (reused from run_eval_prompts.py) ────────────────────────

def load_model(checkpoint_path: Path, device: str, dtype: torch.dtype) -> TransformerLM:
    config_path = checkpoint_path.parent / "model_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    cleaned = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}

    # Handle LoRA checkpoint keys
    is_lora_ckpt = any("original.weight" in k for k in cleaned)
    if is_lora_ckpt:
        remapped = {}
        for k, v in cleaned.items():
            if k.endswith(".original.weight"):
                remapped[k.replace(".original.weight", ".weight")] = v
            elif ".lora_" in k:
                continue
            else:
                remapped[k] = v
        cleaned = remapped

    model = TransformerLM(
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=float(cfg.get("rope_theta", 1000000.0)),
        use_rms_norm=True,
        norm_mode="pre",
        ffn_type="swiglu",
        device=device,
        dtype=dtype,
    ).to(device)
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    return model


# ─── Generation with NO cache (recompute full sequence each step) ───────────

@torch.no_grad()
def generate_no_cache(
    model: TransformerLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
) -> tuple[torch.Tensor, float]:
    """Autoregressive decode without KV Cache — each step recomputes full sequence."""
    generated = prompt_ids.clone()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()

    for _ in range(max_new_tokens):
        # Feed entire sequence, only take last logit
        input_ids = generated[:, -model.context_length:]
        logits = model(input_ids)[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - t0
    return generated, elapsed


# ─── Generation WITH KV Cache (prefill + incremental decode) ────────────────

@torch.no_grad()
def generate_with_cache(
    model: TransformerLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
) -> tuple[torch.Tensor, float, float, float]:
    """Autoregressive decode with KV Cache — returns (output, total_time, prefill_time, decode_time)."""
    generated = prompt_ids.clone()

    # Phase 1: Prefill
    kv_cache = KVCache(len(model.layers))
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_prefill_start = time.perf_counter()

    logits, kv_cache = model.forward(
        prompt_ids, kv_cache=kv_cache, use_cache=True, start_pos=0,
    )
    logits = logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    generated = torch.cat((generated, next_token), dim=1)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    prefill_time = time.perf_counter() - t_prefill_start

    # Phase 2: Decode
    t_decode_start = time.perf_counter()

    for _ in range(max_new_tokens - 1):
        cur_pos = generated.shape[1] - 1
        logits, kv_cache = model.forward(
            next_token, kv_cache=kv_cache, use_cache=True, start_pos=cur_pos,
        )
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    decode_time = time.perf_counter() - t_decode_start
    total_time = prefill_time + decode_time

    return generated, total_time, prefill_time, decode_time


# ─── Benchmark runner ───────────────────────────────────────────────────────

def run_benchmark(
    model: TransformerLM,
    prompt_lengths: list[int],
    gen_lengths: list[int],
    vocab_size: int,
    warmup_runs: int,
    bench_runs: int,
    device: str,
) -> list[dict]:
    results = []

    for prompt_len in prompt_lengths:
        # Create a random prompt token sequence
        torch.manual_seed(42)
        prompt_ids = torch.randint(
            0, vocab_size, (1, prompt_len), device=device, dtype=torch.long,
        )

        for gen_len in gen_lengths:
            # Safety: prompt + gen must fit in context_length
            if prompt_len + gen_len > model.context_length:
                continue

            # Warmup
            for _ in range(warmup_runs):
                generate_no_cache(model, prompt_ids.clone(), min(gen_len, 8))
                generate_with_cache(model, prompt_ids.clone(), min(gen_len, 8))

            # Benchmark runs
            no_cache_times = []
            cache_times = []
            cache_prefill_times = []
            cache_decode_times = []

            for run_idx in range(bench_runs):
                torch.manual_seed(42 + run_idx)

                # No cache
                _, nc_time = generate_no_cache(model, prompt_ids.clone(), gen_len)
                no_cache_times.append(nc_time)

                # With cache
                torch.manual_seed(42 + run_idx)
                _, c_total, c_prefill, c_decode = generate_with_cache(
                    model, prompt_ids.clone(), gen_len,
                )
                cache_times.append(c_total)
                cache_prefill_times.append(c_prefill)
                cache_decode_times.append(c_decode)

            nc_avg = sum(no_cache_times) / len(no_cache_times)
            c_avg = sum(cache_times) / len(cache_times)
            c_prefill_avg = sum(cache_prefill_times) / len(cache_prefill_times)
            c_decode_avg = sum(cache_decode_times) / len(cache_decode_times)
            speedup = nc_avg / c_avg if c_avg > 0 else float("inf")
            decode_tps = gen_len / c_decode_avg if c_decode_avg > 0 else 0
            no_cache_tps = gen_len / nc_avg if nc_avg > 0 else 0

            row = {
                "prompt_len": prompt_len,
                "gen_len": gen_len,
                "no_cache_time_s": round(nc_avg, 4),
                "no_cache_tps": round(no_cache_tps, 1),
                "cache_total_time_s": round(c_avg, 4),
                "cache_prefill_time_s": round(c_prefill_avg, 4),
                "cache_decode_time_s": round(c_decode_avg, 4),
                "cache_decode_tps": round(decode_tps, 1),
                "speedup": round(speedup, 2),
            }
            results.append(row)
            print(
                f"  prompt={prompt_len:>3d}  gen={gen_len:>3d}  "
                f"no_cache={nc_avg:.3f}s ({no_cache_tps:.0f} tok/s)  "
                f"cache={c_avg:.3f}s ({decode_tps:.0f} tok/s decode)  "
                f"speedup={speedup:.2f}x"
            )

    return results


# ─── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    dtype_map = {"float32": torch.float32, "float16": torch.float16}
    dtype = dtype_map[args.dtype]

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.device, dtype)
    print(f"  d_model={model.token_embeddings.weight.shape[1]}, "
          f"layers={len(model.layers)}, ctx={model.context_length}")
    print(f"  device={args.device}, dtype={args.dtype}")

    vocab_size = model.token_embeddings.weight.shape[0]

    # Test matrix
    prompt_lengths = [16, 32, 64, 128, 256]
    gen_lengths = [32, 64, 128, 256]

    print(f"\nRunning benchmark (warmup={args.warmup_runs}, runs={args.bench_runs})...")
    print(f"  prompt_lengths={prompt_lengths}")
    print(f"  gen_lengths={gen_lengths}\n")

    results = run_benchmark(
        model, prompt_lengths, gen_lengths, vocab_size,
        args.warmup_runs, args.bench_runs, args.device,
    )

    # Save CSV
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "kvcache_benchmark.csv"
    if results:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nCSV saved to {csv_path}")

    # Save JSON with metadata
    json_path = args.out_dir / "kvcache_benchmark.json"
    output = {
        "benchmark_config": {
            "checkpoint": str(args.checkpoint),
            "device": args.device,
            "dtype": args.dtype,
            "vocab_size": vocab_size,
            "d_model": model.token_embeddings.weight.shape[1],
            "num_layers": len(model.layers),
            "context_length": model.context_length,
            "warmup_runs": args.warmup_runs,
            "bench_runs": args.bench_runs,
            "prompt_lengths": prompt_lengths,
            "gen_lengths": gen_lengths,
        },
        "results": results,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"JSON saved to {json_path}")

    # Print summary
    if results:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        avg_speedup = sum(r["speedup"] for r in results) / len(results)
        max_speedup = max(r["speedup"] for r in results)
        min_speedup = min(r["speedup"] for r in results)
        avg_decode_tps = sum(r["cache_decode_tps"] for r in results) / len(results)
        print(f"  Speedup range: {min_speedup:.2f}x ~ {max_speedup:.2f}x")
        print(f"  Average speedup: {avg_speedup:.2f}x")
        print(f"  Average decode throughput (cache): {avg_decode_tps:.0f} tok/s")


if __name__ == "__main__":
    main()
