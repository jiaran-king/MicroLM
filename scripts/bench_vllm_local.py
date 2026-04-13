#!/usr/bin/env python3
"""bench_vllm_local.py — Local benchmark for vLLM-served Qwen 1.5B InstructIE LoRA.

Measures throughput (tok/s), Time-To-First-Token (TTFT), latency under
single and multi-concurrent scenarios using the OpenAI-compatible API.

Test matrix:
  Single-concurrency:
    - input_len=128,  output_len=64
    - input_len=512,  output_len=128
    - input_len=1024, output_len=256
  Multi-concurrency:
    - concurrency=4,  input_len=256, output_len=128
    - concurrency=8,  input_len=256, output_len=128

Each config: 1 warmup + 3 formal runs → report mean / min / max / p95.

Usage:
    python scripts/bench_vllm_local.py                          # full benchmark
    python scripts/bench_vllm_local.py --quick                  # single config only
    python scripts/bench_vllm_local.py --base-url http://host:8001
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import requests
except ImportError:
    print("[ERROR] 'requests' not installed. Run: pip install requests")
    sys.exit(1)


# ── Config ────────────────────────────────────────────────────────────────

DEFAULT_BASE_URL = "http://localhost:8000"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_served_model_name(base_url: str) -> str:
    """Query /v1/models to get the actual model ID served by vLLM."""
    try:
        r = requests.get(f"{base_url}/v1/models", timeout=10)
        data = r.json()
        if data.get("data") and len(data["data"]) > 0:
            return data["data"][0]["id"]
    except Exception:
        pass
    return "qwen"  # fallback


def generate_prompt_text(target_length: int) -> str:
    """Generate a prompt of approximately target_length Chinese characters."""
    # Use a repeating Chinese text pattern to reach target length
    base_text = (
        "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。"
        "这些任务包括学习、推理、问题解决、感知和语言理解。"
        "机器学习是人工智能的一个子领域，它使计算机能够从数据中学习和改进，而无需被明确编程。"
        "深度学习是机器学习的一个专门领域，它使用多层神经网络来模拟人脑的工作方式。"
        "自然语言处理（NLP）是人工智能的另一个重要分支，专注于使计算机能够理解、解释和生成人类语言。"
    )
    if target_length <= len(base_text):
        return base_text[:target_length]
    repeats = (target_length // len(base_text)) + 1
    result = (base_text * repeats)[:target_length]
    return result


# ── API Client ────────────────────────────────────────────────────────────

def send_completion(
    base_url: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    model: str = "qwen",
) -> dict:
    """Send a single chat completion request. Returns timing info + response."""
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    t_start = time.perf_counter()
    r = requests.post(url, json=payload, timeout=300)
    t_end = time.perf_counter()
    r.raise_for_status()
    data = r.json()

    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})

    # Estimate output token count from characters (rough for Chinese)
    # vLLM returns actual token counts in usage
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": usage.get("total_tokens", 0),
        "time_total_s": round(t_end - t_start, 4),
        "content_preview": content[:100],
        "status": "ok",
    }


def send_completion_with_streaming_ttft(
    base_url: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    model: str = "qwen",
) -> dict:
    """Send streaming request to measure TTFT accurately."""
    try:
        import sseclient
    except ImportError:
        # Fall back to non-streaming
        return send_completion(base_url, prompt, max_tokens, temperature, model)

    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    t_start = time.perf_counter()
    r = requests.post(url, json=payload, timeout=300, stream=True)
    r.raise_for_status()

    client = sseclient.SSEClient(r)
    ttft = None
    full_content = []
    completion_tokens = 0

    for event in client.events():
        if ttft is None:
            ttft = round(time.perf_counter() - t_start, 4)
        if event.data == "[DONE]":
            break
        try:
            chunk = json.loads(event.data)
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                full_content.append(content)
            token_info = chunk.get("usage") or {}
            completion_tokens = token_info.get("completion_tokens", completion_tokens)
        except (json.JSONDecodeError, KeyError, IndexError):
            continue

    t_end = time.perf_counter()

    return {
        "ttft_s": ttft,
        "completion_tokens": completion_tokens or len("".join(full_content)),
        "time_total_s": round(t_end - t_start, 4),
        "throughput_tok_s": round(completion_tokens / (t_end - t_start), 2) if (t_end - t_start) > 0 else 0,
        "status": "ok",
    }


# ── Benchmark Runner ──────────────────────────────────────────────────────

def run_single_config(
    base_url: str,
    config_name: str,
    input_len: int,
    output_len: int,
    warmup: int = 1,
    runs: int = 3,
    use_streaming: bool = False,
    model_name: str = "qwen",
) -> dict:
    """Run benchmark for a single configuration."""
    prompt = generate_prompt_text(input_len)
    print(f"\n  Config: {config_name} (input={input_len}, output={output_len})")

    results = []

    # Warmup
    for i in range(warmup):
        print(f"    Warmup {i+1}/{warmup}...", end=" ", flush=True)
        try:
            if use_streaming:
                send_completion_with_streaming_ttft(base_url, prompt, output_len, model=model_name)
            else:
                send_completion(base_url, prompt, output_len, model=model_name)
            print("OK")
        except Exception as e:
            print(f"WARN: {e}")

    # Formal runs
    run_results = []
    for i in range(runs):
        print(f"    Run {i+1}/{runs}...", end=" ", flush=True)
        try:
            if use_streaming:
                result = send_completion_with_streaming_ttft(base_url, prompt, output_len, model=model_name)
            else:
                result = send_completion(base_url, prompt, output_len, model=model_name)
            run_results.append(result)
            tok_s = result.get("throughput_tok_s") or (
                result["completion_tokens"] / result["time_total_s"]
                if result["time_total_s"] > 0 else 0
            )
            print(f"OK ({result['time_total_s']}s, ~{tok_s:.1f} tok/s)")
        except Exception as e:
            print(f"FAIL: {e}")
            run_results.append({"status": "error", "error": str(e)})

    if not run_results:
        return {"config": config_name, "status": "all_failed"}

    ok_runs = [r for r in run_results if r.get("status") == "ok"]
    if not ok_runs:
        return {"config": config_name, "status": "all_failed", "errors": [r.get("error") for r in run_results]}

    # Compute statistics
    times = [r["time_total_s"] for r in ok_runs]
    throughputs = [
        r.get("throughput_tok_s") or (r["completion_tokens"] / r["time_total_s"])
        for r in ok_runs
    ]
    ttfts = [r.get("ttft_s") for r in ok_runs if r.get("ttft_s") is not None]

    summary = {
        "config": config_name,
        "input_len": input_len,
        "target_output_len": output_len,
        "status": "ok",
        "runs": len(ok_runs),
        "time_total_s": {
            "mean": round(statistics.mean(times), 4),
            "min": round(min(times), 4),
            "max": round(max(times), 4),
            "p95": round(sorted(times)[int(len(times) * 0.95)] if len(times) > 1 else times[0], 4),
        },
        "throughput_tok_s": {
            "mean": round(statistics.mean(throughputs), 2),
            "min": round(min(throughputs), 2),
            "max": round(max(throughputs), 2),
        },
        "prompt_tokens": ok_runs[0].get("prompt_tokens", 0),
        "completion_tokens_avg": round(sum(r["completion_tokens"] for r in ok_runs) / len(ok_runs), 1),
    }
    if ttfts:
        summary["ttft_s"] = {
            "mean": round(statistics.mean(ttfts), 4),
            "min": round(min(ttfts), 4),
            "max": round(max(ttfts), 4),
        }
    else:
        # Estimate TTFT as fraction of total time (rough heuristic)
        summary["ttft_estimated_s"] = {
            "mean": round(statistics.mean(times) * 0.15, 4),  # prefill typically ~15% of total
        }

    return summary


def run_concurrent_benchmark(
    base_url: str,
    config_name: str,
    concurrency: int,
    input_len: int,
    output_len: int,
    warmup: int = 1,
    runs: int = 3,
    model_name: str = "qwen",
) -> dict:
    """Run benchmark with multiple concurrent requests."""
    prompt = generate_prompt_text(input_len)
    print(f"\n  Config: {config_name} (concurrency={concurrency}, input={input_len}, output={output_len})")

    def _single_request(idx: int) -> dict:
        return send_completion(base_url, prompt, output_len, model=model_name)

    all_run_summaries = []

    for run_idx in range(runs + warmup):
        is_warmup = run_idx < warmup
        label = f"Warmup {run_idx+1}/{warmup}" if is_warmup else f"Run {run_idx-warmup+1}/{runs}"
        print(f"    {label} ({concurrency} concurrent)...", end=" ", flush=True)

        t_batch_start = time.perf_counter()
        batch_results = []
        errors = 0

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(_single_request, i): i for i in range(concurrency)}
            for future in as_completed(futures):
                try:
                    batch_results.append(future.result())
                except Exception as e:
                    errors += 1
                    batch_results.append({"status": "error", "error": str(e)})

        t_batch_end = time.perf_counter()
        batch_time = t_batch_end - t_batch_start

        ok = [r for r in batch_results if r.get("status") == "ok"]
        print(f"OK ({batch_time:.2f}s, {len(ok)}/{concurrency} succeeded{', '+str(errors)+' errors' if errors else ''})")

        if not is_warmup:
            all_run_summaries.append({
                "batch_time_s": round(batch_time, 4),
                "n_success": len(ok),
                "n_errors": errors,
                "individual_times": [r["time_total_s"] for r in ok],
                "individual_throughputs": [
                    r["completion_tokens"] / r["time_total_s"] if r["time_total_s"] > 0 else 0
                    for r in ok
                ],
            })

    if not all_run_summaries:
        return {"config": config_name, "status": "all_failed"}

    # Aggregate across runs
    batch_times = [s["batch_time_s"] for s in all_run_summaries]
    all_throughputs = []
    for s in all_run_summaries:
        all_throughputs.extend(s["individual_throughputs"])

    return {
        "config": config_name,
        "concurrency": concurrency,
        "input_len": input_len,
        "target_output_len": output_len,
        "status": "ok",
        "runs": len(all_run_summaries),
        "batch_time_s": {
            "mean": round(statistics.mean(batch_times), 4),
            "min": round(min(batch_times), 4),
            "max": round(max(batch_times), 4),
        },
        "throughput_per_request_tok_s": {
            "mean": round(statistics.mean(all_throughputs), 2) if all_throughputs else 0,
            "min": round(min(all_throughputs), 2) if all_throughputs else 0,
            "max": round(max(all_throughputs), 2) if all_throughputs else 0,
        } if all_throughputs else {},
        "total_requests": sum(s["n_success"] for s in all_run_summaries),
        "total_errors": sum(s["n_errors"] for s in all_run_summaries),
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM server")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--quick", action="store_true", help="Only run one single-concurrency config")
    parser.add_argument("--output-dir", type=str, default=None, help="Save results directory")
    parser.add_argument("--streaming", action="store_true", help="Use streaming for TTFT measurement")
    args = parser.parse_args()

    print("=" * 60)
    print("  vLLM Benchmark — Qwen 1.5B InstructIE LoRA")
    print(f"  Target: {args.base_url}")
    print(f"  Time:   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Health check
    print("\n[Health Check]")
    try:
        r = requests.get(f"{args.base_url}/health", timeout=10)
        print(f"  Server status: {r.status_code} OK" if r.status_code == 200 else f"  Server status: {r.status_code}")
    except Exception as e:
        print(f"  ERROR: Cannot connect to server — {e}")
        print("  Make sure vLLM server is running: bash scripts/serve_vllm.sh")
        sys.exit(1)

    # Detect served model name
    model_name = get_served_model_name(args.base_url)
    print(f"  Model:   {model_name}")

    all_results = {
        "benchmark_config": {
            "base_url": args.base_url,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "streaming": args.streaming,
        },
        "single_concurrency": [],
        "multi_concurrency": [],
    }

    # ── Single-concurrency benchmarks ──
    print("\n[SINGLE CONCURRENCY]")
    single_configs = [
        ("sc_128_64", 128, 64),
        ("sc_512_128", 512, 128),
        ("sc_1024_256", 1024, 256),
    ]
    if args.quick:
        single_configs = [("sc_quick", 256, 128)]

    for name, inp, out in single_configs:
        result = run_single_config(args.base_url, name, inp, out, use_streaming=args.streaming, model_name=model_name)
        all_results["single_concurrency"].append(result)
        if result["status"] == "ok":
            tp = result["throughput_tok_s"]["mean"]
            tt = result.get("ttft_s", result.get("ttft_estimated_s", {}))
            tt_str = f"TTFT={tt['mean']}" if isinstance(tt, dict) else ""
            print(f"    → {tp:.1f} tok/s avg | total={result['time_total_s']['mean']}s {tt_str}")

    # ── Multi-concurrency benchmarks ──
    if not args.quick:
        print("\n[MULTI CONCURRENCY]")
        multi_configs = [
            ("mc_4conc", 4, 256, 128),
            ("mc_8conc", 8, 256, 128),
        ]

        for name, conc, inp, out in multi_configs:
            result = run_concurrent_benchmark(args.base_url, name, conc, inp, out, model_name=model_name)
            all_results["multi_concurrency"].append(result)
            if result["status"] == "ok":
                bt = result["batch_time_s"]["mean"]
                tp = result.get("throughput_per_request_tok_s", {}).get("mean", 0)
                err = result.get("total_errors", 0)
                print(f"    → batch={bt:.2f}s avg | ~{tp:.1f} tok/s/req | {err} errors")

    # ── Save results ──
    out_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "results" / "vllm_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    json_path = out_dir / f"benchmark_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {json_path}")

    # Also save CSV summary
    csv_path = out_dir / f"benchmark_summary_{timestamp}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow([
            "config", "type", "input_len", "output_len", "concurrency",
            "time_mean_s", "time_min_s", "time_max_s",
            "tok_s_mean", "tok_s_min", "tok_s_max",
            "ttft_mean_s", "runs", "errors",
        ])
        for r in all_results["single_concurrency"]:
            if r["status"] != "ok":
                continue
            tt = r.get("ttft_s", r.get("ttft_estimated_s", {}))
            tt_val = tt.get("mean", "") if isinstance(tt, dict) else ""
            writer.writerow([
                r["config"], "single", r["input_len"], r["target_output_len"], "",
                r["time_total_s"]["mean"], r["time_total_s"]["min"], r["time_total_s"]["max"],
                r["throughput_tok_s"]["mean"], r["throughput_tok_s"]["min"], r["throughput_tok_s"]["max"],
                tt_val, r["runs"], "",
            ])
        for r in all_results["multi_concurrency"]:
            if r["status"] != "ok":
                continue
            tp = r.get("throughput_per_request_tok_s", {})
            writer.writerow([
                r["config"], "multi", r["input_len"], r["target_output_len"], r["concurrency"],
                r["batch_time_s"]["mean"], r["batch_time_s"]["min"], r["batch_time_s"]["max"],
                tp.get("mean", ""), tp.get("min", ""), tp.get("max", ""),
                "", r["runs"], r.get("total_errors", ""),
            ])
    print(f"CSV summary saved to {csv_path}")

    # Print final table
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<18} {'Type':<8} {'In':>5} {'Out':>5} {'Time(s)':>10} {'Tok/s':>10} {'TTFT(s)':>10}")
    print("-" * 68)
    for r in all_results["single_concurrency"]:
        if r["status"] != "ok":
            print(f"{r['config']:<18} {'—':>8}")
            continue
        tt = r.get("ttft_s", r.get("ttft_estimated_s", {}))
        tt_val = f"{tt.get('mean', '')}" if isinstance(tt, dict) else ""
        print(f"{r['config']:<18} {'single':>8} {r['input_len']:>5} {r['target_output_len']:>5} "
              f"{r['time_total_s']['mean']:>10.3f} {r['throughput_tok_s']['mean']:>10.2f} {tt_val:>10}")
    for r in all_results["multi_concurrency"]:
        if r["status"] != "ok":
            print(f"{r['config']:<18} {'—':>8}")
            continue
        tp = r.get("throughput_per_request_tok_s", {})
        print(f"{r['config']:<18} {'multi':>8} {r['input_len']:>5} {r['target_output_len']:>5} "
              f"{r['batch_time_s']['mean']:>10.3f} {tp.get('mean', ''):>10} conc={r['concurrency']}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
