#!/usr/bin/env python3
"""smoke_vllm.py — Smoke test vLLM OpenAI-compatible API.

Sends a few minimal requests to verify the server is running and producing
valid structured output. Uses the OpenAI Python client or raw HTTP.

Usage:
    python scripts/smoke_vllm.py                          # default localhost:8000
    python scripts/smoke_vllm.py --base-url http://host:8001
    python scripts/smoke_vllm.py --structured             # include structured output test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("[ERROR] 'requests' not installed. Run: pip install requests")
    sys.exit(1)


# ── Default config ────────────────────────────────────────────────────────
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


def check_server_health(base_url: str) -> dict:
    """Check /health endpoint."""
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        return r.status_code == 200, r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text
    except requests.ConnectionError:
        return False, {"error": "Connection refused — is the server running?"}
    except Exception as e:
        return False, {"error": str(e)}


def chat_completion(base_url: str, messages: list[dict], **kwargs) -> dict:
    """Send a chat completion request. Returns parsed response."""
    url = f"{base_url}/v1/chat/completions"
    _model = kwargs.pop("model", None) or get_served_model_name(base_url)
    payload = {
        "model": _model,
        "messages": messages,
        "max_tokens": kwargs.pop("max_tokens", 256),
        "temperature": kwargs.pop("temperature", 0.0),
        **kwargs,
    }
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=120)
    elapsed = time.time() - t0
    r.raise_for_status()
    data = r.json()
    data["_elapsed_s"] = round(elapsed, 3)
    return data


def run_smoke_tests(base_url: str, test_structured: bool = False) -> dict:
    """Run smoke test suite against the vLLM server."""
    model_name = get_served_model_name(base_url)
    print(f"Detected model name: {model_name}")
    results = {
        "server_base_url": base_url,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tests": [],
        "all_passed": True,
    }

    # ── Test 1: Health check ──
    print("\n[Test 1] Health Check")
    healthy, health_data = check_server_health(base_url)
    status = "PASS" if healthy else "FAIL"
    print(f"  Status: {status}")
    if healthy:
        print(f"  Response: {json.dumps(health_data, ensure_ascii=False)[:200]}")
    else:
        print(f"  Error: {health_data}")
    results["tests"].append({"name": "health_check", "status": status, "detail": health_data})
    if not healthy:
        results["all_passed"] = False
        return results

    # ── Test 2: Simple chat completion ──
    print("\n[Test 2] Simple Chat Completion")
    try:
        resp = chat_completion(base_url, [
            {"role": "user", "content": "你好，请用一句话介绍你自己。"},
        ], max_tokens=64, temperature=0.3)
        content = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})
        print(f"  Status: PASS")
        print(f"  Response: {content[:150]}{'...' if len(content) > 150 else ''}")
        print(f"  Tokens: prompt={usage.get('prompt_tokens', '?')}, completion={usage.get('completion_tokens', '?')}, total={usage.get('total_tokens', '?')}")
        print(f"  Latency: {resp.get('_elapsed_s', '?')}s")
        results["tests"].append({
            "name": "simple_chat",
            "status": "PASS",
            "response_preview": content[:200],
            "usage": usage,
            "latency_s": resp.get("_elapsed_s"),
        })
    except Exception as e:
        print(f"  Status: FAIL — {e}")
        results["tests"].append({"name": "simple_chat", "status": "FAIL", "error": str(e)})
        results["all_passed"] = False

    # ── Test 3: Structured JSON extraction ──
    print("\n[Test 3] Structured JSON Extraction")
    try:
        resp = chat_completion(base_url, [
            {"role": "system", "content": "你是一个严格遵循 schema 的信息抽取助手。请严格按照给定的 schema 从文本中抽取信息，并以 JSON 格式输出。"},
            {"role": "user", "content": (
                "Schema: [\"姓名\", \"职业\", \"代表作\"]\n"
                "文本: 鲁迅，原名周树人，是中国现代文学的奠基人之一，代表作品有《狂人日记》《阿Q正传》等。\n"
                "请从文本中抽取信息："
            )},
        ], max_tokens=256, temperature=0.0)
        content = resp["choices"][0]["message"]["content"]
        # Try parse as JSON
        try:
            parsed = json.loads(content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
            is_json = True
            fields = list(parsed.keys()) if isinstance(parsed, dict) else f"(list, len={len(parsed)})"
        except (json.JSONDecodeError, AttributeError):
            is_json = False
            fields = None

        status = "PASS" if is_json else "PARTIAL"
        print(f"  Status: {status} (JSON={'yes' if is_json else 'no'})")
        print(f"  Fields: {fields}")
        print(f"  Raw output: {content[:200]}{'...' if len(content) > 200 else ''}")
        results["tests"].append({
            "name": "structured_extraction",
            "status": status,
            "is_valid_json": is_json,
            "fields": fields,
            "raw_output": content[:300],
        })
        if not is_json:
            results["all_passed"] = False
    except Exception as e:
        print(f"  Status: FAIL — {e}")
        results["tests"].append({"name": "structured_extraction", "status": "FAIL", "error": str(e)})
        results["all_passed"] = False

    # ── Test 4: Multi-turn conversation ──
    print("\n[Test 4] Multi-turn Conversation")
    try:
        resp = chat_completion(base_url, [
            {"role": "system", "content": "你是一个严格遵循 schema 的信息抽取助手。"},
            {"role": "user", "content": "请抽取：Schema: [\"名称\", \"类型\"] 文本: 青霉素是一种抗生素药物，由亚历山大·弗莱明发现。"},
            {"role": "assistant", "content": '{"名称": "青霉素", "类型": "抗生素"}'},
            {"role": "user", "content": "现在请抽取这条：Schema: [\"名称\", \"类型\"] 文本: 胰岛素是一种用于治疗糖尿病的激素类药物。"},
        ], max_tokens=128, temperature=0.0)
        content = resp["choices"][0]["message"]["content"]
        print(f"  Status: PASS")
        print(f"  Response: {content[:200]}")
        results["tests"].append({
            "name": "multi_turn",
            "status": "PASS",
            "response_preview": content[:200],
        })
    except Exception as e:
        print(f"  Status: FAIL — {e}")
        results["tests"].append({"name": "multi_turn", "status": "FAIL", "error": str(e)})
        results["all_passed"] = False

    # ── Test 5: Structured output with response_format (optional) ──
    if test_structured:
        print("\n[Test 5] Structured Output (response_format=json_object)")
        try:
            resp = chat_completion(base_url, [
                {"role": "system", "content": "你是一个严格遵循 schema 的信息抽取助手。"},
                {"role": "user", "content": (
                    "Schema: [\"名称\", \"发现者\"]\n"
                    "文本: 盘尼西林（青霉素）是第一种抗生素，由亚历山大·弗莱明在1928年发现。\n"
                    "请抽取："
                )},
            ], max_tokens=256, temperature=0.0, response_format={"type": "json_object"})
            content = resp["choices"][0]["message"]["content"]
            try:
                parsed = json.loads(content.strip())
                is_json = True
            except (json.JSONDecodeError, AttributeError):
                is_json = False
            status = "PASS" if is_json else "PARTIAL"
            print(f"  Status: {status} (JSON={'yes' if is_json else 'no'})")
            print(f"  Output: {content[:200]}")
            results["tests"].append({
                "name": "structured_response_format",
                "status": status,
                "is_valid_json": is_json,
                "raw_output": content[:300],
            })
        except Exception as e:
            # Some servers may not support response_format
            print(f"  Status: SKIP — {e}")
            results["tests"].append({"name": "structured_response_format", "status": "SKIP", "error": str(e)})

    # ── Summary ──
    print(f"\n{'='*50}")
    passed = sum(1 for t in results["tests"] if t["status"] == "PASS")
    partial = sum(1 for t in results["tests"] if t["status"] == "PARTIAL")
    failed = sum(1 for t in results["tests"] if t["status"] in ("FAIL",))
    skipped = sum(1 for t in results["tests"] if t["status"] == "SKIP")
    total = len(results["tests"])
    print(f"Smoke Test Results: {passed} PASSED, {partial} PARTIAL, {failed} FAILED, {skipped} SKIPPED / {total} TOTAL")
    print(f"Overall: {'ALL PASSED' if results['all_passed'] else 'SOME TESTS FAILED'}")
    print(f"{'='*50}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Smoke test vLLM server")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--structured", action="store_true", help="Include response_format=json_object test")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    print(f"vLLM Smoke Test")
    print(f"Target: {args.base_url}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = run_smoke_tests(args.base_url, test_structured=args.structured)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Remove internal fields before saving
        save_results = {k: v for k, v in results.items() if k != "all_passed"}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(save_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {out_path}")

    sys.exit(0 if results["all_passed"] else 1)


if __name__ == "__main__":
    main()
