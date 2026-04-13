#!/usr/bin/env python3
"""check_structured_stability.py — Verify structured output stability on vLLM-served model.

Reuses the InstructIE evaluation prompt set (eval/prompts_instructie.json) to verify
that the vLLM-deployed qwen_lora model maintains its structured output quality.

Runs TWO rounds:
  Round 1: Normal chat completion (no format constraint)
  Round 2: Constrained completion with response_format=json_object (if supported)

For each round, computes:
  - Parse%     (JSON parseable rate)
  - Strict%    (strict schema match rate — all 4 checks pass)
  - Alias-Strict% (alias-normalized strict rate)
  - Per-group breakdown (extraction / schema_constraint / format_following)

Usage:
    python scripts/check_structured_stability.py                          # full test
    python scripts/check_structured_stability.py --rounds 1               # round 1 only
    python scripts/check_structured_stability.py --base-url http://host:8001
    python scripts/check_structured_stability.py --limit 5                # quick check
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    import requests
except ImportError:
    print("[ERROR] 'requests' not installed. Run: pip install requests")
    sys.exit(1)


# ── Paths & Defaults ──────────────────────────────────────────────────────

DEFAULT_BASE_URL = "http://localhost:8000"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_PROMPTS_PATH = PROJECT_ROOT / "eval" / "prompts_instructie.json"


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


# ══════════════════════════════════════════════════════════════════════════
# Detection Functions (adapted from run_instructie_eval.py)
# ══════════════════════════════════════════════════════════════════════════

def clean_model_output(raw: str) -> str:
    s = raw.strip()
    m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", s, re.DOTALL)
    if m:
        s = m.group(1).strip()
    return s


def try_parse_json(text: str):
    try:
        return json.loads(text), True
    except (json.JSONDecodeError, ValueError):
        return None, False


# Alias map (same as run_instructie_eval.py)
FIELD_ALIASES = {
    "创办者": ["创始人", "创建者", "建立者", "发起人"],
    "位于": ["位置", "地点", "所在地", "地址", "所在"],
    "发现者或发明者": ["发现者", "发明者", "发现人", "发明人"],
    "创建或成立时间": ["建造时间", "创建时间", "建立时间", "建成时间"],
    "发生时间": ["时间", "举办时间", "开始时间"],
    "发生地点": ["地点", "位置", "举办地"],
    "常见并发症": ["并发症", "合并症"],
    "症状": ["主要症状", "常见症状", "临床表现"],
    "治疗方法": ["治疗", "疗法", "治疗方式"],
    "成就": ["获奖", "奖项", "荣誉", "成就奖"],
    "子组织": ["旗下", "下属组织", "子公司", "分支机构"],
    "别名": ["又称", "又名", "别称", "也叫"],
    "组成": ["成分", "原料", "组成部分", "构成"],
    "用途": ["应用", "应用领域", "主要用途", "作用"],
    "所属科室": ["科室", "所属科"],
    "线路": ["所属线路", "路线"],
    "车站等级": ["等级"],
    "开通时间": ["启用时间", "运营时间", "通车时间"],
    "保护级别": ["濒危等级", "保护等级"],
    "成立时间": ["成立年份", "创立时间", "创建时间"],
    "出生地": ["出生地点", "籍贯"],
    "出生日期": ["生日", "出生年月"],
    "参与者": ["参加者", "参赛者", "参赛方"],
    "起因": ["原因", "导火索"],
    "导致": ["结果", "后果"],
}


def normalize_field_name(field: str) -> str:
    for canonical, aliases in FIELD_ALIASES.items():
        if field == canonical or field in aliases:
            return canonical
    return field


def extract_all_fields(parsed) -> set[str]:
    fields = set()
    if isinstance(parsed, dict):
        for key, val in parsed.items():
            fields.add(key)
            if isinstance(val, dict):
                fields.update(val.keys())
    return fields


def score_output(raw_output: str, prompt_item: dict) -> dict:
    """Run 4-detection + alias-normalized scoring on a single output."""
    cleaned = clean_model_output(raw_output)
    parsed, is_parseable = try_parse_json(cleaned)

    schema_def = prompt_item.get("schema_def", {})
    required = schema_def.get("required_fields", [])
    allowed = schema_def.get("allowed_fields", [])
    enums = schema_def.get("enum_constraints", {})

    result = {
        "id": prompt_item["id"],
        "group": prompt_item["group"],
        "parsed": is_parseable,
        "raw_output": raw_output[:500],
    }

    if not is_parseable or parsed is None:
        result.update({
            "missing_fields": required,
            "extra_fields": [],
            "schema_strict": False,
            "schema_strict_alias": False,
        })
        return result

    # Collect all fields (handles entity-keyed nested format)
    all_fields_raw = extract_all_fields(parsed)
    missing = [f for f in required if f not in all_fields_raw]
    extra = [f for f in all_fields_raw if f not in set(allowed)]

    # Enum check
    enum_ok = True
    if enums:
        for _entity, fields in (parsed.items() if isinstance(parsed, dict) else []):
            if not isinstance(fields, dict):
                continue
            for fname, fval in fields.items():
                if fname in enums:
                    allowed_vals = set(enums[fname])
                    vals = [fval] if isinstance(fval, str) else fval if isinstance(fval, list) else []
                    for v in vals:
                        if v not in allowed_vals:
                            enum_ok = False

    schema_strict = len(missing) == 0 and len(extra) == 0 and enum_ok

    # Alias-normalized
    all_fields_norm = set(normalize_field_name(f) for f in all_fields_raw)
    missing_alias = [f for f in required if f not in all_fields_norm]
    extra_alias = [f for f in all_fields_raw if normalize_field_name(f) not in set(allowed)]
    schema_strict_alias = len(missing_alias) == 0 and len(extra_alias) == 0 and enum_ok

    result.update({
        "missing_fields": missing,
        "extra_fields": extra,
        "enum_ok": enum_ok if enums else None,
        "schema_strict": schema_strict,
        "schema_strict_alias": schema_strict_alias,
        "missing_fields_alias": missing_alias,
        "extra_fields_alias": extra_alias,
    })
    return result


# ══════════════════════════════════════════════════════════════════════════
# Prompt Builder (same as run_instructie_eval.py)
# ══════════════════════════════════════════════════════════════════════════

def build_prompt_text(prompt_item: dict) -> str:
    instruction = prompt_item.get("instruction", "")
    schema_list = prompt_item.get("schema", [])
    input_text = prompt_item.get("input", "")

    parts = []
    if instruction:
        parts.append(instruction)
    if schema_list and "Schema:" not in instruction and "schema" not in instruction.lower():
        parts.append(f"Schema: {json.dumps(schema_list, ensure_ascii=False)}")
    if "文本:" not in instruction and "从文本" not in instruction:
        parts.append(f"文本: {input_text}")
    else:
        parts.append(input_text)

    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════
# API Client
# ══════════════════════════════════════════════════════════════════════════

def api_chat_completion(
    base_url: str,
    messages: list[dict],
    max_tokens: int = 256,
    temperature: float = 0.0,
    response_format: dict | None = None,
    model: str = "qwen",
) -> tuple[str, dict]:
    """Send chat completion request. Returns (output_text, usage_info)."""
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if response_format:
        payload["response_format"] = response_format

    t0 = time.time()
    r = requests.post(url, json=payload, timeout=120)
    elapsed = time.time() - t0
    r.raise_for_status()
    data = r.json()

    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return content, {"elapsed_s": round(elapsed, 3), **usage}


# ══════════════════════════════════════════════════════════════════════════
# Evaluation Rounds
# ══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = "你是一个严格遵循 schema 的信息抽取助手。请严格按照给定的 schema 从文本中抽取信息，并以 JSON 格式输出。不要在 JSON 前后添加任何解释性文字。"


def run_round(
    base_url: str,
    prompts: list[dict],
    round_name: str,
    use_response_format: bool = False,
    limit: int = 0,
    model_name: str = "qwen",
) -> dict:
    """Run one evaluation round against the vLLM API."""
    print(f"\n{'='*60}")
    print(f"Round: {round_name}")
    print(f"  Mode: {'constrained (response_format=json_object)' if use_response_format else 'normal chat completion'}")
    print(f"  Prompts: {len(prompts) if limit == 0 else min(limit, len(prompts))}")
    print(f"{'='*60}")

    if limit > 0:
        prompts = prompts[:limit]

    scored_results = []
    total_time = 0.0
    errors = 0

    for i, prompt_item in enumerate(prompts):
        prompt_text = build_prompt_text(prompt_item)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]

        try:
            fmt = {"type": "json_object"} if use_response_format else None
            output, usage_info = api_chat_completion(
                base_url, messages,
                max_tokens=256, temperature=0.0,
                response_format=fmt,
                model=model_name,
            )
            total_time += usage_info.get("elapsed_s", 0)
        except Exception as e:
            print(f"  [{i+1}/{len(prompts)}] {prompt_item['id']}: ERROR — {e}")
            errors += 1
            scored_results.append({
                "id": prompt_item["id"],
                "group": prompt_item["group"],
                "parsed": False,
                "schema_strict": False,
                "schema_strict_alias": False,
                "error": str(e),
            })
            continue

        scored = score_output(output, prompt_item)
        scored["latency_s"] = usage_info.get("elapsed_s", 0)
        scored_results.append(scored)

        p_flag = "Y" if scored["parsed"] else "N"
        s_flag = "Y" if scored["schema_strict"] else "N"
        a_flag = "Y" if scored["schema_strict_alias"] else "N"
        preview = output[:80].replace("\n", " ")
        print(f"  [{i+1}/{len(prompts)}] {prompt_item['group'][:4]:>4} {prompt_item['id']}: "
              f"parse={p_flag} strict={s_flag} alias={a_flag} | {preview}...")

    n = len(scored_results)
    if n == 0:
        return {"round": round_name, "status": "all_failed"}

    # Compute summary stats
    parse_count = sum(1 for r in scored_results if r["parsed"])
    strict_count = sum(1 for r in scored_results if r["schema_strict"])
    alias_strict_count = sum(1 for r in scored_results if r.get("schema_strict_alias"))

    summary = {
        "round": round_name,
        "mode": "constrained" if use_response_format else "normal",
        "total": n,
        "errors": errors,
        "parse_rate": round(parse_count / n, 4),
        "strict_rate": round(strict_count / n, 4),
        "alias_strict_rate": round(alias_strict_count / n, 4),
        "total_time_s": round(total_time, 2),
        "avg_latency_s": round(total_time / n, 3) if n > 0 else 0,
        "results": scored_results,
    }

    # Per-group breakdown
    groups = ["extraction", "schema_constraint", "format_following"]
    by_group = {}
    for g in groups:
        group_results = [r for r in scored_results if r["group"] == g]
        gn = len(group_results)
        if gn == 0:
            continue
        by_group[g] = {
            "total": gn,
            "parse_rate": round(sum(1 for r in group_results if r["parsed"]) / gn, 4),
            "strict_rate": round(sum(1 for r in group_results if r["schema_strict"]) / gn, 4),
            "alias_strict_rate": round(sum(1 for r in group_results if r.get("schema_strict_alias")) / gn, 4),
        }
    summary["by_group"] = by_group

    # Print summary
    print(f"\n  --- {round_name} Summary ---")
    print(f"  Parse%:       {summary['parse_rate']:.1%} ({parse_count}/{n})")
    print(f"  Strict%:      {summary['strict_rate']:.1%} ({strict_count}/{n})")
    print(f"  Alias-Strict%:{summary['alias_strict_rate']:.1%} ({alias_strict_count}/{n})")
    print(f"  Errors:       {errors}")
    print(f"  Total time:   {summary['total_time_s']}s")
    if by_group:
        print(f"  By group:")
        for g, gs in by_group.items():
            print(f"    {g:<22} P={gs['parse_rate']:.1%} S={gs['strict_rate']:.1%} A={gs['alias_strict_rate']:.1%}")

    return summary


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Check structured output stability on vLLM")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--eval-file", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=2, help="Number of rounds (1=normal only, 2=+constrained)")
    parser.add_argument("--limit", type=int, default=0, help="Limit prompts per round (0=all)")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    eval_path = Path(args.eval_file) if args.eval_file else EVAL_PROMPTS_PATH
    if not eval_path.exists():
        print(f"[ERROR] Eval prompts not found at {eval_path}")
        sys.exit(1)

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    prompts = eval_data["prompts"]

    gen_params = eval_data.get("generation_params", {})
    print("=" * 60)
    print("  Structured Output Stability Check — vLLM Deployed Model")
    print(f"  Target:      {args.base_url}")
    print(f"  Eval file:   {eval_path.name}")
    print(f"  Prompts:     {len(prompts)}")
    print(f"  Rounds:      {args.rounds}")
    print(f"  Time:        {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Health check
    try:
        r = requests.get(f"{args.base_url}/health", timeout=10)
        assert r.status_code == 200
        print(f"\nServer health: OK")
    except Exception as e:
        print(f"\n[ERROR] Cannot connect to server: {e}")
        sys.exit(1)

    # Detect served model name
    model_name = get_served_model_name(args.base_url)
    print(f"  Model: {model_name}")

    out_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "results" / "vllm_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rounds = []

    # ── Round 1: Normal completion ──
    r1 = run_round(args.base_url, prompts, "Round 1: Normal Chat Completion",
                   use_response_format=False, limit=args.limit, model_name=model_name)
    all_rounds.append(r1)

    # ── Round 2: Constrained completion ──
    if args.rounds >= 2:
        r2 = run_round(args.base_url, prompts, "Round 2: Constrained (response_format=json_object)",
                       use_response_format=True, limit=args.limit, model_name=model_name)
        all_rounds.append(r2)

    # ── Comparison table ──
    print(f"\n{'='*70}")
    print("STABILITY CHECK COMPARISON")
    print(f"{'='*70}")
    print(f"{'Round':<45} {'Parse%':>8} {'Strict%':>9} {'Alias-S%':>10}")
    print("-" * 75)
    for rd in all_rounds:
        mode_tag = " [constrained]" if rd["mode"] == "constrained" else ""
        print(f"{rd['round']:<45}{rd['parse_rate']:>7.1%}{rd['strict_rate']:>8.1%}{rd['alias_strict_rate']:>9.1%}{mode_tag}")

    # Compare with 6C offline results (reference)
    print(f"\n--- Reference: 6C Offline Results (qwen_lora) ---")
    print(f"{'Config':<45} {'Parse%':>8} {'Strict%':>9} {'Alias-S%':>10}")
    print(f"{'6C offline (run_instructie_eval.py)':<45}{'97.5%':>8}{'7.5%':>9}{'15.0%':>10}")

    # ── Save results ──
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Full JSON with results
    save_data = {
        "check_config": {
            "base_url": args.base_url,
            "eval_file": str(eval_path),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rounds": args.rounds,
        },
        "rounds": [
            {
                "round": rd["round"],
                "mode": rd["mode"],
                "total": rd["total"],
                "parse_rate": rd["parse_rate"],
                "strict_rate": rd["strict_rate"],
                "alias_strict_rate": rd["alias_strict_rate"],
                "by_group": rd.get("by_group", {}),
                "avg_latency_s": rd["avg_latency_s"],
            }
            for rd in all_rounds
        ],
        "reference_6c_offline": {
            "model": "qwen_lora",
            "parse_rate": 0.975,
            "strict_rate": 0.075,
            "alias_strict_rate": 0.150,
        },
    }
    json_out = out_dir / f"stability_{timestamp}.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {json_out}")

    # Summary CSV
    csv_out = out_dir / f"stability_summary_{timestamp}.csv"
    import csv
    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "mode", "total", "parse_rate", "strict_rate", "alias_strict_rate",
                         "extraction_P", "extraction_S", "extraction_A",
                         "schema_P", "schema_S", "schema_A",
                         "format_P", "format_S", "format_A",
                         "avg_latency_s"])
        for rd in all_rounds:
            bg = rd.get("by_group", {})
            eg = bg.get("extraction", {})
            sg = bg.get("schema_constraint", {})
            fg = bg.get("format_following", {})
            writer.writerow([
                rd["round"], rd["mode"], rd["total"],
                rd["parse_rate"], rd["strict_rate"], rd["alias_strict_rate"],
                eg.get("parse_rate", ""), eg.get("strict_rate", ""), eg.get("alias_strict_rate", ""),
                sg.get("parse_rate", ""), sg.get("strict_rate", ""), sg.get("alias_strict_rate", ""),
                fg.get("parse_rate", ""), fg.get("strict_rate", ""), fg.get("alias_strict_rate", ""),
                rd["avg_latency_s"],
            ])
    print(f"CSV saved to {csv_out}")


if __name__ == "__main__":
    main()
