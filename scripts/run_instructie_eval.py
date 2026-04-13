#!/usr/bin/env python3
"""run_instructie_eval.py — InstructIE structured evaluation with 4 auto-detection metrics.

Evaluates 4 models across 3 prompt groups (extraction / schema_constraint / format_following)
with unified detection: JSON parseable, missing fields, hallucinated fields, strict schema match.

Usage:
    python scripts/run_instructie_eval.py --config configs/instructie_eval.json
    python scripts/run_instructie_eval.py --skip-microlm   # Qwen models only
    python scripts/run_instructie_eval.py --skip-qwen      # MicroLM models only
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch


# ══════════════════════════════════════════════════════════════════════
# Detection Functions
# ══════════════════════════════════════════════════════════════════════

def clean_model_output(raw: str) -> str:
    """Strip markdown code fences, leading/trailing whitespace, etc."""
    s = raw.strip()
    # Remove ```json ... ``` or ``` ... ``` wrapper
    m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", s, re.DOTALL)
    if m:
        s = m.group(1).strip()
    return s


def try_parse_json(text: str) -> tuple[dict | list | None, bool]:
    """Attempt to parse text as JSON. Returns (parsed, success)."""
    try:
        obj = json.loads(text)
        return obj, True
    except (json.JSONDecodeError, ValueError):
        return None, False


def extract_top_level_fields(obj: dict | list) -> set[str]:
    """Extract all field names from a JSON structure (handles nested entity dicts)."""
    fields = set()
    if isinstance(obj, dict):
        for key, val in obj.items():
            fields.add(key)
            if isinstance(val, dict):
                fields.update(val.keys())
    return fields


def check_missing_fields(
    parsed: dict | list,
    required_fields: list[str],
) -> tuple[bool, list[str]]:
    """Check if all required fields appear in the output. Returns (ok, missing)."""
    if not isinstance(parsed, dict):
        return False, list(required_fields)

    # Collect all field names from top-level and nested values
    all_fields = set()
    for key, val in parsed.items():
        if isinstance(val, dict):
            all_fields.update(val.keys())
        else:
            all_fields.add(key)

    missing = [f for f in required_fields if f not in all_fields]
    return len(missing) == 0, missing


def check_extra_fields(
    parsed: dict | list,
    allowed_fields: list[str],
) -> tuple[bool, list[str]]:
    """Check if output contains fields not in allowed list. Returns (ok, extra)."""
    if not isinstance(parsed, dict):
        return False, []

    allowed = set(allowed_fields)
    # Top-level entity names are OK; check nested field names
    extra = []
    for key, val in parsed.items():
        if isinstance(val, dict):
            for fk in val.keys():
                if fk not in allowed:
                    extra.append(fk)
    return len(extra) == 0, extra


def check_enum_constraints(
    parsed: dict,
    enum_constraints: dict[str, list[str]],
) -> tuple[bool, dict[str, str]]:
    """Check if enum-constrained fields have valid values. Returns (ok, violations)."""
    if not enum_constraints:
        return True, {}
    violations = {}
    allowed_sets = {k: set(v) for k, v in enum_constraints.items()}

    for _entity, fields in (parsed.items() if isinstance(parsed, dict) else []):
        if not isinstance(fields, dict):
            continue
        for fname, fval in fields.items():
            if fname in allowed_sets:
                # Handle single value or list
                vals = [fval] if isinstance(fval, str) else fval if isinstance(fval, list) else []
                for v in vals:
                    if v not in allowed_sets[fname]:
                        violations[fname] = f"got '{v}', expected one of {sorted(allowed_sets[fname])}"
    return len(violations) == 0, violations


def check_value_match(
    parsed: dict,
    gold: dict,
) -> dict[str, bool]:
    """Compare key field values against gold output. Returns per-field match."""
    results = {}
    for entity, fields in gold.items():
        if entity in parsed and isinstance(parsed[entity], dict):
            for fname, fval in fields.items():
                if fname in parsed[entity]:
                    got = parsed[entity][fname]
                    if isinstance(fval, list) and isinstance(got, list):
                        results[fname] = set(str(x) for x in fval) <= set(str(x) for x in got)
                    else:
                        results[fname] = str(got) == str(fval)
                else:
                    results[fname] = False
    return results


def score_single_output(
    raw_output: str,
    prompt_item: dict,
) -> dict:
    """Run all 4 detections + optional value match on a single output."""
    cleaned = clean_model_output(raw_output)
    parsed, is_parseable = try_parse_json(cleaned)

    schema_def = prompt_item.get("schema_def", {})
    required = schema_def.get("required_fields", [])
    allowed = schema_def.get("allowed_fields", [])
    enums = schema_def.get("enum_constraints", {})
    gold = prompt_item.get("gold_output")

    result = {
        "id": prompt_item["id"],
        "group": prompt_item["group"],
        "raw_output": raw_output,
        "clean_output": cleaned,
        "parsed": is_parseable,
    }

    if not is_parseable or parsed is None:
        result.update({
            "missing_fields": required,
            "extra_fields": [],
            "enum_ok": None,
            "schema_strict": False,
            "value_match": None,
        })
        return result

    missing_ok, missing = check_missing_fields(parsed, required)
    extra_ok, extra = check_extra_fields(parsed, allowed)
    enum_ok, enum_violations = check_enum_constraints(parsed, enums)
    schema_strict = missing_ok and extra_ok and enum_ok

    result.update({
        "missing_fields": missing,
        "extra_fields": extra,
        "enum_ok": enum_ok if enums else None,
        "enum_violations": enum_violations if enums else None,
        "schema_strict": schema_strict,
    })

    # Optional: value match
    if gold and isinstance(parsed, dict):
        result["value_match"] = check_value_match(parsed, gold)
    else:
        result["value_match"] = None

    return result


# ══════════════════════════════════════════════════════════════════════
# Alias-Normalized Scorer (Auxiliary Diagnostic)
# ══════════════════════════════════════════════════════════════════════

# Common Chinese field-name aliases encountered in InstructIE evaluation
FIELD_ALIASES: dict[str, list[str]] = {
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
    "别名": ["又称", "又名", "别称", "又称", "也叫"],
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
    """Map an alias field name back to its canonical schema name. Returns original if no alias found."""
    for canonical, aliases in FIELD_ALIASES.items():
        if field == canonical:
            return canonical
        if field in aliases:
            return canonical
    return field


def score_alias_normalized(
    raw_output: str,
    prompt_item: dict,
) -> dict:
    """Alias-normalized scoring: same 4 indicators but with field-name alias resolution.

    This is an auxiliary diagnostic — it does NOT replace the 4 primary indicators.
    It captures structural quality improvements that strict matching misses.
    """
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
    }

    if not is_parseable or parsed is None:
        result.update({
            "missing_fields_alias": required,
            "extra_fields_alias": [],
            "schema_strict_alias": False,
        })
        return result

    if not isinstance(parsed, dict):
        result.update({
            "missing_fields_alias": required,
            "extra_fields_alias": [],
            "schema_strict_alias": False,
        })
        return result

    # Collect all field names (handles both flat and entity-keyed formats), then normalize
    all_fields_raw = set()
    for key, val in parsed.items():
        if isinstance(val, dict):
            all_fields_raw.update(val.keys())
        else:
            all_fields_raw.add(key)

    # Normalize
    all_fields_normalized = set(normalize_field_name(f) for f in all_fields_raw)
    required_normalized = set(required)  # already canonical
    allowed_normalized = set(allowed)

    # Check missing (alias-normalized)
    missing = [f for f in required if f not in all_fields_normalized]

    # Check extra (alias-normalized)
    extra_raw = [f for f in all_fields_raw if normalize_field_name(f) not in allowed_normalized]

    # Enum check (on normalized fields)
    enum_ok = True
    for _entity, fields in parsed.items():
        if not isinstance(fields, dict):
            continue
        for fname, fval in fields.items():
            canon = normalize_field_name(fname)
            if canon in enums:
                allowed_vals = set(enums[canon])
                vals = [fval] if isinstance(fval, str) else fval if isinstance(fval, list) else []
                for v in vals:
                    if v not in allowed_vals:
                        enum_ok = False

    schema_strict_alias = len(missing) == 0 and len(extra_raw) == 0 and enum_ok

    result.update({
        "missing_fields_alias": missing,
        "extra_fields_alias": extra_raw,
        "field_normalization_map": {
            f: normalize_field_name(f) for f in all_fields_raw if normalize_field_name(f) != f
        },
        "enum_ok_alias": enum_ok if enums else None,
        "schema_strict_alias": schema_strict_alias,
    })

    return result


# ══════════════════════════════════════════════════════════════════════
# MicroLM Inference Backend
# ══════════════════════════════════════════════════════════════════════

def load_microlm_model(
    checkpoint_path: Path,
    device: str,
    dtype: torch.dtype,
    lora_adaptor_path: Path | None = None,
    vocab_path: Path | None = None,
    merges_path: Path | None = None,
):
    """Load a MicroLM model (optionally with LoRA). Returns (model, tokenizer, eos_token_id)."""
    import torch.nn as nn
    from microlm.model import TransformerLM
    from microlm.model.lora import load_lora_state_dict, merge_lora
    from microlm.tokenizer import BPETokenizer

    special_tokens = ["</s>"]
    tokenizer = BPETokenizer.from_files(
        str(vocab_path), str(merges_path), special_tokens=special_tokens,
    )

    config_path = checkpoint_path.parent / "model_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        model_config = json.load(f)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    cleaned = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}

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

    ckpt_vocab_size = cleaned["token_embeddings.weight"].shape[0]

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

    # Resize vocab if needed
    tokenizer_vocab = len(tokenizer.id_to_vocab)
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

    if lora_adaptor_path is not None and lora_adaptor_path.exists():
        from microlm.model.lora import apply_lora_to_model
        apply_lora_to_model(model, r=8, alpha=16.0)
        lora_sd = torch.load(lora_adaptor_path, map_location=device, weights_only=True)
        load_lora_state_dict(model, lora_sd)
        merge_lora(model)

    model.eval()

    eos_token_id = tokenizer.vocab_to_id.get("</s>".encode("utf-8"))
    return model, tokenizer, eos_token_id


def microlm_generate(
    model, tokenizer, eos_token_id, prompt_text: str,
    max_new_tokens: int, temperature: float, top_p: float, device: str,
) -> str:
    """Generate text with MicroLM model."""
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

    return tokenizer.decode(new_ids)


# ══════════════════════════════════════════════════════════════════════
# Qwen Inference Backend
# ══════════════════════════════════════════════════════════════════════

def load_qwen_model(
    base_model_path: str,
    adaptor_path: str | None = None,
    device: str = "auto",
):
    """Load Qwen model (optionally with LoRA adaptor). Returns (model, tokenizer)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )

    if adaptor_path is not None:
        model = PeftModel.from_pretrained(model, adaptor_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def qwen_generate(
    model, tokenizer, prompt_text: str,
    max_new_tokens: int, temperature: float, top_p: float,
) -> str:
    """Generate text with Qwen model using chat template."""
    # Build messages in ChatML format
    messages = [
        {"role": "system", "content": "你是一个严格遵循 schema 的信息抽取助手。"},
        {"role": "user", "content": prompt_text},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "top_p": top_p if temperature > 0 else 1.0,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    # Decode only new tokens
    new_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ══════════════════════════════════════════════════════════════════════
# Prompt Builder
# ══════════════════════════════════════════════════════════════════════

def build_prompt_text(prompt_item: dict) -> str:
    """Build the full prompt text from a prompt item."""
    instruction = prompt_item.get("instruction", "")
    schema_list = prompt_item.get("schema", [])
    input_text = prompt_item.get("input", "")

    # If instruction already contains the full template, use it with schema + input
    schema_str = json.dumps(schema_list, ensure_ascii=False)

    # Format: instruction + schema + input
    parts = []
    if instruction:
        parts.append(instruction)
    if schema_list and "Schema:" not in instruction and "schema" not in instruction.lower():
        parts.append(f"Schema: {schema_str}")
    if "文本:" not in instruction and "从文本" not in instruction:
        parts.append(f"文本: {input_text}")
    else:
        # Instruction already references the text, just append input
        parts.append(input_text)

    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════
# Main Evaluation Logic
# ══════════════════════════════════════════════════════════════════════

def run_evaluation(
    eval_file: Path,
    out_dir: Path,
    device: str,
    skip_microlm: bool = False,
    skip_qwen: bool = False,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 42,
    qwen_base_path: str = "./Qwen2.5-1.5B-Instruct",
    qwen_adaptor_path: str = "./outputs/qwen_lora/best_adaptor",
    microlm_sft_path: str = "./outputs/sft_baseline/ckpt_final.pt",
    microlm_lora_path: str = "./outputs/sft_lora/ckpt_final.pt",
    microlm_lora_adaptor: str = "./outputs/sft_lora/lora_adaptor.pt",
    vocab_path: str = "./outputs/tokenizer_full_clean/vocab.json",
    merges_path: str = "./outputs/tokenizer_full_clean/merge.txt",
    limit: int = 0,
):
    torch.manual_seed(seed)

    # Load eval prompts
    with eval_file.open("r", encoding="utf-8") as f:
        eval_data = json.load(f)
    prompts = eval_data["prompts"]
    if limit > 0:
        prompts = prompts[:limit]
        print(f"[LIMIT] Running first {limit} prompts only")

    # Override generation params from eval file
    gen_params = eval_data.get("generation_params", {})
    max_new_tokens = gen_params.get("max_new_tokens", max_new_tokens)
    temperature = gen_params.get("temperature", temperature)
    top_p = gen_params.get("top_p", top_p)
    seed = gen_params.get("seed", seed)

    print(f"Eval params: temp={temperature}, top_p={top_p}, max_tokens={max_new_tokens}, seed={seed}")
    print(f"Total prompts: {len(prompts)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw_outputs"
    scored_dir = out_dir / "scored_results"
    summary_dir = out_dir / "summary"
    raw_dir.mkdir(exist_ok=True)
    scored_dir.mkdir(exist_ok=True)
    summary_dir.mkdir(exist_ok=True)

    all_results = {}

    # ── Model definitions ──
    models_to_eval = []

    if not skip_qwen:
        models_to_eval.append(("qwen_base", "qwen", None))
        models_to_eval.append(("qwen_lora", "qwen", qwen_adaptor_path))

    if not skip_microlm:
        models_to_eval.append(("microlm_sft", "microlm", None))
        models_to_eval.append(("microlm_lora", "microlm", microlm_lora_adaptor))

    for model_name, backend, adaptor in models_to_eval:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name} (backend={backend})")
        print(f"{'='*60}")

        # Load model
        if backend == "qwen":
            model, tokenizer = load_qwen_model(qwen_base_path, adaptor, device=device)
        else:
            model, tokenizer, eos_id = load_microlm_model(
                Path(microlm_sft_path if model_name == "microlm_sft" else microlm_lora_path),
                device=device,
                dtype=torch.float32,
                lora_adaptor_path=Path(adaptor) if adaptor else None,
                vocab_path=Path(vocab_path),
                merges_path=Path(merges_path),
            )

        raw_outputs = []
        scored_results = []
        total_time = 0.0

        for prompt_item in prompts:
            prompt_text = build_prompt_text(prompt_item)
            torch.manual_seed(seed)

            t0 = time.time()
            if backend == "qwen":
                output = qwen_generate(
                    model, tokenizer, prompt_text,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            else:
                output = microlm_generate(
                    model, tokenizer, eos_id, prompt_text,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    device=device,
                )
            elapsed = time.time() - t0
            total_time += elapsed

            # Score
            scored = score_single_output(output, prompt_item)
            scored["latency_s"] = round(elapsed, 3)

            # Alias-normalized score (auxiliary)
            alias_scored = score_alias_normalized(output, prompt_item)
            scored["schema_strict_alias"] = alias_scored.get("schema_strict_alias", False)
            scored["missing_fields_alias"] = alias_scored.get("missing_fields_alias", [])
            scored["extra_fields_alias"] = alias_scored.get("extra_fields_alias", [])

            raw_outputs.append({
                "id": prompt_item["id"],
                "group": prompt_item["group"],
                "prompt": prompt_text,
                "output": output,
            })
            scored_results.append(scored)

            parsed_flag = "✓" if scored["parsed"] else "✗"
            strict_flag = "✓" if scored["schema_strict"] else "✗"
            alias_flag = "✓" if scored.get("schema_strict_alias") else "✗"
            print(f"  [{prompt_item['group'][:4]}] {prompt_item['id']}: "
                  f"parsed={parsed_flag} strict={strict_flag} alias={alias_flag} "
                  f"| {output[:60]}{'...' if len(output) > 60 else ''}")

        # Save raw + scored
        with (raw_dir / f"{model_name}.json").open("w", encoding="utf-8") as f:
            json.dump(raw_outputs, f, indent=2, ensure_ascii=False)
        with (scored_dir / f"{model_name}_scored.json").open("w", encoding="utf-8") as f:
            json.dump(scored_results, f, indent=2, ensure_ascii=False)

        all_results[model_name] = {
            "backend": backend,
            "total_time_s": round(total_time, 2),
            "avg_latency_s": round(total_time / len(prompts), 3),
            "results": scored_results,
        }

        # Free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Summaries ──
    generate_summaries(all_results, prompts, summary_dir)
    print(f"\nAll results saved to {out_dir}")


def generate_summaries(all_results: dict, prompts: list[dict], summary_dir: Path):
    """Generate by_model, by_group, and leaderboard summaries."""

    # ── by_model ──
    by_model = {}
    for model_name, data in all_results.items():
        results = data["results"]
        n = len(results)
        by_model[model_name] = {
            "total": n,
            "parseable": sum(1 for r in results if r["parsed"]),
            "parseable_rate": round(sum(1 for r in results if r["parsed"]) / n, 4) if n else 0,
            "missing_fields_count": sum(len(r["missing_fields"]) for r in results),
            "missing_rate": round(sum(1 for r in results if r["missing_fields"]) / n, 4) if n else 0,
            "extra_fields_count": sum(len(r["extra_fields"]) for r in results),
            "hallucination_rate": round(sum(1 for r in results if r["extra_fields"]) / n, 4) if n else 0,
            "schema_strict_count": sum(1 for r in results if r["schema_strict"]),
            "schema_strict_rate": round(sum(1 for r in results if r["schema_strict"]) / n, 4) if n else 0,
            # Alias-normalized (auxiliary)
            "schema_strict_alias_rate": round(sum(1 for r in results if r.get("schema_strict_alias")) / n, 4) if n else 0,
            "missing_alias_rate": round(sum(1 for r in results if r.get("missing_fields_alias")) / n, 4) if n else 0,
            "hallucination_alias_rate": round(sum(1 for r in results if r.get("extra_fields_alias")) / n, 4) if n else 0,
            "total_time_s": data["total_time_s"],
            "avg_latency_s": data["avg_latency_s"],
        }
    with (summary_dir / "by_model.json").open("w", encoding="utf-8") as f:
        json.dump(by_model, f, indent=2, ensure_ascii=False)

    # ── by_group ──
    groups = ["extraction", "schema_constraint", "format_following"]
    by_group = {}
    for model_name, data in all_results.items():
        by_group[model_name] = {}
        for g in groups:
            group_results = [r for r in data["results"] if r["group"] == g]
            n = len(group_results)
            if n == 0:
                continue
            by_group[model_name][g] = {
                "total": n,
                "parseable_rate": round(sum(1 for r in group_results if r["parsed"]) / n, 4),
                "missing_rate": round(sum(1 for r in group_results if r["missing_fields"]) / n, 4),
                "hallucination_rate": round(sum(1 for r in group_results if r["extra_fields"]) / n, 4),
                "schema_strict_rate": round(sum(1 for r in group_results if r["schema_strict"]) / n, 4),
                # Alias-normalized
                "schema_strict_alias_rate": round(sum(1 for r in group_results if r.get("schema_strict_alias")) / n, 4),
                "missing_alias_rate": round(sum(1 for r in group_results if r.get("missing_fields_alias")) / n, 4),
            }
    with (summary_dir / "by_group.json").open("w", encoding="utf-8") as f:
        json.dump(by_group, f, indent=2, ensure_ascii=False)

    # ── leaderboard (sorted by schema_strict_rate desc) ──
    leaderboard = sorted(
        by_model.items(),
        key=lambda x: (-x[1]["schema_strict_rate"], -x[1]["parseable_rate"]),
    )
    leaderboard_data = [
        {
            "rank": i + 1,
            "model": name,
            "schema_strict_rate": stats["schema_strict_rate"],
            "schema_strict_alias_rate": stats.get("schema_strict_alias_rate", 0),
            "parseable_rate": stats["parseable_rate"],
            "missing_rate": stats["missing_rate"],
            "hallucination_rate": stats["hallucination_rate"],
        }
        for i, (name, stats) in enumerate(leaderboard)
    ]
    with (summary_dir / "leaderboard.json").open("w", encoding="utf-8") as f:
        json.dump(leaderboard_data, f, indent=2, ensure_ascii=False)

    # ── detailed CSV ──
    import csv
    csv_path = summary_dir / "detailed.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "id", "group", "parsed", "missing_fields",
            "extra_fields", "schema_strict", "schema_strict_alias", "raw_output",
        ])
        for model_name, data in all_results.items():
            for r in data["results"]:
                writer.writerow([
                    model_name, r["id"], r["group"],
                    r["parsed"], "; ".join(r["missing_fields"]),
                    "; ".join(r["extra_fields"]),
                    r["schema_strict"],
                    r.get("schema_strict_alias", ""),
                    r["raw_output"][:200],
                ])

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<15} {'Parse%':>8} {'Miss%':>8} {'Hallu%':>8} {'Strict%':>9} {'Alias%':>8}")
    print("-" * 58)
    for name, stats in leaderboard:
        print(f"{name:<15} {stats['parseable_rate']:>7.1%} {stats['missing_rate']:>7.1%} "
              f"{stats['hallucination_rate']:>7.1%} {stats['schema_strict_rate']:>8.1%} "
              f"{stats.get('schema_strict_alias_rate', 0):>7.1%}")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Run InstructIE structured evaluation.")
    parser.add_argument("--eval-file", type=Path, default=Path("eval/prompts_instructie.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/instructie_eval"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-microlm", action="store_true", help="Skip MicroLM models")
    parser.add_argument("--skip-qwen", action="store_true", help="Skip Qwen models")
    parser.add_argument("--qwen-base-path", type=str, default="./Qwen2.5-1.5B-Instruct")
    parser.add_argument("--qwen-adaptor-path", type=str, default="./outputs/qwen_lora/best_adaptor")
    parser.add_argument("--microlm-sft-path", type=str, default="./outputs/sft_baseline/ckpt_final.pt")
    parser.add_argument("--microlm-lora-path", type=str, default="./outputs/sft_lora/ckpt_final.pt")
    parser.add_argument("--microlm-lora-adaptor", type=str, default="./outputs/sft_lora/lora_adaptor.pt")
    parser.add_argument("--vocab-path", type=str, default="./outputs/tokenizer_full_clean/vocab.json")
    parser.add_argument("--merges-path", type=str, default="./outputs/tokenizer_full_clean/merge.txt")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of prompts (0 = all)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        eval_file=args.eval_file,
        out_dir=args.out_dir,
        device=args.device,
        skip_microlm=args.skip_microlm,
        skip_qwen=args.skip_qwen,
        qwen_base_path=args.qwen_base_path,
        qwen_adaptor_path=args.qwen_adaptor_path,
        microlm_sft_path=args.microlm_sft_path,
        microlm_lora_path=args.microlm_lora_path,
        microlm_lora_adaptor=args.microlm_lora_adaptor,
        vocab_path=args.vocab_path,
        merges_path=args.merges_path,
        limit=args.limit,
    )
