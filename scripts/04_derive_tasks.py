"""
04_derive_tasks.py - 从原始样本派生四类 SFT 任务
  1. ie_extraction   — 标准信息抽取
  2. text_to_json    — 强调结构化 JSON 输出
  3. format_following — 强调格式约束
  4. schema_repair   — 对正确输出做可控扰动，构造纠错任务
"""

import json
import sys
import os
import random
import copy
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from conf import (
    TIERED_TRAIN, RAW_SCHEMA, DERIVED_ALL, DERIVE_REPORT,
    SAMPLE, CATE_MAP,
)

# cate -> schema key 映射 (标准化后的 cate -> schema 中的 key)
SCHEMA_KEY_MAP = {v: k for k, v in CATE_MAP.items()}  # {"建筑": "建筑结构"}


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_schema(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def relations_to_output_json(relations):
    """将 relation 列表转为结构化 JSON 字符串 (按 entity 分组)"""
    entity_map = defaultdict(list)
    for r in relations:
        entity_map[r["head"]].append({"relation": r["relation"], "tail": r["tail"]})

    # 转为紧凑格式: {entity: {rel_type: tail_value_or_list}}
    result = {}
    for entity, rels in entity_map.items():
        rel_dict = defaultdict(list)
        for r in rels:
            rel_dict[r["relation"]].append(r["tail"])
        # 单值展开
        for k, v in rel_dict.items():
            if len(v) == 1:
                rel_dict[k] = v[0]
        result[entity] = dict(rel_dict)

    return json.dumps(result, ensure_ascii=False, indent=None)


def get_schema_for_topic(schema, cate):
    """获取 topic 对应的 schema 短字段名列表"""
    # 先直接匹配，再用反向映射
    if cate in schema:
        return schema[cate][1]
    schema_key = SCHEMA_KEY_MAP.get(cate)
    if schema_key and schema_key in schema:
        return schema[schema_key][1]
    return []


def make_ie_extraction(item, schema_fields):
    """任务A: 标准信息抽取"""
    schema_str = json.dumps(schema_fields, ensure_ascii=False)
    prompt = (
        "你是一个信息抽取助手。请根据给定的 schema 从文本中抽取信息，并以 JSON 格式输出。\n"
        f"Schema: {schema_str}\n"
        f"文本: {item['input']}"
    )
    output = relations_to_output_json(item["relation"])
    return {
        "task_type": "ie_extraction",
        "original_id": item["id"],
        "cate": item["cate"],
        "quality_tier": item["quality_tier"],
        "n_relations": len(item["relation"]),
        "input_len": len(item["input"]),
        "prompt": prompt,
        "output": output,
    }


def make_text_to_json(item, schema_fields):
    """任务B: 强调结构化 JSON 输出"""
    # 构造更强调 JSON 格式的 prompt
    schema_str = json.dumps(schema_fields, ensure_ascii=False)
    prompt = (
        "请将以下文本中的信息按照指定 schema 转换为结构化 JSON 对象。\n"
        f"Schema 字段: {schema_str}\n"
        f"要求: 输出合法 JSON，字段名为 schema 中定义的关系类型，值为抽取结果。"
        f"同一实体的多个关系值用列表表示，单个值直接使用字符串。按实体分组。\n"
        f"文本: {item['input']}"
    )
    output = relations_to_output_json(item["relation"])
    return {
        "task_type": "text_to_json",
        "original_id": item["id"],
        "cate": item["cate"],
        "quality_tier": item["quality_tier"],
        "n_relations": len(item["relation"]),
        "input_len": len(item["input"]),
        "prompt": prompt,
        "output": output,
    }


def make_format_following(item, schema_fields):
    """任务C: 强调格式严格遵循"""
    # 随机选择一种格式约束
    constraints = [
        "只输出 JSON，不要附加任何解释文字。",
        "只输出 JSON 格式的结果，不要包含任何额外说明。",
        "严格按照 JSON 格式输出，不要在 JSON 前后添加任何文字。",
        "仅输出结构化 JSON 数据，禁止附加解释、标注或格式化标记。",
    ]
    constraint = random.choice(constraints)

    schema_str = json.dumps(schema_fields, ensure_ascii=False)
    prompt = (
        f"{constraint}\n"
        f"Schema: {schema_str}\n"
        f"从文本中抽取信息并输出 JSON: {item['input']}"
    )
    output = relations_to_output_json(item["relation"])
    return {
        "task_type": "format_following",
        "original_id": item["id"],
        "cate": item["cate"],
        "quality_tier": item["quality_tier"],
        "n_relations": len(item["relation"]),
        "input_len": len(item["input"]),
        "prompt": prompt,
        "output": output,
    }


def perturb_output(output_str, schema_fields, relations):
    """
    对正确的 JSON 输出做可控扰动，生成 schema_repair 任务
    扰动类型:
      1. 字段名拼写错误 (替换一个字符)
      2. 缺失一个字段
      3. 添加一个不在 schema 中的幻觉字段
      4. 将某个字符串值替换为列表（类型错误）
    """
    try:
        output_obj = json.loads(output_str)
    except json.JSONDecodeError:
        return None, None

    perturbation_types = []

    # 找出当前输出中实际出现的关系类型
    used_rel_types = set()
    for entity, rels in output_obj.items():
        if isinstance(rels, dict):
            used_rel_types.update(rels.keys())

    available_schema = set(schema_fields)
    used_in_schema = used_rel_types & available_schema
    not_used_in_schema = available_schema - used_rel_types

    # 扰动1: 字段名拼写错误 (如果有使用中的字段)
    if used_in_schema:
        target_field = random.choice(list(used_in_schema))
        perturbed = copy.deepcopy(output_obj)
        for entity in perturbed:
            if isinstance(perturbed[entity], dict) and target_field in perturbed[entity]:
                # 随机替换一个字符
                field_chars = list(target_field)
                if len(field_chars) > 1:
                    idx = random.randint(0, len(field_chars) - 1)
                    field_chars[idx] = chr(ord(field_chars[idx]) + random.choice([1, -1, 2]))
                wrong_field = "".join(field_chars)
                perturbed[entity][wrong_field] = perturbed[entity].pop(target_field)
                perturbation_desc = f"字段名 '{target_field}' 被错误写成了 '{wrong_field}'"
                return json.dumps(perturbed, ensure_ascii=False), perturbation_desc

    # 扰动2: 缺失一个字段
    if used_in_schema and len(used_in_schema) > 1:
        target_field = random.choice(list(used_in_schema))
        perturbed = copy.deepcopy(output_obj)
        for entity in perturbed:
            if isinstance(perturbed[entity], dict) and target_field in perturbed[entity]:
                del perturbed[entity][target_field]
        perturbation_desc = f"缺少字段 '{target_field}'"
        return json.dumps(perturbed, ensure_ascii=False), perturbation_desc

    # 扰动3: 添加幻觉字段
    if not_used_in_schema:
        fake_field = random.choice(list(not_used_in_schema))
        perturbed = copy.deepcopy(output_obj)
        # 找一个 entity 添加
        entities = list(perturbed.keys())
        if entities:
            target_entity = random.choice(entities)
            if isinstance(perturbed[target_entity], dict):
                perturbed[target_entity][fake_field] = "这是一个不正确的值"
                perturbation_desc = f"实体 '{target_entity}' 中添加了不在原文中的幻觉字段 '{fake_field}'"
                return json.dumps(perturbed, ensure_ascii=False), perturbation_desc

    # 扰动4: 类型错误 - 字符串变列表
    for entity, rels in output_obj.items():
        if isinstance(rels, dict):
            for field, val in rels.items():
                if isinstance(val, str):
                    perturbed = copy.deepcopy(output_obj)
                    perturbed[entity][field] = [val, "多余值"]
                    perturbation_desc = f"字段 '{field}' 的值应该是字符串，但被错误地写成了列表"
                    return json.dumps(perturbed, ensure_ascii=False), perturbation_desc

    return None, None


def make_schema_repair(item, schema_fields):
    """任务D: schema 纠错"""
    output_str = relations_to_output_json(item["relation"])
    perturbed_output, perturbation_desc = perturb_output(output_str, schema_fields, item["relation"])

    if perturbed_output is None:
        return None

    schema_str = json.dumps(schema_fields, ensure_ascii=False)
    prompt = (
        "以下信息抽取结果存在错误，请根据 schema 和原文找出并修正错误。\n"
        f"Schema: {schema_str}\n"
        f"原文: {item['input']}\n"
        f"有错误的抽取结果: {perturbed_output}\n"
        f"错误类型: {perturbation_desc}\n"
        f"请输出修正后的正确 JSON。"
    )

    return {
        "task_type": "schema_repair",
        "original_id": item["id"],
        "cate": item["cate"],
        "quality_tier": item["quality_tier"],
        "n_relations": len(item["relation"]),
        "input_len": len(item["input"]),
        "prompt": prompt,
        "output": output_str,  # 正确答案
        "perturbation": perturbation_desc,
    }


def main():
    print("=" * 60)
    print("04_derive_tasks.py - 派生四类 SFT 任务")
    print("=" * 60)

    random.seed(42)

    data = load_jsonl(TIERED_TRAIN)
    schema = load_schema(RAW_SCHEMA)
    print(f"\n加载分层后数据: {len(data)} 条")
    print(f"Schema topics: {list(schema.keys())}")

    # ── 派生策略 ──────────────────────────────────────
    # 对 high 质量样本: 派生全部4类
    # 对 medium 质量样本: 派生 ie_extraction + text_to_json
    # 对 low 质量样本: 仅派生 ie_extraction
    #
    # 这样自然形成任务配比的大致框架，后续采样时再精确控制

    derived = []
    task_counter = Counter()
    skip_counter = Counter()

    for i, item in enumerate(data):
        cate = item["cate"]
        schema_fields = get_schema_for_topic(schema, cate)

        if not schema_fields:
            skip_counter["no_schema"] += 1
            continue

        tier = item["quality_tier"]

        # 所有质量等级都派生 ie_extraction
        d = make_ie_extraction(item, schema_fields)
        if d:
            derived.append(d)
            task_counter["ie_extraction"] += 1

        if tier in ("high", "medium"):
            # text_to_json
            d = make_text_to_json(item, schema_fields)
            if d:
                derived.append(d)
                task_counter["text_to_json"] += 1

        if tier == "high":
            # format_following
            d = make_format_following(item, schema_fields)
            if d:
                derived.append(d)
                task_counter["format_following"] += 1

            # schema_repair (仅对关系数>=3的样本做扰动)
            if len(item["relation"]) >= 3:
                d = make_schema_repair(item, schema_fields)
                if d:
                    derived.append(d)
                    task_counter["schema_repair"] += 1
                else:
                    skip_counter["schema_repair_failed"] += 1

        if i % 50000 == 0 and i > 0:
            print(f"  已处理 {i}/{len(data)}...")

    # 报告
    print(f"\n派生结果:")
    print(f"  输入样本: {len(data)}")
    print(f"  派生样本总数: {len(derived)}")
    total = len(derived)
    for tt in ["ie_extraction", "text_to_json", "format_following", "schema_repair"]:
        cnt = task_counter[tt]
        print(f"  {tt}: {cnt} ({cnt/total*100:.1f}%)")

    if skip_counter:
        print(f"\n跳过统计:")
        for k, v in skip_counter.items():
            print(f"  {k}: {v}")

    # per-topic 分布
    print(f"\n各 topic 派生数量:")
    topic_counter = Counter(d["cate"] for d in derived)
    for topic, cnt in topic_counter.most_common():
        print(f"  {topic}: {cnt}")

    # 保存
    save_jsonl(derived, DERIVED_ALL)
    print(f"\n保存: {DERIVED_ALL}")

    report = {
        "step": "derive_tasks",
        "input_count": len(data),
        "derived_count": len(derived),
        "task_counts": dict(task_counter),
        "skip_counts": dict(skip_counter),
        "task_ratios": {k: f"{v/total*100:.1f}%" for k, v in task_counter.items()},
        "topic_dist": dict(topic_counter),
    }
    with open(DERIVE_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"报告: {DERIVE_REPORT}")

    print(f"\n[04_derive_tasks] 完成. {len(data)} 条原始样本 -> {len(derived)} 条派生样本")


if __name__ == "__main__":
    main()
