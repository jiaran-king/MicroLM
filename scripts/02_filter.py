"""
02_filter.py - 两层过滤
硬过滤: 空relation、跨集泄漏、极短/极长、极复杂
软过滤: 基于 per-topic 分位数阈值
"""

import json
import sys
import os
import statistics
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from conf import (
    NORM_TRAIN, NORM_VALID, NORM_TEST,
    FILTERED_TRAIN, FILTER_REPORT,
    HARD_FILTER, SOFT_FILTER,
)


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


def compute_output_json_len(item):
    """计算 relation 的 JSON 字符串长度"""
    return len(json.dumps(item["relation"], ensure_ascii=False))


def compute_max_head_tail_len(item):
    """计算最大的 head 或 tail 长度"""
    max_len = 0
    for r in item["relation"]:
        max_len = max(max_len, len(r["head"]), len(r["tail"]))
    return max_len


def compute_per_topic_quantiles(data, fields, pcts):
    """
    计算每个 topic 在各字段上的分位数
    fields: {"input_len": lambda item: ..., ...}
    返回 {topic: {field: {pct: value}}}
    """
    topic_values = defaultdict(lambda: defaultdict(list))
    for item in data:
        topic = item["cate"]
        for fname, fn in fields.items():
            topic_values[topic][fname].append(fn(item))

    result = {}
    for topic in sorted(topic_values.keys()):
        result[topic] = {}
        for fname in sorted(fields.keys()):
            vals = sorted(topic_values[topic][fname])
            result[topic][fname] = {}
            for p in pcts:
                idx = min(int(len(vals) * p / 100), len(vals) - 1)
                result[topic][fname][f"P{p}"] = vals[idx]
    return result


def hard_filter(item, reasons):
    """硬过滤，返回 True 表示保留"""
    rels = item["relation"]
    n_rel = len(rels)
    input_len = len(item["input"])

    if n_rel < HARD_FILTER["min_relations"]:
        reasons.append("empty_relation")
        return False
    if n_rel > HARD_FILTER["max_relations"]:
        reasons.append("too_many_relations")
        return False
    if input_len < HARD_FILTER["min_input_len"]:
        reasons.append("too_short_input")
        return False
    if input_len > HARD_FILTER["max_input_len"]:
        reasons.append("too_long_input")
        return False
    if compute_output_json_len(item) > HARD_FILTER["max_output_json_len"]:
        reasons.append("too_long_output")
        return False
    if compute_max_head_tail_len(item) > HARD_FILTER["max_head_tail_len"]:
        reasons.append("too_long_head_tail")
        return False

    return True


def soft_filter(item, topic_thresholds, reasons):
    """软过滤，基于 per-topic 分位数，返回 True 表示保留"""
    topic = item["cate"]
    if topic not in topic_thresholds:
        return True  # 无阈值则不过滤

    thresholds = topic_thresholds[topic]

    input_len = len(item["input"])
    if "input_len" in thresholds:
        max_val = thresholds["input_len"]["soft_max"]
        if input_len > max_val:
            reasons.append(f"soft_input_len_exceed")
            return False

    n_rel = len(item["relation"])
    if "relation_count" in thresholds:
        max_val = thresholds["relation_count"]["soft_max"]
        if n_rel > max_val:
            reasons.append("soft_relation_count_exceed")
            return False

    output_len = compute_output_json_len(item)
    if "output_len" in thresholds:
        max_val = thresholds["output_len"]["soft_max"]
        if output_len > max_val:
            reasons.append("soft_output_len_exceed")
            return False

    ht_len = compute_max_head_tail_len(item)
    if "head_tail_len" in thresholds:
        max_val = thresholds["head_tail_len"]["soft_max"]
        if ht_len > max_val:
            reasons.append("soft_head_tail_len_exceed")
            return False

    return True


def main():
    print("=" * 60)
    print("02_filter.py - 两层过滤")
    print("=" * 60)

    # 加载标准化数据
    print("\n加载标准化数据...")
    train = load_jsonl(NORM_TRAIN)
    valid = load_jsonl(NORM_VALID)
    test = load_jsonl(NORM_TEST)
    print(f"  train: {len(train)}")
    print(f"  valid: {len(valid)}")
    print(f"  test:  {len(test)}")

    # ── 泄漏检测 ──────────────────────────────────────
    print("\n[泄漏检测]")
    valid_texts = set(item["input"] for item in valid)
    test_texts = set(item["input"] for item in test)
    leak_texts = valid_texts | test_texts
    print(f"  valid 唯一文本: {len(valid_texts)}")
    print(f"  test 唯一文本: {len(test_texts)}")

    # ── 计算分位数 ─────────────────────────────────────
    print("\n[计算 per-topic 分位数]")
    pct = SOFT_FILTER["input_len_pct"]

    fields = {
        "input_len": lambda item: len(item["input"]),
        "output_len": lambda item: compute_output_json_len(item),
        "relation_count": lambda item: len(item["relation"]),
        "head_tail_len": lambda item: compute_max_head_tail_len(item),
    }

    quantiles = compute_per_topic_quantiles(train, fields, [75, 90, 95, 99])

    # 打印分位数表
    print(f"\n  分位数统计 (P{pct} 用作软上限):")
    print(f"  {'Topic':<10} {'输入长度P99':>12} {'输出长度P99':>12} {'关系数P99':>10} {'ht长度P99':>10}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
    for topic in sorted(quantiles.keys()):
        q = quantiles[topic]
        print(f"  {topic:<10} {q['input_len'][f'P{pct}']:>12} "
              f"{q['output_len'][f'P{pct}']:>12} "
              f"{q['relation_count'][f'P{pct}']:>10} "
              f"{q['head_tail_len'][f'P{pct}']:>10}")

    # 构建软过滤阈值
    topic_thresholds = {}
    for topic in quantiles:
        topic_thresholds[topic] = {}
        for fname in quantiles[topic]:
            soft_max = quantiles[topic][fname][f"P{pct}"]
            topic_thresholds[topic][fname] = {"soft_max": soft_max}

    # ── 硬过滤 ────────────────────────────────────────
    print(f"\n[硬过滤] 开始...")
    hard_reason_counter = Counter()
    hard_pass = []
    for item in train:
        reasons = []
        if hard_filter(item, reasons):
            # 同时检查泄漏
            if item["input"] in leak_texts:
                reasons.append("leak_with_valid_test")
                hard_reason_counter["leak_with_valid_test"] += 1
            else:
                hard_pass.append(item)
        for r in reasons:
            hard_reason_counter[r] += 1

    print(f"  硬过滤前: {len(train)}")
    print(f"  硬过滤后: {len(hard_pass)}")
    print(f"  剔除明细:")
    for reason, cnt in hard_reason_counter.most_common():
        print(f"    {reason}: {cnt}")
    total_hard = sum(hard_reason_counter.values())
    print(f"    总剔除: {total_hard} (含交叉)")

    # ── 软过滤 ────────────────────────────────────────
    print(f"\n[软过滤] 开始 (per-topic P{pct} 阈值)...")
    soft_reason_counter = Counter()
    soft_pass = []
    for item in hard_pass:
        reasons = []
        if soft_filter(item, topic_thresholds, reasons):
            soft_pass.append(item)
        for r in reasons:
            soft_reason_counter[r] += 1

    print(f"  软过滤前: {len(hard_pass)}")
    print(f"  软过滤后: {len(soft_pass)}")
    print(f"  剔除明细:")
    for reason, cnt in soft_reason_counter.most_common():
        print(f"    {reason}: {cnt}")

    # ── 统计过滤后 topic 分布 ──────────────────────────
    print(f"\n[过滤后 topic 分布]")
    filtered_cate = Counter(item["cate"] for item in soft_pass)
    for cate, cnt in filtered_cate.most_common():
        orig = sum(1 for t in train if t["cate"] == cate)
        print(f"  {cate}: {cnt} / {orig} (保留 {cnt/orig*100:.1f}%)")

    # 保存
    print(f"\n保存过滤后数据...")
    save_jsonl(soft_pass, FILTERED_TRAIN)
    print(f"  {FILTERED_TRAIN}")

    # 保存报告
    report = {
        "step": "filter",
        "input_count": len(train),
        "after_hard_filter": len(hard_pass),
        "after_soft_filter": len(soft_pass),
        "hard_filter_reasons": dict(hard_reason_counter),
        "soft_filter_reasons": dict(soft_reason_counter),
        "per_topic_quantiles": {
            topic: {fname: vals for fname, vals in topic_data.items()}
            for topic, topic_data in quantiles.items()
        },
        "filtered_topic_dist": dict(filtered_cate),
    }
    os.makedirs(os.path.dirname(FILTER_REPORT), exist_ok=True)
    with open(FILTER_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  {FILTER_REPORT}")

    print(f"\n[02_filter] 完成. {len(train)} -> {len(soft_pass)} (保留 {len(soft_pass)/len(train)*100:.1f}%)")


if __name__ == "__main__":
    main()
