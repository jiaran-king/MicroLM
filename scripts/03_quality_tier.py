"""
03_quality_tier.py - 质量打分与分层
为每条样本打 high/medium/low 标签
"""

import json
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from conf import (
    FILTERED_TRAIN, TIERED_TRAIN, QUALITY_REPORT,
    QUALITY,
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


def compute_match_ratio(item):
    """计算 head/tail 在原文中的匹配率"""
    text = item["input"]
    rels = item["relation"]
    if not rels:
        return 0.0
    matched = 0
    for r in rels:
        if r["head"] in text and r["tail"] in text:
            matched += 1
    return matched / len(rels)


def assign_quality_tier(item):
    """
    质量分层逻辑:
    high:  全部匹配 + 关系数在理想区间 + 输入长度在理想区间
    medium: 匹配率>=80% 或 其他维度稍偏离
    low:   匹配率<80% 或 多维度偏离
    """
    match_ratio = compute_match_ratio(item)
    n_rel = len(item["relation"])
    input_len = len(item["input"])

    ideal_rel = QUALITY["ideal_relation_range"]
    ideal_len = QUALITY["ideal_input_len_range"]

    # 维度得分
    dims = {
        "match": match_ratio >= QUALITY["match_ratio_high"],
        "rel_count": ideal_rel[0] <= n_rel <= ideal_rel[1],
        "input_len": ideal_len[0] <= input_len <= ideal_len[1],
    }

    score = sum(dims.values())

    # 分层
    if match_ratio >= QUALITY["match_ratio_high"] and score >= 2:
        tier = "high"
    elif match_ratio >= QUALITY["match_ratio_medium"] and score >= 1:
        tier = "medium"
    else:
        tier = "low"

    return tier, match_ratio, dims


def main():
    print("=" * 60)
    print("03_quality_tier.py - 质量打分与分层")
    print("=" * 60)

    data = load_jsonl(FILTERED_TRAIN)
    print(f"\n加载过滤后数据: {len(data)} 条")

    # 打分
    print("\n质量分层中...")
    tier_counter = Counter()
    match_ratios = []
    dim_counters = {"match": 0, "rel_count": 0, "input_len": 0}

    for item in data:
        tier, match_ratio, dims = assign_quality_tier(item)
        item["quality_tier"] = tier
        item["match_ratio"] = round(match_ratio, 4)
        item["quality_dims"] = dims
        tier_counter[tier] += 1
        match_ratios.append(match_ratio)
        for k, v in dims.items():
            if v:
                dim_counters[k] += 1

    # 报告
    print(f"\n质量分层结果:")
    for tier in ["high", "medium", "low"]:
        cnt = tier_counter[tier]
        print(f"  {tier}: {cnt} ({cnt/len(data)*100:.1f}%)")

    print(f"\n匹配率统计:")
    print(f"  平均: {sum(match_ratios)/len(match_ratios)*100:.1f}%")
    full_match = sum(1 for r in match_ratios if r == 1.0)
    print(f"  100%匹配: {full_match} ({full_match/len(data)*100:.1f}%)")

    print(f"\n各维度达标率:")
    for k in ["match", "rel_count", "input_len"]:
        cnt = dim_counters[k]
        print(f"  {k}: {cnt}/{len(data)} ({cnt/len(data)*100:.1f}%)")

    # per-topic per-tier
    print(f"\n各 topic 质量分布:")
    topic_tier = {}
    for item in data:
        t = item["cate"]
        q = item["quality_tier"]
        if t not in topic_tier:
            topic_tier[t] = Counter()
        topic_tier[t][q] += 1

    print(f"  {'Topic':<10} {'high':>8} {'medium':>8} {'low':>8} {'high%':>8}")
    for topic in sorted(topic_tier.keys()):
        tc = topic_tier[topic]
        total = sum(tc.values())
        print(f"  {topic:<10} {tc['high']:>8} {tc['medium']:>8} {tc['low']:>8} {tc['high']/total*100:>7.1f}%")

    # 保存
    save_jsonl(data, TIERED_TRAIN)
    print(f"\n保存: {TIERED_TRAIN}")

    report = {
        "step": "quality_tier",
        "input_count": len(data),
        "tier_counts": dict(tier_counter),
        "dim_pass_rates": {k: f"{v/len(data)*100:.1f}%" for k, v in dim_counters.items()},
        "avg_match_ratio": f"{sum(match_ratios)/len(match_ratios)*100:.1f}%",
        "full_match_count": full_match,
        "topic_tier_dist": {t: dict(c) for t, c in topic_tier.items()},
    }
    with open(QUALITY_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"报告: {QUALITY_REPORT}")

    print(f"\n[03_quality_tier] 完成.")


if __name__ == "__main__":
    main()
