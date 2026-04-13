"""
05_stratified_sample.py - 分层采样
按 task_type 配比 + topic 平衡 + quality 优先 + 复杂度控制
"""

import json
import sys
import os
import random
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from conf import DERIVED_ALL, SAMPLED_TRAIN, SAMPLE_REPORT, SAMPLE


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


def classify_complexity(item):
    """按关系数和输入长度分复杂度桶"""
    n_rel = item["n_relations"]
    input_len = item["input_len"]

    if n_rel <= 3 and input_len < 100:
        return "simple"
    elif n_rel <= 6 and input_len < 250:
        return "medium"
    else:
        return "complex"


def main():
    print("=" * 60)
    print("05_stratified_sample.py - 分层采样")
    print("=" * 60)

    random.seed(SAMPLE["random_seed"])

    data = load_jsonl(DERIVED_ALL)
    print(f"\n加载派生数据: {len(data)} 条")

    target = SAMPLE["candidate_target"]
    task_ratio = SAMPLE["task_ratio"]
    print(f"目标采样数: {target}")
    print(f"任务配比: {task_ratio}")

    # ── 步骤1: 计算每个 task_type 的目标数量 ────────────
    task_targets = {}
    for tt, ratio in task_ratio.items():
        task_targets[tt] = int(target * ratio)

    print(f"\n各任务目标数:")
    for tt, t in task_targets.items():
        print(f"  {tt}: {t}")

    # ── 步骤2: 按 task_type 分组 ──────────────────────
    by_task = defaultdict(list)
    for item in data:
        by_task[item["task_type"]].append(item)

    print(f"\n各任务可用数:")
    for tt, items in by_task.items():
        print(f"  {tt}: {len(items)}")

    # ── 步骤3: 在每个 task_type 内做分层采样 ───────────
    # 优先级: high > medium > low
    # 在同 quality 内，优先 medium 复杂度
    # 按 topic 做等比例控制

    sampled = []

    for tt, items in by_task.items():
        tt_target = task_targets.get(tt, 0)
        if tt_target == 0:
            continue

        print(f"\n采样 {tt} (目标 {tt_target})...")

        # 按 (quality_tier, topic, complexity) 分桶
        buckets = defaultdict(list)
        for item in items:
            quality = item.get("quality_tier", "medium")
            topic = item["cate"]
            complexity = classify_complexity(item)
            key = (quality, topic, complexity)
            buckets[key].append(item)

        # 统计
        topic_counts = Counter(item["cate"] for item in items)
        topics = sorted(topic_counts.keys())

        # 每个 topic 的目标数 (等比例)
        per_topic_target = max(1, tt_target // len(topics))

        # 质量优先级排序
        quality_order = ["high", "medium", "low"]
        # 复杂度优先级
        complexity_order = ["medium", "simple", "complex"]

        tt_sampled = []

        # 先按 topic 均衡采样
        for topic in topics:
            topic_collected = []
            remaining = per_topic_target

            # 按 quality -> complexity 优先级采样
            for q in quality_order:
                if remaining <= 0:
                    break
                for c in complexity_order:
                    if remaining <= 0:
                        break
                    key = (q, topic, c)
                    if key in buckets and buckets[key]:
                        pool = buckets[key]
                        n_take = min(remaining, len(pool))
                        random.shuffle(pool)
                        topic_collected.extend(pool[:n_take])
                        remaining -= n_take

            tt_sampled.extend(topic_collected)

        # 如果总采样数不足目标，从剩余样本中补充
        if len(tt_sampled) < tt_target:
            sampled_ids = set((d["original_id"], d["task_type"]) for d in tt_sampled)
            remaining_pool = [d for d in items if (d["original_id"], d["task_type"]) not in sampled_ids]
            random.shuffle(remaining_pool)
            n_more = min(tt_target - len(tt_sampled), len(remaining_pool))
            tt_sampled.extend(remaining_pool[:n_more])

        # 如果超出目标，随机裁剪
        if len(tt_sampled) > tt_target:
            random.shuffle(tt_sampled)
            tt_sampled = tt_sampled[:tt_target]

        sampled.extend(tt_sampled)
        print(f"  采样结果: {len(tt_sampled)}")

    # ── 统计 ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"采样结果汇总")
    print(f"{'='*60}")
    print(f"总采样数: {len(sampled)}")

    print(f"\n任务分布:")
    tt_dist = Counter(d["task_type"] for d in sampled)
    for tt in ["ie_extraction", "text_to_json", "format_following", "schema_repair"]:
        cnt = tt_dist.get(tt, 0)
        print(f"  {tt}: {cnt} ({cnt/len(sampled)*100:.1f}%)")

    print(f"\nTopic 分布:")
    topic_dist = Counter(d["cate"] for d in sampled)
    for topic, cnt in topic_dist.most_common():
        print(f"  {topic}: {cnt} ({cnt/len(sampled)*100:.1f}%)")

    print(f"\n质量分布:")
    q_dist = Counter(d["quality_tier"] for d in sampled)
    for q in ["high", "medium", "low"]:
        cnt = q_dist.get(q, 0)
        print(f"  {q}: {cnt} ({cnt/len(sampled)*100:.1f}%)")

    print(f"\n复杂度分布:")
    c_dist = Counter(classify_complexity(d) for d in sampled)
    for c in ["simple", "medium", "complex"]:
        cnt = c_dist.get(c, 0)
        print(f"  {c}: {cnt} ({cnt/len(sampled)*100:.1f}%)")

    # 保存
    save_jsonl(sampled, SAMPLED_TRAIN)
    print(f"\n保存: {SAMPLED_TRAIN}")

    report = {
        "step": "stratified_sample",
        "input_count": len(data),
        "sampled_count": len(sampled),
        "target": target,
        "task_targets": task_targets,
        "actual_task_dist": dict(tt_dist),
        "topic_dist": dict(topic_dist),
        "quality_dist": dict(q_dist),
        "complexity_dist": dict(c_dist),
    }
    with open(SAMPLE_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"报告: {SAMPLE_REPORT}")

    print(f"\n[05_stratified_sample] 完成. {len(data)} -> {len(sampled)}")


if __name__ == "__main__":
    main()
