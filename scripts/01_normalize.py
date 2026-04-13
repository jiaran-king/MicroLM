"""
01_normalize.py - 统一字段结构
- text -> input
- relation 结构对齐 (保留 head_type/tail_type 为可选字段)
- cate 命名规范化 (建筑结构 -> 建筑)
- 统一输出 JSONL 格式
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from conf import (
    RAW_TRAIN, RAW_VALID, RAW_TEST, RAW_SCHEMA,
    NORM_TRAIN, NORM_VALID, NORM_TEST,
    CATE_MAP, PROC_DIR, REPORT_DIR,
)


def load_jsonl(path):
    """加载 JSONL 或 JSON Lines 文件"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data, path):
    """保存为 JSONL"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def normalize_item(item, source):
    """
    统一单条样本为标准格式:
    {
      "id": str,
      "cate": str (规范化后),
      "input": str,
      "relation": [{"head": str, "relation": str, "tail": str}],
      "head_type": [str] | None,  # 可选，保留 train 的类型信息
      "tail_type": [str] | None,
      "source": "train" | "valid" | "test"
    }
    """
    # 统一输入字段名
    text = item.get("text") or item.get("input", "")
    text = text.strip()

    # 统一 cate
    cate = item.get("cate", "")
    cate = CATE_MAP.get(cate, cate)

    # 统一 relation
    raw_relations = item.get("relation", [])
    relations = []
    head_types = []
    tail_types = []

    for r in raw_relations:
        rel = {
            "head": r.get("head", "").strip(),
            "relation": r.get("relation", "").strip(),
            "tail": r.get("tail", "").strip(),
        }
        relations.append(rel)
        # 保留类型信息（如有）
        head_types.append(r.get("head_type", ""))
        tail_types.append(r.get("tail_type", ""))

    result = {
        "id": str(item.get("id", "")),
        "cate": cate,
        "input": text,
        "relation": relations,
        "source": source,
    }

    # 仅当有类型信息时才附加
    if any(t for t in head_types):
        result["head_types"] = head_types
        result["tail_types"] = tail_types

    return result


def main():
    print("=" * 60)
    print("01_normalize.py - 标准化原始数据")
    print("=" * 60)

    # 加载
    print("\n加载原始文件...")
    train = load_jsonl(RAW_TRAIN)
    valid = load_jsonl(RAW_VALID)
    test = load_jsonl(RAW_TEST)
    print(f"  train: {len(train)}")
    print(f"  valid: {len(valid)}")
    print(f"  test:  {len(test)}")

    # 标准化
    print("\n标准化...")
    norm_train = [normalize_item(item, "train") for item in train]
    norm_valid = [normalize_item(item, "valid") for item in valid]
    norm_test = [normalize_item(item, "test") for item in test]

    # 验证
    for name, data in [("train", norm_train), ("valid", norm_valid), ("test", norm_test)]:
        # 检查字段一致性
        assert all("input" in d for d in data), f"{name} 仍有 text 字段"
        assert all("relation" in d for d in data), f"{name} 缺 relation"
        assert all(isinstance(d["relation"], list) for d in data), f"{name} relation 不是 list"
        # 检查 cate 命名
        cates = set(d["cate"] for d in data)
        assert "建筑结构" not in cates, f"{name} 仍有 建筑结构 cate"

    # 统计
    print("\n标准化后 cate 分布:")
    for name, data in [("train", norm_train), ("valid", norm_valid), ("test", norm_test)]:
        from collections import Counter
        cate_dist = Counter(d["cate"] for d in data)
        print(f"\n  {name} ({len(data)} 条):")
        for cate, cnt in cate_dist.most_common():
            print(f"    {cate}: {cnt}")

    # 保存
    print("\n保存标准化文件...")
    save_jsonl(norm_train, NORM_TRAIN)
    save_jsonl(norm_valid, NORM_VALID)
    save_jsonl(norm_test, NORM_TEST)
    print(f"  {NORM_TRAIN}")
    print(f"  {NORM_VALID}")
    print(f"  {NORM_TEST}")

    # 统计摘要
    report = {
        "step": "normalize",
        "input_counts": {"train": len(train), "valid": len(valid), "test": len(test)},
        "output_counts": {"train": len(norm_train), "valid": len(norm_valid), "test": len(norm_test)},
        "cate_map_applied": CATE_MAP,
    }
    report_path = os.path.join(REPORT_DIR, "normalize_report.json")
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告: {report_path}")

    print("\n[01_normalize] 完成.")


if __name__ == "__main__":
    main()
