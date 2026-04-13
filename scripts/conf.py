"""
全局阈值配置 - 所有清洗脚本共用
"""

import os

# ── 路径 ──────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "instructie")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
CAND_DIR = os.path.join(BASE_DIR, "data", "sft_candidate")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

# 原始文件
RAW_TRAIN = os.path.join(RAW_DIR, "train_zh.json")
RAW_VALID = os.path.join(RAW_DIR, "valid_zh.json")
RAW_TEST = os.path.join(RAW_DIR, "test_zh.json")
RAW_SCHEMA = os.path.join(RAW_DIR, "schema_zh.json")

# 标准化中间文件
NORM_TRAIN = os.path.join(PROC_DIR, "normalized_train.jsonl")
NORM_VALID = os.path.join(PROC_DIR, "normalized_valid.jsonl")
NORM_TEST = os.path.join(PROC_DIR, "normalized_test.jsonl")

# 过滤后
FILTERED_TRAIN = os.path.join(PROC_DIR, "filtered_train.jsonl")
FILTER_REPORT = os.path.join(REPORT_DIR, "filter_report.json")

# 质量分层
TIERED_TRAIN = os.path.join(PROC_DIR, "tiered_train.jsonl")
QUALITY_REPORT = os.path.join(REPORT_DIR, "quality_report.json")

# 派生任务
DERIVED_ALL = os.path.join(PROC_DIR, "derived_all.jsonl")
DERIVE_REPORT = os.path.join(REPORT_DIR, "derive_report.json")

# 采样
SAMPLED_TRAIN = os.path.join(PROC_DIR, "sampled_train.jsonl")
SAMPLE_REPORT = os.path.join(REPORT_DIR, "sample_report.json")

# 最终交付
FINAL_TRAIN = os.path.join(CAND_DIR, "train.jsonl")
FINAL_VALID = os.path.join(CAND_DIR, "valid.jsonl")
FINAL_METADATA = os.path.join(CAND_DIR, "metadata.json")

# ── cate 名称映射 ────────────────────────────────────
CATE_MAP = {
    "建筑结构": "建筑",
}

# ── 硬过滤阈值 ───────────────────────────────────────
HARD_FILTER = {
    "min_relations": 1,        # 空关系直接剔除
    "max_relations": 25,       # 超复杂样本剔除
    "min_input_len": 15,       # 极短文本剔除
    "max_input_len": 800,      # 极长文本剔除
    "max_output_json_len": 2500,
    "max_head_tail_len": 100,  # 单个head/tail最大字符数
}

# ── 软过滤阈值 (分位数) ──────────────────────────────
SOFT_FILTER = {
    "input_len_pct": 99,
    "output_len_pct": 99,
    "relation_count_pct": 99,
    "head_tail_len_pct": 99,
}

# ── 质量分层 ─────────────────────────────────────────
QUALITY = {
    # head/tail 全部在原文中 -> high; >=80% -> medium; 否则 -> low
    "match_ratio_high": 1.0,
    "match_ratio_medium": 0.8,
    # 理想关系数量区间
    "ideal_relation_range": (2, 10),
    # 理想输入长度区间
    "ideal_input_len_range": (30, 400),
}

# ── 采样策略 ─────────────────────────────────────────
SAMPLE = {
    "candidate_target": 30000,    # 候选集目标条数
    "final_target": 15000,        # 正式训练集目标
    "internal_valid_ratio": 0.05, # 内部valid比例
    "random_seed": 42,
    # 任务配比
    "task_ratio": {
        "ie_extraction": 0.50,
        "text_to_json": 0.25,
        "format_following": 0.15,
        "schema_repair": 0.10,
    },
}
