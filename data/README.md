# Data

本项目使用 **两个外部数据源**，分别服务于两条训练链路。

---

## 数据源总览

| 数据源 | 用途 | 服务链路 | 原始规模 | 处理后规模 |
|--------|------|----------|----------|------------|
| [MiniMind](#1-minimind) | 预训练 + SFT 对话微调 | 自研 MicroLM（主线 A → B） | ~141 万条 | Pretrain: ~140 万 / SFT: MiniMind 对话数据 |
| [InstructIE](#2-instructie) | 结构化信息抽取 SFT | Qwen 迁移线（主线 C → D） | 171,471 条 | 28.5K train + 1.5K valid |

---

## 1. MiniMind

**用途**：自研 MicroLM 的预训练语料与 SFT 对话数据。

### 1.1 预训练语料

- **文件**：`pretrain_hq.jsonl`
- **格式**：每行 `{"text": "..."}`，中文对话/指令/知识文本
- **原始规模**：**1,413,103 条**（~1.6 GB）
- **来源**：[匠数科技](https://github.com/jingyaogong/minimind) 大模型数据集中文部分，经 MiniMind 项目清洗为 `pretrain_hq.jsonl`
- **下载方式**：

```bash
# 方式 A — ModelScope（推荐，国内速度快）
pip install modelscope
modelscope download --dataset gongjy/minimind_dataset pretrain_hq.jsonl

# 方式 B — HuggingFace
pip install datasets
# 访问 https://huggingface.co/datasets/jingyaogong/minimind_dataset 下载

# 方式 C — GitHub（含完整项目与数据）
git clone https://github.com/xuyongfu/minimind-20250320.git
```

SFT 对话数据（`sft_t2t_mini.jsonl`）在同一个数据集中，一并下载即可。

- **处理流程**（`scripts/prepare_pretrain_jsonl.py`）：

```
pretrain_hq.jsonl (1,413,103 条)
  → 控制字符清理 + HTML 标签清理 + 空白压缩
  → 长度过滤 + SHA256 精确去重
  → SHA1 哈希确定性划分 train/valid (99:1)
  → 文档间插入 EOS 分隔符
  ↓
pretrain_clean_v2/
  ├── train.txt      (1,398,265 条)
  ├── valid.txt      (13,981 条)
  └── tokenizer_corpus.txt
```

清洗统计：
- HTML 标签清理：11,427 条（0.81%）
- 空白压缩：55,681 条（3.94%）
- 精确去重：821 条（0.058%）
- 总过滤率：0.06%（整体质量已较高，清洗主要起规范化作用）

### 1.2 SFT 对话数据

- **目录**：`minimind_sft/`
- **文件**：`gongjy/minimind_dataset/sft_t2t_mini.jsonl`
- **用途**：MicroLM SFT 全参微调 / LoRA 微调的训练对话数据
- **处理**：经 `sft.py` 的 `SFTDataset` 渲染为 chat prompt，构建 assistant-only masked loss

---

## 2. InstructIE

**用途**：Qwen2.5-1.5B-Instruct 结构化信息抽取 LoRA 微调。

- **来源**：HuggingFace — [`zjunlp/InstructIE`](https://huggingface.co/datasets/zjunlp/InstructIE)
- **原始规模**：train **171,471** 条 / valid 1,004 条 / test 1,002 条
- **覆盖主题**：12 个（人物、组织、地点、事件、作品、医学、自然科学等）
- **语言**：中英双语
- **数据格式**：每条包含 input text + 抽取 schema + gold JSON output
- **下载方式**：

```bash
# 方式 A — HuggingFace datasets 库（推荐）
pip install datasets
python -c "from datasets import load_dataset; load_dataset('zjunlp/InstructIE')"

# 方式 B — HuggingFace 网页手动下载
# 访问 https://huggingface.co/datasets/zjunlp/InstructIE ，在 Files and versions 中下载

# 方式 C — ModelScope（国内镜像，数据集名 IEPile）
pip install modelscope
modelscope download --dataset ZJUNLP/IEPile
```

### 处理 Pipeline（6 步）

```
InstructIE 原始数据 (171,471 条)
  │
  ├─ Step 1: 01_normalize.py   字段标准化 (text→input, relation 对齐, cate 归一化)
  ├─ Step 2: 02_filter.py       两层过滤 (硬过滤 3,585 条 + P99 软过滤 4,257 条)
  │                             → 163,629 条
  ├─ Step 3: 03_quality_tier.py 质量三档分层 (high 95.5% / medium 3.9% / low 0.6%)
  │                             → 156,275 条 high
  ├─ Step 4: 04_derive_tasks.py 四类任务派生 (每个样本 → 4 条 SFT 训练样本)
  │                             ie_extraction(50%) / text_to_json(25%)
  │                             format_following(15%) / schema_repair(10%)
  │                             → 623,650 条
  ├─ Step 5: 05_stratified_sample.py  分层采样 (task_type + topic 12均衡 + quality)
  │                             → 30,000 条
  └─ Step 6: 06_to_chat_jsonl.py 格式转写 + valid 切分 (5%)
                                全量 JSON 合法性校验: 100% 通过
  ↓
sft_candidate/
  ├── train.jsonl     (28,500 条)
  ├── valid.jsonl     (1,500 条)
  └── metadata.json
```

所有步骤的阈值集中配置在 `scripts/conf.py`，每步产出独立 JSON 统计报告。

---

## 目录结构

```
data/
├── pretrain_hq.jsonl              # MiniMind 预训练原始语料 (~141万条)
├── pretrain_clean_v2/             # 清洗后的预训练文本 (train.txt / valid.txt)
│   └── tokenized_full/            # BPE 编码后的 token IDs (.npy memmap)
├── minimind_sft/                  # MiniMind SFT 对话数据
├── instructie/                    # InstructIE 原始数据集
├── processed/                     # 6 步 pipeline 中间产物 (normalized / filtered / tiered / derived / sampled)
├── sft_candidate/                 # 最终 SFT 数据集 (28.5K train + 1.5K valid)
├── smoke/                         # Smoke test 用的小规模验证数据
├── pretrain / pretrain_clean / pretrain_quick  # 早期实验中间数据
└── sft_smoke                      # SFT smoke test 数据
```

---

## 引用

若使用本项目的数据处理结果，请同时引用原始数据源：

- **MiniMind**:
  - 项目主页：[jingyaogong/minimind](https://github.com/jingyaogong/minimind)
  - 数据集（ModelScope）：[gongjy/minimind_dataset](https://www.modelscope.cn/datasets/gongjy/minimind_dataset)
  - 数据集（HuggingFace）：[jingyaogong/minimind_dataset](https://huggingface.co/datasets/jingyaogong/minimind_dataset)
  - 含数据说明的 fork：[xuyongfu/minimind-20250320](https://github.com/xuyongfu/minimind-20250320)

- **InstructIE**:
  - Wang, Y. et al. *InstructIE: A Bilingual Instruction-based Information Extraction Dataset*
  - HuggingFace：[zjunlp/InstructIE](https://huggingface.co/datasets/zjunlp/InstructIE)
  - ModelScope（国内镜像）：[ZJUNLP/IEPile](https://modelscope.cn/datasets/ZJUNLP/IEPile)
  - GitHub（DeepKE 生态）：[zjunlp/IEPile](https://github.com/zjunlp/IEPile)
