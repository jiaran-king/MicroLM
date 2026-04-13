# MicroLM

> 轻量级 LLM 训练、微调、评测与部署全链路项目

从零搭建一个能训练、能微调、能推理的完整 LLM 链路——tokenizer 训练、语料处理、pretrain、SFT、LoRA、推理优化、评测、部署，每一个环节亲手实现。

---

## 项目架构

项目采用**双轨并行**设计，用自研链路和开源迁移链路验证同一套方法论的可迁移性：

```
┌─ 自研 MicroLM 链路（31.7M 参数）──────────────────────────┐
│                                                            │
│  原始语料 (~141万条)                                        │
│    → 清洗切分 → BPE tokenizer (vocab=6400)                 │
│    → Pretrain (8层 Transformer, RoPE+SwiGLU)              │
│    → SFT 全参微调 / LoRA 微调 (0.83% 可训练参数)            │
│    → 推理 (KV Cache 加速, 平均 3.86x) → chat REPL          │
│                                                            │
└────────────────────────────────────────────────────────────┘

┌─ Qwen 迁移与部署链路（1.55B 参数）─────────────────────────┐
│                                                            │
│  InstructIE 数据 (171K 条)                                  │
│    → 6步数据 pipeline → SFT 数据集 (30K)                    │
│    → Qwen2.5-1.5B LoRA 微调 (0.14% 可训练参数)              │
│    → 结构化评测 (4模型 × 40prompt × 4指标)                   │
│    → vLLM 部署 → smoke/benchmark/稳定性验证                  │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

## 核心成果

### 自研 MicroLM 链路

| 指标 | 数值 |
|------|------|
| 模型参数量 | 31.7M（8层 Transformer, RoPE+SwiGLU+RMSNorm） |
| LoRA 可训练参数占比 | 0.83%（adaptor 仅 1.0 MB） |
| KV Cache 平均加速 | **3.86x**（最大 9.08x） |
| SFT 生成质量提升 | 平均评分 +81%（pretrain 1.13 → baseline 2.04） |

### Qwen 迁移链路

| 指标 | 数值 |
|------|------|
| 基座模型 | Qwen2.5-1.5B-Instruct |
| LoRA 可训练参数占比 | 0.14%（adaptor 仅 8.3 MB） |
| val_loss 降幅 | 0.40 → **0.15**（61.6%） |
| vLLM 部署验证 | smoke 5/5 通过，stability Parse% **100%** |

---

## 目录结构

```
micro_LM/
├── microlm/                  # 核心 Python 包
│   ├── model/                #   TransformerLM, LoRA, KV Cache
│   ├── tokenizer/            #   BPE tokenizer (训练 + 推理)
│   ├── training/             #   优化器、调度器、SFT 数据协议、loss
│   └── inference/            #   推理辅助
├── scripts/                  # 可执行脚本
│   ├── train_pretrain.py     #   预训练入口
│   ├── train_sft.py          #   SFT 微调入口
│   ├── train_qwen_lora.py    #   Qwen LoRA 微调入口
│   ├── train_tokenizer.py    #   BPE tokenizer 训练
│   ├── chat.py               #   交互式聊天
│   ├── 01_normalize.py       #   数据处理 pipeline (6步)
│   ├── ...
│   └── smoke_vllm.py         #   vLLM 部署验证
├── tests/                    # 测试套件 (6个测试文件)
├── configs/                  # 训练/推理配置 JSON
├── eval/                     # 评测 prompt 模板
├── results/                  # 评测结果
├── data/                     # 训练数据 (大部分需自行准备, 见下方说明)
├── Readme/                   # 详细中文文档
│   ├── 项目全景图/            #   项目总览、阶段路线图、复盘
│   └── 核心代码解析/          #   各模块代码详解
└── pyproject.toml
```

## 快速开始

### 环境配置

```bash
# 克隆仓库
git clone https://github.com/<your-username>/MicroLM.git
cd MicroLM

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -e .
```

依赖项：`torch`, `einops`, `numpy`, `regex`, `wandb`, `pytest`

### 运行测试

```bash
# 使用 smoke 数据运行测试
pytest tests/
```

### 数据准备

仓库仅包含 `data/smoke/` 和 `data/sft_smoke/` 中的小型测试数据。完整的训练数据需要自行准备：

- **预训练语料**：JSONL 格式（`{"text": "..."}`）
- **SFT 数据**：多轮对话 JSONL
- **InstructIE 数据**：结构化信息抽取数据集
- **Qwen2.5-1.5B-Instruct**：从 [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) 下载

详见 `data/README.md`。

## 里程碑

- **Milestone A**（2026-04-10）：自研链路基线冻结 — tokenizer + pretrain + SFT + LoRA + KV Cache + chat
- **Milestone B**（2026-04-12）：InstructIE 结构化闭环 — 30K 数据集 → Qwen LoRA 微调 → 结构化评测 → vLLM 部署

## 详细文档

完整中文文档位于 [`Readme/`](Readme/) 目录：

- [项目总览](Readme/项目全景图/01-项目总览.md)
- [自研 MicroLM 主线](Readme/项目全景图/02-自研%20MicroLM%20主线.md)
- [推理与系统能力增强](Readme/项目全景图/03-推理与系统能力增强.md)
- [Qwen 迁移与结构化输出](Readme/项目全景图/04-Qwen%20迁移与结构化输出主线.md)
- [评测、验证与部署闭环](Readme/项目全景图/05-评测、验证与部署闭环.md)
- [项目复盘与总结](Readme/项目全景图/06-项目复盘与总结.md)

核心代码解析：[transformer.py](Readme/核心代码解析/01-transformer.py%20模型主干.md)、[lora.py](Readme/核心代码解析/02-lora.py%20LoRA%20参数高效微调.md)、[sft.py](Readme/核心代码解析/03-sft.py%20SFT%20数据协议.md) 等。

## License

MIT License. See [LICENSE](LICENSE).
