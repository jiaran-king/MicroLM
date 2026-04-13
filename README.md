# MicroLM

> 从零实现 LLM 训练链路，并将同一方法论迁移到 Qwen 工业工具栈完成结构化输出部署。

两条主线并行验证同一套工程方法论——自研链路用纯 PyTorch 从字节到服务全手写，迁移链路用 HF/PEFT/vLLM 完成相同流程并聚焦结构化信息抽取。

---

## 核心成果

**自研链路（31.7M 参数）** — 从 BPE tokenizer 到 KV Cache 推理加速，全部自研：

- 8 层 TransformerLM（RoPE + SwiGLU + RMSNorm pre-norm），自定义 einsum Linear 层
- LoRA 仅 **0.83%** 可训练参数达到全参 SFT **85%** 的生成质量，adaptor 存储 **1.0 MB**（节省 99.7%）
- KV Cache 增量解码平均加速 **3.86x**（最大 9.08x），decode 吞吐恒定 ~100 tok/s
- SFT 使生成评分提升 **+81%**（pretrain 1.13 → baseline 2.04 / 满分 5）

**迁移链路（Qwen2.5-1.5B-Instruct）** — 聚焦结构化信息抽取，完成从数据到服务的闭环：

- 6 步数据 pipeline 将 InstructIE 171K 条原始数据转化为 **28.5K** 高质量 SFT 数据集（全量 JSON 校验 100% 通过）
- 仅训练 **0.14%** 参数完成结构化能力塑形，val_loss 降幅 **61.6%**（0.40 → 0.15）
- 自动评测推荐部署 `qwen_lora`：Alias-Strict 命中率 **15.0%**（为 base 的 **2 倍**），结构化质量全面领先
- vLLM 服务化部署：smoke test **5/5** 通过，稳定性验证 Parse% **100%**，单并发吞吐 ~30 tok/s

**工程闭环** — 不是一组散落的实验脚本，而是一条可复现的完整管线：

```
数据获取 → 清洗/切分 → tokenizer 训练 → 预训练 → SFT/LoRA 微调 → 自动评测 → vLLM 部署 → 性能量化
```

---

## 3 分钟快速体验

```bash
# 1. 安装
git clone https://github.com/jiaran-king/MicroLM.git
cd MicroLM
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[all]"

# 2. 跑通测试（验证核心模块可用）
pytest tests/

# 3. 用 smoke 数据跑最小链路（自研线）
python scripts/train_pretrain.py --config configs/pretrain_smoke.json

# 4. 想看最终服务效果（Qwen 线，需先准备数据和基座模型）
python scripts/export_final_model.py          # 导出合并模型
bash scripts/serve_vllm.sh                     # 启动 vLLM 服务
python scripts/smoke_vllm.py                   # 5 项功能验证
```

> 完整的数据下载和安装说明见下方「环境配置」和 [`data/README.md`](data/README.md)。

---

## 为什么这个项目不是脚本拼盘

| 方法论 | 在项目中的体现 |
|--------|--------------|
| **配置驱动** | 17 个 JSON 配置文件覆盖全部阶段，`resolved_train_config.json` 固化每次实验的完整运行条件 |
| **协议显式化** | SFT 数据协议（`sft.py`）和 Qwen 三段式格式先定义再编码，数据长什么样写在文档里而非隐含在代码中 |
| **Smoke First** | 每个阶段都遵循 smoke test → 正式训练模式，避免"跑了 10 小时发现 loss mask 写反了" |
| **评测先行** | 自研线固定 prompt 人工评分 + Qwen 线 4 项自动检测指标，让"变好了"从感受变成可复现数字 |
| **双轨并行** | 两条链路共享设计模式但技术栈完全独立，互为对照系 |

---

## 先看哪些代码

| 想了解什么 | 看这里 |
|-----------|--------|
| Transformer 模型主干（RoPE / SwiGLU / RMSNorm） | `microlm/model/transformer.py` |
| 自研 LoRA 实现（LoRALinear / merge-unmerge） | `microlm/model/lora.py` |
| SFT 数据协议（chat prompt 渲染 / assistant-only masked loss） | `microlm/training/sft.py` |
| BPE tokenizer 训练与推理 | `microlm/tokenizer/` |
| KV Cache 增量推理 + 自回归生成循环 | `microlm/inference/generate_text.py` |
| 交互式多轮对话系统 | `scripts/chat.py` |
| Qwen 数据 pipeline（6 步处理） | `scripts/01_normalize.py` → `scripts/06_to_chat_jsonl.py` |
| Qwen LoRA 微调入口 | `scripts/train_qwen_lora.py` |
| 模型导出 + vLLM 部署 | `scripts/export_final_model.py` / `scripts/serve_vllm.sh` / `scripts/smoke_vllm.py` |

---

## 复现路径

### 自研 MicroLM 线

```
pip install -e .                                    # 核心依赖（纯 PyTorch）
↓
下载数据 → data/pretrain_hq.jsonl (~1.6GB)         # 见下方数据准备
↓
python scripts/train_tokenizer.py                   # 训练 BPE tokenizer (vocab=6400)
python scripts/prepare_pretrain_jsonl.py            # 清洗语料 + 切分 train/valid
python scripts/train_pretrain.py                    # 预训练 (31.7M 参数)
python scripts/train_sft.py                         # SFT 全参微调
# 或 python scripts/train_sft.py --use-lora          # SFT LoRA 微调 (0.83% 参数)
↓
python scripts/chat.py                               # 交互式对话
```

### Qwen 迁移线

```
pip install -e ".[all]"                             # 含 HF / PEFT / vLLM
↓
下载数据 → InstructIE (171K 条) + Qwen2.5-1.5B     # 见下方数据准备
↓
python scripts/01_normalize.py                       # Step 1: 字段标准化
python scripts/02_filter.py                          # Step 2: 过滤
python scripts/03_quality_tier.py                    # Step 3: 质量分层
python scripts/04_derive_tasks.py                    # Step 4: 任务派生
python scripts/05_stratified_sample.py               # Step 5: 分层采样
python scripts/06_to_chat_jsonl.py                  # Step 6: 格式转写
↓
python scripts/train_qwen_lora.py                    # LoRA 微调 (0.14% 参数)
python scripts/run_instructie_eval.py                # 结构化自动评测
↓
python scripts/export_final_model.py                 # 导出合并模型
bash scripts/serve_vllm.sh                           # 启动 vLLM 服务
python scripts/smoke_vllm.py                         # 部署验证
```

---

## 项目亮点

### 从零实现

不依赖 HuggingFace 模型层——BPE tokenizer（byte-level → 6400 词表）、TransformerLM（einsum Linear / RoPE / SwiGLU / RMSNorm）、AdamW 优化器 + cosine scheduler + gradient clipping、assistant-only masked loss 全部手写。证明了对架构的理解可以落实到每一行代码。

### 方法论可迁移

自研链路的配置组织（JSON config → resolved config）、日志格式（JSONL per-step）、训练循环结构（step loop → val → checkpoint）在迁移到 HF/PEFT/vLLM 工具栈时几乎原样复用。需要重新适配的是接口层（tokenizer / chat template / LoRA 框架），不是思维方式。

### 结构化评测替代主观打分

Qwen 线上完全放弃人工评分，改用 4 项自动检测指标（JSON 可解析率 / 缺字段率 / 幻觉字段率 / 严格 Schema 命中率）对 4 个模型做统一量化对比，并附加 alias 归一化和结构化质量分析。部署决策基于 Alias-Strict 15.0%（base 的 2 倍）+ 结构化质量全面领先，而非主观感受。

### 服务化部署

从 checkpoint 到 HTTP API 的完整路径：PEFT merge_and_unload → HF 格式 → vLLM 加载 → OpenAI 兼容 API → smoke/benchmark/stability 全通过。单并发 ~30 tok/s，8 并发 ~122 tok/s，零错误率。

---

## 目录结构

```
micro_LM/
├── microlm/                  # 核心 Python 包
│   ├── model/                #   transformer.py, lora.py, kv_cache.py
│   ├── tokenizer/            #   BPE tokenizer (训练 + 编码/解码)
│   ├── training/             #   optimizer, scheduler, sft.py, data_loader, loss
│   └── inference/            #   generate_text.py, prompting.py
├── scripts/                  # 可执行脚本入口
│   ├── train_pretrain.py     #   预训练
│   ├── train_sft.py          #   SFT 微调 (支持 --use-lora)
│   ├── train_qwen_lora.py    #   Qwen LoRA 微调
│   ├── train_tokenizer.py    #   BPE tokenizer 训练
│   ├── chat.py               #   交互式对话 REPL
│   ├── 01~06_*.py            #   InstructIE 数据 pipeline (6步)
│   ├── export_final_model.py #   LoRA 合并导出
│   ├── serve_vllm.sh         #   vLLM 服务启动
│   └── smoke_vllm.py         #   vLLM 功能验证
├── tests/                    # 测试套件
├── configs/                  # 训练/推理 JSON 配置
├── eval/                     # 评测 prompt 模板
├── data/                     # 训练数据 (需自行下载, 见下方)
├── Readme/                   # 详细中文文档
│   ├── 项目全景图/            #   01-06 全景文档
│   └── 核心代码解析/          #   各模块代码详解
└── pyproject.toml
```

---

## 环境配置

### 依赖安装

```bash
pip install -e .              # 核心依赖（纯 PyTorch，自研链路够用）
pip install -e ".[qwen]"      # +transformers, peft, datasets（Qwen 迁移链路）
pip install -e "[dev]"        # +pytest（开发测试）
pip install -e "[all]"        # 以上全部（推荐）
```

可选：
- **vLLM 部署**：`pip install vllm`
- **ModelScope 数据下载**：`pip install modelscope`（国内用户）

### 数据准备

仓库仅包含 `data/smoke/` 和 `data/sft_smoke/` 小型测试数据。完整训练数据需自行下载：

| 数据源 | 用途 | 下载方式 |
|--------|------|----------|
| [MiniMind](https://github.com/jingyaogong/minimind) | 预训练 + SFT 对话（自研线） | ModelScope: `gongjy/minimind_dataset` |
| [InstructIE](https://huggingface.co/datasets/zjunlp/InstructIE) | 结构化抽取（Qwen 线） | HuggingFace: `zjunlp/InstructIE` |
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | 基座模型 | HuggingFace: `Qwen/Qwen2.5-1.5B-Instruct` |

详细处理流程、目录结构和命令见 [`data/README.md`](data/README.md)。

---

## 详细文档

| 想看什么 | 文档 |
|---------|------|
| **3 分钟了解全貌** | [01-项目总览](Readme/项目全景图/01-项目总览.md) — 双轨架构、里程碑、核心量化成果 |
| **从零实现训练链路** | [02-自研 MicroLM 主线](Readme/项目全景图/02-自研%20MicroLM%20主线.md) — 数据/Tokenizer/模型/Pretrain/SFT/LoRA/能力边界 |
| **推理与系统实现** | [03-推理与系统能力增强](Readme/项目全景图/03-推理与系统能力增强.md) — 文本生成/KV Cache/chat.py |
| **Qwen 迁移与结构化输出** | [04-Qwen 迁移线](Readme/项目全景图/04-Qwen%20迁移与结构化输出主线.md) — 数据pipeline/LoRA微调/导出部署 |
| **评测与部署闭环** | [05-评测部署](Readme/项目全景图/05-评测、验证与部署闭环.md) — 自动评测/Alias归一化/vLLM benchmark |
| **方法论与复盘** | [06-复盘总结](Readme/项目全景图/06-项目复盘与总结.md) — 成果/Bug清册/方法论收获/扩展方向 |

核心代码解析：[transformer.py](Readme/核心代码解析/01-transformer.py%20模型主干.md) / [lora.py](Readme/核心代码解析/02-lora.py%20LoRA%20参数高效微调.md) / [sft.py](Readme/核心代码解析/03-sft.py%20SFT%20数据协议.md) / [data_loader & loss](Readme/核心代码解析/04-data_loader.py%20与%20loss.py.md) / [generate_text.py](Readme/核心代码解析/05-generate_text.py%20推理链路.md) / [chat.py](Readme/核心代码解析/06-chat.py%20多轮对话系统.md) / [train_qwen_lora.py](Readme/核心代码解析/07-train_qwen_lora.py%20Qwen%20迁移线核心.md) / [数据 pipeline](Readme/核心代码解析/08-数据%20pipeline%20六步处理.md)

---

## License

MIT License. See [LICENSE](LICENSE).
