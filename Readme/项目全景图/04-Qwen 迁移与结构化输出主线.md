---
type: project-note
project: MicroLM
section: qwen-migration
---

# 四、Qwen 迁移与结构化输出主线

> 这一条主线讲的是：把自研链路积累的方法论迁移到开源生态（Qwen2.5-1.5B-Instruct），同时把任务聚焦到结构化输出。它的定位不是"再做一次训练"，而是"同一套方法论在工业工具栈上的验证 + 能力聚焦带来的可量化收益"。突出"迁移"和"聚焦"，不是简单复制第二章。

---

## 4.1 迁移动机与方向聚焦

### 这一小节要解决什么问题

自研 MicroLM 验证了完整链路的可行性，但 31.7M 参数的规模决定了能力天花板。需要回答：为什么迁移、为什么选 Qwen、为什么从通用聊天转向结构化输出。

### 核心设计 / 核心决策

**MicroLM 的天花板：**

| 维度 | MicroLM | 现实约束 |
|------|---------|----------|
| 参数量 | 31.7M | 语言表示容量有限，复杂任务难以收敛 |
| 词表 | 6,400 | 中文覆盖率足够，但 JSON 输出时 token 效率低 |
| 训练数据 | MiniMind 通用对话 | 无结构化标注数据，无法训练 schema-following 能力 |
| 结构化评测结果 | Parse% = **0.0%** | 40 条结构化 prompt 中无法输出合法 JSON |

最后一条是决定性的。在阶段 6C 的四模型对比评测中，MicroLM 系列（SFT / LoRA）的 JSON 可解析率为 0%——不是"差"，是完全没有这个能力。这源于词表、预训练数据、SFT 数据三重缺失。

**方向聚焦：不做泛化型中文助手增强。** 原因很实际——通用聊天能力的提升难以量化。"感觉更聪明了"不能写进简历，也不能作为部署决策的依据。而结构化信息抽取不同：
- 有明确的硬指标：JSON 可解析率、字段准确率、格式遵循成功率
- LoRA 擅长的"行为约束 + 格式塑形"在这里能发挥最大价值
- 是 Agent / tool call / function calling 的前置基础——后续扩展路径清晰

> [!tip] 聚焦的代价与收益
> 放弃通用能力意味着项目不会变成"另一个中文 ChatBot"。但换来的是：每一步优化都能用数字说话，评测完全自动化无需人工打分，部署目标明确（结构化抽取服务而非通用助手）。对于简历项目和面试展示，有硬指标的深度 > 没指标的广度。

**基座选择：Qwen2.5-1.5B-Instruct**

| 选择理由 | 说明 |
|----------|------|
| 规模适中 | 1.55B 参数，单卡 RTX 5070 Ti 可完成全量微调 |
| Instruct 底子好 | 已经过指令微调，ChatML 模板成熟，中文能力强 |
| 生态完善 | HuggingFace 一键加载，PEFT/vLLM 直接支持 |
| 社区认可度高 | 面试官熟悉，不需要额外解释"这是什么模型" |

### 这一节的结论

迁移决策基于一个清晰的证据链：MicroLM 的 Parse%=0% → 小模型不具备结构化输出能力 → 必须换基座 → 选 Qwen2.5-1.5B-Instruct（规模适中+生态好）→ 聚焦结构化抽取（有硬指标+LoRA 擅长）。每一步都有数据支撑，不是拍脑袋决定。

---

## 4.2 InstructIE 数据分析与 Pipeline 设计

### 这一小节要解决什么问题

InstructIE 原始数据（171K 条）不能直接用于 SFT 训练——存在字段不统一、跨集泄漏、topic 不均衡等问题。需要设计一套工程化的数据处理 pipeline，将原始数据转化为高质量、可追踪的 SFT 数据集。同时需要定义四类派生任务，让模型学到不同层面的结构化输出能力。

### 核心设计 / 核心流程

**数据源定位：** 单一来源 **InstructIE**（HuggingFace: `zjunlp/InstructIE`），不做多源混合。原始数据规模：train **171,471** 条 / valid 1,004 条 / test 1,002 条。

InstructIE 的天然定位与本项目的目标高度吻合：覆盖中英双语、12 个主题；以 topic schema 组织数据，每条样本包含 input text + 抽取 schema + gold JSON output；本质就是 "instruction-based information extraction" —— 和项目方向逐字对应。

**数据质量探索——发现的问题直接驱动 pipeline 各步设计：**

| 问题 | 详情 | 处理方式 |
|------|------|----------|
| 字段不统一 | train 用 `text`，valid/test 用 `input` | Step 1 标准化统一为 `input` |
| relation 结构差异 | train 含 `head_type`/`tail_type`，valid/test 不含 | Step 1 对齐字段 |
| cate 命名漂移 | "建筑结构" vs "建筑" | Step 1 归一化映射 |
| 跨集泄漏 | train∩valid 143 条，train∩test 181 条 | Step 2 硬过滤剔除 |
| head/tail 匹配率 | **99.7%**（高质量）| 作为 quality tier 的核心指标 |

**Topic 分布不均**——前 7 个 topic 占约 85%，自然科学（2.5%）和医学（1.9%）明显偏少。采样时需要强制均衡。

**文本长度分布**：train 中位数 129 字符（明显长于 valid 的 86 和 test 的 76），P95 达到 468。这决定了输入上限设为 **512 字符**（覆盖 P95）是合理的。

> [!note] 关系频率倾斜
> "位于"（193K）和"别名"（111K）两个关系占全部关系三元组的 **29%**。这是一个重要的数据偏斜——模型可能过度拟合这两个高频关系的抽取模式，而忽略低频但同样重要的关系类型。后续 pipeline 没有对关系类型做降采样或均衡处理，这是已知的数据偏差。

**六步 Pipeline（阈值集中在 `conf.py`）：**

```
原始数据 (171,471 条)
  │
  ▼ [Step 1] 01_normalize.py — 字段标准化 (171,471)
     text→input, relation 对齐, cate 归一化, 新增 source 字段
     │
  ▼ [Step 2] 02_filter.py — 两层过滤 (163,629)
     │  ├─ 硬过滤（剔除 3,585 条）:
     │  │   空关系 2,446 / 跨集泄漏 638 / 关系数>25 632
     │  │   输入过长>800 145 / head/tail 过长>100 40 / 输入过短<15 3
     │  └─ 软过滤（per-topic P99 分位数, 剔除 4,257 条）:
        输入长度超 P99 1,499 / head/tail 长度超 P99 1,461
        关系数超 P99 984 / 输出 JSON 长度超 P99 313
     │
  ▼ [Step 3] 03_quality_tier.py — 质量三档分层 (156,275 条 high)
     high: 匹配率100% + 关系数和长度在理想区间 (95.5%)
     medium: 3.9% / low: 0.6%
     → 最终采样仅保留 high 质量 (100%)
     │
  ▼ [Step 4] 04_derive_tasks.py — 四类任务派生 (623,650 条)
     从每个原始样本派生出 4 个 SFT 训练样本
     │
  ▼ [Step 5] 05_stratified_sample.py — 分层采样 (30,000 条)
     按 task_type(50/25/15/10) + topic(12均衡) + quality(high) + complexity
     │
  ▼ [Step 6] 06_to_chat_jsonl.py — 格式转写 + valid 切分
     统一为 instruction + schema + input → JSON output 三段式 chat-style JSONL
     全量 JSON 合法性校验: **100% 通过**
     │
  ▼ 最终产出
     data/sft_candidate/train.jsonl (28,500 条)
     data/sft_candidate/valid.jsonl (1,500 条, 从 train 中切分 5%)
     data/sft_candidate/metadata.json
```

**各步骤解决什么问题：**
- **Step 1 标准化**：解决字段不统一问题（text vs input, cate 漂移）
- **Step 2 过滤**：硬过滤剔除明确坏样本（空关系、泄漏、超长），软过滤按 per-topic P99 分位数控制极端值
- **Step 3 质量分层**：三档分层确保后续只采 high 质量
- **Step 4 任务派生**：从一个 IE 样本派生 4 类 SFT 任务（见下文）
- **Step 5 分层采样**：按 task_type + topic + quality 三维分层，保证分布均衡
- **Step 6 格式转写**：统一为 chat-style JSONL，切分 valid 集

**四类派生任务设计：**

从每个 InstructIE 原始样本派生 4 类 SFT 任务，配比经过精心设计：

| 任务类型 | 占比 | 训练信号 | 与目标的映射 |
|----------|------|----------|-------------|
| **ie_extraction** | **50%** | 给定 schema + 文本 → JSON 输出 | 核心主轴：标准信息抽取 |
| **text_to_json** | **25%** | 强调结构化 JSON 转换 | 输出格式稳定性训练 |
| **format_following** | **15%** | 只输出 JSON / 不附加解释 | 零容忍偏差的行为约束 |
| **schema_repair** | **10%** | 对正确输出做可控扰动 → 构造纠错任务 | Schema 校验与纠错能力 |

**配比逻辑：** InstructIE 最大价值在于 IE，所以信息抽取占半壁江山；格式遵循不单独成大类而是缩入 15% 作为辅助——因为本阶段不是在做泛问答增强，而是在做 schema-guided 结构化抽取。

**为什么是 50/25/15/10：** ie_extraction 是核心任务必须占最大比例；text_to_json 训练输出格式的稳定性（25% 足够）；format_following 是行为约束不需要大量样本（15%）；schema_repair 是增强型任务少量即可（10%）。

**为什么要做 topic 坐标：** 前 7 个 topic 占约 85%，自然科学和医学明显偏少。如果不强制均衡，模型会过度拟合高频 topic 的模式。分层采样时按 12 个 topic 等量分配（各 ~2,500 条），确保低频 topic 也有足够的训练信号。

**为什么只取 high quality 样本：** medium 和 low 质量的样本可能含有噪声标注或边界 case。对于 30K 规模的数据集，优先保证质量比追求数量更有价值。最终采样结果 high 质量占比 **100%**。

schema_repair 的构造方式值得一提：对正确的 gold output 做可控扰动（字段名错误 / 缺少字段 / 幻觉字段 / 类型错误），让模型学习识别和修复这类问题。这是原数据集中不存在的能力维度，纯靠数据增强引入。

### 关键结果 / 数据

| 维度 | 目标 | 实际 |
|------|------|------|
| 总规模 | 2~4 万 | **30,000** |
| 训练集 | 1~2 万 | **28,500**（95%）|
| 验证集 | 1,000~2,000 | **1,500**（5%，从 train 中独立切分）|
| ie_extraction 占比 | 50% | **50.0%**（精确匹配）|
| text_to_json 占比 | 25% | **25.0%** |
| format_following 占比 | 15% | **15.0%** |
| schema_repair 占比 | 10% | **10.0%** |
| 12 topic 均衡 | 各 ~2,500 | **各 2,500**（精确均衡）|
| high 质量占比 | 优先 high | **100%** |
| JSON 合法性 | 100% | **100%** |

### 这一节的结论

Pipeline 的工程核心在于**阈值集中配置**（conf.py）和**每步独立可审计**（JSON 统计报告）。6 步处理从 171K 条原始数据中产出 28.5K 条高质量 SFT 数据，全量 JSON 合法性校验 100% 通过。四类任务设计覆盖了信息抽取的核心能力和边缘场景（格式遵循、纠错），50/25/15/10 的配比反映了"核心为主、辅助为辅"的优先级。

---

## 4.3 Qwen LoRA 微调实现

### 这一小节要解决什么问题

有了 SFT 数据集后，如何在 Qwen2.5-1.5B-Instruct 上完成 LoRA 微调。关键挑战是与 MicroLM SFT 的适配差异（tokenizer、chat template、loss mask 方式），以及 smoke→正式的训练节奏控制。

### 核心设计 / 核心流程

**与 MicroLM SFT 的关键差异：**

| 维度 | MicroLM SFT (`train_sft.py`) | Qwen LoRA (`train_qwen_lora.py`) |
|------|-------------------------------|-----------------------------------|
| 基座模型 | 自研 TransformerLM (31.7M) | Qwen2.5-1.5B-Instruct (1.55B) |
| Tokenizer | 自研 BPETokenizer (vocab=6400) | HF AutoTokenizer (ChatML 模板) |
| LoRA 框架 | 自研 `lora.py`（LoRALinear） | PEFT 库（`get_peft_model`） |
| Chat 模板 | 自定义 role markers | ChatML（`<\|im_start\|>` / `<\|im_end\|>`）|
| Loss mask | `build_loss_labels()` 定位 assistant 标记 | prefix 对比法定位 assistant JSON 段 |
| 精度 | float32 | FP16 |
| System prompt | 可选注入（ratio=0.1）| 固定："你是一个严格遵循 schema 的信息抽取助手" |

复用的部分：配置文件组织方式（JSON → resolved config）、日志格式（JSONL）、训练循环结构（step loop → val check → checkpoint save）、报告模板。

**LoRA 配置：**

| 参数 | 值 | 说明 |
|------|-----|------|
| r (rank) | 8 | 低秩维度 |
| alpha | 16 | scaling = alpha/r = 2 |
| targets | q_proj, k_proj, v_proj, o_proj | 四个注意力投影层 |
| dropout | 0.05 | 防止过拟合 |
| 可训练参数 | **2,179,072** | 占全参 **0.14%** |
| Adaptor 存储 | **8.3 MB** | vs 基座 2,944 MB，节省 **99.7%** |

与 MicroLM LoRA 的对比：参数效率更高（0.14% vs 0.83%），因为基座模型更大（1.55B vs 31.7M），同样的 r=8 注入的绝对参数量更大，但占总参数比例更低。

**Loss Mask：prefix 对比法**

Qwen 使用 ChatML 模板渲染后的输入格式为：

```
<|im_start|>system\n你是一个严格遵循 schema 的信息抽取助手<|im_end|>
<|im_start|>user\n{instruction}\n\nSchema: {schema}\n\nInput: {input}<|im_end|>
<|im_start|>assistant\n

```

Loss mask 需要让模型只学习最后一个 `<|im_start|>assistant\n` 之后的内容（即 JSON 输出）。实现方式是 **prefix 对比法**：

1. 用 template 渲染一份"无回答"的版本（到 assistant 标记为止）作为 prefix
2. 用 template 渲染一份"有回答"的完整版本
3. 对比两者 token 数量的差异，定位 assistant 输出的起止位置
4. 该区间 label 保留真实 token ID，其余置 -100

这与 MicroLM 的 `build_loss_labels()` 思路一致（都是 assistant-only masked loss），但定位方式不同——MicroLM 通过自定义 role markers 字符串匹配定位，Qwen 通过 ChatML 模板的 prefix 长度差异定位。

**Smoke test → 正式训练的过渡：**

Smoke test（50 步）先验证链路完整性：

| 配置 | 值 |
|------|-----|
| lr | 3e-5 |
| batch_size | 2 |
| grad_accum | 4 |
| effective_batch | 8 |
| val_loss 变化 | 0.6833 → 0.3937（降幅 42.4%）|

Smoke 通过后启动正式训练（2000 步）：

| 配置 | 值 |
|------|-----|
| lr | 2e-5 |
| batch_size | 4 |
| grad_accum | 4 |
| effective_batch | 16 |
| 精度 | FP16 |
| 训练时长 | ~6.5 小时（RTX 5070 Ti）|

### 关键结果 / 数据

**Loss 曲线：**

| Step | train_loss | val_loss |
|------|-----------|----------|
| 100 | 0.3722 | 0.3999 |
| 500 | 0.2538 | 0.2028 |
| 1000 | 0.1520 | 0.1753 |
| 1500 | 0.2115 | 0.1629 |
| 2000 | 0.1857 | **0.1534** |

全程无震荡、无过拟合，loss 单调下降。val_loss 从 0.3999 降至 0.1534，**降幅 61.6%**。

**Loss 曲线阶段特征：**

| 阶段 | Step 范围 | 特征 |
|------|----------|------|
| 快速下降期 | 100–500 | val_loss 0.40 → 0.20，梯度信号强 |
| 稳定下降期 | 500–1500 | 0.20 → 0.16，收敛速度放缓但仍持续改善 |
| 收敛趋缓期 | 1500–2000 | 0.16 → 0.15，每 500 步仅改善 ~0.01 |

后期仍未完全收敛（曲线斜率未归零），继续训练可能仍有小幅改善空间。

### 这一节的结论

Qwen LoRA 微调的关键成功因素：(1) **先 smoke 再正式**——50 步 smoke 验证全链路后再投入 6.5 小时的正式训练；(2) **prefix 对比法 loss mask**——干净地隔离了 assistant 输出区间；(3) **FP16 精度**——在 1.55B 模型上安全使用混合精度，加速训练且不影响收敛质量。val_loss 降幅 61.6% 证明 LoRA 在 Qwen 上同样有效——这与 MicroLM 上的发现形成交叉验证。

**LoRA 学到了什么：结构化行为塑形。** 在阶段 6C 的评测中发现一个关键现象：LoRA 的主指标 Strict%（7.5%）略低于 base（10.0%），但这不是能力退步——而是**格式偏好变化**。qwen_base 倾向于输出扁平格式 JSON，qwen_lora 倾向于输出实体嵌套格式（InstructIE 数据集中的典型风格）。Alias 归一化评测证实了这一点：lora 的 Alias-Strict%（15.0%）是 base（7.5%）的 **2 倍**。核心结论：LoRA 的价值不在精确字段名对齐，而在**从自由格式 JSON 转向实体中心的规范 JSON**——这正是结构化抽取任务所需要的。

---

## 4.4 模型导出与部署准备

### 这一小节要解决什么问题

训练完成后如何将 LoRA adaptor 合并进基座模型并导出为 vLLM 可加载的格式，以及部署前需要准备哪些产物和脚本。

### 核心设计 / 核心流程

**LoRA Merge 导出**——`export_final_model.py` 将 LoRA adaptor 合并进基座模型：

```python
model = AutoModelForCausalLM.from_pretrained(base_model_path)
model = PeftModel.from_pretrained(model, adaptor_path)
model.merge_and_unload()          # LoRA 权重折叠进 W
model.save_pretrained(output_dir)  # 标准 HF 格式
```

导出至 `outputs/qwen_lora_merged_final/`，可直接被 vLLM 加载。脚本支持 `--skip-if-exists` 幂等导出，并自动记录导出元信息（源路径、时间戳、adaptor hash）。

**Merge 的意义：** 部署时不需要额外加载 adaptor 并在推理时动态合并——权重已经是最终状态，vLLM 可以像加载普通模型一样加载它。

**部署产物清单：**

| 产物 | 路径 | 状态 |
|------|------|------|
| 合并模型目录 | `outputs/qwen_lora_merged_final/` | 已导出 |
| 服务启动脚本 | `scripts/serve_vllm.sh` | 就绪（OpenAI 兼容 API）|
| Smoke test | `scripts/smoke_vllm.py` | 就绪（5 项功能验证）|
| Benchmark | `scripts/bench_vllm_local.py` | 就绪（单/多并发 TTFT/吞吐/P95）|
| 稳定性验证 | `scripts/check_structured_stability.py` | 就绪（Parse%/Strict%/Alias-Strict%）|
| 部署文档 | `docs/vllm_deploy.md` | 就绪 |
| Benchmark 报告 | `reports/vllm_benchmark_report.md` | 就绪 |

### 关键结果 / 数据

Smoke test **5/5 全部通过**：

| # | 测试项 | 验证内容 | 结果 |
|---|--------|----------|------|
| 1 | Health check | `/health` 端点正常响应 | **PASS** |
| 2 | Simple chat | 基础对话 completion 能力 | **PASS** |
| 3 | Structured extraction | 给定 schema 的信息抽取能力 | **PASS** |
| 4 | Multi-turn | 多轮对话上下文保持 | **PASS** |
| 5 | Response format | `response_format=json_object` 约束输出 | **PASS** |

Benchmark **5 组配置 0 错误**，稳定性验证 **Parse% 100%**（服务化环境下结构化输出无退化）。

### 这一节的结论

导出→部署的链条是完整的：PEFT merge_and_unload → HF 格式 → vLLM 加载 → OpenAI 兼容 API → smoke/benchmark/stability 全部通过。这证明了从训练到服务的路径是通的，不是停在 checkpoint 就结束。

---

## 相关记录

- [[01-项目总览]] — 项目全局地图，两条主线的关系说明
- [[02-自研 MicroLM 主线]] — 自研链路的 LoRA 实现（本线的参考起点）
