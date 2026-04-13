---
type: project-note
project: MicroLM
section: self-developed-mainline
---

# 二、自研 MicroLM 主线

> 这一章专门讲从零搭的那条线。它的定位是"证明你真的理解并实现过完整训练链路"。按"数据 → 模型 → 训练 → 微调 → 能力边界"一条线讲清楚。

---

## 2.1 预训练数据处理与 Tokenizer

### 这一小节要解决什么问题

原始语料（~141 万条中文文本）如何变成可训练的 token 流。这涉及三个环节：清洗与切分、BPE tokenizer 训练、文本编码为模型可消费的 token ID 序列。

### 核心设计 / 核心流程

**语料处理管线**——`prepare_pretrain_jsonl.py` 对 `data/pretrain_hq.jsonl`（约 1.6GB / 141 万条）执行清洗与切分：

| 清洗规则 | CLI 控制 | 在 141 万条上的效果 |
|----------|----------|---------------------|
| 控制字符清理 | 默认开启 | 移除 U+0000-U+001F（保留 \t\n\r）|
| HTML 标签清理 | `--clean-html` | 命中 11,427 条 |
| 空白压缩 | `--compress-whitespace` | 命中 55,681 条 |
| 长度过滤 | `--min/max-length` | 过滤 36 条 |
| 精确去重 | `--dedup` | 去重 821 条 |

整体过滤率仅 0.06%，但规范化了 4.7% 的文档。所有规则通过 CLI 参数控制，向后兼容——不加参数时行为与初始版本一致。

切分阶段使用 **SHA1 哈希确定性划分** train/valid，避免随机切分导致的分布偏移。文档之间插入 EOS 分隔符，让模型在 pretrain 阶段就能感知文档边界。

**BPE Tokenizer 训练**——`bpe.py` 实现标准 byte-level BPE：

1. 初始化 byte-level 词表（前 256 个 token）
2. GPT-2 风格预分词：按空格/标点边界切分，**中文字符独立成词**
3. 统计 pair 频率，迭代 merge 至目标大小 **6400**
4. 通过 `bytes_to_unicode` 编码保存 vocab.json 与 merge.txt

**为什么词表选 6400：** vocab_size=6400 是一个刻意的小词表选择。对于 31.7M 的微型模型，大词表意味着 embedding 层占用过多参数比例，而中文场景下 6400 个 token 足以覆盖常用字和常见子词组合。

**文本编码**——`tokenizer.py` 实现：
- 编码：文本 → BPE merge → token IDs，支持流式处理（`.npy` memmap），适合百万级语料
- 解码：token IDs → 文本，处理 special token 和未知 token
- 截断与 padding：统一到 context_length=512，EOS token id 填充

### 关键结果 / 数据

产出物：`data/pretrain_clean_v2/train.txt` + `valid.txt` + `tokenizer_corpus.txt` + `tokenized_full/train_ids.npy` + `valid_ids.npy`

### 这一节的结论

数据处理管线的设计体现了两个原则：(1) **确定性**——SHA1 哈希切分保证可复现；(2) **可配置性**——所有清洗规则通过 CLI 参数控制，不破坏已有行为。小词表（6400）的选择是在模型容量和 token 效率之间的权衡——对 31.7M 模型合理，但也意味着后续 JSON 输出时 token 效率较低（这是迁移到 Qwen 的动机之一）。

---

## 2.2 TransformerLM 模型设计

### 这一小节要解决什么问题

设计一个 31.7M 参数的微型 Transformer 语言模型，在有限规模下尽可能保留现代 LLM 的核心架构特征（RoPE、SwiGLU、RMSNorm pre-norm），同时让所有组件都便于后续 LoRA 注入和 KV Cache 推理加速。

### 核心设计 / 核心流程

**模型整体结构：**

```
TransformerLM (31,729,152 参数)
├── Embedding           — 自实现（非 nn.Embedding），截断正态初始化 std=1.0
├── TransformerBlock × 8
│   ├── RMSNorm (pre-norm)     — float32 计算，cast 回输入 dtype
│   ├── MultiHeadSelfAttention (8 heads, d_head=64)
│   │   ├── q/k/v/output proj  — 自定义 Linear（einsum 实现）
│   │   └── RoPE               — 预计算 cos/sin 表，register_buffer
│   ├── RMSNorm (pre-norm)
│   └── SwiGLU FFN (d_ff=1344)
│       └── FFN(x) = W2(SiLU(W1·x)) * W3·x
├── RMSNorm (final)
└── lm_head (Linear → logits over 6400 vocab)
```

**自定义 Linear 层。** 使用 `torch.einsum("... i, o i -> ... o", x, weight)` 替代标准 `F.linear`。权重初始化采用截断正态分布：std = sqrt(2/(in+out))，边界在 ±3×std。这个选择让后续 LoRA 注入时的权重操作更透明——不需要绕过 nn.Linear 的内部封装。

**RoPE 位置编码。** rope_theta = 1,000,000（较大的 base，适合短上下文场景）。cos/sin 表预计算到 max_seq_len 并通过 register_buffer 挂载，随模型自动迁移设备。

**Pre-norm + SwiGLU。** 采用 LLaMA/Qwen 系列的标准配置：RMSNorm 在 attention 和 FFN 之前（pre-norm），FFN 使用 SwiGLU 门控激活函数。post-norm 在代码中显式抛出 NotImplementedError——因为 post-norm 与 KV Cache 的交互存在已知问题，不做支持。

**KV Cache 增量推理支持。** `forward()` 支持 `use_cache=True` 模式：
- **Prefill 阶段**：完整 prompt 一次性送入，缓存每层的 K/V 张量
- **Decode 阶段**：每次只输入 1 个新 token，通过 `start_pos` 计算 RoPE 位置，将新 K/V 拼接到缓存末尾
- Decode 时跳过 causal mask（只有 1 个 query token，无需掩码）

### 关键结果 / 数据

**核心超参数表：**

| 参数 | 值 | 说明 |
|------|-----|------|
| vocab_size | 6,400 | BPE 词表大小 |
| context_length | 512 | 最大序列长度 |
| d_model | 512 | 隐藏维度 |
| num_layers | 8 | Transformer 层数 |
| num_heads | 8 | 注意力头数（d_head=64）|
| d_ff | 1,344 | FFN 中间维度（≈2.625 × d_model）|
| rope_theta | 1,000,000 | RoPE 基频 |

总参数量：**31,729,152（~31.7M）**

### 这一节的结论

模型设计的核心决策链是：**einsum Linear → 透明权重操作 → LoRA 注入便利；pre-norm + register_buffer → KV Cache 兼容；小词表 + 短上下文 → 适合微型模型但限制了结构化输出能力**。每一个选择都是在"理解原理"和"工程可用性"之间的平衡。

---

## 2.3 Pretrain 训练流程

### 这一小节要解决什么问题

有了 tokenizer 和模型结构后，如何在百万级语料上完成预训练。涉及数据加载方式、训练循环组织、优化器/调度器/梯度裁剪的配合、以及 checkpoint 和日志管理。

### 核心设计 / 核心流程

**数据加载**——`data_loader.py` 的 `get_batch()` 从连续的 token ID 序列（.npy memmap）中**随机截取固定长度窗口**：

```python
x = ids[i : i + context_length]      # 输入：前 512 个 token
y = ids[i + 1 : i + context_length + 1]  # 标签：后 512 个 token（shifted by 1）
```

每个 batch 都是数据流中的一个随机位置，天然实现了 data shuffling。这种方式比先切分样本再 shuffle 更节省内存，适合百万级语料的流式训练。**x/y 错开一位的原因**：语言建模的本质是"给定前 n 个 token，预测第 n+1 个"，所以标签序列就是输入序列右移一位。

**训练循环**——`train_pretrain.py` 组织完整流程：

| 组件 | 实现 | 关键参数 |
|------|------|----------|
| Loss | 全序列 cross_entropy | 每个 position 都参与预测下一个 token |
| Optimizer | 自实现 AdamW | bias correction + weight decay 解耦 |
| Scheduler | linear warmup + cosine decay | lr: 2e-4 → min_lr=2e-5, warmup_iters=2000 |
| Gradient clipping | 全局 L2 范数裁剪 | max_norm=1.0 |
| Checkpoint | save/load 含优化器状态 | 支持断点续训；另有 load_model_state 仅加载模型权重供 SFT init |

**Cross entropy 怎么算：** 对序列中每个 position 的预测 logit 与真实 label 做 cross_entropy，然后取 mean。全序列参与 loss（不同于 SFT 阶段的 assistant-only masked loss）。

**三者的配合关系：** AdamW 负责 parameter update（含 weight decay 解耦和 bias correction），cosine scheduler 控制 learning rate 从 warmup 峰值平滑衰减到最小值，gradient clipping 在每步更新前将全局梯度范数裁剪到 max_norm=1.0 以防止梯度爆炸。三者共同保证训练稳定收敛。

训练日志以 JSONL 格式写入 `train_log.jsonl`，每步记录 step / train_loss / val_loss / lr / timestamp，支持 wandb 同步。

### 关键结果 / 数据

产出物目录：

```
outputs/pretrain_full_corpus/
├── model_config.json          # 模型结构配置（冻结）
├── resolved_train_config.json # 实际运行参数（含路径解析结果）
├── ckpt.pt                    # 中间 checkpoint
└── ckpt_final.pt              # 最终 checkpoint（31.7M 参数）
```

### 这一节的结论

Pretrain 训练流程的设计亮点是 **get_batch 的流式采样**——不需要预先构建 dataset/dataloader，直接从 memmap 中随机位置截取窗口。这在百万级语料场景下既节省内存又实现了有效 shuffle。完整的 optimizer-scheduler-clipping 配合保证了训练稳定性，checkpoint 机制支持断点续训和 SFT 阶段的权重继承。

---

## 2.4 SFT 数据协议与训练机制

### 这一小节要解决什么问题

Pretrain 完成后，模型具备基础语言建模能力但没有对话意识。SFT（Supervised Fine-Tuning）阶段需要解决的是：如何将对话数据转化为模型可学习的格式，以及如何保证微调不会破坏 pretrain 学到的语言表示。

### 核心设计 / 核心流程

`sft.py` 是 SFT 阶段的"数据协议层"，定义对话格式标准和 loss mask 规则。这是整条链路中最关键的设计之一——它决定了 pretrain 权重如何被安全地适配到对话任务上。

**Chat Prompt 格式**使用自定义 role markers（非 ChatML）：

| role | 格式 |
|------|------|
| system | 以 `system` 标记开头 |
| user | 以 `user` 标记开头 |
| assistant | 以 `assistant` 标记开头，回复后接 EOS 换行 |
| tool | 以 `tool` 标记开头 |

**SFTDataset 的完整处理流程**（每条样本）：

1. 按 offset 从 JSONL 读取一行
2. `normalize_conversations()` — 校验合法性，统一 role 小写，过滤空轮次
3. `maybe_add_system_prompt()` — 以 ratio=0.1 概率注入默认 system prompt
4. `render_chat_prompt()` — 结构化对话渲染为线性文本
5. BPETokenizer.encode — 文本转 token IDs，截断到 max_length=512
6. EOS token id padding
7. `build_loss_labels()` — **核心步骤**：定位 assistant 区间的起止位置，只在该区间保留真实 token ID 作为 label，其余位置置为 -100
8. 返回 `(input_ids, labels)`

**为什么只让 assistant 区间参与 loss：** 让梯度只流向 assistant 回答部分，防止 prompt（system/user 标记和内容）的梯度污染 pretrain 学到的语言表示。如果全序列参与 loss，模型会学到"预测 system/user 标记"这种无意义的模式，浪费容量。

**训练格式与推理格式如何保持一致：** 推理侧的 `build_generation_prompt()` 保证格式一致——在末尾补上 assistant 标记，让模型从正确的上下文开始生成。任何格式偏差都会静默降低生成质量（见 Bug 2.2 的教训）。

**全参 SFT 结果：**

| 对比项 | Pretrain | SFT baseline |
|--------|----------|---------------|
| 数据格式 | 连续 token 流（next-token prediction）| 结构化对话 → chat prompt |
| Loss | 全序列 cross_entropy | assistant-only masked_cross_entropy |
| 学习率 | 2e-4 | 1e-5（微调用更小学习率）|
| 最终 val_loss | — | **2.3265**（step 1000）|

SFT 复用了 pretrain 的 tokenizer、模型结构和 checkpoint 机制，唯一变化的是数据格式和 loss 计算方式。

### 关键结果 / 数据

Baseline val_loss = **2.3265**（step 1000）。SFT 使模型获得了对话格式理解、中文输出能力和基本指令响应能力——在固定 Prompt 评测中平均评分从 pretrain 的 1.13 提升到 **2.04（+81%）**。

### 这一节的结论

SFT 数据协议的核心价值在于 **loss mask 的精确控制**。通过 build_loss_labels() 定位 assistant 区间边界，只让回答部分参与梯度更新，保护了 pretrain 权重不被 prompt 部分的噪声梯度污染。这个设计模式后来在 Qwen 迁移线上以 prefix 对比法的形式复用——思路一致，定位方式不同。

---

## 2.5 LoRA 接入与参数高效微调

### 这一小节要解决什么问题

全参 SFT 需要 31.7M 参数全部参与训练。LoRA（Low-Rank Adaptation）的目标是用极少的额外参数（0.83%）达到接近全参微调的效果，同时保持 adaptor 可独立存储和快速切换。

### 核心设计 / 核心流程

`lora.py` 实现纯 PyTorch 的 LoRA，不依赖 PEFT 库。

**LoRALinear 的计算公式：**

$$\text{output} = W \cdot x + \frac{\alpha}{r} \cdot B \cdot A \cdot x$$

- 原始权重 $W$ 冻结（requires_grad=False）
- $A$ 矩阵：(r, in_features)，Kaiming uniform 初始化
- $B$ 矩阵：(out_features, r)，**零初始化**（确保初始 delta 为零，模型行为不变）
- scaling = $\alpha / r$

**为什么 A/B 这样初始化：** A 用 Kaiming uniform 保证初始时有合理的梯度传播路径；B 用零初始化确保 LoRA 的初始输出 delta 为零——即注入 LoRA 后模型行为与原始模型完全一致，训练从"等价起点"开始。

**默认注入目标：** 四个注意力投影层 —— `q_proj`, `k_proj`, `v_proj`, `output_proj`

**核心 API：**

| 函数 | 功能 |
|------|------|
| `apply_lora_to_model(model, r, alpha, target_names)` | 遍历替换匹配 Linear 为 LoRALinear，冻结全部原参 |
| `get_lora_params(model)` | 返回 A/B 参数列表供优化器使用 |
| `merge_lora() / unmerge_lora()` | 批量合并/撤销合并（直接修改 W 权重）|
| `get_lora_state_dict() / load_lora_state_dict()` | 仅序列化/反序列化 LoRA 权重（~1MB）|

**Merge / Unmerge 怎么做：** merge 时直接修改原始 W 权重：`W = W + (α/r) · B · A`；unmerge 时反向操作：`W = W - (α/r) · B · A`。这两种操作都是原地修改，不创建新的权重副本。merge 后的模型在推理时不再有额外的 LoRA 前向传播开销。

### 关键结果 / 数据

**MicroLM LoRA 效率数据：**

| 指标 | 数值 |
|------|------|
| 总参数量 | 31,729,152 |
| 可训练参数 | 262,144 |
| 可训练占比 | **0.83%** |
| Adaptor 存储 | 1.0 MB（vs 全参 377 MB）|
| 存储节省 | **99.7%** |
| LoRA val_loss | **2.5377**（vs baseline 2.3265，差距 9%）|

**Loss 曲线对比（逐步记录）：**

| Step | Baseline val_loss | LoRA val_loss | 差值 |
|------|-----------------|-------------|------|
| 50 | 2.5787 | 2.7593 | +0.1806 |
| 100 | 2.5251 | 2.7312 | +0.2061 |
| 200 | 2.4735 | 2.6727 | +0.1992 |
| 300 | 2.4385 | 2.6372 | +0.1987 |
| 500 | 2.3897 | 2.5925 | +0.1782 |
| 700 | 2.3585 | 2.5657 | +0.2072 |
| 1000 | **2.3265** | **2.5377** | **+0.2112** |

差距全程恒定（~0.21），两条曲线走势完全一致——说明 LoRA 没有训练不稳定或发散，只是以更少的参数在更高的 loss 平面上收敛。

### 这一节的结论

LoRA 在 MicroLM 上验证了参数高效微调的核心假设：**低秩适配空间足以捕获微调所需的方向调整**。0.83% 的可训练参数达到全参 SFT 约 85% 的效果（val_loss 差距仅 9%，生成质量在同一量级），而 adaptor 存储仅需 1.0 MB（节省 99.7%）。merge/unmerge 机制支持两种推理模式，灵活适配不同部署场景。

---

## 2.6 自研链路的能力边界

### 这一小节要解决什么问题

31.7M 参数的模型能做到什么、做不到什么。这不是"模型不好"的自我批评，而是基于实测数据的理性评估——它直接驱动了后续向 Qwen 迁移的决策。

### 核心设计 / 核心流程

基于 13 条固定 prompt × 3 个模型（pretrain / baseline / lora）的统一评测结果，结合结构化评测（阶段 6C）的数据，绘制能力边界图。

**Pretrain → SFT → LoRA 的能力差异：**

| 维度 | Pretrain | SFT Baseline | LoRA (0.83%参数) |
|------|----------|-------------|------------------|
| 对话意识 | 无（纯续写）| 有（指令跟随）| 有 |
| 中文输出 | 符号乱码 | 流畅中文 | 流畅中文 |
| 平均评分 | 1.13 | **2.04 (+81%)**| 1.73 (+53%) |
| 格式遵循 | N/A | 较好（列表等）| 概念解释类更强 |
| JSON 输出 | 不可能 | Parse%=**0%** | Parse%=**0%** |
| 长输出质量 | N/A | ~64-128 token 后崩坏 | 同左 |

**Baseline vs LoRA 的分化：**

| 维度 | Baseline (全参) | LoRA (0.83%参数) |
|------|----------------|-----------------|
| 优势领域 | 指令遵循（2.08 vs 1.00）、续写（2.19 vs 1.25）| 中文表达与总结（**2.42** vs 2.25）|
| 典型场景 | 列表格式输出更规范 | 概念解释类任务结构化程度更高 |
| 共同弱点 | 128 token 后出现 **repetition loop**（重复崩溃）| 同左 |

具体到单条 prompt 的典型差异：
- "解释人工智能"：lora 得分 **14/20**（全场最佳），提到 ML/NLP/CV 分支且结构化程度最高
- "列出三种水果"：baseline 成功用列表格式输出苹果（10/20），lora 输出乱码（4/20）
- "春天的早晨"续写：baseline 意境更好（"蝴蝶翩翩起舞"，9/16），lora 跑题到咖啡推荐（5/16）

### 关键结果 / 数据

**能力边界总结表：**

| 能做到 | 做不好 | 做不到 |
|--------|--------|--------|
| 简单事实问答（"最大海洋"→太平洋）、概念解释（AI 定义）、格式化列表 | 精确翻译、JSON 输出、多步推理 | 长文连贯叙事、复杂指令组合、无 repetition 的长输出 |

**Repetition Loop 现象：** MicroLM 的 SFT baseline 和 LoRA 在多数 prompt 上都会在生成约 64~128 token 后进入重复循环模式（重复最后一个短语或片段，如"诗以诗以诗以"、"444444..."）。这不是某个模块能修好的问题，而是模型容量不足的表现——31.7M 参数无法维持长程生成的语义一致性。

### 这一节的结论

核心瓶颈是模型容量。31.7M 参数限制了知识存储和推理深度——多数失败不是"完全不会"，而是"开头对但后续崩"或"知道但说不准"。结构化评测中 Parse%=0% 是最硬的证据：小词表（6400）+ 无结构化预训练 + 无 schema-guided SFT 数据 = 完全不具备 JSON 输出能力。这也直接驱动了后续向 Qwen 迁移的决策，以及在 Qwen 线上聚焦结构化输出（短 JSON 输出不需要长程连贯性）的策略选择。

---

## 相关记录

- [[01-项目总览]] — 项目全局地图与 Qwen 迁移线
- [[产物]] — 所有 checkpoint、数据集、报告索引
