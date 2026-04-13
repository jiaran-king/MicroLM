---
type: project-note
project: MicroLM
section: core-code-analysis
priority: P0
file: microlm/training/sft.py (218行)
---

# sft.py — SFT 数据协议

> 这是 SFT 阶段最有"设计味道"的文件。它不只是数据预处理，而是在定义一套**对话数据协议**：normalize → render → encode → build_loss_labels。面试里比训练循环更有区分度，因为它能体现你理解"监督微调到底在优化什么"。

---

## 协议总览

```
原始 JSONL（conversations 列表）
    │
    ▼ normalize_conversations()     — 校验合法性，统一 role 小写
    │
    ▼ maybe_add_system_prompt()     — 以 ratio=0.1 概率注入 system prompt
    │
    ▼ render_chat_prompt()          — 结构化渲染为线性文本
    │
    ▼ BPETokenizer.encode           — 文本 → token IDs
    │
    ▼ EOS padding                   — 截断/填充到 max_length
    │
    ▼ build_loss_labels()           — ★ 核心：assistant 区间 label 保留，其余 -100
    │
    ▼ (input_ids, labels)           — 返回给训练循环
```

---

## 逐段源码与解析

### 1. Role Markers 定义（L18-23）

```python
ROLE_MARKERS = {
    "system": "


\n",
    "user": "



\n",
    "assistant": "



\n",
    "tool": "<|tool|>\n",
}
```

自定义 role markers（非 ChatML）。每条消息以标记开头 + 内容 + 换行。assistant 消息额外追加 EOS 标记。

**格式示例：**
```
system
你是一个知识助手

user
什么是深度学习？

assistant
深度学习是机器学习的子领域...

```

### 2. normalize_conversations（L26-49）

```python
def normalize_conversations(conversations: list[dict[str, str]]) -> list[dict[str, str]]:
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("conversations must be a non-empty list")
    normalized = []
    for index, message in enumerate(conversations):
        if not isinstance(message, dict):
            raise ValueError(f"conversation turn {index} must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"turn {index} must contain string role/content")
        role = role.strip().lower()
        if role not in ROLE_MARKERS:
            raise ValueError(f"unsupported conversation role {role!r}")
        content = content.strip()
        if not content:
            continue                    # 跳过空内容
        normalized.append({"role": role, "content": content})

    if not normalized:
        raise ValueError("list becomes empty after normalization")
    return normalized
```

**守卫逻辑：** 类型检查 → role 合法性 → 大小写统一 → 空内容过滤 → 非空校验。这是数据进入系统的第一道防线。

### 3. maybe_add_system_prompt（L52-70）

```python
def maybe_add_system_prompt(conversations, rng, system_prompt_ratio,
                            system_prompts=None):
    if not conversations:
        return conversations
    if conversations[0]["role"] == "system":
        return conversations              # 已有 system，不重复添加
    if system_prompt_ratio <= 0.0:
        return conversations
    prompts = system_prompts or DEFAULT_CHAT_SYSTEM_PROMPTS
    if rng.random() >= system_prompt_ratio:
        return conversations             # 未命中概率
    injected = {"role": "system", "content": rng.choice(prompts)}
    return [injected, *conversations]
```

以 `ratio=0.1` 的概率随机注入 system prompt。使用确定性 RNG（`seed + index`），保证同一样本每次处理结果一致。

### 4. render_chat_prompt（L73-92）

```python
def render_chat_prompt(conversations, eos_token="


", add_generation_prompt=False):
    parts = []
    for message in conversations:
        role = message["role"]
        content = message["content"]
        parts.append(ROLE_MARKERS[role])
        parts.append(content)
        parts.append("\n")
        if role == "assistant":
            parts.append(eos_token)       # assistant 回复后接 EOS
            parts.append("\n")

    if add_generation_prompt:
        parts.append(ROLE_MARKERS["assistant"])  # 推理时末尾补 assistant 标记

    return "".join(parts)
```

**关键：assistant 后的 EOS 标记。** 训练数据中每条 assistant 回复都以 EOS 结尾，模型学到的是"输出一段文字然后遇到 EOS 就停"。推理时如果 prompt 中遗漏这个标记，生成质量会静默下降（这是 Bug 2）。

**add_generation_prompt 参数：**
- `False`（训练时）：正常渲染所有消息
- `True`（推理时）：在末尾追加 `assistant` 标记，触发模型开始生成

### 5. ★ build_loss_labels（L105-130）— 最核心函数

```python
def build_loss_labels(input_ids, tokenizer, max_length,
                      assistant_header_ids, eos_boundary_ids, pad_token_id):
    labels = [-100] * len(input_ids)         # ① 全部初始化为 -100
    index = 0
    while index < len(input_ids):
        header_index = _find_subsequence(
            input_ids, assistant_header_ids, start=index)   # ② 搜索 assistant 标记
        if header_index < 0:
            break
        start = header_index + len(assistant_header_ids)     # ③ assistant 内容起点
        end = _find_subsequence(
            input_ids, eos_boundary_ids, start=start)        # ④ 搜索 EOS 边界
        if end < 0:
            end = len(input_ids)                             # 无 EOS 则到序列末尾
            boundary = end
        else:
            boundary = min(end + len(eos_boundary_ids), max_length)

        # ⑤ 只在 assistant 区间内保留真实 token ID 作为 label
        for position in range(start, min(boundary, len(input_ids))):
            if input_ids[position] != pad_token_id:
                labels[position] = input_ids[position]

        index = boundary if end >= 0 else len(input_ids)
    return labels
```

**这是整条链路最关键的设计决策之一。**

**为什么只让 assistant 区间参与 loss 计算：**

```
输入序列：
[system] [sys_content] [user] [question] [assistant] [answer] [EOS]

labels:
[-100]    [-100]       [-100]   [-100]      [answer_tokens]   [-100]
          ↑ 不参与 loss                        ↑ 只有这里参与梯度更新
```

如果全序列参与 loss，模型会学到"预测 system/user 标记"这种无意义的模式，浪费容量并污染 pretrain 学到的语言表示。

**算法步骤：**

| 步骤 | 操作 | 目的 |
|------|------|------|
| ① | 全部初始化 -100 | PyTorch 的 cross_entropy 忽略 -100 位置 |
| ② | 在 input_ids 中搜索 `assistant` 标记的 token 序列 | 定位回答区间起点 |
| ③ | 起点 = 标记长度之后 | 排除标记本身 |
| ④ | 搜索 EOS 边界 (`"\n

"`) | 定位回答区间终点 |
| ⑤ | 区间内非 pad 位置赋值真实 token ID | 让这些位置参与 loss |

支持多轮对话：while 循环会找到所有 assistant 区间并分别标注。

### 6. SFTDataset.__getitem__（L193-207）— 完整数据处理管线

```python
def __getitem__(self, index):
    sample = self._read_sample(index)                          # 从 JSONL 读一行
    conversations = self._prepare_conversations(sample, index)  # normalize + inject system
    rendered = render_chat_prompt(
        conversations, eos_token=self.eos_token,
        add_generation_prompt=False)                           # 渲染为文本
    input_ids = self.tokenizer.encode(rendered)[:self.max_length]  # 编码 + 截断
    input_ids += [self.pad_token_id] * (self.max_length - len(input_ids))  # padding
    labels = build_loss_labels(                                # ★ 构造 loss mask
        input_ids=input_ids,
        tokenizer=self.tokenizer,
        max_length=self.max_length,
        assistant_header_ids=self.assistant_header_ids,
        eos_boundary_ids=self.eos_boundary_ids,
        pad_token_id=self.pad_token_id,
    )
    return torch.tensor(input_ids, dtype=torch.long), \
           torch.tensor(labels, dtype=torch.long)
```

8 行代码串起整个 SFT 数据协议。返回的 `(input_ids, labels)` 直接送入训练循环。

### 7. build_generation_prompt（L210-217）— 推理侧对称接口

```python
def build_generation_prompt(conversations, eos_token="


"):
    normalized = normalize_conversations(conversations)
    if normalized[-1]["role"] == "assistant":
        raise ValueError("generation prompt should end with user/system turns")
    return render_chat_prompt(normalized, eos_token=eos_token,
                              add_generation_prompt=True)
```

与 `render_chat_prompt` 对称设计。推理时在末尾补上 `assistant` 标记，让模型从正确的上下文开始生成。**任何格式偏差都会静默降低生成质量。**

---

## 训练 vs 推理格式一致性

这是 SFT 阶段最容易出 bug 的地方：

| 阶段 | 函数 | 格式 |
|------|------|------|
| 训练 | `render_chat_prompt(add_generation_prompt=False)` | 完整对话，assistant 后有 EOS |
| 推理 | `build_generation_prompt()` | 历史对话 + 末尾 `assistant` 标记（无 EOS） |

两者的 assistant 区间边界定义必须完全一致——否则 loss mask 和实际生成目标错位。

---

## 面试高频追问清单

| 问题 | 回答要点 |
|------|----------|
| 为什么只对 assistant 段算 loss？ | 防止 prompt 部分（system/user 标记和内容）的梯度污染 pretrain 权重 |
| loss mask 怎么构造？ | 先全部置 -100，搜索 assistant 标记定位区间，区间内赋值真实 token ID |
| -100 有什么特殊含义？ | PyTorch cross_entropy 默认忽略 index=-100 的位置 |
| 训练和推理格式怎么保持一致？ | 用同一个 render_chat_prompt 函数；推理时通过 add_generation_prompt 补标记 |
| 多轮对话怎么处理？ | while 循环找到所有 assistant 区间，分别标注 |
| system prompt 怎么注入？ | 以 ratio=0.1 概率随机注入，用确定性 RNG 保证可复现 |

---

## 相关记录

- [[01-transformer.py 模型主干]] — 消费本文件产出的 (input_ids, labels)
- [[02-lora.py LoRA 参数高效微调]] — LoRA 微调复用相同的数据协议
- [[04-data_loader.py 与 loss.py]] — Pretrain 的数据加载和 loss 计算（对比参考）
- [[07-train_qwen_lora.py Qwen 迁移线]] — prefix 对比法实现同样的 assistant-only loss（不同定位方式）
