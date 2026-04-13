---
type: project-note
project: MicroLM
section: core-code-analysis
priority: P0
file: microlm/training/data_loader.py (19行) + loss.py (30行)
---

# data_loader.py 与 loss.py — 训练最小闭环

> 这两个文件对应"训练最小闭环"：数据怎么取、loss 怎么算。代码量小但概念密集，`get_batch()` 的 x/y 键位和 `masked_cross_entropy` 的 mask 机制都是面试经典题。

---

## 1. data_loader.py — 流式数据采样（19 行）

### 源码

```python
import torch
import numpy as np
import numpy.typing as npt

def get_batch(
        dataset: npt.NDArray,
        batch_size: int,
        context_length: int,
        device: str
) -> tuple[torch.Tensor, torch.Tensor]:

    dataset_len = len(dataset)
    max_id = dataset_len - context_length - 1          # ① 防止越界
    ix = torch.randint(0, max_id + 1, (batch_size,))   # ② 随机取 batch_size 个起始位置
    x_stack = [dataset[i:i+context_length] for i in ix] # ③ 截取输入窗口
    y_stack = [dataset[i+1:i+context_length+1] for i in ix]  # ④ 截取标签窗口（右移一位）
    x = torch.from_numpy(np.array(x_stack)).to(device).long()
    y = torch.from_numpy(np.array(y_stack)).to(device).long()
    return x, y
```

### 解析

**核心操作：从连续 token 流中随机截取固定长度窗口。**

```
token 流: [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ...]
                                    ↑
                            random start = 4
                            context_length = 3

x (input):  [t4, t5, t6]     ← 模型看到的上下文
y (target): [t5, t6, t7]     ← 模型需要预测的（右移一位）
```

**为什么 x/y 错开一位：** 语言建模的本质是"给定前 n 个 token，预测第 n+1 个"。所以标签序列就是输入序列向右移动一个位置。

| 步骤 | 操作 | 说明 |
|------|------|------|
| ① `max_id = len - ctx - 1` | 计算最大合法起始位置 | 保证 `i + context_length + 1` 不越界 |
| ② `randint(0, max_id+1)` | 随机采样起始位置 | 每个 batch 是数据流中的随机位置 |
| ③ `dataset[i : i+ctx_len]` | 截取输入窗口 | 长度 = context_length |
| ④ `dataset[i+1 : i+ctx_len+1]` | 截取标签窗口 | 同样长度，内容右移一位 |

**流式采样的优势：** 不需要预先构建 Dataset/Dataloader，直接从 .npy memmap 中随机位置截取。百万级语料场景下既节省内存又实现了有效 shuffle——每个 batch 都是数据流中的不同位置，天然打散了数据顺序。

---

## 2. loss.py — Loss 计算（30 行）

### 源码

```python
def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Pretrain 用：全序列 next-token prediction loss"""
    shifted = logits - logits.max(dim=-1, keepdim=True).values   # 数值稳定
    logsumexp = torch.log(torch.exp(shifted).sum(dim=-1))
    target_logits = shifted.gather(
        dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (logsumexp - target_logits).mean()


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """SFT 用：仅对非 -100 位置计算 loss"""
    if logits.shape[:-1] != targets.shape:
        raise ValueError("logits and targets must agree on batch/time dimensions")
    if targets.shape != loss_mask.shape:
        raise ValueError("targets and loss_mask must have the same shape")

    shifted = logits - logits.max(dim=-1, keepdim=True).values
    logsumexp = torch.log(torch.exp(shifted).sum(dim=-1))
    safe_targets = targets.clamp_min(0)           # 将 -100 clamp 到 0（后续被 mask 掉）
    target_logits = shifted.gather(
        dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    losses = logsumexp - target_logits            # 逐位置 loss

    mask = loss_mask.to(losses.dtype)
    denom = mask.sum().clamp_min(1.0)             # 防止除零
    return (losses * mask).sum() / denom           # 只平均有意义的位置
```

### 解析

#### cross_entropy — Pretrain 用

**手写 cross_entropy 的数值稳定实现：**

$$\text{CE} = \log\left(\sum_j e^{z_j - z_{\max}}\right) - (z_{\text{target}} - z_{\max})$$

| 步骤 | 操作 | 目的 |
|------|------|------|
| 减去 max | `logits - max` | 防止 exp 溢出 |
| logsumexp | `log(sum(exp(...)))` | 归一化项的对数 |
| gather | 取出目标位置的 logit | 提取预测值 |
| 相减取均值 | `(logsumexp - target_logit).mean()` | 最终标量 loss |

全序列每个 position 都参与 loss 计算——这是 pretrain 的 next-token prediction 任务。

#### masked_cross_entropy — SFT 用

**在 cross_entropy 基础上增加 mask 机制：**

```python
safe_targets = targets.clamp_min(0)    # -100 → 0（临时替换，避免 gather 越界）
losses = logsumexp - target_logits     # 计算逐位置 loss
mask = loss_mask.to(losses.dtype)      # 有效位置=1，无效位置=0
return (losses * mask).sum() / mask.sum()  # 只平均有效位置
```

**关键技巧：`clamp_min(0)` + mask 配合。**

targets 中 -100 表示"该位置不参与 loss"。但直接用 - 做 gather 会取出错误的 logit 值。解决方案：
1. 先将 -100 clamp 到 0（得到一个占位值）
2. 正常计算所有位置的 loss
3. 用 mask 把无效位置的 loss 清零
4. 只对有效位置求平均

这等价于 PyTorch 的 `F.cross_entropy(ignore_index=-100)`，但完全手写实现。

---

## 两个 loss 的对比

| 维度 | cross_entropy | masked_cross_entropy |
|------|--------------|---------------------|
| 使用阶段 | Pretrain | SFT |
| 参与计算的 position | **全部** | 仅 assistant 区间 |
| targets 特征 | 全部是合法 token ID | 合法 ID 与 -100 混合 |
| 分母 | 总 token 数 | 有效 token 数（mask sum）|
| 对应的数据协议 | 连续 token 流 | sft.py 的 build_loss_labels |

---

## 面试高频追问清单

| 问题 | 回答要点 |
|------|----------|
| get_batch 怎么取数据的？ | 从连续 token 流中随机取起始位置，截取固定长度窗口；x 和 y 错开一位 |
| 为什么 x/y 错开一位？ | 语言建模：给定前 n 个 token，预测第 n+1 个 |
| 为什么不用 DataLoader？ | 流式采样更省内存，百万级语料不需要预先构建 dataset |
| cross_entropy 怎么保证数值稳定？ | 先减去 max（防止 exp 溢出），再算 logsumexp |
| masked_cross_entropy 的 mask 怎么工作？ | -100 位置先 clamp 到 0 做 gather，再用 mask 把无效位置 loss 清零，最后只对有效位置求平均 |
| 为什么不直接用 F.cross_entropy？ | 手写实现展示了对原理的理解；且可以灵活控制 mask 逻辑 |

---

## 相关记录

- [[01-transformer.py 模型主干]] — 消费 get_batch 产出的 (x, y)
- [[03-sft.py SFT 数据协议]] — 产出 masked_cross_entropy 所需的 labels/mask
- [[05-generate_text.py 推理链路]] — Pretrain 模型的推理入口
