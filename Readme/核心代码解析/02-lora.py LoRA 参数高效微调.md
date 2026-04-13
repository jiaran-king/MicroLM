---
type: project-note
project: MicroLM
section: core-code-analysis
priority: P0
file: microlm/model/lora.py (173行)
---

# lora.py — LoRA 参数高效微调

> 典型的"面试很爱让你讲，甚至让你写个简化版"的模块。逻辑相对独立，手写难度适中。核心就一个类 `LoRALinear`（53 行），加上一组工具函数完成注入/合并/序列化。

---

## 核心公式

$$\text{output} = W \cdot x + \frac{\alpha}{r} \cdot B \cdot A \cdot x$$

- 原始权重 $W$ **冻结**（requires_grad=False）
- $A$ 矩阵：(r, in_features)，Kaiming uniform 初始化
- $B$ 矩阵：(out_features, r)，**零初始化**
- scaling = $\alpha / r$

---

## 逐段源码与解析

### 1. LoRALinear 类（L20-76）— 核心，仅 53 行

```python
class LoRALinear(nn.Module):
    """Drop-in replacement for Linear that adds a low-rank adapter."""

    def __init__(self, original: Linear, r: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.original = original
        self.original.weight.requires_grad_(False)   # ← 冻结原始权重

        out_features, in_features = original.weight.shape
        self.r = r
        self.scaling = alpha / r                     # ← scaling factor
        device = original.weight.device

        # A: (r, in_features),  B: (out_features, r)
        self.lora_A = nn.Parameter(torch.empty(r, in_features, device=device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self._merged = False
```

**三个关键设计决策：**

| 决策 | 代码 | 为什么 |
|------|------|--------|
| 冻结 W | L31 `requires_grad_(False)` | 保持预训练知识不被破坏 |
| A 用 Kaiming 初始化 | L41 | 保证初始时有合理的梯度传播路径 |
| B 零初始化 | L40 `torch.zeros` | 确保 LoRA 初始输出 delta=0 → 模型行为不变 |

> [!warning] B 零初始化是 LoRA 正确工作的前提
> 如果 A 和 B 都用随机初始化，注入 LoRA 后模型会立即偏离预训练状态。B 零初始化确保 `delta = B·A·x = 0`，训练从"等价起点"开始。这是面试高频追问点。

### 2. forward（L47-52）

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    original_out = torch.einsum("... i, o i -> ... o", x, self.original.weight)
    if self._merged:
        return original_out                    # merge 后跳过 LoRA 计算
    lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
    return original_out + lora_out
```

**计算路径：**
- 未 merge：`output = W·x + (α/r) · (x·A^T) · B^T`
- 已 merge：直接返回 `W·x`（此时 W 已包含 LoRA delta）

注意这里用了矩阵乘法 `@` 而非 einsum——因为 A/B 的形状是标准二维矩阵，不需要 einsum 的广播能力。

### 3. merge / unmerge（L56-72）

```python
@torch.no_grad()
def merge(self) -> None:
    """Fold LoRA weights into the original weight (for inference)."""
    if self._merged:
        return
    delta = (self.lora_B @ self.lora_A) * self.scaling
    self.original.weight.add_(delta)     # ← 原地修改 W
    self._merged = True

@torch.no_grad()
def unmerge(self) -> None:
    """Undo merge (restore original weight)."""
    if not self._merged:
        return
    delta = (self.lora_B @ self.lora_A) * self.scaling
    self.original.weight.sub_(delta)     # ← 原地减去 delta
    self._merged = False
```

**merge 的意义：** 推理时不再有额外的 LoRA 前向传播开销。merge 后的模型在推理时和普通模型完全一样。

**原地操作（add_ / sub_）：** 不创建新的权重副本，直接修改 W 张量的存储。这意味着 unmerge 可以精确恢复原始权重（只要中间没有其他修改）。

### 4. apply_lora_to_model（L92-118）— 模型级注入

```python
_DEFAULT_TARGETS = {"q_proj", "k_proj", "v_proj", "output_proj"}

def apply_lora_to_model(model, r=8, alpha=16.0, target_names=None):
    # 1. 全局冻结
    for p in model.parameters():
        p.requires_grad_(False)

    if target_names is None:
        target_names = _DEFAULT_TARGETS
    target_set = set(target_names)

    # 2. 先收集所有匹配项（不能边迭代边修改）
    replacements = []
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in target_set and isinstance(module, Linear):
            replacements.append((name, LoRALinear(module, r=r, alpha=alpha)))

    # 3. 执行替换
    for name, lora_layer in replacements:
        _replace_module(model, name, lora_layer)
```

**注入流程：**

```
原始模型 (31.7M 参数, 全部可训练)
    │
    ▼ ① 全部冻结 requires_grad=False
    │
    ▼ ② 遍历 named_modules，匹配 q_proj/k_proj/v_proj/output_proj
    │   每个 Linear → 替换为 LoRALinear
    │
    ▼ 结果：
    总参数量仍为 31.7M（W 冻结 + A/B 新增）
    可训练参数 = 262K（仅 A/B 矩阵）
    可训练占比 = 0.83%
```

**为什么选这四个目标层：** 注意力层的四个投影参数量最大（4 × d_model²），且对任务适应最敏感。FFN 层也可以注入，但收益通常不如注意力层明显。

### 5. 工具函数集（L121-172）

```python
def get_lora_params(model):          # 返回 A/B 参数列表供优化器使用
def get_lora_state_dict(model):      # 仅序列化 LoRA 权重（~1MB）
def load_lora_state_dict(model, sd): # 反序列化 LoRA 权重
def merge_lora(model):               # 批量合并所有 LoRALinear
def unmerge_lora(model):             # 批量撤销合并
def print_trainable_params(model):   # 打印总参数 vs 可训练参数统计
```

**state_dict 只存 A/B 的意义：** 全参 377 MB vs LoRA adaptor 仅 **1.0 MB**，节省 **99.7%** 存储。可以快速切换不同任务的 adaptor 而不需要保存完整模型。

---

## MicroLM 上的实测数据

| 指标 | 数值 |
|------|------|
| 总参数量 | 31,729,152 |
| 可训练参数 | 262,144 |
| 可训练占比 | **0.83%** |
| Adaptor 存储 | **1.0 MB**（vs 全参 377 MB）|
| 存储节省 | **99.7%** |
| LoRA val_loss | 2.5377（vs baseline 2.3265，差距 9%）|

Loss 曲线全程恒定差距 ~0.21，两条走势完全一致——LoRA 没有训练不稳定或发散，只是以更少的参数在更高的 loss 平面上收敛。

---

## 面试高频追问清单

| 问题 | 回答要点 |
|------|----------|
| LoRA 的核心思想是什么？ | 在冻结的预训练权重旁添加低秩适配矩阵，只训练这些小矩阵 |
| 为什么 A 用 Kaiming、B 用零初始化？ | Kaiming 保证梯度通路；零初始化确保初始 delta=0，从等价起点开始 |
| scaling = α/r 的作用？ | 控制适配强度。α 是超参数控制整体幅度，r 是秩控制容量 |
| 默认注入哪些层？为什么？ | q_proj/k_proj/v_proj/output_proj — 注意力层参数量大且对任务适应敏感 |
| merge 和 unmerge 怎么工作？ | merge: W += (B@A)*α/r；unmerge: W -= 同样的值。都是原地操作 |
| merge 后推理有什么变化？ | 不再有额外的 LoRA 前向开销，和普通模型一样推理 |
| LoRA adaptor 多大？ | MicroLM 上仅 1MB（全参 377MB 的 0.27%）|
| 0.83% 参数达到什么效果？ | val_loss 差距仅 9%，生成质量达到全参 SFT 的 ~85% |

---

## 相关记录

- [[01-transformer.py 模型主干]] — LoRA 注入的目标就是本文件的 Linear 层
- [[03-sft.py SFT 数据协议]] — LoRA 微调使用的数据协议与全参 SFT 相同
- [[07-train_qwen_lora.py Qwen 迁移线]] — PEFT 库实现的 LoRA（工业生态版本）
