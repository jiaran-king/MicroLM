---
type: project-note
project: MicroLM
section: core-code-analysis
priority: P0
file: microlm/model/transformer.py (416行)
---

# transformer.py — 自研模型主干

> 这是整个自研链路的核心中的核心。面试最容易追问的实现点全部集中在这里：注意力怎么写、QKV 怎么拆头、RoPE 怎么加、pre-norm vs post-norm、为什么用 SwiGLU、generate 为什么只取最后一个位置的 logits、KV Cache 怎么接进 forward。

---

## 架构总览

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

**超参数表：**

| 参数 | 值 | 说明 |
|------|-----|------|
| vocab_size | 6,400 | BPE 词表大小 |
| context_length | 512 | 最大序列长度 |
| d_model | 512 | 隐藏维度 |
| num_layers | 8 | Transformer 层数 |
| num_heads | 8 | 注意力头数（d_head=64）|
| d_ff | 1,344 | FFN 中间维度（≈2.625 × d_model）|
| rope_theta | 1,000,000 | RoPE 基频 |

---

## 逐段源码与解析

### 1. Linear 层（L8-17）

```python
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        std = math.sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("... i, o i -> ... o", x, self.weight)
```

**为什么用 einsum 而非 F.linear：** 让权重操作完全透明。后续 LoRA 注入时需要直接操作 `weight` 张量，不需要绕过 `nn.Linear` 的内部封装。初始化用截断正态分布（Xavier 风格），`std = sqrt(2/(in+out))`。

### 2. Embedding 层（L20-28）

```python
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
```

同样自实现。初始化 std=1.0（比 Linear 大），因为 embedding 层需要更大的初始变化范围来区分不同 token。

### 3. RMSNorm（L31-43）

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_float = x.to(torch.float32)
        rms = torch.sqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = (x_float / rms) * self.weight.to(torch.float32)
        return out.to(in_dtype)
```

**关键设计：float32 计算 + cast 回原 dtype。** 即使模型是 FP16/BF16，RMSNorm 内部也用 float32 做均方根计算，避免精度损失。`eps=1e-5` 防止除零。

RMSNorm vs LayerNorm 的区别：RMSNorm 不做均值中心化（没有减去 mean），只做方差归一化。计算量更少，效果相当。

### 4. SwiGLU FFN（L55-63）

```python
def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))
```

**公式：FFN(x) = W2(SiLU(W1·x)) · W3(x)**

三矩阵结构（比标准 FFN 多一个 W3）。SiLU 门控信号和线性通路分别计算后相乘，让网络有选择性地激活/抑制信息。这是 LLaMA/Qwen 系列的标准配置，比普通 ReLU FFN 效果更好。

d_ff=1344 ≈ 2.625 × d_model（SwiGLU 需要更大中间维度才能达到与 ReLU FFN 相当的参数量）。

### 5. RoPE 位置编码（L89-116）

```python
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even for RoPE")
        indices = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)
        inv_freq = theta ** (-indices / d_k)
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        angles = torch.outer(positions, inv_freq)
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        if x.ndim > cos.ndim and cos.ndim >= 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        x_pairs = rearrange(x, "... seq (pair two) -> ... seq pair two", two=2)
        cos = cos.unsqueeze(-1)
        sin = sin.unsqueeze(-1)
        x_even = x_pairs[..., 0:1]
        x_odd = x_pairs[..., 1:2]
        rotated = torch.cat((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
        return rearrange(rotated, "... seq pair two -> ... seq (pair two)")
```

**RoPE 核心思想：** 将位置信息编码为旋转矩阵，通过复数乘法（等价的 2D 旋转）注入到 attention 的 Q/K 中。

**实现要点：**
- **预计算 cos/sin 表**（L98-99）：`angles = outer(positions, inv_freq)`，其中 `inv_freq = θ^(-2i/d_k)`。`register_buffer` 让张量随模型自动迁移设备，但不参与梯度更新。
- **pair 拆分旋转**（L110-116）：将 d_k 维度每两个一组 `(x_{2i}, x_{2i+1})`，做 2D 旋转：
  ```
  x'_even = x_even * cos - x_odd * sin
  x'_odd  = x_even * sin + x_odd * cos
  ```
  这就是标准的 RoPE 旋转变换。

- **theta=1,000,000**：较大的基频值，适合短上下文场景（max_seq_len=512）。

> [!tip] register_buffer vs nn.Parameter
> `register_buffer` 注册的张量是"常量"——随模型迁移设备、可被序列化保存，但**不会被优化器更新**。RoPE 的 cos/sin 表在训练过程中不变，所以用 buffer 而非 parameter。

### 6. MultiHeadSelfAttention（L145-199）

```python
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int | None = None,
                 theta: float | None = None, device=None, dtype=None):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = None
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_head,
                                                  max_seq_len=max_seq_len, device=device)

    def forward(self, x, token_positions=None, past_k=None, past_v=None, use_cache=False):
        seq_len = x.shape[-2]
        leading_shape = x.shape[:-2]

        # QKV 投影 + 拆头
        q = rearrange(self.q_proj(x), "... seq (head d) -> ... head seq d", head=self.num_heads)
        k = rearrange(self.k_proj(x), "... seq (head d) -> ... head seq d", head=self.num_heads)
        v = rearrange(self.v_proj(x), "... seq (head d) -> ... head seq d", head=self.num_heads)

        # RoPE 位置编码
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
                token_positions = token_positions.view(*([1] * len(leading_shape)), seq_len).expand(*leading_shape, seq_len)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # KV Cache 分支 vs 标准分支
        if use_cache:
            if past_k is not None:
                k = torch.cat([past_k, k], dim=-2)
                v = torch.cat([past_v, v], dim=-2)
            attn_out = scaled_dot_product_attention(q, k, v, mask=None)  # 无 causal mask
            new_k, new_v = k, v
        else:
            causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))
            attn_out = scaled_dot_product_attention(q, k, v, mask=causal_mask)
            new_k, new_v = None, None

        attn_out = rearrange(attn_out, "... head seq d -> ... seq (head d)")
        out = self.output_proj(attn_out)

        if use_cache:
            return out, new_k, new_v
        return out
```

**前向传播流程：**

| 步骤 | 操作 | 输出形状 |
|------|------|----------|
| 1 | QKV 线性投影 | (batch, seq, d_model) |
| 2 | rearrange 拆头 | (batch, head, seq, d_head) |
| 3 | RoPE 旋转 Q 和 K | 同上 |
| 4 | **use_cache=True**：拼接历史 K/V；**False**：构建下三角 mask | K/V 变为 (batch, head, cached_seq, d_head) |
| 5 | scaled_dot_product_attention | (batch, head, seq, d_head) |
| 6 | 合并头 + output 投影 | (batch, seq, d_model) |

**KV Cache 关键细节：**
- `use_cache=True` 时**跳过 causal mask**（L187）：decode 阶段只有 1 个 query token，不需要掩码
- 新 K/V 通过 `torch.cat` 拿到缓存末尾（L185-186）
- 返回 `(out, new_k, new_v)` 三元组供外层管理

### 7. Scaled Dot-Product Attention（L125-142）

```python
def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    attn_dtype = v.dtype
    scores = torch.einsum(
        "... q d, ... k d -> ... q k",
        q.to(torch.float32),
        k.to(torch.float32),
    ) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    probs = softmax(scores, dim=-1).to(attn_dtype)
    return torch.einsum("... q k, ... k d -> ... q d", probs, v.to(attn_dtype))
```

**三个数值稳定性设计：**
1. 注意力分数计算用 **float32**（即使模型是 FP16）— L136-137
2. softmax 前减去最大值防止溢出 — L120（softmax 函数内）
3. mask 填充用 `finfo.min` 而非 `-inf` — L140

### 8. TransformerBlock（L202-254）

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta,
                 use_rms_norm=True, norm_mode="pre", ffn_type="swiglu",
                 device=None, dtype=None):
        super().__init__()
        self.norm_mode = norm_mode
        norm_cls = lambda: RMSNorm(d_model, device=device, dtype=dtype) if use_rms_norm else Identity()
        self.ln1 = norm_cls()
        self.ln2 = norm_cls()
        self.attn = MultiHeadSelfAttention(...)
        self.ffn = SwiGLU(d_model, d_ff, ...)  # 或 SiLU_FFN

    def forward(self, x, token_positions=None, past_k=None, past_v=None, use_cache=False):
        if self.norm_mode == "post":
            raise NotImplementedError("post-norm + kv cache 先别接，先跑通 pre-norm")

        h = self.ln1(x)
        if use_cache:
            attn_out, new_k, new_v = self.attn(h, ..., use_cache=True)
            x = x + attn_out          # 残差连接 1
            x = x + self.ffn(self.ln2(x))  # 残差连接 2
            return x, new_k, new_v
        else:
            x = x + self.attn(h, ...)
            x = x + self.ffn(self.ln2(x))
            return x
```

**Pre-norm 结构：** RMSNorm 在 attention/FFN **之前**。好处是梯度流经 norm 层时更稳定，训练更深网络时收敛更好。Post-norm 显式抛出 NotImplementedError（L236）—— 因为 post-norm 与 KV Cache 的交互有已知问题。

### 9. TransformerLM 主模型（L257-335）

```python
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers,
                 num_heads, d_ff, rope_theta, ...):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, ...)
        self.layers = nn.ModuleList([TransformerBlock(...) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, ...)
        self.lm_head = Linear(d_model, vocab_size, ...)

    def forward(self, token_ids, kv_cache=None, use_cache=False, start_pos=0):
        seq_len = token_ids.shape[-1]
        if seq_len > self.context_length:
            raise ValueError("input sequence length exceeds context length")

        leading_shape = token_ids.shape[:-1]
        token_positions = torch.arange(start_pos, start_pos + seq_len, device=token_ids.device)
        token_positions = token_positions.view(*([1] * len(leading_shape)), seq_len).expand(*leading_shape, seq_len)

        x = self.token_embeddings(token_ids)

        if use_cache and kv_cache is None:
            kv_cache = KVCache(len(self.layers))

        for layer_idx, layer in enumerate(self.layers):
            if use_cache:
                x, new_k, new_v = layer(
                    x, token_positions=token_positions,
                    past_k=kv_cache.k[layer_idx], past_v=kv_cache.v[layer_idx],
                    use_cache=True,
                )
                kv_cache.k[layer_idx] = new_k
                kv_cache.v[layer_idx] = new_v
            else:
                x = layer(x, token_positions=token_positions)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        if use_cache:
            return logits, kv_cache
        return logits
```

**forward 的两种模式：**

| 模式 | 用途 | 行为 |
|------|------|------|
| `use_cache=False` | 训练 / 无缓存推理 | 标准 forward，返回 logits |
| `use_cache=True` | KV Cache 推理 | 返回 (logits, kv_cache)，每层缓存 K/V |

**start_pos 的作用：** decode 阶段每次只送入 1 个新 token，`start_pos` 告诉 RoPE 当前 token 在完整序列中的绝对位置（L308-309）。Prefill 时 start_pos=0，第 N 次 decode 时 start_pos=prompt_len+N-1。

### 10. generate 方法（L338-389）— 自回归生成

```python
@torch.no_grad()
def generate(self, prompt_ids, max_new_tokens, eos_token_id=None,
             temperature=1.0, top_p=1.0):
    self.eval()
    generated = prompt_ids.clone()
    kv_cache = KVCache(len(self.layers))

    # ── Phase 1: Prefill ──
    logits, kv_cache = self.forward(prompt_ids, kv_cache=kv_cache,
                                     use_cache=True, start_pos=0)
    logits = logits[:, -1, :]       # ← 只取最后位置的 logits

    # ── Phase 2: Decode Loop ──
    for _ in range(max_new_tokens):
        if temperature != 1.0:
            logits = logits / (temperature + 1e-8)
        if top_p < 1.0:
            logits = self._top_p_filter(logits, top_p)
        probs = softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat((generated, next_token), dim=1)
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

        cur_pos = generated.shape[1] - 1
        logits, kv_cache = self.forward(next_token, kv_cache=kv_cache,
                                        use_cache=True, start_pos=cur_pos)
        logits = logits[:, -1, :]   # ← 每次都只取最后位置

    return generated
```

**generate 流程图：**

```
prompt_ids (shape: [1, seq_len])
    │
    ▼ [Prefill]
model.forward(prompt_ids, use_cache=True, start_pos=0)
→ 完整 prompt 过所有层，每层缓存 K/V
→ 取 logits[:, -1, :] （仅最后一个位置）
    │
    ▼ [Decode Loop] × max_new_tokens
① temperature 缩放 logits
② top-p nucleus sampling（累积概率截断）
③ softmax + multinomial 采样 → next_token
④ EOS 检查：命中则终止
⑤ model.forward([next_token], kv_cache=cached, start_pos=cur_pos)
→ 只算 1 个 token，新 K/V 拼到缓存末尾
→ 取 logits[:, -1, :]
    │
    ▼
generated_ids (prompt + new_tokens)
```

**为什么只取 `logits[:, -1, :]`：** 自回归模型的本质是"给定前 n 个 token，预测第 n+1 个"。prefill 后最后一个位置的 logits 就是"下一个 token 的概率分布"。decode 阶段输入只有 1 个 token，`[-1]` 就是这唯一的位置。

### 11. Top-P 截断（L391-415）

```python
def _top_p_filter(self, logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    # "右移一位"技巧：保留第一个越界 token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, float('-inf'))
    return logits
```

**右移一位技巧（L407）：** 将 remove 标记右移一格，确保至少保留最高概率的 token + 第一个导致超标的 token。否则极端情况下可能 mask 掉所有 token。

---

## 面试高频追问清单

| 问题 | 对应代码位置 | 回答要点 |
|------|-------------|----------|
| 注意力怎么写的？ | L125-199 | einsum QKV投影 → rearrange拆头 → RoPE → float32 softmax → einsum加权求和 |
| QKV 怎么拆头的？ | L172-174 | `rearrange(..., seq (head d) -> ... head seq d)` |
| RoPE 怎么实现的？ | L89-116 | 预计算cos/sin表 → pair拆分 → 2D旋转矩阵 |
| pre-norm vs post-norm 区别？ | L202-254 | pre-norm: norm在attention之前，训练更稳定；本项目只用pre-norm |
| 为什么用 SwiGLU？ | L55-63 | 门控激活函数，三矩阵结构，LLaMA/Qwen标配，效果优于ReLU FFN |
| generate 为什么只取最后一个位置？ | L362, L387 | 自回归只需预测下一个token，最后一个位置就是预测目标 |
| KV Cache 怎么接入 forward？ | L296-335 | use_cache标志控制：True时每层返回(new_k,new_v)，外层循环更新cache |
| decode 时为什么跳过 causal mask？ | L187 | 只有1个query token，不存在"看到未来"的问题 |
| einsum Linear vs nn.Linear？ | L8-17 | 权重操作透明，方便LoRA直接修改W张量 |
| RMSNorm 为什么内部用 float32？ | L38-42 | 即使FP16模型也用float32做归一化，避免精度损失 |

---

## 相关记录

- [[02-lora.py LoRA 参数高效微调]] — LoRA 注入的目标层就在本文件的 Linear 上
- [[03-sft.py SFT 数据协议]] — 训练数据格式决定模型学什么
- [[05-generate_text.py 推理链路]] — 本文件 generate() 的调用方
- [[06-chat.py 多轮对话系统]] — 本文件的上层应用
