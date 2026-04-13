import math

import torch
import torch.nn as nn
from einops import rearrange


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        std = math.sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("... i, o i -> ... o", x, self.weight)


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


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


class Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


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


class SiLU_FFN(nn.Module):
    """SiLU-only FFN 变体: FFN(x) = W2(SiLU(W1·x))，d_ff = 4 * d_model 以匹配 SwiGLU 参数量"""
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)))


class KVCache:
    def __init__(self, num_layers: int):
        self.k = [None] * num_layers
        self.v = [None] * num_layers

    def reset(self):
        for i in range(len(self.k)):
            self.k[i] = None
            self.v[i] = None



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
## 简单说，register_buffer 让这些张量享受和模型参数一样的"生命周期管理"（设备、精度、序列化），
# 但又不会被优化器当作可训练参数去更新梯度。这对 RoPE 这种预计算的常量张量来说正好合适。

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        # Insert head dimension for multi-head attention (4D+ inputs)
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


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    shifted = x - x.max(dim=dim, keepdim=True).values
    exp_shifted = torch.exp(shifted)
    return exp_shifted / exp_shifted.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = q.shape[-1]
    # Use float32 for the score/softmax path so fp16 inference stays numerically stable.
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


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int | None = None, theta: float | None = None,
                 device=None, dtype=None):
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
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_head, max_seq_len=max_seq_len, device=device)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
        past_k: torch.Tensor | None = None,
        past_v: torch.Tensor | None = None,
        use_cache: bool = False,
    ):
        seq_len = x.shape[-2]
        leading_shape = x.shape[:-2]

        q = rearrange(self.q_proj(x), "... seq (head d) -> ... head seq d", head=self.num_heads)
        k = rearrange(self.k_proj(x), "... seq (head d) -> ... head seq d", head=self.num_heads)
        v = rearrange(self.v_proj(x), "... seq (head d) -> ... head seq d", head=self.num_heads)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
                token_positions = token_positions.view(*([1] * len(leading_shape)), seq_len).expand(*leading_shape, seq_len)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        if use_cache:
            if past_k is not None:
                k = torch.cat([past_k, k], dim=-2)
                v = torch.cat([past_v, v], dim=-2)
            attn_out = scaled_dot_product_attention(q, k, v, mask=None)
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


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float,
                 use_rms_norm: bool = True, norm_mode: str = "pre", ffn_type: str = "swiglu",
                 device=None, dtype=None):
        super().__init__()
        self.norm_mode = norm_mode

        norm_cls = lambda: RMSNorm(d_model, device=device, dtype=dtype) if use_rms_norm else Identity()
        self.ln1 = norm_cls()
        self.ln2 = norm_cls()

        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )

        if ffn_type == "silu":
            self.ffn = SiLU_FFN(d_model=d_model, d_ff=4 * d_model, device=device, dtype=dtype)
        else:
            self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
        past_k: torch.Tensor | None = None,
        past_v: torch.Tensor | None = None,
        use_cache: bool = False,
    ):
        if self.norm_mode == "post":
            raise NotImplementedError("post-norm + kv cache 先别接，先跑通 pre-norm")

        h = self.ln1(x)

        if use_cache:
            attn_out, new_k, new_v = self.attn(
                h,
                token_positions=token_positions,
                past_k=past_k,
                past_v=past_v,
                use_cache=True,
            )
            x = x + attn_out
            x = x + self.ffn(self.ln2(x))
            return x, new_k, new_v
        else:
            x = x + self.attn(h, token_positions=token_positions)
            x = x + self.ffn(self.ln2(x))
            return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        use_rms_norm: bool = True,
        norm_mode: str = "pre",
        ffn_type: str = "swiglu",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    use_rms_norm=use_rms_norm,
                    norm_mode=norm_mode,
                    ffn_type=ffn_type,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype) if use_rms_norm else Identity()
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        token_ids: torch.Tensor,
        kv_cache: KVCache | None = None,
        use_cache: bool = False,
        start_pos: int = 0,
    ):
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
                    x,
                    token_positions=token_positions,
                    past_k=kv_cache.k[layer_idx],
                    past_v=kv_cache.v[layer_idx],
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

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int = None,
        temperature: float = 1.0,
        top_p: float = 1.0
    ) -> torch.Tensor:
        self.eval()

        if prompt_ids.shape[1] + max_new_tokens > self.context_length:
            raise ValueError("当前 KV cache 版本暂不支持超过 context_length 的生成")

        generated = prompt_ids.clone()

        kv_cache = KVCache(len(self.layers))

        # prefill
        logits, kv_cache = self.forward(
            prompt_ids,
            kv_cache=kv_cache,
            use_cache=True,
            start_pos=0,
        )
        logits = logits[:, -1, :]

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

            logits, kv_cache = self.forward(
                next_token,
                kv_cache=kv_cache,
                use_cache=True,
                start_pos=cur_pos,
            )
            logits = logits[:, -1, :]

        return generated

    def _top_p_filter(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """内部工具函数：执行 Top-P 截断"""
        # 对词表分值进行降序排序
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        
        # 计算累积概率分布
        cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)
        
        # 创建掩码：我们要去掉累积概率超过 p 的 Token
        # 逻辑：保留最小的集合 V(p)，使其概率之和 >= p
        # 我们把所有超过 p 的位置标记为 True（需要移除）
        sorted_indices_to_remove = cumulative_probs > p
        
        # 关键修正：确保至少保留第一个词（最高概率词），
        # 并且我们要保留第一个“使概率超过 p”的那个词。
        # 做法是把标记位向右移动一格。
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # 将被移除的 Token 分数设为负无穷
        # 这里需要利用 scatter 将排序后的掩码映射回原始词表索引位置
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        return logits
