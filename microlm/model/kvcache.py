import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleKVCache:
    def __init__(self):
        self.k = None  # [B, H, T, D]
        self.v = None  # [B, H, T, D]

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor):
        # k_new, v_new: [B, H, T_new, D]
        if self.k is None:
            self.k = k_new
            self.v = v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=-2)
            self.v = torch.cat([self.v, v_new], dim=-2)
        return self.k, self.v

    def reset(self):
        self.k = None
        self.v = None


class MiniAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] -> [B, H, T, Hd]
        B, T, D = x.shape
        x = x.view(B, T, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, T, Hd] -> [B, T, D]
        B, H, T, Hd = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(B, T, H * Hd)

    def forward(self, x: torch.Tensor, cache: SimpleKVCache | None = None):
        """
        x:
          prefill 时可以是 [B, T, D]
          decode 时通常是 [B, 1, D]
        """
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if cache is not None:
            k_all, v_all = cache.update(k, v)
        else:
            k_all, v_all = k, v

        # 用 PyTorch 官方 SDPA 做注意力
        out = F.scaled_dot_product_attention(
            q, k_all, v_all,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True
        )

        out = self._merge_heads(out)
        return self.o_proj(out)


def demo():
    torch.manual_seed(0)

    B, T, D, H = 2, 5, 32, 4
    attn = MiniAttention(d_model=D, num_heads=H)
    cache = SimpleKVCache()

    # 1) Prefill：一次输入整段
    x_prefill = torch.randn(B, T, D)
    y_prefill = attn(x_prefill, cache=cache)
    print("prefill output:", y_prefill.shape)
    print("cache k shape after prefill:", cache.k.shape)  # [B, H, T, Hd]

    # 2) Decode：每次只输入一个新 token
    x_next = torch.randn(B, 1, D)
    y_next = attn(x_next, cache=cache)
    print("decode output:", y_next.shape)
    print("cache k shape after one decode step:", cache.k.shape)  # [B, H, T+1, Hd]

if __name__ == "__main__":
    demo()