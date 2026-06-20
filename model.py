"""
Uzbek GPT — a from-scratch, decoder-only transformer.

Modern GPT variant: RMSNorm (pre-norm) + Rotary Position Embeddings (RoPE)
+ SwiGLU feed-forward + multi-head causal self-attention (fused SDPA).

103M params at the default config (n_embd=768, n_layer=12, n_head=12).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


class RoPE(nn.Module):
    """Rotary position embeddings, applied to Q and K."""
    def __init__(self, head_size, max_len=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_size, 2).float() / head_size))
        angles = torch.arange(max_len).float()[:, None] * inv_freq[None, :]
        self.register_buffer("cos", torch.cos(angles).repeat_interleave(2, dim=-1))
        self.register_buffer("sin", torch.sin(angles).repeat_interleave(2, dim=-1))

    def forward(self, x):
        T = x.shape[-2]
        return (x * self.cos[:T]) + (rotate_half(x) * self.sin[:T])


class MultiHeadAttention(nn.Module):
    """Multi-head causal self-attention. All heads computed in one batched op."""
    def __init__(self, n_embd, num_heads):
        super().__init__()
        self.n_head = num_heads
        self.head_size = n_embd // num_heads
        self.key   = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.rope  = RoPE(self.head_size)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = self.rope(q)
        k = self.rope(k)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return out.transpose(1, 2).contiguous().view(B, T, C)


class RMSNorm(nn.Module):
    def __init__(self, n_embd, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_embd))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        hidden = int(n_embd * 8 / 3)
        self.w_gate = nn.Linear(n_embd, hidden, bias=False)
        self.w_up   = nn.Linear(n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, n_embd, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        self.sa  = MultiHeadAttention(n_embd, num_heads)
        self.ffn = SwiGLU(n_embd)
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, num_heads, n_layers):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, n_embd=n_embd,
                           block_size=block_size, num_heads=num_heads, n_layers=n_layers)
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.blocks    = nn.Sequential(*[Block(n_embd, num_heads) for _ in range(n_layers)])
        self.ln_f      = RMSNorm(n_embd)
        self.head      = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        x = self.token_emb(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)
