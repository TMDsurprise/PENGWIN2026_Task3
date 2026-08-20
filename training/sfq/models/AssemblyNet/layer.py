"""Transformer layer for Point Cloud Assembly."""

try:
    import flash_attn_interface as flash_attn
    flash_version = 3
except ImportError:
    try:
        import flash_attn
        flash_version = 2
    except ImportError:
        flash_attn = None
        flash_version = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from .embedding import MultiHeadRMSNorm


class _GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class _FeedForward(nn.Module):
    def __init__(self, dim: int, out_dim: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            _GEGLU(),
            nn.Linear(dim * mult, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerLayer(nn.Module):
    """Transformer layer with Intra- and Inter-Fragment Attention."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        qkv_proj_bias: bool = False,
        attn_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.dropout = dropout
        self.attn_dtype = attn_dtype

        self.intra_prenorm = nn.LayerNorm(dim)
        self.intra_qkv_proj = nn.Linear(dim, 3 * self.inner_dim, bias=qkv_proj_bias)
        self.intra_q_norm = MultiHeadRMSNorm(self.attention_head_dim, num_attention_heads)
        self.intra_k_norm = MultiHeadRMSNorm(self.attention_head_dim, num_attention_heads)
        self.intra_out_proj = nn.Linear(self.inner_dim, dim, bias=qkv_proj_bias)

        self.inter_prenorm = nn.LayerNorm(dim)
        self.inter_qkv_proj = nn.Linear(dim, 3 * self.inner_dim, bias=qkv_proj_bias)
        self.inter_q_norm = MultiHeadRMSNorm(self.attention_head_dim, num_attention_heads)
        self.inter_k_norm = MultiHeadRMSNorm(self.attention_head_dim, num_attention_heads)
        self.inter_out_proj = nn.Linear(self.inner_dim, dim, bias=qkv_proj_bias)

        self.ff_norm = nn.LayerNorm(dim)
        self.ff = _FeedForward(dim, dim, mult=4)

    def _run_flash_attn_varlen(self, q, k, v, cu_seqlens, max_seqlen, dtype):
        if flash_attn is None:
            return self._run_attention_standard(q, k, v, cu_seqlens, max_seqlen, dtype)

        if flash_version == 3:
            attn_output = flash_attn.flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                softmax_scale=None, causal=False,
            )[0]
        else:
            attn_output = flash_attn.flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                dropout_p=self.dropout if self.training else 0.0,
                softmax_scale=None, causal=False,
            )
        return attn_output.to(dtype)

    def _run_attention_standard(self, q, k, v, cu_seqlens, max_seqlen, dtype):
        """Fallback SDPA for when flash_attn is not installed. Pads to max_seqlen."""
        batch_size = len(cu_seqlens) - 1
        num_heads = q.shape[1]
        head_dim = q.shape[2]
        device = q.device

        # Pack into [B, max_seqlen, H, D]
        q_pad = q.new_zeros(batch_size, max_seqlen, num_heads, head_dim)
        k_pad = k.new_zeros(batch_size, max_seqlen, num_heads, head_dim)
        v_pad = v.new_zeros(batch_size, max_seqlen, num_heads, head_dim)
        # True = ignore (padding), False = attend
        key_mask = torch.ones(batch_size, max_seqlen, dtype=torch.bool, device=device)

        for i in range(batch_size):
            s, e = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
            seq_len = e - s
            q_pad[i, :seq_len] = q[s:e]
            k_pad[i, :seq_len] = k[s:e]
            v_pad[i, :seq_len] = v[s:e]
            key_mask[i, :seq_len] = False

        # [B, H, S, D]
        q_pad = q_pad.permute(0, 2, 1, 3)
        k_pad = k_pad.permute(0, 2, 1, 3)
        v_pad = v_pad.permute(0, 2, 1, 3)

        # [B, 1, 1, S] broadcast over heads and query positions
        attn_bias = key_mask.unsqueeze(1).unsqueeze(2).float() * -1e9

        out = F.scaled_dot_product_attention(q_pad, k_pad, v_pad, attn_mask=attn_bias)

        # [B, H, S, D] -> [B, S, H, D]
        out = out.permute(0, 2, 1, 3)

        # Unpack back to packed format
        result = q.new_zeros(q.shape[0], num_heads, head_dim)
        for i in range(batch_size):
            s, e = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
            seq_len = e - s
            result[s:e] = out[i, :seq_len]

        return result.to(dtype)

    def _run_attention(self, hidden_states, prenorm_fn, qkv_proj_fn, q_norm, k_norm, out_proj_fn, cu_seqlens, max_seqlen):
        n_points = hidden_states.shape[0]
        dtype = hidden_states.dtype

        x = prenorm_fn(hidden_states)
        qkv = qkv_proj_fn(x).reshape(n_points, 3, self.num_attention_heads, self.attention_head_dim)
        q, k, v = qkv.unbind(dim=1)

        q = q_norm(q).to(v.dtype)
        k = k_norm(k).to(v.dtype)

        attn_output = self._run_flash_attn_varlen(q, k, v, cu_seqlens, max_seqlen, dtype)
        return hidden_states + out_proj_fn(attn_output.flatten(1))

    def forward(self, hidden_states, intra_attn_cu_seqlens, intra_attn_max_seqlen,
                inter_attn_cu_seqlens, inter_attn_max_seqlen, batch=None):
        hidden_states = self._run_attention(
            hidden_states, self.intra_prenorm, self.intra_qkv_proj,
            self.intra_q_norm, self.intra_k_norm, self.intra_out_proj,
            intra_attn_cu_seqlens, intra_attn_max_seqlen,
        )
        hidden_states = self._run_attention(
            hidden_states, self.inter_prenorm, self.inter_qkv_proj,
            self.inter_q_norm, self.inter_k_norm, self.inter_out_proj,
            inter_attn_cu_seqlens, inter_attn_max_seqlen,
        )
        x = self.ff_norm(hidden_states)
        return hidden_states + self.ff(x)
