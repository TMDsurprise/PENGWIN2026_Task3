"""Point embedding utilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadRMSNorm(nn.Module):
    """Multi-head RMS normalization layer."""
    def __init__(self, dim: int, heads: int = 1):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head RMS normalization."""
        return F.normalize(x, dim=-1) * self.gamma * self.scale


class PointCloudEmbedding:
    """Sinusoidal positional embedding for point clouds."""
    def __init__(self, include_input=True, input_dims=3, max_freq_log2=9, num_freqs=10, log_sampling=True, periodic_fns=None):
        self.include_input = include_input
        self.input_dims = input_dims
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.periodic_fns = periodic_fns or [torch.sin, torch.cos]
        self._create_embedding_fn()

    def _create_embedding_fn(self):
        embed_fns = []
        d = self.input_dims
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d
        if self.log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, self.max_freq_log2, steps=self.num_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**self.max_freq_log2, steps=self.num_freqs)
        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d
        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        """Embed input tensor using sinusoidal encoding."""
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class PointCloudEncodingManager(nn.Module):
    """Encoding Manager for Point Cloud Assembly."""
    def __init__(self, embed_dim: int, multires: int = 10, max_parts: int = 50):
        super().__init__()
        self.embed_dim = embed_dim

        self.coord_embedding = PointCloudEmbedding(input_dims=3, max_freq_log2=multires - 1, num_freqs=multires)
        self.normal_embedding = PointCloudEmbedding(input_dims=3, max_freq_log2=multires - 1, num_freqs=multires)

        self.part_embed_dim = 16
        self.part_embedding = nn.Embedding(max_parts, self.part_embed_dim)
        nn.init.trunc_normal_(self.part_embedding.weight, std=0.02)

        embed_input_dim = (
            self.coord_embedding.out_dim +
            self.normal_embedding.out_dim +
            self.part_embed_dim
        )

        self.emb_proj = nn.Sequential(
            nn.Linear(embed_input_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, coords, normals, point_part_ids):
        """Forward pass for point cloud encoding."""
        c_pos_emb = self.coord_embedding.embed(coords)
        normal_emb = self.normal_embedding.embed(normals)

        safe_part_ids = torch.clamp(point_part_ids.long(), max=self.part_embedding.num_embeddings - 1)
        part_emb = self.part_embedding(safe_part_ids)

        fused = torch.cat([c_pos_emb, normal_emb, part_emb], dim=-1)
        return self.emb_proj(fused)
