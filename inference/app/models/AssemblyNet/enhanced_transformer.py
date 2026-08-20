"""Pure Point Cloud Transformer for Assembly."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import PointCloudEncodingManager
from .layer import TransformerLayer


class FragmentContextAdapter(nn.Module):
    """Equal-fragment context path with an identity-preserving initialization."""

    def __init__(
            self,
            embed_dim: int,
            context_dim: int,
            num_heads: int,
            dropout_rate: float = 0.0,
    ):
        super().__init__()
        if context_dim % num_heads != 0:
            raise ValueError("fragment context_dim must be divisible by num_heads")
        self.input_norm = nn.LayerNorm(embed_dim)
        self.down = nn.Linear(embed_dim, context_dim)
        self.token_norm = nn.LayerNorm(context_dim)
        self.attention = nn.MultiheadAttention(
            context_dim,
            num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(context_dim)
        self.ffn = nn.Sequential(
            nn.Linear(context_dim, context_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(context_dim * 2, context_dim),
        )
        self.up = nn.Linear(context_dim, embed_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    @staticmethod
    def _pool_fragments(
            point_features: torch.Tensor,
            point_part_ids: torch.Tensor,
            num_parts: int,
    ) -> torch.Tensor:
        pooled = torch.zeros(
            num_parts,
            point_features.shape[-1],
            dtype=torch.float32,
            device=point_features.device,
        )
        pooled.index_add_(0, point_part_ids, point_features.float())
        counts = torch.bincount(
            point_part_ids, minlength=num_parts
        ).float().unsqueeze(-1).clamp_min(1.0)
        return pooled / counts

    def forward(
            self,
            point_features: torch.Tensor,
            point_part_ids: torch.Tensor,
            parts_per_case: torch.Tensor,
    ) -> torch.Tensor:
        num_parts = int(parts_per_case.sum().item())
        if num_parts == 0:
            return point_features

        fragment_tokens = self._pool_fragments(
            point_features,
            point_part_ids,
            num_parts,
        )
        fragment_tokens = self.down(self.input_norm(fragment_tokens))

        batch_size = int(parts_per_case.numel())
        max_parts = int(parts_per_case.max().item())
        padded = fragment_tokens.new_zeros(batch_size, max_parts, fragment_tokens.shape[-1])
        padding_mask = torch.ones(
            batch_size, max_parts, dtype=torch.bool, device=fragment_tokens.device
        )
        offset = 0
        for case_id, count_tensor in enumerate(parts_per_case):
            count = int(count_tensor.item())
            if count:
                padded[case_id, :count] = fragment_tokens[offset:offset + count]
                padding_mask[case_id, :count] = False
                offset += count

        normalized = self.token_norm(padded)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        padded = padded + attended
        padded = padded + self.ffn(self.ffn_norm(padded))

        contextual_tokens = []
        for case_id, count_tensor in enumerate(parts_per_case):
            count = int(count_tensor.item())
            if count:
                contextual_tokens.append(padded[case_id, :count])
        contextual_tokens = torch.cat(contextual_tokens, dim=0)
        point_updates = self.up(contextual_tokens)[point_part_ids]
        return point_features + point_updates.to(point_features.dtype)


class AssemblyTransformer(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_layers: int,
            num_heads: int,
            dropout_rate: float = 0.0,
             attn_dtype: torch.dtype = torch.float16,
             max_parts: int = 50,
             output_type: str = "coords",
             fragment_context_enabled: bool = False,
             fragment_context_start_layer: int = 8,
             fragment_context_dim: int = 192,
             fragment_context_heads: int = 4,
             point_reliability_enabled: bool = False,
     ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.output_type = output_type
        self.fragment_context_enabled = bool(fragment_context_enabled)
        self.fragment_context_start_layer = int(fragment_context_start_layer)
        self.point_reliability_enabled = bool(point_reliability_enabled)

        self.encoding_manager = PointCloudEncodingManager(embed_dim=embed_dim, max_parts=max_parts)

        self.transformer_layers = nn.ModuleList([
            TransformerLayer(
                dim=embed_dim, num_attention_heads=num_heads,
                attention_head_dim=embed_dim // num_heads,
                dropout=dropout_rate, attn_dtype=attn_dtype,
            ) for _ in range(num_layers)
        ])

        if self.fragment_context_enabled:
            if not 0 <= self.fragment_context_start_layer < num_layers:
                raise ValueError("fragment_context_start_layer must index a Transformer layer")
            self.fragment_context_adapters = nn.ModuleDict({
                str(layer_index): FragmentContextAdapter(
                    embed_dim=embed_dim,
                    context_dim=fragment_context_dim,
                    num_heads=fragment_context_heads,
                    dropout_rate=dropout_rate,
                )
                for layer_index in range(self.fragment_context_start_layer, num_layers)
            })
        else:
            self.fragment_context_adapters = nn.ModuleDict()

        if self.point_reliability_enabled:
            reliability_hidden = max(32, embed_dim // 3)
            self.point_reliability_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, reliability_hidden),
                nn.SiLU(),
                nn.Linear(reliability_hidden, 1),
            )
            nn.init.zeros_(self.point_reliability_head[-1].weight)
            nn.init.zeros_(self.point_reliability_head[-1].bias)
        else:
            self.point_reliability_head = None

        if self.output_type == "coords":
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, 3, bias=False)
            )
        elif self.output_type == "pose":
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, 7)
            )
        else:
            raise ValueError("output_type must be 'coords' or 'pose'")

    def _generate_point_part_ids(self, points_per_part):
        B, max_part = points_per_part.shape
        part_idx = torch.arange(max_part, device=points_per_part.device).unsqueeze(0).expand(B, max_part)
        valid_mask = points_per_part.flatten() > 0
        return torch.repeat_interleave(part_idx.flatten()[valid_mask], points_per_part.flatten()[valid_mask])

    def forward(self, input_coords, input_normals, points_per_part, return_features=False):
        point_part_ids = self._generate_point_part_ids(points_per_part)
        x = self.encoding_manager(input_coords, input_normals, point_part_ids)

        part_valids = points_per_part != 0
        self_attn_seqlen = points_per_part[part_valids]
        self_attn_cu_seqlens = nn.functional.pad(torch.cumsum(self_attn_seqlen, 0), (1, 0)).to(torch.int32)

        global_attn_seqlen = points_per_part.sum(dim=1)
        global_attn_cu_seqlens = nn.functional.pad(torch.cumsum(global_attn_seqlen, 0), (1, 0)).to(torch.int32)
        part_batch_ids = torch.repeat_interleave(torch.arange(len(self_attn_seqlen), device=x.device), self_attn_seqlen)

        parts_per_case = part_valids.sum(dim=1)
        for layer_index, layer in enumerate(self.transformer_layers):
            x = layer(
                hidden_states=x,
                intra_attn_cu_seqlens=self_attn_cu_seqlens,
                intra_attn_max_seqlen=self_attn_seqlen.max(),
                inter_attn_cu_seqlens=global_attn_cu_seqlens,
                inter_attn_max_seqlen=global_attn_seqlen.max(),
                batch=part_batch_ids
            )
            adapter_key = str(layer_index)
            if adapter_key in self.fragment_context_adapters:
                x = self.fragment_context_adapters[adapter_key](
                    x,
                    part_batch_ids,
                    parts_per_case,
                )

        if self.output_type == "coords":
            pred_coords = self.head(x.float())
            output = {"pred_coords": pred_coords, "part_batch_ids": part_batch_ids}
            if self.point_reliability_head is not None:
                output["point_reliability_logits"] = self.point_reliability_head(
                    x.float()
                ).squeeze(-1)
            if return_features:
                num_valid_parts = len(self_attn_seqlen)
                part_features = torch.zeros(
                    num_valid_parts, self.embed_dim, device=x.device, dtype=torch.float32
                )
                # FP16 accumulation over thousands of points can overflow even
                # when every point feature is finite.
                part_features.index_add_(0, part_batch_ids, x.float())
                counts = torch.bincount(
                    part_batch_ids, minlength=num_valid_parts
                ).float().unsqueeze(-1).clamp(min=1)
                output["part_features"] = part_features / counts
                output["point_features"] = x
            return output

        elif self.output_type == "pose":
            num_valid_parts = len(self_attn_seqlen)
            part_features = torch.zeros(num_valid_parts, self.embed_dim, device=x.device, dtype=x.dtype)
            part_features.index_add_(0, part_batch_ids, x)
            counts = torch.bincount(part_batch_ids, minlength=num_valid_parts).unsqueeze(-1).clamp(min=1)
            part_features = part_features / counts

            pose_pred = self.head(part_features.float())
            trans = torch.clamp(pose_pred[:, :3], -10.0, 10.0)
            quat = F.normalize(pose_pred[:, 3:], p=2, dim=-1)

            return {"trans": trans, "quat": quat, "part_batch_ids": part_batch_ids}
