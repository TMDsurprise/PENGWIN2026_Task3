"""Pure Point Cloud Transformer for Assembly."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import PointCloudEncodingManager
from .layer import TransformerLayer


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
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.output_type = output_type

        self.encoding_manager = PointCloudEncodingManager(embed_dim=embed_dim, max_parts=max_parts)

        self.transformer_layers = nn.ModuleList([
            TransformerLayer(
                dim=embed_dim, num_attention_heads=num_heads,
                attention_head_dim=embed_dim // num_heads,
                dropout=dropout_rate, attn_dtype=attn_dtype,
            ) for _ in range(num_layers)
        ])

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

        for layer in self.transformer_layers:
            x = layer(
                hidden_states=x,
                intra_attn_cu_seqlens=self_attn_cu_seqlens,
                intra_attn_max_seqlen=self_attn_seqlen.max(),
                inter_attn_cu_seqlens=global_attn_cu_seqlens,
                inter_attn_max_seqlen=global_attn_seqlen.max(),
                batch=part_batch_ids
            )

        if self.output_type == "coords":
            pred_coords = self.head(x.float())
            output = {"pred_coords": pred_coords, "part_batch_ids": part_batch_ids}
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
