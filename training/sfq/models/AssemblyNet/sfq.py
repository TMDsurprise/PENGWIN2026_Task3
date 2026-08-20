"""Small-fragment query residual pose module built on a frozen e25 backbone."""

from __future__ import annotations

import math
import os
from functools import partial
from typing import Sequence

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import extract_poses_from_coords, quaternion_to_matrix


def _scatter_mean(values: torch.Tensor, ids: torch.Tensor, count: int) -> torch.Tensor:
    if values.ndim == 1:
        sums = values.new_zeros(count).index_add_(0, ids, values)
        counts = torch.bincount(ids, minlength=count).to(values.dtype).clamp_min(1.0)
        return sums / counts
    sums = values.new_zeros(count, values.shape[-1]).index_add_(0, ids, values)
    counts = torch.bincount(ids, minlength=count).to(values.dtype).clamp_min(1.0)
    return sums / counts[:, None]


def _rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    first = F.normalize(rotation_6d[..., :3], dim=-1, eps=1e-6)
    second_raw = rotation_6d[..., 3:]
    second = second_raw - (first * second_raw).sum(dim=-1, keepdim=True) * first
    second = F.normalize(second, dim=-1, eps=1e-6)
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def _limit_rotation(
    rotation: torch.Tensor, gate: torch.Tensor, max_angle_deg: float
) -> torch.Tensor:
    """Slerp from identity to a near-identity 6D rotation with an angle cap."""
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    qw = (0.5 * torch.sqrt((1.0 + trace).clamp_min(1e-6))).clamp_min(1e-4)
    xyz = torch.stack(
        (
            rotation[:, 2, 1] - rotation[:, 1, 2],
            rotation[:, 0, 2] - rotation[:, 2, 0],
            rotation[:, 1, 0] - rotation[:, 0, 1],
        ),
        dim=-1,
    ) / (4.0 * qw[:, None])
    quaternion = F.normalize(torch.cat((qw[:, None], xyz), dim=-1), dim=-1)
    quaternion = torch.where(quaternion[:, :1] < 0.0, -quaternion, quaternion)

    sin_half = torch.linalg.vector_norm(quaternion[:, 1:], dim=-1)
    half_angle = torch.atan2(sin_half, quaternion[:, 0].clamp_min(1e-6))
    max_half = math.radians(float(max_angle_deg)) * 0.5
    fraction = (max_half / half_angle.clamp_min(1e-6)).clamp_max(1.0) * gate
    scaled_half = half_angle * fraction
    ratio = torch.where(
        sin_half > 1e-6,
        torch.sin(scaled_half) / sin_half.clamp_min(1e-6),
        fraction,
    )
    scaled = torch.cat(
        (torch.cos(scaled_half)[:, None], quaternion[:, 1:] * ratio[:, None]), dim=-1
    )
    return quaternion_to_matrix(F.normalize(scaled, dim=-1))


def _rotation_error_deg(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    relative = prediction @ target.transpose(-1, -2)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(
        -1.0 + 1e-6, 1.0 - 1e-6
    )
    return torch.rad2deg(torch.acos(cosine))


def _matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert arbitrary proper rotation matrices to stable wxyz quaternions."""
    m00, m01, m02 = matrix[:, 0, 0], matrix[:, 0, 1], matrix[:, 0, 2]
    m10, m11, m12 = matrix[:, 1, 0], matrix[:, 1, 1], matrix[:, 1, 2]
    m20, m21, m22 = matrix[:, 2, 0], matrix[:, 2, 1], matrix[:, 2, 2]
    q_abs = torch.sqrt(
        torch.stack(
            (
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ),
            dim=-1,
        ).clamp_min(0.0)
    )
    candidates = torch.stack(
        (
            torch.stack((q_abs[:, 0].square(), m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, q_abs[:, 1].square(), m10 + m01, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m10 + m01, q_abs[:, 2].square(), m12 + m21), dim=-1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[:, 3].square()), dim=-1),
        ),
        dim=1,
    )
    candidates = candidates / (2.0 * q_abs.clamp_min(0.1))[:, :, None]
    best = q_abs.argmax(dim=-1)
    quaternion = candidates[
        torch.arange(matrix.shape[0], device=matrix.device), best
    ]
    quaternion = F.normalize(quaternion, dim=-1)
    return torch.where(quaternion[:, :1] < 0.0, -quaternion, quaternion)


def _quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _weighted_kabsch_from_coords(
    source: torch.Tensor,
    target: torch.Tensor,
    part_ids: torch.Tensor,
    logits: torch.Tensor,
    uniform_floor: float = 0.20,
):
    """Differentiable per-fragment Kabsch with a uniform safety floor."""
    source = source.detach().float()
    target = target.detach().float()
    part_ids = part_ids.detach()
    logits = logits.float().reshape(-1)
    num_parts = int(part_ids.max().item()) + 1
    rotations = source.new_zeros(num_parts, 3, 3)
    translations = source.new_zeros(num_parts, 3)
    point_weights = source.new_zeros(source.shape[0])
    identity = torch.eye(3, device=source.device, dtype=torch.float32)
    for part_index in range(num_parts):
        mask = part_ids == part_index
        src = source[mask]
        dst = target[mask]
        count = int(src.shape[0])
        if count < 3:
            rotations[part_index] = identity
            continue
        weights = torch.softmax(logits[mask], dim=0)
        weights = (1.0 - uniform_floor) * weights + uniform_floor / count
        point_weights[mask] = weights
        src_mean = (weights[:, None] * src).sum(dim=0)
        dst_mean = (weights[:, None] * dst).sum(dim=0)
        src_centered = src - src_mean
        dst_centered = dst - dst_mean
        covariance = (weights[:, None] * src_centered).T @ dst_centered
        u, _, vh = torch.linalg.svd(covariance)
        determinant = torch.det(vh.transpose(-1, -2) @ u.transpose(-1, -2))
        correction = torch.diag(
            torch.stack(
                (
                    determinant.new_tensor(1.0),
                    determinant.new_tensor(1.0),
                    torch.where(determinant < 0.0, -determinant.new_tensor(1.0), determinant.new_tensor(1.0)),
                )
            )
        )
        rotation = vh.transpose(-1, -2) @ correction @ u.transpose(-1, -2)
        translation = dst_mean - rotation @ src_mean
        rotations[part_index] = rotation
        translations[part_index] = translation
    return _matrix_to_quaternion(rotations), translations, point_weights


class SmallFragmentQueryModule(L.LightningModule):
    """Correct the frozen e25 rigid update with fragment-level SE(3) residuals.

    Variants are cumulative only when explicitly requested: ``r0`` uses a pooled
    fragment token, ``f`` adds equal-weight fragment self-attention, ``x`` adds
    spatial cross-attention, and ``fx`` uses both. The residual head is exactly
    zero-initialized, so the initial prediction is the original e25 prediction.
    """

    def __init__(
        self,
        transformer_model: nn.Module,
        optimizer: "partial[torch.optim.Optimizer]",
        checkpoint: str,
        variant: str = "r0",
        hidden_dim: int = 384,
        max_parts: int = 50,
        num_heads: int = 8,
        context_layers: int = 2,
        patches_per_fragment: int = 16,
        local_points: int = 128,
        local_radii: Sequence[float] = (0.05, 0.10, 0.20),
        qsmall_threshold_mm: float = 91.578,
        qsmall_gate_center_mm: float = 105.0,
        qsmall_gate_width_mm: float = 18.0,
        qsmall_loss_boost: float = 2.0,
        max_rotation_deg: float = 30.0,
        max_translation_mm: float = 30.0,
        rotation_weight: float = 2.0,
        translation_weight: float = 0.5,
        paired_weight: float = 0.5,
        preserve_weight: float = 0.2,
        residual_weight: float = 0.02,
        correction_confidence_weight: float = 0.2,
        point_confidence_weight: float = 0.1,
        confidence_logit_bias: float = 4.0,
        kabsch_uniform_floor: float = 0.20,
        train_new_heads_only: bool = False,
    ):
        super().__init__()
        variant = str(variant).lower()
        if variant not in {
            "r0", "f", "x", "fx", "l", "xl", "fxg", "fxh", "fxgh"
        }:
            raise ValueError(f"Unsupported SFQ variant: {variant!r}")
        self.save_hyperparameters(ignore=("transformer_model", "optimizer"))
        self.transformer_model = transformer_model
        self.optimizer_factory = optimizer
        self.variant = variant
        self.hidden_dim = int(hidden_dim)
        self.max_parts = int(max_parts)
        self.patches_per_fragment = int(patches_per_fragment)
        self.local_points = int(local_points)
        self.local_radii = tuple(float(radius) for radius in local_radii)
        if len(self.local_radii) != 3:
            raise ValueError("SFQ local encoder expects exactly three normalized radii")
        self.qsmall_threshold_mm = float(qsmall_threshold_mm)
        self.qsmall_gate_center_mm = float(qsmall_gate_center_mm)
        self.qsmall_gate_width_mm = float(qsmall_gate_width_mm)
        self.qsmall_loss_boost = float(qsmall_loss_boost)
        self.max_rotation_deg = float(max_rotation_deg)
        self.max_translation_mm = float(max_translation_mm)
        self.rotation_weight = float(rotation_weight)
        self.translation_weight = float(translation_weight)
        self.paired_weight = float(paired_weight)
        self.preserve_weight = float(preserve_weight)
        self.residual_weight = float(residual_weight)
        self.correction_confidence_weight = float(correction_confidence_weight)
        self.point_confidence_weight = float(point_confidence_weight)
        self.confidence_logit_bias = float(confidence_logit_bias)
        self.kabsch_uniform_floor = float(kabsch_uniform_floor)
        self.train_new_heads_only = bool(train_new_heads_only)

        self._load_coordinate_checkpoint(checkpoint)
        for parameter in self.transformer_model.parameters():
            parameter.requires_grad_(False)
        self.transformer_model.eval()

        backbone_dim = int(self.transformer_model.embed_dim)
        self.backbone_projection = nn.Sequential(
            nn.LayerNorm(backbone_dim), nn.Linear(backbone_dim, hidden_dim)
        )
        # centroid(3), diameter/count(2), covariance(3), observability/qsmall(2),
        # base rotation 6D(6), base translation(3), and bone one-hot(3).
        self.metadata_projection = nn.Sequential(
            nn.Linear(22, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.bone_embedding = nn.Embedding(3, hidden_dim)
        self.fragment_slot_embedding = nn.Embedding(max_parts, hidden_dim)

        if "f" in variant:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.fragment_context = nn.TransformerEncoder(
                layer, num_layers=context_layers, norm=nn.LayerNorm(hidden_dim)
            )
        else:
            self.fragment_context = None

        if "x" in variant:
            self.patch_feature_projection = nn.Sequential(
                nn.LayerNorm(backbone_dim), nn.Linear(backbone_dim, hidden_dim)
            )
            self.patch_geometry_projection = nn.Sequential(
                nn.Linear(6, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
            )
            self.cross_attention = nn.MultiheadAttention(
                hidden_dim, num_heads=num_heads, dropout=0.0, batch_first=True
            )
            self.cross_norm = nn.LayerNorm(hidden_dim)
        else:
            self.patch_feature_projection = None
            self.patch_geometry_projection = None
            self.cross_attention = None
            self.cross_norm = None

        if "l" in variant:
            # Three radii x (count, distance mean/std, normal dot, normal-direction)
            # plus covariance eigenvalue ratios, planarity and linearity.
            self.local_point_encoder = nn.Sequential(
                nn.Linear(20, 128),
                nn.SiLU(),
                nn.LayerNorm(128),
                nn.Linear(128, 128),
                nn.SiLU(),
            )
            self.local_attention = nn.Linear(128, 1)
            self.local_projection = nn.Sequential(
                nn.LayerNorm(128), nn.Linear(128, hidden_dim)
            )
        else:
            self.local_point_encoder = None
            self.local_attention = None
            self.local_projection = None

        if "h" in variant:
            self.point_confidence_head = nn.Sequential(
                nn.LayerNorm(backbone_dim),
                nn.Linear(backbone_dim, 128),
                nn.SiLU(),
                nn.Linear(128, 1),
            )
            nn.init.zeros_(self.point_confidence_head[-1].weight)
            nn.init.zeros_(self.point_confidence_head[-1].bias)
        else:
            self.point_confidence_head = None

        if "g" in variant:
            self.correction_confidence_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 128),
                nn.SiLU(),
                nn.Linear(128, 1),
            )
            nn.init.zeros_(self.correction_confidence_head[-1].weight)
            nn.init.zeros_(self.correction_confidence_head[-1].bias)
        else:
            self.correction_confidence_head = None

        self.residual_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 9),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

        if self.train_new_heads_only:
            if self.correction_confidence_head is None and self.point_confidence_head is None:
                raise ValueError("train_new_heads_only requires a G or H variant")
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            for head in (
                self.correction_confidence_head,
                self.point_confidence_head,
            ):
                if head is not None:
                    for parameter in head.parameters():
                        parameter.requires_grad_(True)

    def _load_coordinate_checkpoint(self, checkpoint_path: str) -> None:
        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"e25 coordinate checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        prefix = "transformer_model."
        stripped = {
            key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)
        }
        missing, unexpected = self.transformer_model.load_state_dict(stripped, strict=False)
        loaded = len(stripped) - len(unexpected)
        if loaded <= 0:
            raise RuntimeError(f"No e25 transformer tensors loaded from {checkpoint_path}")
        if os.environ.get("LOCAL_RANK", "0") == "0":
            print(
                f"[SFQ] e25 loaded={loaded} missing={len(missing)} "
                f"unexpected={len(unexpected)} from {checkpoint_path}"
            )

    def train(self, mode: bool = True):
        super().train(mode)
        self.transformer_model.eval()
        return self

    @staticmethod
    def _layout(points_per_part: torch.Tensor):
        valid = points_per_part > 0
        part_to_case = torch.nonzero(valid, as_tuple=False)[:, 0].long()
        counts = points_per_part[valid].long()
        part_ids = torch.repeat_interleave(
            torch.arange(counts.numel(), device=counts.device), counts
        )
        local_slots = torch.zeros_like(part_to_case)
        anchor = torch.zeros_like(part_to_case, dtype=torch.bool)
        for case_idx in range(points_per_part.shape[0]):
            local = torch.nonzero(part_to_case == case_idx, as_tuple=False).flatten()
            if local.numel():
                local_slots[local] = torch.arange(local.numel(), device=counts.device)
                anchor[local[0]] = True
        return valid, part_to_case, counts, part_ids, local_slots, anchor

    @staticmethod
    def _part_covariance_features(
        points: torch.Tensor, part_ids: torch.Tensor, num_parts: int
    ) -> torch.Tensor:
        features = points.new_zeros(num_parts, 3)
        for part_idx in range(num_parts):
            local = points[part_ids == part_idx]
            centered = local - local.mean(dim=0, keepdim=True)
            covariance = centered.T @ centered / max(1, local.shape[0])
            eigenvalues = torch.linalg.eigvalsh(covariance.float()).clamp_min(1e-8)
            features[part_idx] = (eigenvalues / eigenvalues.sum()).to(points.dtype)
        return features

    def _fragment_attention(
        self,
        token: torch.Tensor,
        part_to_case: torch.Tensor,
        local_slots: torch.Tensor,
        batch_size: int,
        max_parts: int,
    ) -> torch.Tensor:
        padded = token.new_zeros(batch_size, max_parts, token.shape[-1])
        padding_mask = torch.ones(batch_size, max_parts, dtype=torch.bool, device=token.device)
        padded[part_to_case, local_slots] = token
        padding_mask[part_to_case, local_slots] = False
        encoded = self.fragment_context(padded, src_key_padding_mask=padding_mask)
        return encoded[part_to_case, local_slots]

    @staticmethod
    def _fps_indices(points: torch.Tensor, count: int) -> torch.Tensor:
        if points.shape[0] <= count:
            base = torch.arange(points.shape[0], device=points.device)
            if points.shape[0] == count:
                return base
            repeats = base[torch.arange(count - points.shape[0], device=points.device) % points.shape[0]]
            return torch.cat((base, repeats), dim=0)
        selected = torch.empty(count, dtype=torch.long, device=points.device)
        centre = points.mean(dim=0, keepdim=True)
        distance = (points - centre).square().sum(dim=-1)
        selected[0] = distance.argmax()
        minimum = (points - points[selected[0]]).square().sum(dim=-1)
        for index in range(1, count):
            selected[index] = minimum.argmax()
            candidate = (points - points[selected[index]]).square().sum(dim=-1)
            minimum = torch.minimum(minimum, candidate)
        return selected

    def _spatial_cross_attention(
        self,
        query: torch.Tensor,
        point_features: torch.Tensor,
        base_points: torch.Tensor,
        base_normals: torch.Tensor,
        part_ids: torch.Tensor,
        part_to_case: torch.Tensor,
        local_slots: torch.Tensor,
        bone: torch.Tensor,
    ) -> torch.Tensor:
        memories = []
        for part_idx in range(query.shape[0]):
            point_index = torch.nonzero(part_ids == part_idx, as_tuple=False).flatten()
            chosen = point_index[self._fps_indices(base_points[point_index], self.patches_per_fragment)]
            patch = self.patch_feature_projection(point_features[chosen].float())
            patch = patch + self.patch_geometry_projection(
                torch.cat((base_points[chosen], base_normals[chosen]), dim=-1)
            )
            patch = patch + self.bone_embedding(bone[part_idx])
            patch = patch + self.fragment_slot_embedding(local_slots[part_idx])
            memories.append(patch)

        output = query.clone()
        for case_idx in range(int(part_to_case.max().item()) + 1):
            local = torch.nonzero(part_to_case == case_idx, as_tuple=False).flatten()
            if local.numel() == 0:
                continue
            memory = torch.cat([memories[int(index)] for index in local], dim=0)[None]
            attended, _ = self.cross_attention(
                query[local][None], memory, memory, need_weights=False
            )
            output[local] = self.cross_norm(query[local] + attended.squeeze(0))
        return output

    def _local_invariant_tokens(
        self,
        points: torch.Tensor,
        normals: torch.Tensor,
        part_ids: torch.Tensor,
        num_parts: int,
    ) -> torch.Tensor:
        descriptors = []
        with torch.autocast(device_type=points.device.type, enabled=False):
            points = points.detach().float()
            normals = F.normalize(normals.detach().float(), dim=-1, eps=1e-6)
            for part_idx in range(num_parts):
                point_index = torch.nonzero(part_ids == part_idx, as_tuple=False).flatten()
                chosen = point_index[
                    self._fps_indices(points[point_index], min(self.local_points, point_index.numel()))
                ]
                local_points = points[chosen]
                local_normals = normals[chosen]
                diameter = torch.linalg.vector_norm(
                    local_points.max(dim=0).values - local_points.min(dim=0).values
                ).clamp_min(1e-4)
                offset = local_points[None, :, :] - local_points[:, None, :]
                distance = torch.linalg.vector_norm(offset, dim=-1) / diameter
                direction = offset / (distance[..., None] * diameter).clamp_min(1e-6)
                normal_dot = torch.abs(local_normals @ local_normals.T)
                normal_direction = torch.abs(
                    (local_normals[:, None, :] * direction).sum(dim=-1)
                )
                not_self = ~torch.eye(
                    local_points.shape[0], dtype=torch.bool, device=points.device
                )

                scale_features = []
                largest_mask = None
                for radius in self.local_radii:
                    mask = (distance <= radius) & not_self
                    largest_mask = mask
                    weight = mask.float()
                    count = weight.sum(dim=1).clamp_min(1.0)
                    mean_distance = (weight * distance).sum(dim=1) / count
                    variance = (
                        weight * (distance - mean_distance[:, None]).square()
                    ).sum(dim=1) / count
                    scale_features.extend(
                        (
                            (count / max(1, local_points.shape[0] - 1))[:, None],
                            mean_distance[:, None],
                            variance.sqrt()[:, None],
                            ((weight * normal_dot).sum(dim=1) / count)[:, None],
                            ((weight * normal_direction).sum(dim=1) / count)[:, None],
                        )
                    )

                weight = largest_mask.float()
                weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(1.0)
                neighbour_mean = torch.einsum("ij,ijk->ik", weight, offset)
                centered = offset - neighbour_mean[:, None, :]
                covariance = torch.einsum(
                    "ij,ijk,ijl->ikl", weight, centered, centered
                )
                eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(1e-8)
                ratios = eigenvalues / eigenvalues.sum(dim=-1, keepdim=True)
                planarity = (
                    (eigenvalues[:, 1] - eigenvalues[:, 0])
                    / eigenvalues[:, 2].clamp_min(1e-8)
                )[:, None]
                linearity = (
                    (eigenvalues[:, 2] - eigenvalues[:, 1])
                    / eigenvalues[:, 2].clamp_min(1e-8)
                )[:, None]
                descriptor = torch.cat(
                    (*scale_features, ratios, planarity, linearity), dim=-1
                )
                if descriptor.shape[-1] != 20:
                    raise RuntimeError(
                        f"Expected 20 invariant local features, got {descriptor.shape[-1]}"
                    )
                descriptors.append(descriptor)

        tokens = []
        for descriptor in descriptors:
            encoded = self.local_point_encoder(descriptor)
            attention = torch.softmax(self.local_attention(encoded).squeeze(-1), dim=0)
            tokens.append((attention[:, None] * encoded).sum(dim=0))
        return torch.stack(tokens, dim=0)

    def forward(self, data_dict: dict):
        points_per_part = data_dict["points_per_part"]
        source = data_dict["pointclouds"].float()
        normals = data_dict["pointclouds_normals"].float()
        valid, part_to_case, counts, part_ids, local_slots, anchor = self._layout(
            points_per_part
        )
        num_parts = int(counts.numel())

        with torch.no_grad():
            base = self.transformer_model(
                input_coords=source,
                input_normals=normals,
                points_per_part=points_per_part,
                return_features=True,
            )
            base_points = base["pred_coords"].float()
            # Candidate generation in the verified e25 submission uses this exact
            # SVD Kabsch path. Keeping it here makes a zero residual reproduce the
            # original four-candidate pool, not merely the coordinate prediction.
            model_base_quaternion, model_base_translation = extract_poses_from_coords(
                source, base_points, part_ids
            )
            override_quaternion = data_dict.get("base_quaternion_override")
            override_translation = data_dict.get("base_translation_override")
            if (override_quaternion is None) != (override_translation is None):
                raise RuntimeError("Base pose override requires quaternion and translation")
            if override_quaternion is not None:
                override_quaternion = override_quaternion.float()
                override_translation = override_translation.float()
                if override_quaternion.shape != model_base_quaternion.shape:
                    raise RuntimeError(
                        "Base quaternion override shape mismatch: "
                        f"{tuple(override_quaternion.shape)} vs "
                        f"{tuple(model_base_quaternion.shape)}"
                    )
                if override_translation.shape != model_base_translation.shape:
                    raise RuntimeError(
                        "Base translation override shape mismatch: "
                        f"{tuple(override_translation.shape)} vs "
                        f"{tuple(model_base_translation.shape)}"
                    )
                reference_base_quaternion = F.normalize(override_quaternion, dim=-1)
                reference_base_translation = override_translation
            else:
                reference_base_quaternion = model_base_quaternion
                reference_base_translation = model_base_translation
            reference_base_rotation = quaternion_to_matrix(reference_base_quaternion)

        point_confidence_logits = None
        point_weights = None
        if self.point_confidence_head is not None:
            point_confidence_logits = self.point_confidence_head(
                base["point_features"].float()
            ).squeeze(-1)
            with torch.autocast(device_type=source.device.type, enabled=False):
                weighted_quaternion, weighted_translation, point_weights = (
                    _weighted_kabsch_from_coords(
                        source,
                        base_points,
                        part_ids,
                        point_confidence_logits,
                        uniform_floor=self.kabsch_uniform_floor,
                    )
                )
                weighted_rotation = quaternion_to_matrix(weighted_quaternion)
                model_base_rotation = quaternion_to_matrix(model_base_quaternion)
                correction_rotation = weighted_rotation @ model_base_rotation.transpose(-1, -2)
                correction_translation = weighted_translation - torch.bmm(
                    correction_rotation, model_base_translation[:, :, None]
                ).squeeze(-1)
                # A zero-initialized H head must be a bit-exact parent control.
                # The confidence KL still trains the point logits on this first
                # step; the weighted pose path opens as soon as they move away
                # from zero.
                if float(point_confidence_logits.detach().abs().max()) <= 1e-8:
                    base_quaternion = reference_base_quaternion
                    base_translation = reference_base_translation
                    base_rotation = reference_base_rotation
                else:
                    base_rotation = correction_rotation @ reference_base_rotation
                    base_translation = torch.bmm(
                        correction_rotation, reference_base_translation[:, :, None]
                    ).squeeze(-1) + correction_translation
                    base_quaternion = _matrix_to_quaternion(base_rotation)
        else:
            base_quaternion = reference_base_quaternion
            base_translation = reference_base_translation
            base_rotation = reference_base_rotation

        bone = data_dict["bonetype"][valid].long().clamp(0, 2)
        diameter = data_dict["fragment_diameter_mm"][valid].float()
        scales = data_dict["norm_scale"].float().reshape(-1)[part_to_case].clamp_min(1e-6)
        base_centroid = _scatter_mean(base_points, part_ids, num_parts)
        covariance = self._part_covariance_features(source, part_ids, num_parts)
        observability = (
            (1.0 / covariance.square().sum(dim=-1).clamp_min(1e-8) - 1.0) * 0.5
        ).clamp(0.0, 1.0)
        pose_valid = counts >= 3
        qsmall_strength = torch.sigmoid(
            (self.qsmall_gate_center_mm - diameter) / self.qsmall_gate_width_mm
        )
        rotation_6d = torch.cat((base_rotation[:, :, 0], base_rotation[:, :, 1]), dim=-1)
        metadata = torch.cat(
            (
                base_centroid,
                (diameter / 200.0)[:, None],
                (torch.log1p(counts.float()) / 10.0)[:, None],
                covariance,
                observability[:, None],
                qsmall_strength[:, None],
                rotation_6d,
                base_translation,
                F.one_hot(bone, num_classes=3).float(),
            ),
            dim=-1,
        )
        if metadata.shape[-1] != 22:
            raise RuntimeError(f"Expected 22 SFQ metadata values, got {metadata.shape[-1]}")

        token = self.backbone_projection(base["part_features"].float())
        token = token + self.metadata_projection(metadata)
        token = token + self.bone_embedding(bone)
        token = token + self.fragment_slot_embedding(local_slots.clamp_max(self.max_parts - 1))

        if self.local_point_encoder is not None:
            local_token = self._local_invariant_tokens(
                source, normals, part_ids, num_parts
            )
            token = token + self.local_projection(local_token)

        if self.fragment_context is not None:
            token = self._fragment_attention(
                token,
                part_to_case,
                local_slots,
                int(points_per_part.shape[0]),
                int(points_per_part.shape[1]),
            )

        if self.cross_attention is not None:
            base_normals = torch.bmm(
                base_rotation[part_ids], normals.unsqueeze(-1)
            ).squeeze(-1)
            token = self._spatial_cross_attention(
                token,
                base["point_features"],
                base_points,
                base_normals,
                part_ids,
                part_to_case,
                local_slots,
                bone,
            )

        raw = self.residual_head(token).float()
        correction_confidence_logits = None
        if self.correction_confidence_head is not None:
            correction_confidence_logits = (
                self.correction_confidence_head(token).float().squeeze(-1)
                + self.confidence_logit_bias
            )
            confidence_normalizer = torch.sigmoid(
                raw.new_tensor(self.confidence_logit_bias)
            )
            correction_confidence = (
                torch.sigmoid(correction_confidence_logits) / confidence_normalizer
            ).clamp(max=1.0)
        else:
            correction_confidence = raw.new_ones(raw.shape[0])
        # Pose composition stays in FP32 even under Lightning mixed precision.
        # Otherwise a zero residual changes e25 coordinates by roughly 1e-4.
        with torch.autocast(device_type=source.device.type, enabled=False):
            size_gate = 0.25 + 0.75 * qsmall_strength.float()
            observability_gate = 0.5 + 0.5 * observability.float().sqrt().clamp(0.0, 1.0)
            structural_gate = size_gate * observability_gate * pose_valid.float()
            structural_gate = torch.where(
                anchor, torch.zeros_like(structural_gate), structural_gate
            )
            gate = structural_gate * correction_confidence.float()

            identity_6d = raw.new_tensor((1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
            raw_rotation = _rotation_6d_to_matrix(
                identity_6d[None] + 0.5 * torch.tanh(raw[:, :6])
            )
            proposal_residual_rotation = _limit_rotation(
                raw_rotation, structural_gate, self.max_rotation_deg
            )
            proposal_residual_translation_mm = (
                torch.tanh(raw[:, 6:])
                * self.max_translation_mm
                * structural_gate[:, None]
            )
            residual_rotation = _limit_rotation(raw_rotation, gate, self.max_rotation_deg)
            residual_translation_mm = (
                torch.tanh(raw[:, 6:]) * self.max_translation_mm * gate[:, None]
            )
            residual_translation = residual_translation_mm / scales.float()[:, None]
            proposal_residual_translation = (
                proposal_residual_translation_mm / scales.float()[:, None]
            )

            identity = torch.eye(3, device=source.device, dtype=torch.float32)[None]
            centered = base_points.float() - base_centroid.float()[part_ids]

            def apply_residual(residual_rot, residual_trans):
                rotation_delta = residual_rot - identity
                points = (
                    base_points.float()
                    + torch.bmm(
                        rotation_delta[part_ids], centered.unsqueeze(-1)
                    ).squeeze(-1)
                    + residual_trans[part_ids]
                )
                rotation = residual_rot @ base_rotation.float()
                translation = (
                    base_translation.float()
                    + torch.bmm(
                        rotation_delta,
                        (base_translation.float() - base_centroid.float()).unsqueeze(-1),
                    ).squeeze(-1)
                    + residual_trans
                )
                return points, rotation, translation

            proposal_points, proposal_rotation, proposal_translation = apply_residual(
                proposal_residual_rotation, proposal_residual_translation
            )
            final_points, final_rotation, final_translation = apply_residual(
                residual_rotation, residual_translation
            )
            residual_quaternion = _matrix_to_quaternion(residual_rotation)
            final_quaternion = _quaternion_multiply(
                residual_quaternion, base_quaternion.float()
            )
        return {
            "pred_coords": final_points,
            "base_pred_coords": base_points,
            "part_batch_ids": part_ids,
            "part_to_case": part_to_case,
            "local_slots": local_slots,
            "anchor_mask": anchor,
            "base_rotation": base_rotation,
            "base_translation": base_translation,
            "reference_base_rotation": reference_base_rotation,
            "reference_base_translation": reference_base_translation,
            "final_rotation": final_rotation,
            "final_translation": final_translation,
            "proposal_pred_coords": proposal_points,
            "proposal_rotation": proposal_rotation,
            "proposal_translation": proposal_translation,
            "quat": final_quaternion,
            "trans": final_translation,
            "residual_rotation": residual_rotation,
            "residual_translation_mm": residual_translation_mm,
            "qsmall_strength": qsmall_strength,
            "diameter_mm": diameter,
            "gate": gate,
            "structural_gate": structural_gate,
            "correction_confidence": correction_confidence,
            "correction_confidence_logits": correction_confidence_logits,
            "point_confidence_logits": point_confidence_logits,
            "point_weights": point_weights,
        }

    @staticmethod
    def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return (values * weights).sum() / weights.sum().clamp_min(1e-6)

    def _loss_and_metrics(self, output: dict, data_dict: dict):
        gt_rotation = quaternion_to_matrix(data_dict["quat_gt"][data_dict["points_per_part"] > 0].float())
        gt_translation = data_dict["trans_gt"][data_dict["points_per_part"] > 0].float()
        scales = data_dict["norm_scale"].float().reshape(-1)[output["part_to_case"]]
        active = ~output["anchor_mask"]
        qsmall = (output["diameter_mm"] <= self.qsmall_threshold_mm) & active
        qlarge = (output["diameter_mm"] > self.qsmall_threshold_mm) & active
        weights = (1.0 + self.qsmall_loss_boost * output["qsmall_strength"])[active]

        final_rotation = output["final_rotation"]
        final_translation = output["final_translation"]
        rotation_chordal = (final_rotation - gt_rotation).square().sum(dim=(-2, -1)) / 8.0
        translation_delta_mm = (final_translation - gt_translation) * scales[:, None]
        translation_per_part = F.smooth_l1_loss(
            translation_delta_mm / 30.0,
            torch.zeros_like(translation_delta_mm),
            beta=0.1,
            reduction="none",
        ).mean(dim=-1)

        point_scale = scales[output["part_batch_ids"]]
        point_delta_mm = (
            output["pred_coords"] - data_dict["pointclouds_gt"].float()
        ) * point_scale[:, None]
        point_loss = F.smooth_l1_loss(
            point_delta_mm / 30.0,
            torch.zeros_like(point_delta_mm),
            beta=0.1,
            reduction="none",
        ).mean(dim=-1)
        paired_per_part = _scatter_mean(
            point_loss, output["part_batch_ids"], final_rotation.shape[0]
        )

        with torch.no_grad():
            base_rot_deg = _rotation_error_deg(output["base_rotation"], gt_rotation)
            base_trans_mm = torch.linalg.vector_norm(
                (output["base_translation"] - gt_translation) * scales[:, None], dim=-1
            )
        final_rot_deg = _rotation_error_deg(final_rotation, gt_rotation)
        final_trans_mm = torch.linalg.vector_norm(translation_delta_mm, dim=-1)
        final_tre_mm = _scatter_mean(
            torch.linalg.vector_norm(point_delta_mm, dim=-1),
            output["part_batch_ids"],
            final_rotation.shape[0],
        )
        base_point_delta_mm = (
            output["base_pred_coords"] - data_dict["pointclouds_gt"].float()
        ) * point_scale[:, None]
        base_tre_mm = _scatter_mean(
            torch.linalg.vector_norm(base_point_delta_mm, dim=-1),
            output["part_batch_ids"],
            final_rotation.shape[0],
        )

        zero = final_rot_deg.sum() * 0.0
        loss_correction_confidence = zero
        confidence_accuracy = zero
        confidence_positive = zero
        if output["correction_confidence_logits"] is not None:
            with torch.no_grad():
                proposal_rot_deg = _rotation_error_deg(
                    output["proposal_rotation"], gt_rotation
                )
                proposal_trans_mm = torch.linalg.vector_norm(
                    (output["proposal_translation"] - gt_translation)
                    * scales[:, None],
                    dim=-1,
                )
                proposal_point_delta_mm = (
                    output["proposal_pred_coords"]
                    - data_dict["pointclouds_gt"].float()
                ) * point_scale[:, None]
                proposal_tre_mm = _scatter_mean(
                    torch.linalg.vector_norm(proposal_point_delta_mm, dim=-1),
                    output["part_batch_ids"],
                    final_rotation.shape[0],
                )
                base_quality = (
                    base_rot_deg / 30.0
                    + base_trans_mm / 30.0
                    + base_tre_mm / 30.0
                )
                proposal_quality = (
                    proposal_rot_deg / 30.0
                    + proposal_trans_mm / 30.0
                    + proposal_tre_mm / 30.0
                )
                confidence_target = (proposal_quality < base_quality).float()
            confidence_logits = output["correction_confidence_logits"]
            loss_correction_confidence = F.binary_cross_entropy_with_logits(
                confidence_logits[active], confidence_target[active]
            )
            confidence_accuracy = (
                (confidence_logits[active] >= 0.0)
                == (confidence_target[active] >= 0.5)
            ).float().mean()
            confidence_positive = confidence_target[active].mean()

        loss_point_confidence = zero
        if output["point_weights"] is not None:
            source_points = data_dict["pointclouds"].float()
            paired_error_mm = torch.linalg.vector_norm(base_point_delta_mm, dim=-1)
            point_losses = []
            for part_index in torch.nonzero(active, as_tuple=False).flatten().tolist():
                mask = output["part_batch_ids"] == part_index
                local_source = source_points[mask]
                centered_source = local_source - local_source.mean(dim=0, keepdim=True)
                radius = torch.linalg.vector_norm(centered_source, dim=-1)
                leverage = 0.25 + 0.75 * torch.sqrt(
                    radius / radius.max().clamp_min(1e-6)
                )
                reliability = torch.exp(-paired_error_mm[mask].detach() / 5.0)
                target_weights = (reliability * leverage).clamp_min(1e-8)
                target_weights = target_weights / target_weights.sum()
                predicted_weights = output["point_weights"][mask].clamp_min(1e-8)
                point_losses.append(
                    (
                        target_weights
                        * (target_weights.log() - predicted_weights.log())
                    ).sum()
                )
            if point_losses:
                loss_point_confidence = torch.stack(point_losses).mean()

        preserve = (
            F.relu(final_rot_deg - base_rot_deg.detach() - 0.25) / 30.0
            + F.relu(final_trans_mm - base_trans_mm.detach() - 0.25) / 30.0
            + F.relu(final_tre_mm - base_tre_mm.detach() - 0.10) / 30.0
        )
        residual_angle = _rotation_error_deg(
            output["residual_rotation"],
            torch.eye(3, device=self.device)[None].expand_as(output["residual_rotation"]),
        )
        residual_regularizer = (
            residual_angle / max(self.max_rotation_deg, 1e-6)
            + torch.linalg.vector_norm(output["residual_translation_mm"], dim=-1)
            / max(self.max_translation_mm, 1e-6)
        ) * (1.0 - output["qsmall_strength"])

        loss_rotation = self._weighted_mean(rotation_chordal[active], weights)
        loss_translation = self._weighted_mean(translation_per_part[active], weights)
        loss_paired = self._weighted_mean(paired_per_part[active], weights)
        loss_preserve = preserve[active].mean()
        loss_residual = residual_regularizer[active].mean()
        loss = (
            self.rotation_weight * loss_rotation
            + self.translation_weight * loss_translation
            + self.paired_weight * loss_paired
            + self.preserve_weight * loss_preserve
            + self.residual_weight * loss_residual
            + self.correction_confidence_weight * loss_correction_confidence
            + self.point_confidence_weight * loss_point_confidence
        )

        def selected_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            mask = mask & active
            return values[mask].mean() if mask.any() else values.sum() * 0.0

        return {
            "loss": loss,
            "loss_rotation": loss_rotation,
            "loss_translation": loss_translation,
            "loss_paired": loss_paired,
            "loss_preserve": loss_preserve,
            "loss_residual": loss_residual,
            "loss_correction_confidence": loss_correction_confidence,
            "loss_point_confidence": loss_point_confidence,
            "rot_deg": selected_mean(final_rot_deg, active),
            "trans_mm": selected_mean(final_trans_mm, active),
            "tre_mm": selected_mean(final_tre_mm, active),
            "base_rot_deg": selected_mean(base_rot_deg, active),
            "base_trans_mm": selected_mean(base_trans_mm, active),
            "base_tre_mm": selected_mean(base_tre_mm, active),
            "qsmall_rot_deg": selected_mean(final_rot_deg, qsmall),
            "qsmall_trans_mm": selected_mean(final_trans_mm, qsmall),
            "qsmall_tre_mm": selected_mean(final_tre_mm, qsmall),
            "qsmall_rot_gt30": selected_mean((final_rot_deg > 30.0).float(), qsmall),
            "qsmall_rot_gt60": selected_mean((final_rot_deg > 60.0).float(), qsmall),
            "base_qsmall_rot_deg": selected_mean(base_rot_deg, qsmall),
            "base_qsmall_trans_mm": selected_mean(base_trans_mm, qsmall),
            "base_qsmall_tre_mm": selected_mean(base_tre_mm, qsmall),
            "qlarge_rot_deg": selected_mean(final_rot_deg, qlarge),
            "qlarge_trans_mm": selected_mean(final_trans_mm, qlarge),
            "qlarge_tre_mm": selected_mean(final_tre_mm, qlarge),
            "gate": output["gate"][active].mean(),
            "correction_confidence": output["correction_confidence"][active].mean(),
            "confidence_accuracy": confidence_accuracy,
            "confidence_positive": confidence_positive,
        }

    def _shared_step(self, data_dict: dict, stage: str):
        output = self(data_dict)
        with torch.autocast(device_type=self.device.type, enabled=False):
            metrics = self._loss_and_metrics(output, data_dict)
        batch_size = int(data_dict["points_per_part"].shape[0])
        for name, value in metrics.items():
            self.log(
                f"{stage}/{name}",
                value,
                on_step=stage == "train" and name == "loss",
                on_epoch=True,
                prog_bar=name in {"loss", "qsmall_rot_deg", "qsmall_trans_mm"},
                sync_dist=True,
                batch_size=batch_size,
            )
        return metrics["loss"]

    def training_step(self, data_dict: dict, batch_idx: int):
        return self._shared_step(data_dict, "train")

    def validation_step(self, data_dict: dict, batch_idx: int):
        return self._shared_step(data_dict, "val")

    def configure_optimizers(self):
        trainable = [parameter for parameter in self.parameters() if parameter.requires_grad]
        return self.optimizer_factory(trainable)

    def on_fit_start(self):
        if self.trainer.is_global_zero:
            trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
            total = sum(parameter.numel() for parameter in self.parameters())
            print(
                f"[SFQ] variant={self.variant} trainable={trainable/1e6:.3f}M/"
                f"{total/1e6:.3f}M max_rot={self.max_rotation_deg:g}deg "
                f"max_trans={self.max_translation_mm:g}mm"
            )
