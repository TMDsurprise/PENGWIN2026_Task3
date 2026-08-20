"""Fragment-level candidate quality prediction for iterative pelvic reduction."""

from __future__ import annotations

import json
import math
import os
from functools import partial
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import extract_poses_from_coords, quaternion_to_matrix, rot_geodesic_deg


class FragmentConfidenceModule(nn.Module):
    """Freeze a coordinate model and learn which fragment pose candidate to keep.

    Candidate 0 is always the unmodified input. The other candidates are SE(3)
    fractions of the coordinate model's rigid Kabsch update. No inference-time
    prior is added to candidate 0; preserving it must be justified by geometry.
    """

    def __init__(
        self,
        transformer_model: nn.Module,
        optimizer: "partial[torch.optim.Optimizer]",
        checkpoint: str,
        candidate_alphas: Sequence[float] = (0.0, 0.5, 1.0, 1.25),
        axis_offsets_deg: Sequence[float] = (),
        ranking_checkpoint: Optional[str] = None,
        hidden_dim: int = 256,
        num_context_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_parts: int = 50,
        utility_scales: Sequence[float] = (2.53, 2.76, 2.67, 3.37),
        pair_margin: float = 0.05,
        ce_weight: float = 1.0,
        pair_weight: float = 0.5,
        regression_weight: float = 0.25,
        severe_weight: float = 0.2,
        chamfer_points: int = 64,
    ):
        super().__init__()
        self.transformer_model = transformer_model
        self.optimizer = optimizer
        self.candidate_alphas = tuple(float(x) for x in candidate_alphas)
        self.axis_offsets_deg = tuple(float(x) for x in axis_offsets_deg)
        self.num_candidates = len(self.candidate_alphas) + 3 * len(self.axis_offsets_deg)
        self.utility_scales = tuple(float(x) for x in utility_scales)
        self.pair_margin = float(pair_margin)
        self.ce_weight = float(ce_weight)
        self.pair_weight = float(pair_weight)
        self.regression_weight = float(regression_weight)
        self.severe_weight = float(severe_weight)
        self.chamfer_points = int(chamfer_points)
        self.max_parts = int(max_parts)

        if checkpoint:
            self._load_coordinate_checkpoint(checkpoint)
        for parameter in self.transformer_model.parameters():
            parameter.requires_grad_(False)
        self.transformer_model.eval()

        backbone_dim = int(self.transformer_model.embed_dim)
        self.backbone_projection = nn.Sequential(
            nn.LayerNorm(backbone_dim),
            nn.Linear(backbone_dim, hidden_dim),
        )
        self.geometry_projection = nn.Sequential(
            nn.Linear(27, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.bone_embedding = nn.Embedding(3, hidden_dim)
        self.candidate_embedding = nn.Embedding(self.num_candidates, hidden_dim)
        self.fragment_slot_embedding = nn.Embedding(max_parts, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_context_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.metric_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 4)
        )
        self.severe_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, 1)
        )
        if ranking_checkpoint:
            self._load_ranking_checkpoint(ranking_checkpoint)

    def train(self, mode: bool = True):
        super().train(mode)
        self.transformer_model.eval()
        return self

    def _load_coordinate_checkpoint(self, checkpoint_path: str):
        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Coordinate checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = checkpoint.get("state_dict", checkpoint)
        prefix = "transformer_model."
        stripped = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        missing, unexpected = self.transformer_model.load_state_dict(stripped, strict=False)
        loaded = len(stripped) - len(unexpected)
        if loaded <= 0:
            raise RuntimeError(f"No transformer weights loaded from {checkpoint_path}")
        if os.environ.get("LOCAL_RANK", "0") == "0":
            print(
                f"[FragmentConfidence] loaded={loaded} missing={len(missing)} "
                f"unexpected={len(unexpected)} from {checkpoint_path}"
            )

    def _load_ranking_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        source = checkpoint.get("state_dict", checkpoint)
        current = self.state_dict()
        compatible = {}
        for key, value in source.items():
            if key == "candidate_embedding.weight":
                rows = min(value.shape[0], current[key].shape[0])
                expanded = current[key].clone()
                expanded[:rows] = value[:rows]
                compatible[key] = expanded
            elif key in current and current[key].shape == value.shape:
                compatible[key] = value
        missing, unexpected = self.load_state_dict(compatible, strict=False)
        if os.environ.get("LOCAL_RANK", "0") == "0":
            print(
                f"[FragmentConfidence] ranking_init={len(compatible)} "
                f"missing={len(missing)} unexpected={len(unexpected)} from {checkpoint_path}"
            )

    @staticmethod
    def _scatter_mean(values: torch.Tensor, ids: torch.Tensor, count: int) -> torch.Tensor:
        if values.ndim == 1:
            sums = torch.zeros(count, dtype=values.dtype, device=values.device)
            sums.scatter_add_(0, ids, values)
            counts = torch.bincount(ids, minlength=count).to(values.dtype).clamp_min(1.0)
            return sums / counts
        sums = torch.zeros(count, values.shape[-1], dtype=values.dtype, device=values.device)
        sums.index_add_(0, ids, values)
        counts = torch.bincount(ids, minlength=count).to(values.dtype).clamp_min(1.0)
        return sums / counts.unsqueeze(-1)

    @staticmethod
    def _scaled_quaternion(quaternion: torch.Tensor, alpha: float) -> torch.Tensor:
        quaternion = F.normalize(quaternion.float(), dim=-1)
        quaternion = torch.where(quaternion[:, :1] < 0.0, -quaternion, quaternion)
        half_angle = torch.acos(quaternion[:, :1].clamp(-1.0 + 1e-7, 1.0 - 1e-7))
        sin_half = torch.sin(half_angle)
        fallback_axis = torch.zeros_like(quaternion[:, 1:])
        fallback_axis[:, 0] = 1.0
        axis = torch.where(
            sin_half > 1e-6,
            quaternion[:, 1:] / sin_half.clamp_min(1e-6),
            fallback_axis,
        )
        scaled_half = half_angle * float(alpha)
        result = torch.cat([torch.cos(scaled_half), axis * torch.sin(scaled_half)], dim=-1)
        return F.normalize(result, dim=-1)

    @staticmethod
    def _apply_part_pose(
        points: torch.Tensor,
        quaternion: torch.Tensor,
        translation: torch.Tensor,
        part_ids: torch.Tensor,
    ) -> torch.Tensor:
        rotation = quaternion_to_matrix(quaternion)
        point_rotation = rotation[part_ids]
        point_translation = translation[part_ids]
        return torch.bmm(point_rotation, points.unsqueeze(-1)).squeeze(-1) + point_translation

    @staticmethod
    def _quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        lw, lx, ly, lz = left.unbind(-1)
        rw, rx, ry, rz = right.unbind(-1)
        return torch.stack(
            (
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ),
            dim=-1,
        )

    def _fragment_centres_and_axes(
        self, points: torch.Tensor, part_ids: torch.Tensor, num_parts: int
    ):
        centres = self._scatter_mean(points, part_ids, num_parts)
        axes = points.new_empty(num_parts, 3, 3)
        for part_idx in range(num_parts):
            local = points[part_ids == part_idx] - centres[part_idx]
            covariance = local.T @ local / max(1, local.shape[0])
            _, eigenvectors = torch.linalg.eigh(covariance.float())
            axes[part_idx] = eigenvectors.to(points.dtype)
        return centres, axes

    def _build_candidates(self, data_dict, backbone_output):
        input_coords = data_dict["pointclouds"].float()
        pred_coords = backbone_output["pred_coords"].float()
        part_ids = backbone_output["part_batch_ids"].long()
        pred_quat, pred_trans = extract_poses_from_coords(input_coords, pred_coords, part_ids)

        candidate_points = []
        candidate_quat = []
        candidate_trans = []
        for alpha in self.candidate_alphas:
            quat = self._scaled_quaternion(pred_quat, alpha)
            trans = pred_trans * alpha
            candidate_points.append(self._apply_part_pose(input_coords, quat, trans, part_ids))
            candidate_quat.append(quat)
            candidate_trans.append(trans)

        if self.axis_offsets_deg:
            num_parts = pred_quat.shape[0]
            centres, principal_axes = self._fragment_centres_and_axes(
                input_coords, part_ids, num_parts
            )
            base_rotation = quaternion_to_matrix(pred_quat)
            base_centres = torch.bmm(
                base_rotation, centres.unsqueeze(-1)
            ).squeeze(-1) + pred_trans
            world_axes = torch.bmm(base_rotation, principal_axes)
            for angle_deg in self.axis_offsets_deg:
                half_angle = math.radians(angle_deg) * 0.5
                for axis_idx in range(3):
                    axis = F.normalize(world_axes[:, :, axis_idx].float(), dim=-1)
                    delta = torch.cat(
                        [
                            axis.new_full((num_parts, 1), math.cos(half_angle)),
                            axis * math.sin(half_angle),
                        ],
                        dim=-1,
                    )
                    quat = F.normalize(
                        self._quaternion_multiply(delta, pred_quat.float()), dim=-1
                    )
                    rotation = quaternion_to_matrix(quat)
                    trans = base_centres - torch.bmm(
                        rotation, centres.float().unsqueeze(-1)
                    ).squeeze(-1)
                    candidate_points.append(
                        self._apply_part_pose(input_coords, quat, trans, part_ids)
                    )
                    candidate_quat.append(quat)
                    candidate_trans.append(trans)
        return (
            torch.stack(candidate_points, dim=1),
            torch.stack(candidate_quat, dim=1),
            torch.stack(candidate_trans, dim=1),
            part_ids,
            pred_coords,
        )

    def _candidate_targets(
        self,
        data_dict,
        candidate_points,
        candidate_quat,
        candidate_trans,
        part_ids,
    ):
        valid_parts = data_dict["points_per_part"] > 0
        part_to_case = torch.nonzero(valid_parts, as_tuple=False)[:, 0].long()
        scales = data_dict["norm_scale"].float().reshape(-1)[part_to_case]
        gt_quat = data_dict["quat_gt"][valid_parts].float()
        gt_trans = data_dict["trans_gt"][valid_parts].float()
        gt_points = data_dict["pointclouds_gt"].float()
        num_parts, num_candidates = candidate_quat.shape[:2]

        point_distance = torch.linalg.norm(candidate_points - gt_points[:, None, :], dim=-1)
        tre = torch.stack(
            [self._scatter_mean(point_distance[:, k], part_ids, num_parts) for k in range(num_candidates)],
            dim=1,
        ) * scales[:, None]

        rot = rot_geodesic_deg(
            candidate_quat.reshape(-1, 4),
            gt_quat[:, None, :].expand(-1, num_candidates, -1).reshape(-1, 4),
        ).reshape(num_parts, num_candidates)
        trans = torch.linalg.norm(candidate_trans - gt_trans[:, None, :], dim=-1) * scales[:, None]

        chamfer = torch.empty_like(tre)
        for part_idx in range(num_parts):
            indices = torch.nonzero(part_ids == part_idx, as_tuple=False).flatten()
            if indices.numel() > self.chamfer_points:
                selection = torch.linspace(
                    0, indices.numel() - 1, self.chamfer_points, device=indices.device
                ).round().long()
                indices = indices[selection]
            candidate_subset = candidate_points[indices].permute(1, 0, 2)
            gt_subset = gt_points[indices].unsqueeze(0).expand(num_candidates, -1, -1)
            distances = torch.cdist(candidate_subset, gt_subset)
            chamfer[part_idx] = 0.5 * (
                distances.amin(dim=2).mean(dim=1) + distances.amin(dim=1).mean(dim=1)
            ) * scales[part_idx]

        metrics = torch.stack([tre, trans, rot, chamfer], dim=-1)
        scale = metrics.new_tensor(self.utility_scales)
        utility = (metrics / scale).mean(dim=-1)
        severe = ((rot > 30.0) | (trans > 20.0)).float()
        return metrics.detach(), utility.detach(), severe.detach(), part_to_case

    def _geometry_features(
        self,
        data_dict,
        candidate_points,
        candidate_quat,
        candidate_trans,
        part_ids,
        pred_coords,
        part_to_case,
    ):
        num_parts, num_candidates = candidate_quat.shape[:2]
        valid_parts = data_dict["points_per_part"] > 0
        scales = data_dict["norm_scale"].float().reshape(-1)[part_to_case]
        bone = data_dict["bonetype"][valid_parts].long().clamp(0, 2)
        diameter = data_dict["fragment_diameter_mm"][valid_parts].float()
        point_counts = data_dict["points_per_part"][valid_parts].float()

        pred_residual = torch.linalg.norm(candidate_points - pred_coords[:, None, :], dim=-1)
        residual_mean = torch.stack(
            [self._scatter_mean(pred_residual[:, k], part_ids, num_parts) for k in range(num_candidates)],
            dim=1,
        ) * scales[:, None]
        residual_rms = torch.stack(
            [
                self._scatter_mean(pred_residual[:, k].square(), part_ids, num_parts).sqrt()
                for k in range(num_candidates)
            ],
            dim=1,
        ) * scales[:, None]

        centroids = torch.stack(
            [self._scatter_mean(candidate_points[:, k], part_ids, num_parts) for k in range(num_candidates)],
            dim=1,
        )
        global_centroids = torch.zeros_like(centroids)
        minimum_same = torch.full(
            (num_parts, num_candidates), 2.0, device=centroids.device, dtype=centroids.dtype
        )
        minimum_other = torch.full_like(minimum_same, 2.0)
        for case_idx in range(int(data_dict["points_per_part"].shape[0])):
            local = torch.nonzero(part_to_case == case_idx, as_tuple=False).flatten()
            if local.numel() == 0:
                continue
            case_centroid = centroids.index_select(0, local).mean(dim=0, keepdim=True)
            global_centroids.index_copy_(
                0, local, case_centroid.expand(local.numel(), -1, -1)
            )
            for candidate_idx in range(num_candidates):
                distances = torch.cdist(
                    centroids[local, candidate_idx], centroids[local, candidate_idx]
                )
                distances.fill_diagonal_(float("inf"))
                local_bone = bone[local]
                same_mask = local_bone[:, None] == local_bone[None, :]
                other_mask = ~same_mask
                minimum_same[local, candidate_idx] = distances.masked_fill(
                    ~same_mask, float("inf")
                ).amin(dim=1).nan_to_num(posinf=2.0)
                minimum_other[local, candidate_idx] = distances.masked_fill(
                    ~other_mask, float("inf")
                ).amin(dim=1).nan_to_num(posinf=2.0)

        radial = torch.linalg.norm(centroids - global_centroids, dim=-1)
        predicted_rot = rot_geodesic_deg(
            candidate_quat.reshape(-1, 4),
            candidate_quat.new_tensor([1.0, 0.0, 0.0, 0.0]).expand(num_parts * num_candidates, -1),
        ).reshape(num_parts, num_candidates)
        predicted_trans = torch.linalg.norm(candidate_trans, dim=-1) * scales[:, None]
        candidate_steps = self.candidate_alphas + (1.0,) * (
            3 * len(self.axis_offsets_deg)
        )
        alpha = candidate_quat.new_tensor(candidate_steps)[None, :].expand(num_parts, -1)
        bone_one_hot = F.one_hot(bone, num_classes=3).float()[:, None, :].expand(-1, num_candidates, -1)

        features = torch.cat(
            [
                candidate_quat,
                candidate_trans,
                alpha[..., None],
                (residual_mean / 100.0)[..., None],
                (residual_rms / 100.0)[..., None],
                centroids,
                radial[..., None],
                minimum_same[..., None],
                minimum_other[..., None],
                (diameter / 200.0)[:, None, None].expand(-1, num_candidates, -1),
                (torch.log1p(point_counts) / 10.0)[:, None, None].expand(-1, num_candidates, -1),
                (point_counts / 5000.0)[:, None, None].expand(-1, num_candidates, -1),
                bone_one_hot,
                (predicted_rot / 180.0)[..., None],
                (predicted_trans / 100.0)[..., None],
                ((centroids - global_centroids) / 2.0),
            ],
            dim=-1,
        )
        if features.shape[-1] != 27:
            raise RuntimeError(f"Expected 27 geometry features, got {features.shape[-1]}")
        return features, bone

    def forward(self, data_dict):
        with torch.no_grad():
            backbone_output = self.transformer_model(
                input_coords=data_dict["pointclouds"],
                input_normals=data_dict["pointclouds_normals"],
                points_per_part=data_dict["points_per_part"],
                return_features=True,
            )
            candidate_points, candidate_quat, candidate_trans, part_ids, pred_coords = (
                self._build_candidates(data_dict, backbone_output)
            )
            valid_parts = data_dict["points_per_part"] > 0
            part_to_case = torch.nonzero(valid_parts, as_tuple=False)[:, 0].long()
            geometry, bone = self._geometry_features(
                data_dict,
                candidate_points,
                candidate_quat,
                candidate_trans,
                part_ids,
                pred_coords,
                part_to_case,
            )

        num_parts = geometry.shape[0]
        part_feature = self.backbone_projection(backbone_output["part_features"].float())
        token = part_feature[:, None, :] + self.geometry_projection(geometry)
        token = token + self.bone_embedding(bone)[:, None, :]
        candidate_ids = torch.arange(self.num_candidates, device=token.device)
        token = token + self.candidate_embedding(candidate_ids)[None, :, :]

        batch_size = int(data_dict["points_per_part"].shape[0])
        max_parts = int(data_dict["points_per_part"].shape[1])
        padded = token.new_zeros(batch_size, max_parts, self.num_candidates, token.shape[-1])
        valid = torch.zeros(batch_size, max_parts, dtype=torch.bool, device=token.device)
        local_slot = torch.zeros(num_parts, dtype=torch.long, device=token.device)
        for case_idx in range(batch_size):
            local = torch.nonzero(part_to_case == case_idx, as_tuple=False).flatten()
            count = int(local.numel())
            if count:
                padded[case_idx, :count] = token[local]
                valid[case_idx, :count] = True
                local_slot[local] = torch.arange(count, device=token.device)
        slot_ids = torch.arange(max_parts, device=token.device).clamp_max(self.max_parts - 1)
        padded = padded + self.fragment_slot_embedding(slot_ids)[None, :, None, :]
        flat = padded.reshape(batch_size, max_parts * self.num_candidates, -1)
        padding_mask = (~valid[:, :, None].expand(-1, -1, self.num_candidates)).reshape(batch_size, -1)
        encoded = self.context_encoder(flat, src_key_padding_mask=padding_mask)
        encoded = encoded.reshape(batch_size, max_parts, self.num_candidates, -1)
        encoded_valid = encoded[part_to_case, local_slot]

        scores = self.score_head(encoded_valid).squeeze(-1)
        selected = scores.argmax(dim=1)
        output = {
            "scores": scores,
            "metric_prediction": F.softplus(self.metric_head(encoded_valid)),
            "severe_logits": self.severe_head(encoded_valid).squeeze(-1),
            "selected_candidate": selected,
            "candidate_quat": candidate_quat,
            "candidate_trans": candidate_trans,
            "quat": candidate_quat.gather(
                1, selected[:, None, None].expand(-1, 1, 4)
            ).squeeze(1),
            "trans": candidate_trans.gather(
                1, selected[:, None, None].expand(-1, 1, 3)
            ).squeeze(1),
        }
        if all(key in data_dict for key in ("pointclouds_gt", "quat_gt", "trans_gt")):
            metrics, utility, severe, _ = self._candidate_targets(
                data_dict, candidate_points, candidate_quat, candidate_trans, part_ids
            )
            output.update({"metrics": metrics, "utility": utility, "severe": severe})
        return output

    def _losses(self, output):
        scores = output["scores"]
        utility = output["utility"]
        oracle = utility.argmin(dim=1)
        ce = F.cross_entropy(scores, oracle)

        difference = utility[:, :, None] - utility[:, None, :]
        score_difference = scores[:, :, None] - scores[:, None, :]
        pair_mask = difference.abs() > self.pair_margin
        target_sign = -difference.sign()
        pair = F.softplus(-target_sign * score_difference)
        pair = pair[pair_mask].mean() if pair_mask.any() else scores.sum() * 0.0

        regression_target = torch.log1p(output["metrics"])
        regression = F.smooth_l1_loss(output["metric_prediction"], regression_target)
        severe = F.binary_cross_entropy_with_logits(output["severe_logits"], output["severe"])
        total = (
            self.ce_weight * ce
            + self.pair_weight * pair
            + self.regression_weight * regression
            + self.severe_weight * severe
        )
        return {"loss": total, "ce": ce, "pair": pair, "regression": regression, "severe": severe}

    @staticmethod
    def _gather(values, indices):
        return values.gather(1, indices[:, None]).squeeze(1)

    def _ranking_diagnostics(self, scores, utility):
        difference = utility[:, :, None] - utility[:, None, :]
        score_difference = scores[:, :, None] - scores[:, None, :]
        upper = torch.triu(
            torch.ones_like(difference, dtype=torch.bool), diagonal=1
        )
        valid_pairs = upper & (difference.abs() > self.pair_margin)
        pair_correct = ((score_difference * difference) < 0.0) & valid_pairs
        pairwise_accuracy = pair_correct.float().sum() / valid_pairs.float().sum().clamp_min(1.0)

        quality = -utility
        quality_rank = quality.argsort(dim=1).argsort(dim=1).float()
        score_rank = scores.argsort(dim=1).argsort(dim=1).float()
        quality_rank = quality_rank - quality_rank.mean(dim=1, keepdim=True)
        score_rank = score_rank - score_rank.mean(dim=1, keepdim=True)
        spearman = (
            (quality_rank * score_rank).sum(dim=1)
            / (
                quality_rank.square().sum(dim=1).sqrt()
                * score_rank.square().sum(dim=1).sqrt()
            ).clamp_min(1e-6)
        ).mean()
        return pairwise_accuracy, spearman

    def _shared_step(self, data_dict, stage):
        output = self.forward(data_dict)
        losses = self._losses(output)
        batch_size = int(data_dict["points_per_part"].shape[0])
        sync = self.trainer.world_size > 1
        for name, value in losses.items():
            self.log(
                f"{stage}/{name}", value, on_step=stage == "train", on_epoch=True,
                prog_bar=name in ("loss",), sync_dist=sync, batch_size=batch_size,
            )

        if stage == "val":
            selected = output["scores"].argmax(dim=1)
            oracle = output["utility"].argmin(dim=1)
            pairwise_accuracy, spearman = self._ranking_diagnostics(
                output["scores"], output["utility"]
            )
            selected_utility = self._gather(output["utility"], selected)
            oracle_utility = self._gather(output["utility"], oracle)
            current_utility = output["utility"][:, 0]
            selected_metrics = output["metrics"].gather(
                1, selected[:, None, None].expand(-1, 1, 4)
            ).squeeze(1)
            oracle_metrics = output["metrics"].gather(
                1, oracle[:, None, None].expand(-1, 1, 4)
            ).squeeze(1)
            current_metrics = output["metrics"][:, 0]
            severe_current = output["severe"][:, 0] > 0.5
            recovered = severe_current & (self._gather(output["severe"], selected) < 0.5)

            validation = {
                "oracle_regret": (selected_utility - oracle_utility).mean(),
                "top1_accuracy": (selected == oracle).float().mean(),
                "pairwise_accuracy": pairwise_accuracy,
                "spearman": spearman,
                "selected_utility": selected_utility.mean(),
                "oracle_utility": oracle_utility.mean(),
                "current_utility": current_utility.mean(),
                "improvement": (current_utility - selected_utility).mean(),
                "severe_coverage": recovered.float().sum() / severe_current.float().sum().clamp_min(1.0),
                "preserve_rate": (selected == 0).float().mean(),
            }
            metric_names = ("tre_mm", "trans_mm", "rot_deg", "cd_proxy_mm")
            for metric_idx, metric_name in enumerate(metric_names):
                validation[f"selected_{metric_name}"] = selected_metrics[:, metric_idx].mean()
                validation[f"oracle_{metric_name}"] = oracle_metrics[:, metric_idx].mean()
                validation[f"current_{metric_name}"] = current_metrics[:, metric_idx].mean()
            num_parts = int(output["scores"].shape[0])
            for name, value in validation.items():
                self.log(
                    f"val/{name}", value, on_epoch=True, prog_bar=name in ("oracle_regret", "improvement"),
                    sync_dist=sync, batch_size=num_parts,
                )
        return losses["loss"]

    def training_step(self, data_dict, batch_idx):
        return self._shared_step(data_dict, "train")

    def validation_step(self, data_dict, batch_idx):
        return self._shared_step(data_dict, "val")

    def on_train_epoch_start(self):
        loader = self.trainer.train_dataloader
        dataset = getattr(loader, "dataset", None)
        if dataset is not None and hasattr(dataset, "reshuffle"):
            dataset.reshuffle()

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking or not self.trainer.is_global_zero:
            return
        names = (
            "val/loss", "val/oracle_regret", "val/top1_accuracy",
            "val/pairwise_accuracy", "val/spearman", "val/improvement",
            "val/severe_coverage", "val/preserve_rate", "val/current_utility",
            "val/selected_utility", "val/oracle_utility", "val/current_tre_mm",
            "val/selected_tre_mm", "val/oracle_tre_mm", "val/current_trans_mm",
            "val/selected_trans_mm", "val/oracle_trans_mm", "val/current_rot_deg",
            "val/selected_rot_deg", "val/oracle_rot_deg", "val/current_cd_proxy_mm",
            "val/selected_cd_proxy_mm", "val/oracle_cd_proxy_mm",
        )
        record = {"epoch": int(self.current_epoch), "global_step": int(self.global_step)}
        for name in names:
            value = self.trainer.callback_metrics.get(name)
            if value is not None:
                record[name] = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
        path = os.path.join(self.trainer.default_root_dir, "confidence_validation.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def configure_optimizers(self):
        trainable = [parameter for parameter in self.parameters() if parameter.requires_grad]
        optimizer = self.optimizer(trainable)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(self.trainer.max_epochs)), eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def on_fit_start(self):
        if self.trainer.is_global_zero:
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            frozen = sum(p.numel() for p in self.transformer_model.parameters())
            print(
                f"[FragmentConfidence] alphas={self.candidate_alphas} "
                f"axis_offsets_deg={self.axis_offsets_deg} candidates={self.num_candidates} "
                f"trainable={trainable / 1e6:.2f}M frozen_backbone={frozen / 1e6:.2f}M"
            )
