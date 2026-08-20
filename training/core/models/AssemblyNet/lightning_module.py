"""Pure Transformer Lightning Module for Point Cloud Assembly."""

from __future__ import annotations

import os
import json
import math
import glob
import copy
from functools import partial
from typing import Optional
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import *

class AssemblyLightningModule(L.LightningModule):
    def __init__(
            self,
            transformer_model: nn.Module,
            optimizer: "partial[torch.optim.Optimizer]",
            output_type: str = "coords",
            training_mode: str = "full",
            lora_rank: int = 16,
            lora_alpha: float = 32.0,
            save_vis_path: str = "./output",
            debug_vis: bool = False,
            checkpoint: Optional[str] = None,
            hard_mining_enabled: bool = False,
            hard_loss_weight: float = 0.3,
            hard_warmup_epochs: int = 1,
            schedule_warmup_epochs: float = 0.25,
            schedule_decay_epochs: float = 5.0,
            schedule_initial_lr_ratio: float = 0.04,
            schedule_min_lr_ratio: float = 0.02,
            anti_forgetting_enabled: bool = False,
            anti_forgetting_mode: str = "teacher",
            anti_forgetting_weight: float = 0.05,
            anti_forgetting_every_n_steps: int = 4,
            anti_forgetting_warmup_epochs: float = 2.0,
            rotation_teacher_margin_deg: float = 0.25,
            rotation_teacher_max_error_deg: float = 15.0,
            rotation_aware_enabled: bool = False,
            rotation_warmup_epochs: int = 3,
            fragment_balanced_weight: float = 0.25,
            centered_weight: float = 0.010,
            pair_weight: float = 0.004,
            covariance_weight: float = 0.005,
            horn_rot_weight: float = 0.010,
            horn_trans_weight: float = 0.0015,
            rotation_reference_deg: float = 10.0,
            translation_reference_mm: float = 50.0,
            paired_points_per_fragment: int = 24,
            paired_max_pairs: int = 96,
            horn_observability_floor: float = 0.05,
            qsmall_finetune_enabled: bool = False,
            qsmall_weight_boost: float = 0.75,
            qsmall_weight_threshold_mm: float = 120.0,
            qsmall_weight_span_mm: float = 60.0,
            qsmall_eval_threshold_mm: float = 91.578,
             qsmall_unfreeze_more_epoch: int = 2,
             qsmall_base_lr: float = 5.0e-6,
             qsmall_opened_lr: float = 1.0e-6,
             qsmall_adapter_lr: float = 2.0e-4,
             qsmall_trainable_last_layers: int = 4,
             qsmall_train_coordinate_head: bool = True,
             reliability_loss_weight: float = 0.0,
             reliability_coord_blend: float = 0.0,
             reliability_error_temperature_mm: float = 5.0,
             reliability_uniform_floor: float = 0.25,
     ):
        super().__init__()
        self.transformer_model = transformer_model
        self.optimizer = optimizer
        self.output_type = output_type
        self.training_mode = training_mode
        self.debug_vis = debug_vis
        self.hard_mining_enabled = hard_mining_enabled
        self.hard_loss_weight = hard_loss_weight
        self.hard_warmup_epochs = hard_warmup_epochs
        self.schedule_warmup_epochs = schedule_warmup_epochs
        self.schedule_decay_epochs = schedule_decay_epochs
        self.schedule_initial_lr_ratio = schedule_initial_lr_ratio
        self.schedule_min_lr_ratio = schedule_min_lr_ratio
        self.anti_forgetting_enabled = bool(anti_forgetting_enabled)
        self.anti_forgetting_mode = str(anti_forgetting_mode)
        if self.anti_forgetting_mode not in ("teacher", "l2sp", "rotation_teacher"):
            raise ValueError(
                f"Unknown anti_forgetting_mode={self.anti_forgetting_mode!r}"
            )
        self.anti_forgetting_weight = float(anti_forgetting_weight)
        self.anti_forgetting_every_n_steps = max(1, int(anti_forgetting_every_n_steps))
        self.anti_forgetting_warmup_epochs = float(anti_forgetting_warmup_epochs)
        self.rotation_teacher_margin_deg = float(rotation_teacher_margin_deg)
        self.rotation_teacher_max_error_deg = float(rotation_teacher_max_error_deg)
        self.rotation_aware_enabled = bool(rotation_aware_enabled)
        self.rotation_warmup_epochs = max(1, int(rotation_warmup_epochs))
        self.fragment_balanced_weight = float(fragment_balanced_weight)
        self.centered_weight = float(centered_weight)
        self.pair_weight = float(pair_weight)
        self.covariance_weight = float(covariance_weight)
        self.horn_rot_weight = float(horn_rot_weight)
        self.horn_trans_weight = float(horn_trans_weight)
        self.rotation_reference_rad = math.radians(float(rotation_reference_deg))
        self.translation_reference_mm = float(translation_reference_mm)
        self.paired_points_per_fragment = int(paired_points_per_fragment)
        self.paired_max_pairs = int(paired_max_pairs)
        self.horn_observability_floor = float(horn_observability_floor)
        self.qsmall_finetune_enabled = bool(qsmall_finetune_enabled)
        self.qsmall_weight_boost = float(qsmall_weight_boost)
        self.qsmall_weight_threshold_mm = float(qsmall_weight_threshold_mm)
        self.qsmall_weight_span_mm = float(qsmall_weight_span_mm)
        self.qsmall_eval_threshold_mm = float(qsmall_eval_threshold_mm)
        self.qsmall_unfreeze_more_epoch = int(qsmall_unfreeze_more_epoch)
        self.qsmall_base_lr = float(qsmall_base_lr)
        self.qsmall_opened_lr = float(qsmall_opened_lr)
        self.qsmall_adapter_lr = float(qsmall_adapter_lr)
        self.qsmall_trainable_last_layers = max(0, int(qsmall_trainable_last_layers))
        self.qsmall_train_coordinate_head = bool(qsmall_train_coordinate_head)
        self.reliability_loss_weight = float(reliability_loss_weight)
        self.reliability_coord_blend = float(reliability_coord_blend)
        self.reliability_error_temperature_mm = float(reliability_error_temperature_mm)
        self.reliability_uniform_floor = float(reliability_uniform_floor)
        if not 0.0 <= self.reliability_coord_blend <= 1.0:
            raise ValueError("reliability_coord_blend must be in [0, 1]")
        if not 0.0 <= self.reliability_uniform_floor <= 1.0:
            raise ValueError("reliability_uniform_floor must be in [0, 1]")
        self._epoch_hard_records = []
        self._qsmall_opened_parameter_ids = set()
        self._qsmall_adapter_parameter_ids = set()

        self.base_vis_path = save_vis_path
        self.save_vis_path_val = None

        if checkpoint is not None:
            self._load_pretrained_weights(checkpoint)

        if self.anti_forgetting_enabled:
            teacher = copy.deepcopy(self.transformer_model).eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
            # Keep the teacher outside Lightning's state dict and optimizer.
            object.__setattr__(self, "_anti_forgetting_teacher", teacher)
        else:
            object.__setattr__(self, "_anti_forgetting_teacher", None)

        if self.qsmall_finetune_enabled:
            self._configure_qsmall_trainable_parameters()

        if training_mode == "lora":
            from .lora import apply_lora_to_transformer
            apply_lora_to_transformer(self.transformer_model, rank=lora_rank, alpha=lora_alpha)

    def on_fit_start(self):
        if self._anti_forgetting_teacher is not None:
            self._anti_forgetting_teacher.to(self.device).eval()
        if self.trainer.is_global_zero:
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            mode_str = f"mode={self.output_type}  training={self.training_mode}"
            param_str = f"params={trainable/1e6:.1f}M trainable / {total/1e6:.1f}M total"
            print(f"[Model] {mode_str}  {param_str}")
            if self.rotation_aware_enabled:
                print(
                    "[RotationAwareE659] "
                    f"fragment={self.fragment_balanced_weight:g} "
                    f"centered={self.centered_weight:g} pair={self.pair_weight:g} "
                    f"cov={self.covariance_weight:g} rot={self.horn_rot_weight:g} "
                    f"trans={self.horn_trans_weight:g} warmup={self.rotation_warmup_epochs}"
                )
            if self.anti_forgetting_mode == "rotation_teacher":
                print(
                    "[RotationTeacher] "
                    f"weight={self.anti_forgetting_weight:g} "
                    f"every={self.anti_forgetting_every_n_steps} "
                    f"margin={self.rotation_teacher_margin_deg:g}deg "
                    f"max_teacher_error={self.rotation_teacher_max_error_deg:g}deg"
                )
            if self.qsmall_finetune_enabled:
                print(
                    "[Qsmall] "
                    f"boost={self.qsmall_weight_boost:g} "
                    f"threshold={self.qsmall_weight_threshold_mm:g}mm "
                    f"eval_threshold={self.qsmall_eval_threshold_mm:g}mm "
                    f"last_layers={self.qsmall_trainable_last_layers} "
                    f"head={self.qsmall_train_coordinate_head} "
                    f"unfreeze_more_epoch={self.qsmall_unfreeze_more_epoch} "
                    f"base_lr={self.qsmall_base_lr:g} opened_lr={self.qsmall_opened_lr:g} "
                    f"adapter_lr={self.qsmall_adapter_lr:g}"
                )
            if getattr(self.transformer_model, "point_reliability_head", None) is not None:
                print(
                    "[PointReliability] "
                    f"loss_weight={self.reliability_loss_weight:g} "
                    f"coord_blend={self.reliability_coord_blend:g} "
                    f"temperature={self.reliability_error_temperature_mm:g}mm "
                    f"uniform_floor={self.reliability_uniform_floor:g}"
                )

    def _configure_qsmall_trainable_parameters(self):
        layers = self.transformer_model.transformer_layers
        if self.qsmall_trainable_last_layers > len(layers):
            raise ValueError("qsmall_trainable_last_layers exceeds the Transformer depth")
        for parameter in self.transformer_model.parameters():
            parameter.requires_grad_(False)
        trainable_layers = []
        if self.qsmall_trainable_last_layers:
            trainable_layers = list(layers[-self.qsmall_trainable_last_layers:])
            for layer in trainable_layers:
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)
        if self.qsmall_train_coordinate_head:
            for parameter in self.transformer_model.head.parameters():
                parameter.requires_grad_(True)

        delayed_count = max(0, len(trainable_layers) - 2)
        self._qsmall_opened_parameter_ids = {
            id(parameter)
            for layer in trainable_layers[:delayed_count]
            for parameter in layer.parameters()
        }

        adapter_modules = [self.transformer_model.fragment_context_adapters]
        reliability_head = getattr(self.transformer_model, "point_reliability_head", None)
        if reliability_head is not None:
            adapter_modules.append(reliability_head)
        for module in adapter_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
                self._qsmall_adapter_parameter_ids.add(id(parameter))

    def _load_pretrained_weights(self, checkpoint_path: str):
        """Load pretrained weights into the transformer before LoRA wrapping."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        prefix = "transformer_model."
        stripped = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                stripped[k[len(prefix):]] = v
            else:
                stripped[k] = v

        missing, unexpected = self.transformer_model.load_state_dict(stripped, strict=False)

        if os.environ.get("LOCAL_RANK", "0") == "0":
            loaded = len(stripped) - len(missing)
            print(f"[Pretrained] Loaded {loaded} keys from {checkpoint_path}")
            if missing:
                print(f"[Pretrained]   Missing keys: {missing}")
            if unexpected:
                print(f"[Pretrained]   Unexpected keys: {unexpected}")

    def setup(self, stage: str):
        """Set up output directories for validation/testing stages."""
        if stage in ['fit', 'validate']:
            self.save_vis_path_val = os.path.join(self.base_vis_path, "valsets_results")
            os.makedirs(self.save_vis_path_val, exist_ok=True)

    def on_train_epoch_start(self):
        """Dynamic reshuffling at the start of each epoch to increase data variety."""
        super().on_train_epoch_start()

        if self.qsmall_finetune_enabled and self.trainer.is_global_zero:
            active_layers = self.qsmall_trainable_last_layers
            if self.current_epoch < self.qsmall_unfreeze_more_epoch:
                active_layers = min(2, active_layers)
            phase = (
                f"last{active_layers}"
                f"+{'head' if self.qsmall_train_coordinate_head else 'frozen-head'}"
                "+adapters"
            )
            print(f"[Qsmall] epoch={self.current_epoch} trainable_phase={phase}")

        if self.hard_mining_enabled:
            self._merge_persistent_hard_pool()

        train_loader = self.trainer.train_dataloader
        if train_loader is not None:
            dataset = train_loader.dataset
            if hasattr(dataset, 'datasets'):
                for ds in dataset.datasets:
                    if getattr(ds, 'split', None) == 'train' and hasattr(ds, 'reshuffle'):
                        ds.reshuffle()
            else:
                if getattr(dataset, 'split', None) == 'train' and hasattr(dataset, 'reshuffle'):
                    dataset.reshuffle()

    def on_after_backward(self):
        if not self.qsmall_finetune_enabled:
            return
        if self.current_epoch >= self.qsmall_unfreeze_more_epoch:
            return
        for parameter in self.transformer_model.parameters():
            if id(parameter) in self._qsmall_opened_parameter_ids:
                parameter.grad = None

    def forward(self, data_dict: dict):
        """Single-step forward pass through the transformer."""
        points_per_part = data_dict["points_per_part"]
        input_coords = data_dict["pointclouds"]
        input_normals = data_dict["pointclouds_normals"]

        output_dict = self.transformer_model(
            input_coords=input_coords,
            input_normals=input_normals,
            points_per_part=points_per_part,
        )

        if "part_batch_ids" not in output_dict:
            part_valids = points_per_part != 0
            seq_len = points_per_part[part_valids]
            output_dict["part_batch_ids"] = torch.repeat_interleave(
                torch.arange(len(seq_len), device=self.device), seq_len
            )

        if self.output_type == "pose":
            if self.debug_vis:
                pred_assemble = apply_pose_to_points(
                    input_coords,
                    output_dict["trans"],
                    output_dict["quat"],
                    output_dict["part_batch_ids"]
                )
                debug_visualize_pred_assemble(input_coords, output_dict["part_batch_ids"])
                debug_visualize_coords_vs_pred(input_coords, pred_assemble, output_dict["part_batch_ids"])
        elif self.output_type == "coords":
            if not self.training or self.debug_vis:
                quat, trans = extract_poses_from_coords(
                    input_coords,
                    output_dict["pred_coords"],
                    output_dict["part_batch_ids"]
                )
                output_dict["quat"] = quat
                output_dict["trans"] = trans
            if self.debug_vis:
                debug_visualize_coords_vs_pred(
                    input_coords,
                    output_dict["pred_coords"],
                    output_dict["part_batch_ids"]
                )
        return output_dict

    def _rotation_aux_scale(self) -> float:
        if self.rotation_warmup_epochs <= 1:
            return 1.0
        return min(1.0, float(self.current_epoch) / float(self.rotation_warmup_epochs - 1))

    @staticmethod
    def _fragment_mean(values: torch.Tensor, part_ids: torch.Tensor, n_parts: int):
        sums = values.new_zeros(n_parts).index_add_(0, part_ids, values)
        counts = torch.bincount(part_ids, minlength=n_parts).to(values.dtype).clamp_min(1.0)
        return sums / counts

    def _qsmall_part_weights(self, data_dict: dict, dtype: torch.dtype):
        valid_parts = data_dict["points_per_part"] > 0
        diameters = data_dict["fragment_diameter_mm"][valid_parts].to(dtype=dtype)
        strength = (
            (self.qsmall_weight_threshold_mm - diameters)
            / max(self.qsmall_weight_span_mm, 1e-6)
        ).clamp(0.0, 1.0)
        weights = 1.0 + self.qsmall_weight_boost * strength
        return weights, strength, diameters

    def _point_reliability_losses(self, output_dict: dict, data_dict: dict):
        """Calibrate point quality and robustify the fragment-balanced coordinate term."""
        pred = output_dict["pred_coords"].float()
        logits = output_dict.get("point_reliability_logits")
        zero = pred.sum() * 0.0
        if logits is None:
            return {
                "loss_reliability": zero,
                "loss_coord_reliability": zero,
                "reliability_entropy": zero.detach(),
                "reliability_effective_points": zero.detach(),
            }

        logits = logits.float()
        target = data_dict["pointclouds_gt"].detach().float()
        source = data_dict["pointclouds"].detach().float()
        part_ids = output_dict["part_batch_ids"].detach().long()
        n_parts = int(part_ids.max().item()) + 1 if part_ids.numel() else 0
        if n_parts == 0:
            return {
                "loss_reliability": zero,
                "loss_coord_reliability": zero,
                "reliability_entropy": zero.detach(),
                "reliability_effective_points": zero.detach(),
            }

        valid_parts = data_dict["points_per_part"] > 0
        part_to_case = torch.nonzero(valid_parts, as_tuple=False)[:, 0].long()
        part_scales = data_dict["norm_scale"].float().reshape(-1)[part_to_case]
        point_scales = part_scales[part_ids]
        paired_error_mm = torch.linalg.vector_norm(pred.detach() - target, dim=-1) * point_scales
        point_mse = (pred - target).square().mean(dim=-1)
        part_weights, _, _ = self._qsmall_part_weights(data_dict, pred.dtype)

        kl_terms = []
        coord_terms = []
        entropy_terms = []
        effective_terms = []
        temperature = max(self.reliability_error_temperature_mm, 1e-3)
        uniform_floor = min(max(self.reliability_uniform_floor, 0.0), 1.0)
        for part_id in range(n_parts):
            mask = part_ids == part_id
            count = int(mask.sum().item())
            if count == 0:
                continue

            src = source[mask]
            src_centered = src - src.mean(dim=0, keepdim=True)
            radius = torch.linalg.vector_norm(src_centered, dim=-1)
            radius_ref = torch.quantile(radius, 0.9).clamp_min(1e-6)
            leverage = 0.25 + 0.75 * torch.sqrt((radius / radius_ref).clamp(0.0, 1.0))

            quality = torch.exp(-paired_error_mm[mask] / temperature) * leverage
            target_distribution = quality.clamp_min(1e-8)
            target_distribution = target_distribution / target_distribution.sum()

            predicted_distribution = torch.softmax(logits[mask], dim=0)
            predicted_distribution = (
                (1.0 - uniform_floor) * predicted_distribution
                + uniform_floor / float(count)
            )
            predicted_distribution = predicted_distribution.clamp_min(1e-8)
            predicted_distribution = predicted_distribution / predicted_distribution.sum()

            kl_terms.append(torch.sum(
                target_distribution
                * (target_distribution.log() - predicted_distribution.log())
            ))
            coord_terms.append(torch.sum(predicted_distribution * point_mse[mask]))
            entropy = -torch.sum(
                predicted_distribution * predicted_distribution.log()
            )
            entropy_terms.append(entropy / max(math.log(max(count, 2)), 1e-6))
            effective_terms.append(1.0 / predicted_distribution.square().sum())

        if not kl_terms:
            return {
                "loss_reliability": zero,
                "loss_coord_reliability": zero,
                "reliability_entropy": zero.detach(),
                "reliability_effective_points": zero.detach(),
            }

        kl_terms = torch.stack(kl_terms)
        coord_terms = torch.stack(coord_terms)
        loss_reliability = (kl_terms * part_weights).sum() / part_weights.sum().clamp_min(1e-6)
        loss_coord_reliability = (
            coord_terms * part_weights
        ).sum() / part_weights.sum().clamp_min(1e-6)
        return {
            "loss_reliability": loss_reliability,
            "loss_coord_reliability": loss_coord_reliability,
            "reliability_entropy": torch.stack(entropy_terms).mean().detach(),
            "reliability_effective_points": torch.stack(effective_terms).mean().detach(),
        }

    def _rotation_aware_losses(self, output_dict: dict, data_dict: dict):
        pred = output_dict["pred_coords"].float()
        target = data_dict["pointclouds_gt"].detach().float()
        source = data_dict["pointclouds"].detach().float()
        part_ids = output_dict["part_batch_ids"].detach().long()
        n_parts = int(part_ids.max().item()) + 1 if part_ids.numel() else 0
        zero = pred.sum() * 0.0
        if n_parts == 0:
            return {name: zero for name in (
                "loss_coord_global", "loss_coord_fragment", "loss_centered",
                "loss_pair", "loss_covariance", "loss_horn_rot", "loss_horn_trans",
                "horn_rot_deg", "horn_trans_mm", "horn_observability",
            )}

        point_mse = (pred - target).square().mean(dim=-1)
        loss_coord_global = point_mse.mean()
        fragment_mse = self._fragment_mean(point_mse, part_ids, n_parts)
        part_weights, qsmall_strength, part_diameters = self._qsmall_part_weights(
            data_dict,
            pred.dtype,
        )
        loss_coord_fragment = (
            fragment_mse * part_weights
        ).sum() / part_weights.sum().clamp_min(1e-6)
        qsmall_eval_mask = part_diameters <= self.qsmall_eval_threshold_mm
        if qsmall_eval_mask.any():
            loss_coord_qsmall = fragment_mse[qsmall_eval_mask].mean()
        else:
            loss_coord_qsmall = zero

        valid_parts = data_dict["points_per_part"] > 0
        part_to_case = torch.nonzero(valid_parts, as_tuple=False)[:, 0].long()
        anchor_parts = torch.zeros(n_parts, dtype=torch.bool, device=pred.device)
        for case_id in range(int(valid_parts.shape[0])):
            local_parts = torch.nonzero(part_to_case == case_id, as_tuple=False).flatten()
            if local_parts.numel():
                anchor_parts[local_parts[0]] = True

        centered_terms, pair_terms, covariance_terms = [], [], []
        centered_weights, pair_weights, covariance_weights = [], [], []
        geometry_weights = torch.ones(pred.shape[0], device=pred.device)
        for part_id in range(n_parts):
            mask = part_ids == part_id
            src, dst, out = source[mask], target[mask], pred[mask]
            if src.shape[0] < 3:
                continue
            src_c = src - src.mean(dim=0, keepdim=True)
            radii_sq = src_c.square().sum(dim=-1)
            radius_ref = torch.quantile(radii_sq, 0.9).clamp_min(1e-6)
            geometry_weights[mask] = (radii_sq / radius_ref).clamp(0.25, 1.0)
            if anchor_parts[part_id]:
                continue

            dst_c = dst - dst.mean(dim=0, keepdim=True)
            out_c = out - out.mean(dim=0, keepdim=True)
            diameter = torch.linalg.vector_norm(
                src.max(dim=0).values - src.min(dim=0).values
            ).clamp_min(0.05)
            centered_terms.append(F.smooth_l1_loss(
                out_c / diameter, dst_c / diameter, beta=0.02
            ))
            centered_weights.append(part_weights[part_id])

            pred_cov = src_c.T @ out_c
            target_cov = src_c.T @ dst_c
            cov_scale = torch.linalg.matrix_norm(target_cov).detach().clamp_min(1e-6)
            covariance_terms.append(F.mse_loss(pred_cov / cov_scale, target_cov / cov_scale))
            covariance_weights.append(part_weights[part_id])

            n_select = min(self.paired_points_per_fragment, int(src.shape[0]))
            selected = torch.topk(radii_sq, k=n_select, largest=True).indices
            pair_index = torch.triu_indices(n_select, n_select, offset=1, device=src.device)
            if pair_index.shape[1] == 0:
                continue
            first, second = selected[pair_index[0]], selected[pair_index[1]]
            baselines = torch.linalg.vector_norm(src[first] - src[second], dim=-1)
            n_keep = min(self.paired_max_pairs, int(baselines.numel()))
            keep = torch.topk(baselines, k=n_keep, largest=True).indices
            first, second = first[keep], second[keep]
            leverage = baselines[keep].square()
            leverage = (leverage / leverage.mean().clamp_min(1e-8)).clamp(0.25, 4.0)
            pred_vectors = (out[first] - out[second]) / diameter
            target_vectors = (dst[first] - dst[second]) / diameter
            vector_error = F.smooth_l1_loss(
                pred_vectors, target_vectors, beta=0.02, reduction="none"
            ).mean(dim=-1)
            direction_error = 1.0 - F.cosine_similarity(
                pred_vectors, target_vectors, dim=-1, eps=1e-6
            )
            pair_terms.append((leverage * (vector_error + 0.25 * direction_error)).mean())
            pair_weights.append(part_weights[part_id])

        def reduce_terms(terms, weights):
            if not terms:
                return zero
            stacked_terms = torch.stack(terms)
            stacked_weights = torch.stack(weights).to(stacked_terms.dtype)
            return (stacked_terms * stacked_weights).sum() / stacked_weights.sum().clamp_min(1e-6)

        pred_rot, pred_trans, observability, pred_valid = weighted_horn_matrices(
            source, pred, part_ids, geometry_weights
        )
        with torch.no_grad():
            target_rot, target_trans, _, target_valid = weighted_horn_matrices(
                source, target, part_ids, geometry_weights
            )
        pose_valid = pred_valid & target_valid & (~anchor_parts)
        pose_valid &= observability >= self.horn_observability_floor
        if pose_valid.any():
            pose_weight = observability[pose_valid].sqrt().clamp(0.25, 1.0)
            pose_weight = pose_weight * part_weights[pose_valid]
            rot_per_part = (
                pred_rot[pose_valid] - target_rot[pose_valid]
            ).square().sum(dim=(-2, -1)) / 8.0
            loss_horn_rot = (pose_weight * rot_per_part).sum() / pose_weight.sum()
            part_scales = data_dict["norm_scale"].float().reshape(-1)[part_to_case]
            trans_delta_mm = (
                pred_trans[pose_valid] - target_trans[pose_valid]
            ) * part_scales[pose_valid, None]
            loss_horn_trans = F.smooth_l1_loss(
                trans_delta_mm / self.translation_reference_mm,
                torch.zeros_like(trans_delta_mm), beta=0.2,
            )
            relative = pred_rot[pose_valid] @ target_rot[pose_valid].transpose(-1, -2)
            cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
            horn_rot_deg = torch.rad2deg(torch.acos(cosine)).mean().detach()
            horn_trans_mm = torch.linalg.vector_norm(trans_delta_mm, dim=-1).mean().detach()
            horn_observability = observability[pose_valid].mean().detach()
        else:
            loss_horn_rot = loss_horn_trans = horn_rot_deg = horn_trans_mm = zero
            horn_observability = zero

        return {
            "loss_coord_global": loss_coord_global,
            "loss_coord_fragment": loss_coord_fragment,
            "loss_coord_qsmall": loss_coord_qsmall,
            "loss_centered": reduce_terms(centered_terms, centered_weights),
            "loss_pair": reduce_terms(pair_terms, pair_weights),
            "loss_covariance": reduce_terms(covariance_terms, covariance_weights),
            "loss_horn_rot": loss_horn_rot,
            "loss_horn_trans": loss_horn_trans,
            "horn_rot_deg": horn_rot_deg,
            "horn_trans_mm": horn_trans_mm,
            "horn_observability": horn_observability,
            "qsmall_weight_mean": part_weights.mean().detach(),
            "qsmall_strength_mean": qsmall_strength.mean().detach(),
        }

    def loss(self, output_dict: dict, data_dict: dict):
        """Compute the regression loss based on the output type."""
        gt_coords = data_dict["pointclouds_gt"]

        if self.output_type == "coords":
            pred_coords = output_dict["pred_coords"]
            if not self.rotation_aware_enabled:
                loss = F.mse_loss(pred_coords, gt_coords, reduction="mean")
                return {"loss": loss}

            with torch.autocast(device_type=pred_coords.device.type, enabled=False):
                terms = self._rotation_aware_losses(output_dict, data_dict)
                reliability_terms = self._point_reliability_losses(output_dict, data_dict)
                terms.update(reliability_terms)
            fragment_weight = min(max(self.fragment_balanced_weight, 0.0), 1.0)
            scale = pred_coords.new_tensor(self._rotation_aux_scale())
            reliability_blend = min(
                max(self.reliability_coord_blend, 0.0), 1.0
            ) * scale
            if "point_reliability_logits" not in output_dict:
                reliability_blend = reliability_blend * 0.0
            fragment_coord_loss = (
                (1.0 - reliability_blend) * terms["loss_coord_fragment"]
                + reliability_blend * terms["loss_coord_reliability"]
            )
            coord_loss = (
                (1.0 - fragment_weight) * terms["loss_coord_global"]
                + fragment_weight * fragment_coord_loss
            )
            loss = coord_loss + scale * (
                self.centered_weight * terms["loss_centered"]
                + self.pair_weight * terms["loss_pair"]
                + self.covariance_weight * terms["loss_covariance"]
                + self.horn_rot_weight * terms["loss_horn_rot"]
                + self.horn_trans_weight * terms["loss_horn_trans"]
                + self.reliability_loss_weight * terms["loss_reliability"]
            )
            return {
                "loss": loss,
                "loss_coord": coord_loss,
                "reliability_coord_blend": reliability_blend,
                "rotation_scale": scale,
                **terms,
            }

        elif self.output_type == "pose":
            pred_quat, pred_trans = output_dict["quat"], output_dict["trans"]

            pred_coords = apply_pose_to_points(
                data_dict["pointclouds"],
                pred_trans,
                pred_quat,
                output_dict["part_batch_ids"]
            )
            loss = F.mse_loss(pred_coords, gt_coords, reduction="mean")
            return {"loss": loss}

    def training_step(self, data_dict: dict, batch_idx: int, dataloader_idx: int = 0):
        output_dict = self.forward(data_dict)
        loss_dict = self.loss(output_dict, data_dict)

        if self.hard_mining_enabled and self.output_type == "coords":
            hard_dict = self._online_hard_mining(
                output_dict,
                data_dict,
                loss_dict["loss"],
                coordinate_loss=loss_dict.get("loss_coord"),
            )
            loss_dict.update(hard_dict)

        if (
            self._anti_forgetting_teacher is not None
            and self.output_type == "coords"
            and batch_idx % self.anti_forgetting_every_n_steps == 0
        ):
            consistency = self._anti_forgetting_loss(output_dict, data_dict)
            if consistency is not None:
                if self.anti_forgetting_warmup_epochs > 0:
                    ramp = min(
                        1.0,
                        (float(self.current_epoch) + 1.0)
                        / self.anti_forgetting_warmup_epochs,
                    )
                else:
                    ramp = 1.0
                consistency_weight = self.anti_forgetting_weight * ramp
                loss_dict["loss"] = loss_dict["loss"] + consistency_weight * consistency
                loss_dict["loss_consistency"] = consistency.detach()
                loss_dict["consistency_weight"] = loss_dict["loss"].new_tensor(
                    consistency_weight
                )
                if self.anti_forgetting_mode == "rotation_teacher":
                    loss_dict.update(getattr(self, "_rotation_teacher_metrics", {}))

        loss_val = loss_dict["loss"]
        batch_size = data_dict["points_per_part"].shape[0]

        # Sync NaN state across ALL ranks — if ANY rank has non-finite loss, ALL skip
        is_finite_int = torch.isfinite(loss_val).int()
        if self.trainer.world_size > 1:
            torch.distributed.all_reduce(is_finite_int, op=torch.distributed.ReduceOp.MIN)
        is_finite = is_finite_int.bool()

        if not is_finite:
            print(f"[Rank {self.global_rank}] Non-finite loss at step {self.global_step}, all ranks skipping.")
            self.log("train/loss", float('nan'), on_step=True, on_epoch=True, prog_bar=True,
                     sync_dist=True, batch_size=batch_size)
            return loss_val * 0.0

        self.log("train/loss", loss_val, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=batch_size)

        for key in ("loss_base", "loss_hard", "difficulty", "hard_weight", "rot_mm_score",
                    "trans_mm_score", "tre_mm_score", "source_natural", "source_proposal",
                    "source_qsmall", "source_replay", "loss_consistency", "consistency_weight", "loss_coord",
                    "loss_coord_global", "loss_coord_fragment", "loss_centered", "loss_pair",
                    "loss_covariance", "loss_horn_rot", "loss_horn_trans", "rotation_scale",
                    "horn_rot_deg", "horn_trans_mm", "horn_observability",
                    "loss_coord_qsmall", "qsmall_weight_mean", "qsmall_strength_mean",
                    "loss_reliability", "loss_coord_reliability",
                    "reliability_entropy", "reliability_effective_points",
                    "reliability_coord_blend",
                    "rotation_teacher_coverage", "rotation_teacher_student_deg",
                    "rotation_teacher_teacher_deg"):
            if key in loss_dict:
                self.log(
                    f"train/{key}", loss_dict[key], on_step=True, on_epoch=True,
                    prog_bar=key in ("difficulty",), sync_dist=True, batch_size=batch_size,
                )

        if "loss_rot" in loss_dict:
            self.log("train/loss_rot", loss_dict["loss_rot"], on_step=True, on_epoch=False, sync_dist=True,
                     batch_size=batch_size)
        if "loss_trans" in loss_dict:
            self.log("train/loss_trans", loss_dict["loss_trans"], on_step=True, on_epoch=False, sync_dist=True,
                     batch_size=batch_size)

        return loss_val

    @staticmethod
    def _scatter_mean(values, ids, count):
        sums = torch.zeros(count, dtype=values.dtype, device=values.device)
        sums.scatter_add_(0, ids, values)
        counts = torch.bincount(ids, minlength=count).to(values.dtype).clamp_min(1.0)
        return sums / counts

    @staticmethod
    def _bucket_id(bone, diameter_mm, rot_err_deg, trans_err_mm):
        size_bin = 0 if diameter_mm < 60.0 else (1 if diameter_mm < 120.0 else 2)
        rot_score = float(rot_err_deg) / 15.0
        trans_score = float(trans_err_mm) / 10.0
        if rot_score < 1.0 and trans_score < 1.0:
            failure_mode = 0
        elif rot_score >= 1.25 * trans_score:
            failure_mode = 1
        elif trans_score >= 1.25 * rot_score:
            failure_mode = 2
        else:
            failure_mode = 3
        return int(bone) * 12 + size_bin * 4 + failure_mode

    def _anti_forgetting_loss(self, output_dict, data_dict):
        if self.anti_forgetting_mode == "l2sp":
            reference = dict(self._anti_forgetting_teacher.named_parameters())
            normalized_terms = []
            for name, parameter in self.transformer_model.named_parameters():
                if not parameter.requires_grad or parameter.ndim <= 1:
                    continue
                anchor = reference[name]
                denominator = anchor.float().square().mean().clamp_min(1e-8)
                normalized_terms.append(
                    (parameter.float() - anchor.float()).square().mean() / denominator
                )
            if not normalized_terms:
                return None
            return torch.stack(normalized_terms).mean()

        natural_cases = data_dict["sample_source_id"].long() == 0
        if (
            self.anti_forgetting_mode != "rotation_teacher"
            and not bool(natural_cases.any())
        ):
            return None
        point_to_case = torch.repeat_interleave(
            torch.arange(natural_cases.numel(), device=self.device),
            data_dict["points_per_sample"].long(),
        )
        natural_points = natural_cases[point_to_case]
        with torch.no_grad():
            teacher_output = self._anti_forgetting_teacher(
                input_coords=data_dict["pointclouds"],
                input_normals=data_dict["pointclouds_normals"],
                points_per_part=data_dict["points_per_part"],
            )
        if self.anti_forgetting_mode == "rotation_teacher":
            source = data_dict["pointclouds"].detach().float()
            student = output_dict["pred_coords"].float()
            teacher = teacher_output["pred_coords"].detach().float()
            target = data_dict["pointclouds_gt"].detach().float()
            part_ids = output_dict["part_batch_ids"].detach().long()

            with torch.no_grad():
                student_rot, _, student_obs, student_valid = weighted_horn_matrices(
                    source, student.detach(), part_ids
                )
                teacher_rot, _, teacher_obs, teacher_valid = weighted_horn_matrices(
                    source, teacher, part_ids
                )
                target_rot, _, _, target_valid = weighted_horn_matrices(
                    source, target, part_ids
                )

            valid_parts = data_dict["points_per_part"] > 0
            part_to_case = torch.nonzero(valid_parts, as_tuple=False)[:, 0].long()
            natural_parts = natural_cases[part_to_case]

            def matrix_angle_deg(first, second):
                relative = first @ second.transpose(-1, -2)
                cosine = (
                    (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5
                ).clamp(-1.0, 1.0)
                return torch.rad2deg(torch.acos(cosine))

            teacher_gt_deg = matrix_angle_deg(teacher_rot, target_rot).detach()
            student_gt_deg = matrix_angle_deg(student_rot, target_rot).detach()
            preserve = natural_parts & student_valid & teacher_valid & target_valid
            preserve &= teacher_obs >= self.horn_observability_floor
            preserve &= student_obs >= self.horn_observability_floor
            preserve &= teacher_gt_deg <= self.rotation_teacher_max_error_deg
            preserve &= (
                teacher_gt_deg + self.rotation_teacher_margin_deg <= student_gt_deg
            )
            # Horn is used only for detached gating. Backpropagate through
            # centered coordinates so translation remains unconstrained and
            # the graph does not contain 48 Horn power iterations.
            n_parts = int(preserve.numel())
            counts = torch.bincount(part_ids, minlength=n_parts).to(student.dtype)
            counts = counts.clamp_min(1.0)
            student_mean = student.new_zeros((n_parts, 3)).index_add_(
                0, part_ids, student
            ) / counts[:, None]
            teacher_mean = teacher.new_zeros((n_parts, 3)).index_add_(
                0, part_ids, teacher
            ) / counts[:, None]
            source_min = source.new_full((n_parts, 3), float("inf"))
            source_max = source.new_full((n_parts, 3), float("-inf"))
            source_min.scatter_reduce_(
                0, part_ids[:, None].expand(-1, 3), source, reduce="amin", include_self=True
            )
            source_max.scatter_reduce_(
                0, part_ids[:, None].expand(-1, 3), source, reduce="amax", include_self=True
            )
            diameter = torch.linalg.vector_norm(source_max - source_min, dim=-1).clamp_min(0.05)
            preserve_points = preserve[part_ids]
            student_centered = (
                student - student_mean[part_ids]
            ) / diameter[part_ids, None]
            teacher_centered = (
                teacher - teacher_mean[part_ids]
            ) / diameter[part_ids, None]
            radius = torch.linalg.vector_norm(
                source - source.new_zeros((n_parts, 3)).index_add_(
                    0, part_ids, source
                )[part_ids] / counts[part_ids, None],
                dim=-1,
            )
            radius_weight = radius / self._fragment_mean(
                radius, part_ids, n_parts
            )[part_ids].clamp_min(1e-6)
            radius_weight = radius_weight.clamp(0.25, 3.0)
            vector_error = F.smooth_l1_loss(
                student_centered,
                teacher_centered,
                beta=0.02,
                reduction="none",
            ).mean(dim=-1)
            gated_weight = radius_weight * preserve_points.to(radius_weight.dtype)
            loss = (gated_weight * vector_error).sum() / gated_weight.sum().clamp_min(1e-6)
            preserve_count = preserve.sum().to(student.dtype)
            self._rotation_teacher_metrics = {
                "rotation_teacher_coverage": preserve.float().mean().detach(),
                "rotation_teacher_student_deg": (
                    (student_gt_deg * preserve).sum() / preserve_count.clamp_min(1.0)
                ).detach(),
                "rotation_teacher_teacher_deg": (
                    (teacher_gt_deg * preserve).sum() / preserve_count.clamp_min(1.0)
                ).detach(),
            }
            return loss
        return F.mse_loss(
            output_dict["pred_coords"][natural_points],
            teacher_output["pred_coords"][natural_points],
            reduction="mean",
        )

    def _online_hard_mining(
        self, output_dict, data_dict, base_loss, coordinate_loss=None
    ):
        pred_coords = output_dict["pred_coords"]
        gt_coords = data_dict["pointclouds_gt"]
        part_ids = output_dict["part_batch_ids"].long()
        batch_size = int(data_dict["points_per_part"].shape[0])
        num_parts = int(part_ids.max().item()) + 1

        point_mse = (pred_coords - gt_coords).pow(2).mean(dim=-1)
        frag_mse = self._scatter_mean(point_mse, part_ids, num_parts)

        points_per_sample = data_dict["points_per_sample"].long()
        point_to_case = torch.repeat_interleave(
            torch.arange(batch_size, device=pred_coords.device), points_per_sample
        )
        case_mse = self._scatter_mean(point_mse, point_to_case, batch_size)

        valid_parts = data_dict["points_per_part"] > 0
        part_to_case = torch.nonzero(valid_parts, as_tuple=False)[:, 0].long()
        scales = data_dict["norm_scale"].float().reshape(-1)
        part_scales = scales[part_to_case]

        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=False):
            pred_quat, pred_trans = extract_poses_from_coords(
                data_dict["pointclouds"].float(),
                pred_coords.detach().float(),
                part_ids,
            )
            gt_quat = data_dict["quat_gt"][valid_parts].float()
            gt_trans = data_dict["trans_gt"][valid_parts].float()
            rot_err = rot_geodesic_deg(pred_quat.float(), gt_quat)
            trans_mm = torch.norm(pred_trans.float() - gt_trans, dim=-1) * part_scales
            tre_mm = torch.sqrt(frag_mse.detach().float().clamp_min(0.0)) * part_scales

            frag_difficulty = (
                0.50 * (rot_err / 45.0).clamp(0.0, 3.0)
                + 0.25 * (trans_mm / 30.0).clamp(0.0, 3.0)
                + 0.25 * (tre_mm / 20.0).clamp(0.0, 3.0)
            )

            case_difficulty = []
            worst_part_indices = []
            for case_idx in range(batch_size):
                local_indices = torch.nonzero(part_to_case == case_idx, as_tuple=False).flatten()
                local_scores = frag_difficulty[local_indices]
                top_count = min(2, int(local_scores.numel()))
                top_values, top_positions = torch.topk(local_scores, k=top_count)
                case_difficulty.append(0.7 * top_values.mean() + 0.3 * local_scores.mean())
                worst_part_indices.append(int(local_indices[top_positions[0]].item()))
            case_difficulty = torch.stack(case_difficulty)

        hard_weights = 1.0 + 1.5 * torch.sigmoid((case_difficulty - 0.5) / 0.5)
        hard_weights = hard_weights.clamp(1.0, 3.0).detach()
        weighted_case_loss = (case_mse * hard_weights).sum() / hard_weights.sum().clamp_min(1e-6)

        if self.current_epoch < self.hard_warmup_epochs:
            mix_weight = 0.0
        else:
            ramp = min(1.0, (self.current_epoch - self.hard_warmup_epochs + 1) / 2.0)
            mix_weight = self.hard_loss_weight * ramp
        if coordinate_loss is None:
            coordinate_loss = base_loss
        rotation_aux_loss = base_loss - coordinate_loss
        total_loss = (
            (1.0 - mix_weight) * coordinate_loss
            + mix_weight * weighted_case_loss
            + rotation_aux_loss
        )

        sample_indices = data_dict["sample_index"].detach().cpu().tolist()
        aug_seeds = data_dict["augmentation_seed"].detach().cpu().tolist()
        pose_mode_ids = data_dict["pose_mode_id"].detach().cpu().tolist()
        source_ids = data_dict["sample_source_id"].detach().cpu().tolist()
        bones = data_dict["bonetype"][valid_parts].detach().cpu()
        diameters = data_dict["fragment_diameter_mm"][valid_parts].detach().cpu()
        mode_names = {0: "baseline", 1: "natural", 2: "proposal", 3: "qsmall_hard"}

        for case_idx, worst_idx in enumerate(worst_part_indices):
            self._epoch_hard_records.append({
                "sample_index": int(sample_indices[case_idx]),
                "augmentation_seed": int(aug_seeds[case_idx]),
                "pose_mode": mode_names.get(int(pose_mode_ids[case_idx]), "natural"),
                "source_id": int(source_ids[case_idx]),
                "difficulty": float(case_difficulty[case_idx].item()),
                "bucket": self._bucket_id(
                    int(bones[worst_idx].item()),
                    float(diameters[worst_idx].item()),
                    float(rot_err[worst_idx].item()),
                    float(trans_mm[worst_idx].item()),
                ),
                "rot_err_deg": float(rot_err[worst_idx].item()),
                "trans_err_mm": float(trans_mm[worst_idx].item()),
                "tre_err_mm": float(tre_mm[worst_idx].item()),
                "epoch": int(self.current_epoch),
            })

        return {
            "loss": total_loss,
            "loss_base": base_loss.detach(),
            "loss_hard": weighted_case_loss.detach(),
            "loss_rotation_aux": rotation_aux_loss.detach(),
            "rotation_aux_retention": base_loss.new_tensor(1.0),
            "difficulty": case_difficulty.mean(),
            "hard_weight": hard_weights.mean(),
            "rot_mm_score": rot_err.mean(),
            "trans_mm_score": trans_mm.mean(),
            "tre_mm_score": tre_mm.mean(),
            "source_natural": (data_dict["sample_source_id"] == 0).float().mean(),
            "source_proposal": (data_dict["pose_mode_id"] == 2).float().mean(),
            "source_qsmall": (data_dict["pose_mode_id"] == 3).float().mean(),
            "source_replay": (data_dict["sample_source_id"] == 2).float().mean(),
        }

    def on_train_epoch_end(self):
        if not self.hard_mining_enabled:
            return

        # Keep one replay pool per rank. Object collectives at this point can
        # deadlock when DDP ranks finish their final batch at different times.
        all_records = list(self._epoch_hard_records)

        loader = self.trainer.train_dataloader
        dataset = getattr(loader, "dataset", None)
        if dataset is not None and hasattr(dataset, "update_hard_pool"):
            dataset.update_hard_pool(all_records, current_epoch=int(self.current_epoch))
            pool = dataset.get_hard_pool()
            self.log("train/hard_pool_size", float(len(pool)), on_epoch=True, sync_dist=False)

            pool_dir = os.path.join(self.base_vis_path, "hard_pool")
            os.makedirs(pool_dir, exist_ok=True)
            pool_path = os.path.join(
                pool_dir,
                f"hard_pool_rank_{self.global_rank}_epoch_{self.current_epoch:04d}.json",
            )
            temp_path = pool_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(pool, f, indent=2)
            os.replace(temp_path, pool_path)

        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
            self.log("train/peak_memory_gb", peak_gb, on_epoch=True, sync_dist=False)
            torch.cuda.reset_peak_memory_stats(self.device)

        self._epoch_hard_records.clear()

    def _merge_persistent_hard_pool(self):
        loader = self.trainer.train_dataloader
        dataset = getattr(loader, "dataset", None)
        if dataset is None or not hasattr(dataset, "update_hard_pool"):
            return
        pool_dir = os.path.join(self.base_vis_path, "hard_pool")
        if not os.path.isdir(pool_dir):
            return

        latest_by_rank = {}
        for path in glob.glob(os.path.join(pool_dir, "hard_pool_rank_*_epoch_*.json")):
            name = os.path.basename(path)
            try:
                rank = int(name.split("_rank_")[1].split("_epoch_")[0])
                epoch = int(name.split("_epoch_")[1].split(".json")[0])
            except (IndexError, ValueError):
                continue
            if epoch >= int(self.current_epoch):
                continue
            if rank not in latest_by_rank or epoch > latest_by_rank[rank][0]:
                latest_by_rank[rank] = (epoch, path)

        merged_records = []
        for _, path in latest_by_rank.values():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    merged_records.extend(json.load(f))
            except (OSError, json.JSONDecodeError):
                continue
        if merged_records:
            dataset.update_hard_pool(
                merged_records,
                current_epoch=int(self.current_epoch),
            )

    def _log_qsmall_validation_batch(self, output_dict: dict, data_dict: dict):
        if "quat" not in output_dict or "trans" not in output_dict:
            return
        valid_parts = data_dict["points_per_part"] > 0
        part_to_case = torch.nonzero(valid_parts, as_tuple=False)[:, 0].long()
        n_parts = int(part_to_case.numel())
        if n_parts == 0:
            return

        gt_quat = data_dict["quat_gt"][valid_parts].float()
        gt_trans = data_dict["trans_gt"][valid_parts].float()
        pred_quat = output_dict["quat"].float()
        pred_trans = output_dict["trans"].float()
        part_ids = output_dict["part_batch_ids"].long()
        scales = data_dict["norm_scale"].float().reshape(-1)[part_to_case]

        rot_err = rot_geodesic_deg(pred_quat, gt_quat)
        trans_mm = torch.linalg.vector_norm(pred_trans - gt_trans, dim=-1) * scales
        point_tre = torch.linalg.vector_norm(
            output_dict["pred_coords"].float() - data_dict["pointclouds_gt"].float(),
            dim=-1,
        )
        tre_mm = self._fragment_mean(point_tre, part_ids, n_parts) * scales

        diameters = data_dict["fragment_diameter_mm"][valid_parts].float()
        anchor_parts = torch.zeros(n_parts, dtype=torch.bool, device=self.device)
        for case_id in range(int(valid_parts.shape[0])):
            local_parts = torch.nonzero(part_to_case == case_id, as_tuple=False).flatten()
            if local_parts.numel():
                anchor_parts[local_parts[0]] = True
        qsmall = (diameters <= self.qsmall_eval_threshold_mm) & (~anchor_parts)

        self.log(
            "val/all_rot_deg_official_scale",
            rot_err.mean(),
            on_epoch=True,
            sync_dist=True,
            batch_size=n_parts,
        )
        self.log(
            "val/all_trans_mm",
            trans_mm.mean(),
            on_epoch=True,
            sync_dist=True,
            batch_size=n_parts,
        )
        self.log(
            "val/all_tre_mm",
            tre_mm.mean(),
            on_epoch=True,
            sync_dist=True,
            batch_size=n_parts,
        )

        # Pelvis batches consistently contain Qsmall fragments. Keep the call
        # count identical on every rank and let Lightning reduce weighted means.
        q_count = int(qsmall.sum().item())
        q_denom = qsmall.sum().clamp_min(1).to(rot_err.dtype)
        q_batch_size = max(q_count, 1)
        q_metrics = {
            "val/qsmall_rot_deg": (rot_err * qsmall).sum() / q_denom,
            "val/qsmall_trans_mm": (trans_mm * qsmall).sum() / q_denom,
            "val/qsmall_tre_mm": (tre_mm * qsmall).sum() / q_denom,
            "val/qsmall_rot_gt30_rate": ((rot_err > 30.0) * qsmall).sum() / q_denom,
            "val/qsmall_rot_gt60_rate": ((rot_err > 60.0) * qsmall).sum() / q_denom,
        }
        for name, value in q_metrics.items():
            self.log(
                name,
                value,
                on_epoch=True,
                sync_dist=True,
                batch_size=q_batch_size,
            )

    def validation_step(self, data_dict: dict, batch_idx: int, dataloader_idx: int = 0):
        output_dict = self.forward(data_dict)
        loss_dict = self.loss(output_dict, data_dict)
        batch_size = data_dict["points_per_part"].shape[0]

        self.log("val/loss", loss_dict["loss"], on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        for key in (
            "loss_coord", "loss_coord_global", "loss_coord_fragment", "loss_centered",
            "loss_pair", "loss_covariance", "loss_horn_rot", "loss_horn_trans",
            "rotation_scale", "horn_rot_deg", "horn_trans_mm", "horn_observability",
            "loss_coord_qsmall", "qsmall_weight_mean", "qsmall_strength_mean",
            "loss_reliability", "loss_coord_reliability",
            "reliability_entropy", "reliability_effective_points",
            "reliability_coord_blend",
        ):
            if key in loss_dict:
                self.log(
                    f"val/{key}", loss_dict[key], on_epoch=True, sync_dist=True,
                    batch_size=batch_size,
                )

        if "quat" in output_dict and "trans" in output_dict:
            valid_mask = data_dict["points_per_part"] > 0
            gt_quat = data_dict["quat_gt"][valid_mask]
            gt_trans = data_dict["trans_gt"][valid_mask]
            pred_quat = output_dict["quat"]
            pred_trans = output_dict["trans"]

            with torch.autocast(device_type=self.device.type, enabled=False):
                rot_err = rot_geodesic_deg(pred_quat.float(), gt_quat.float())
                trans_err = torch.norm(pred_trans.float() - gt_trans.float(), dim=-1)

                if "pred_coords" in output_dict:
                    pred_assembled = output_dict["pred_coords"].float()
                else:
                    pred_assembled = apply_pose_to_points(
                        data_dict["pointclouds"].float(),
                        pred_trans.float(), pred_quat.float(),
                        output_dict["part_batch_ids"],
                    )
                tre = torch.sqrt(F.mse_loss(
                    pred_assembled, data_dict["pointclouds_gt"].float(), reduction="none"
                ).mean(dim=-1)).mean()

            n_parts = rot_err.shape[0]
            self.log("val/rot_err_deg", rot_err.mean(), on_epoch=True, sync_dist=True, batch_size=n_parts)
            self.log("val/trans_err", trans_err.mean(), on_epoch=True, sync_dist=True, batch_size=n_parts)
            self.log("val/tre", tre, on_epoch=True, sync_dist=True, batch_size=batch_size)

            self._log_qsmall_validation_batch(output_dict, data_dict)

        if self.debug_vis and batch_idx < 10:
            if self.output_type == "coords":
                pred_assemble = enforce_rigid_transform_svd(
                    src_points=data_dict["pointclouds"],
                    pred_points=output_dict["pred_coords"],
                    part_batch_ids=output_dict["part_batch_ids"]
                )
            else:
                pred_assemble = apply_pose_to_points(
                    data_dict["pointclouds"],
                    output_dict["trans"],
                    output_dict["quat"],
                    output_dict["part_batch_ids"]
                )

            output_results(
                pred_assemble=pred_assemble,
                data_dict=data_dict,
                output_dict=output_dict,
                save_path=self.save_vis_path_val,
                save_all=False,
                save_gt=True
            )

        return loss_dict["loss"]

    def configure_optimizers(self):
        """Use a short LR adaptation schedule that is independent of max_epochs."""
        # Exclude biases and 1D params (LayerNorm, etc.) from weight decay.
        grouped = {}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            no_decay = p.ndim <= 1 or name.endswith(".bias")
            if self.qsmall_finetune_enabled:
                if id(p) in self._qsmall_adapter_parameter_ids:
                    lr = self.qsmall_adapter_lr
                elif id(p) in self._qsmall_opened_parameter_ids:
                    lr = self.qsmall_opened_lr
                else:
                    lr = self.qsmall_base_lr
            else:
                lr = self.optimizer.keywords.get("lr", 1e-4)
            grouped.setdefault((lr, no_decay), []).append(p)

        wd = self.optimizer.keywords.get("weight_decay", 0.01)
        param_groups = []
        for (lr, no_decay), parameters in sorted(grouped.items(), key=lambda item: item[0]):
            param_groups.append({
                "params": parameters,
                "lr": lr,
                "weight_decay": 0.0 if no_decay else wd,
            })
        optimizer = self.optimizer(param_groups)

        total_steps = int(self.trainer.estimated_stepping_batches)
        max_epochs = max(1, int(self.trainer.max_epochs or 1))
        steps_per_epoch = max(1, int(math.ceil(total_steps / max_epochs)))
        warmup_steps = max(1, int(round(self.schedule_warmup_epochs * steps_per_epoch)))
        decay_end_step = max(warmup_steps + 1, int(round(self.schedule_decay_epochs * steps_per_epoch)))

        def lr_lambda(step):
            if step < warmup_steps:
                progress = step / warmup_steps
                return self.schedule_initial_lr_ratio + progress * (
                    1.0 - self.schedule_initial_lr_ratio
                )
            if step < decay_end_step:
                progress = (step - warmup_steps) / (decay_end_step - warmup_steps)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return self.schedule_min_lr_ratio + (
                    1.0 - self.schedule_min_lr_ratio
                ) * cosine
            return self.schedule_min_lr_ratio

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}
        }
