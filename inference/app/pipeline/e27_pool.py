"""Shared model and geometry helpers for the e27 simulation candidate pool."""

from __future__ import annotations

import hashlib
import random
from functools import partial
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from models.AssemblyNet.fragment_confidence import FragmentConfidenceModule
from models.AssemblyNet.enhanced_transformer import AssemblyTransformer
from models.AssemblyNet.utils import (
    extract_poses_from_coords,
    quaternion_to_matrix,
)


MAX_PARTS = 50
NUM_CANDIDATES = 4
METRIC_SCALES = np.asarray([2.53, 2.76, 2.67, 3.37], dtype=np.float32)


def deterministic_seed(*parts: object) -> int:
    payload = ":".join(str(value) for value in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def seed_sample(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def move_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _base_transformer(**overrides) -> AssemblyTransformer:
    config = {
        "max_parts": MAX_PARTS,
        "embed_dim": 384,
        "num_layers": 12,
        "num_heads": 8,
        "dropout_rate": 0.0,
        "output_type": "coords",
        "fragment_context_enabled": False,
        "fragment_context_start_layer": 8,
        "fragment_context_dim": 192,
        "fragment_context_heads": 4,
        "point_reliability_enabled": False,
    }
    config.update(overrides)
    return AssemblyTransformer(**config)


def load_e3_ranker(checkpoint: str | Path, device: torch.device) -> FragmentConfidenceModule:
    checkpoint = str(checkpoint)
    model = FragmentConfidenceModule(
        transformer_model=_base_transformer(),
        optimizer=partial(torch.optim.AdamW, lr=2e-5, weight_decay=0.01),
        checkpoint=checkpoint,
        ranking_checkpoint=checkpoint,
        candidate_alphas=(0.0, 0.5, 1.0, 1.25),
        axis_offsets_deg=(),
        hidden_dim=256,
        num_context_layers=3,
        num_heads=8,
        dropout=0.1,
        max_parts=MAX_PARTS,
        utility_scales=tuple(float(value) for value in METRIC_SCALES),
        pair_margin=0.05,
        ce_weight=1.0,
        pair_weight=0.5,
        regression_weight=0.25,
        severe_weight=0.2,
        chamfer_points=64,
        c2_metric_margin=0.05,
        c2_severe_tolerance=0.0,
        c2_severe_threshold=0.2,
    )
    model.to(device).eval()
    return model


def load_coordinate_backbone(
    checkpoint: str | Path,
    device: torch.device,
    *,
    point_reliability: bool,
) -> AssemblyTransformer:
    model = _base_transformer(
        fragment_context_enabled=True,
        point_reliability_enabled=point_reliability,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload)
    prefix = "transformer_model."
    stripped = {
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if not stripped:
        raise RuntimeError(f"No transformer_model tensors in {checkpoint}")
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    allowed_missing = {
        key for key in missing if key.startswith("point_reliability_head.")
    }
    real_missing = sorted(set(missing) - allowed_missing)
    if real_missing or unexpected:
        raise RuntimeError(
            f"Coordinate checkpoint mismatch for {checkpoint}: "
            f"missing={real_missing[:8]} unexpected={unexpected[:8]}"
        )
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def split_fragments(sample: dict, key: str) -> list[np.ndarray]:
    counts = np.asarray(sample["points_per_part"], dtype=np.int64)
    counts = counts[counts > 0]
    values = np.asarray(sample[key], dtype=np.float32)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    return [values[offsets[i] : offsets[i + 1]].copy() for i in range(len(counts))]


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def apply_rotation(normals: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return normals @ transform[:3, :3].T


def build_tinit(centroid: np.ndarray, scale: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.eye(3, dtype=np.float64) / float(scale)
    transform[:3, 3] = -np.asarray(centroid, dtype=np.float64) / float(scale)
    return transform


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    flat = matrix.reshape(-1, 3, 3)
    m00, m01, m02 = flat[:, 0, 0], flat[:, 0, 1], flat[:, 0, 2]
    m10, m11, m12 = flat[:, 1, 0], flat[:, 1, 1], flat[:, 1, 2]
    m20, m21, m22 = flat[:, 2, 0], flat[:, 2, 1], flat[:, 2, 2]
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
    quaternion = candidates[torch.arange(flat.shape[0], device=flat.device), best]
    quaternion = F.normalize(quaternion, dim=-1)
    quaternion = torch.where(quaternion[:, :1] < 0.0, -quaternion, quaternion)
    return quaternion.reshape(matrix.shape[:-2] + (4,))


def _make_inference_batch(
    coords: np.ndarray,
    normals: np.ndarray,
    counts: np.ndarray,
    bones: np.ndarray,
    diameters_mm: np.ndarray,
    norm_scale_mm: float,
    device: torch.device,
) -> dict:
    padded_counts = np.zeros(MAX_PARTS, dtype=np.int32)
    padded_counts[: len(counts)] = counts
    padded_bones = np.zeros(MAX_PARTS, dtype=np.int64)
    padded_bones[: len(bones)] = bones
    padded_diameters = np.zeros(MAX_PARTS, dtype=np.float32)
    padded_diameters[: len(diameters_mm)] = diameters_mm
    return {
        "pointclouds": torch.from_numpy(coords.astype(np.float32)).to(device),
        "pointclouds_normals": torch.from_numpy(normals.astype(np.float32)).to(device),
        "points_per_part": torch.from_numpy(padded_counts).unsqueeze(0).to(device),
        "bonetype": torch.from_numpy(padded_bones).unsqueeze(0).to(device),
        "fragment_diameter_mm": torch.from_numpy(padded_diameters).unsqueeze(0).to(device),
        "norm_scale": torch.tensor([norm_scale_mm], dtype=torch.float32, device=device),
    }


def _e3_c2_selected(model: FragmentConfidenceModule, output: dict, points_per_part: torch.Tensor) -> torch.Tensor:
    selected = model._c2_select(output, points_per_part).clone()
    if selected.numel():
        selected[0] = 0
    return selected


@torch.inference_mode()
def rollout_final_pose(
    model: torch.nn.Module,
    fragment_points: Sequence[np.ndarray],
    fragment_normals: Sequence[np.ndarray],
    bones: np.ndarray,
    diameters_mm: np.ndarray,
    physical_scale_mm: float,
    device: torch.device,
    *,
    ranker: bool,
    max_iters: int = 10,
    convergence_mm: float = 2.0,
) -> tuple[np.ndarray, int]:
    current_points = [value.copy() for value in fragment_points]
    current_normals = [value.copy() for value in fragment_normals]
    # Match the frozen e26 inference path exactly.  In particular, e26 keeps
    # all iterative transforms in float32 and centers by the full-cloud
    # bounding-box centre (not by the arithmetic mean).
    cumulative = np.repeat(np.eye(4, dtype=np.float32)[None], len(current_points), axis=0)
    amp_enabled = device.type == "cuda"

    for iteration in range(max_iters):
        concatenated = np.concatenate(current_points, axis=0).astype(np.float32)
        centroid = (concatenated.max(axis=0) + concatenated.min(axis=0)) / 2.0
        centered = concatenated - centroid[None]
        scale = np.linalg.norm(centered, axis=1).max()
        if not np.isfinite(scale) or scale <= 1e-8:
            raise RuntimeError("Invalid rollout normalization scale")
        normalized = centered.copy()
        normalized /= scale
        normals = np.concatenate(current_normals, axis=0).astype(np.float32)
        counts = np.asarray([len(value) for value in current_points], dtype=np.int32)
        batch = _make_inference_batch(
            normalized,
            normals,
            counts,
            bones,
            diameters_mm,
            scale * physical_scale_mm,
            device,
        )
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            if ranker:
                output = model(batch)
                selected = _e3_c2_selected(model, output, batch["points_per_part"])
                rows = torch.arange(selected.numel(), device=device)
                quaternion = output["candidate_quat"][rows, selected].float()
                translation = output["candidate_trans"][rows, selected].float()
            else:
                output = model(
                    input_coords=batch["pointclouds"],
                    input_normals=batch["pointclouds_normals"],
                    points_per_part=batch["points_per_part"],
                )
                quaternion, translation = extract_poses_from_coords(
                    batch["pointclouds"].float(),
                    output["pred_coords"].float(),
                    output["part_batch_ids"].long(),
                )
                quaternion = quaternion.float()
                translation = translation.float()

        quaternion_cpu = quaternion.float().detach().cpu()
        rotation = quaternion_to_matrix(quaternion_cpu).numpy().astype(np.float32)
        translation_np = translation.float().detach().cpu().numpy().astype(np.float32)
        tinit = np.eye(4, dtype=np.float32)
        tinit[:3, :3] = np.diag(
            1.0 / np.asarray([scale, scale, scale], dtype=np.float32)
        )
        tinit[:3, 3] = -centroid / scale
        tinit_inv = np.linalg.inv(tinit)
        max_translation_mm = 0.0
        for part_index in range(len(current_points)):
            normalized_step = np.eye(4, dtype=np.float32)
            normalized_step[:3, :3] = rotation[part_index]
            normalized_step[:3, 3] = translation_np[part_index]
            world_step = tinit_inv @ normalized_step @ tinit
            cumulative[part_index] = world_step @ cumulative[part_index]
            points_h = np.concatenate(
                (
                    current_points[part_index],
                    np.ones((len(current_points[part_index]), 1), dtype=np.float32),
                ),
                axis=1,
            )
            current_points[part_index] = (world_step @ points_h.T).T[:, :3]
            current_normals[part_index] = (
                world_step[:3, :3] @ current_normals[part_index].T
            ).T
            max_translation_mm = max(
                max_translation_mm,
                float(np.linalg.norm(world_step[:3, 3])) * physical_scale_mm,
            )
        if max_translation_mm < convergence_mm:
            return cumulative, iteration + 1
    return cumulative, max_iters


def anchor_normalize(transforms: np.ndarray) -> np.ndarray:
    anchor_inverse = np.linalg.inv(transforms[0])
    normalized = np.einsum("ij,njk->nik", anchor_inverse, transforms)
    normalized[0] = np.eye(4, dtype=np.float64)
    return normalized


@torch.inference_mode()
def build_cached_ranker_record(
    e3: FragmentConfidenceModule,
    sample: dict,
    final_poses: Sequence[np.ndarray],
    source_alphas: Sequence[float],
    device: torch.device,
) -> dict[str, np.ndarray]:
    input_parts = split_fragments(sample, "pointclouds")
    normal_parts = split_fragments(sample, "pointclouds_normals")
    gt_parts = split_fragments(sample, "pointclouds_gt")
    counts = np.asarray([len(value) for value in input_parts], dtype=np.int32)
    num_parts = len(counts)
    bones = np.asarray(sample["bonetype"][:num_parts], dtype=np.int64)
    diameters = np.asarray(sample["fragment_diameter_mm"][:num_parts], dtype=np.float32)
    physical_scale_mm = float(np.asarray(sample["norm_scale"]))

    final_poses = [anchor_normalize(np.asarray(value, dtype=np.float64)) for value in final_poses]
    baseline = final_poses[0]
    current_parts = [
        apply_transform(points, baseline[index]).astype(np.float32)
        for index, points in enumerate(input_parts)
    ]
    current_normals = [
        apply_rotation(normals, baseline[index]).astype(np.float32)
        for index, normals in enumerate(normal_parts)
    ]
    current = np.concatenate(current_parts, axis=0).astype(np.float64)
    centroid = current.mean(axis=0)
    centered = current - centroid[None]
    canonical_scale = float(np.linalg.norm(centered, axis=1).max())
    if not np.isfinite(canonical_scale) or canonical_scale <= 1e-8:
        raise RuntimeError("Invalid canonical normalization scale")
    canonical_points = (centered / canonical_scale).astype(np.float32)
    canonical_normals = np.concatenate(current_normals, axis=0).astype(np.float32)
    canonical_gt = (
        (np.concatenate(gt_parts, axis=0).astype(np.float64) - centroid[None])
        / canonical_scale
    ).astype(np.float32)

    tinit = build_tinit(centroid, canonical_scale)
    tinit_inv = np.linalg.inv(tinit)
    candidate_matrices = []
    for source in final_poses:
        steps = np.stack(
            [source[index] @ np.linalg.inv(baseline[index]) for index in range(num_parts)],
            axis=0,
        )
        candidate_matrices.append(
            np.einsum("ij,njk,kl->nil", tinit, steps, tinit_inv)
        )
    candidate_matrices = np.stack(candidate_matrices, axis=1)
    candidate_matrices[0] = np.eye(4, dtype=np.float64)[None]

    batch = _make_inference_batch(
        canonical_points,
        canonical_normals,
        counts,
        bones,
        diameters,
        canonical_scale * physical_scale_mm,
        device,
    )
    batch["pointclouds_gt"] = torch.from_numpy(canonical_gt).to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        backbone = e3.transformer_model(
            input_coords=batch["pointclouds"],
            input_normals=batch["pointclouds_normals"],
            points_per_part=batch["points_per_part"],
            return_features=True,
        )
    part_ids = backbone["part_batch_ids"].long()
    gt_quat, gt_trans = extract_poses_from_coords(
        batch["pointclouds"].float(), batch["pointclouds_gt"].float(), part_ids
    )
    padded_quat = torch.zeros(1, MAX_PARTS, 4, dtype=torch.float32, device=device)
    padded_quat[..., 0] = 1.0
    padded_trans = torch.zeros(1, MAX_PARTS, 3, dtype=torch.float32, device=device)
    padded_quat[0, :num_parts] = gt_quat.float()
    padded_trans[0, :num_parts] = gt_trans.float()
    batch["quat_gt"] = padded_quat
    batch["trans_gt"] = padded_trans

    matrices = torch.from_numpy(candidate_matrices.astype(np.float32)).to(device)
    candidate_quat = matrix_to_quaternion(matrices[..., :3, :3])
    candidate_trans = matrices[..., :3, 3]
    candidate_points = torch.stack(
        [
            e3._apply_part_pose(
                batch["pointclouds"].float(),
                candidate_quat[:, candidate],
                candidate_trans[:, candidate],
                part_ids,
            )
            for candidate in range(NUM_CANDIDATES)
        ],
        dim=1,
    )
    valid_parts = batch["points_per_part"] > 0
    part_to_case = torch.nonzero(valid_parts, as_tuple=False)[:, 0].long()
    original_alphas = e3.candidate_alphas
    e3.candidate_alphas = tuple(float(value) for value in source_alphas)
    try:
        geometry, returned_bones = e3._geometry_features(
            batch,
            candidate_points,
            candidate_quat,
            candidate_trans,
            part_ids,
            backbone["pred_coords"].float(),
            part_to_case,
        )
    finally:
        e3.candidate_alphas = original_alphas
    metrics, utility, severe, _ = e3._candidate_targets(
        batch,
        candidate_points,
        candidate_quat,
        candidate_trans,
        part_ids,
    )
    if not torch.equal(returned_bones.cpu(), torch.from_numpy(bones)):
        raise RuntimeError("Bone labels changed while constructing cached features")
    loss_mask = np.ones(num_parts, dtype=bool)
    loss_mask[0] = False
    return {
        "part_features": backbone["part_features"].float().cpu().numpy().astype(np.float16),
        "geometry": geometry.float().cpu().numpy().astype(np.float16),
        "bone": bones.astype(np.int16),
        "metrics": metrics.float().cpu().numpy().astype(np.float32),
        "utility": utility.float().cpu().numpy().astype(np.float32),
        "severe": severe.float().cpu().numpy().astype(np.float32),
        "candidate_quat": candidate_quat.float().cpu().numpy().astype(np.float32),
        "candidate_trans": candidate_trans.float().cpu().numpy().astype(np.float32),
        "loss_mask": loss_mask,
        "diameters_mm": diameters.astype(np.float32),
        "source_alphas": np.asarray(source_alphas, dtype=np.float32),
        "rollout_iters": np.asarray([], dtype=np.int16),
    }


def summarize_oracle(records: Iterable[dict[str, np.ndarray]]) -> dict:
    baseline_metrics = []
    oracle_metrics = []
    baseline_utility = []
    oracle_utility = []
    selected = []
    for record in records:
        mask = np.asarray(record["loss_mask"], dtype=bool)
        metrics = np.asarray(record["metrics"], dtype=np.float64)[mask]
        utility = np.asarray(record["utility"], dtype=np.float64)[mask]
        oracle = utility.argmin(axis=1)
        rows = np.arange(len(oracle))
        baseline_metrics.append(metrics[:, 0])
        oracle_metrics.append(metrics[rows, oracle])
        baseline_utility.append(utility[:, 0])
        oracle_utility.append(utility[rows, oracle])
        selected.append(oracle)
    if not baseline_metrics:
        return {}
    baseline_metrics_np = np.concatenate(baseline_metrics)
    oracle_metrics_np = np.concatenate(oracle_metrics)
    selected_np = np.concatenate(selected)
    return {
        "fragments": int(len(selected_np)),
        "baseline_metrics": baseline_metrics_np.mean(axis=0).tolist(),
        "oracle_metrics": oracle_metrics_np.mean(axis=0).tolist(),
        "metric_improvements": (
            baseline_metrics_np.mean(axis=0) - oracle_metrics_np.mean(axis=0)
        ).tolist(),
        "baseline_utility": float(np.concatenate(baseline_utility).mean()),
        "oracle_utility": float(np.concatenate(oracle_utility).mean()),
        "oracle_slot_rates": [float(np.mean(selected_np == index)) for index in range(NUM_CANDIDATES)],
    }
