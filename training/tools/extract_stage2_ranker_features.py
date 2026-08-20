#!/usr/bin/env python3
"""Replay frozen Clinical states and score an arbitrary four-candidate pose pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


MAX_PARTS = 50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_npz(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def apply_pose(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return points @ pose[:3, :3].T + pose[:3, 3]


def build_tinit(centroid: np.ndarray, scale: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.eye(3, dtype=np.float32) / float(scale)
    transform[:3, 3] = -np.asarray(centroid).reshape(3) / float(scale)
    return transform


def reconstruct_state(fragment_data, bones, policy, iteration, anchor_index):
    current = np.asarray(policy["current_relative"][iteration], dtype=np.float64)
    if int(policy["selected"][iteration, anchor_index]) != 0:
        raise RuntimeError("Canonical anchor must select slot 0")
    if not np.allclose(current[anchor_index], np.eye(4), atol=1e-8):
        raise RuntimeError("Canonical state is not anchor-normalized")

    posed_coords = [
        apply_pose(points.astype(np.float64), pose).astype(np.float32)
        for (points, _), pose in zip(fragment_data, current)
    ]
    posed_normals = [
        (normals.astype(np.float64) @ pose[:3, :3].T).astype(np.float32)
        for (_, normals), pose in zip(fragment_data, current)
    ]
    all_points = np.concatenate(posed_coords, axis=0).astype(np.float32)
    all_normals = np.concatenate(posed_normals, axis=0).astype(np.float32)
    counts = np.asarray([len(points) for points in posed_coords], dtype=np.int32)
    diameters = np.asarray(policy["diameters_mm"][iteration], dtype=np.float32)

    canonical_candidates = np.asarray(
        policy["candidate_relative"][iteration], dtype=np.float64
    )
    candidate_steps = np.stack(
        [
            canonical_candidates[index] @ np.linalg.inv(current[index])
            for index in range(len(current))
        ]
    )
    identity = np.eye(3, dtype=np.float64)
    systems = []
    targets = []
    for part_index in range(len(current)):
        base_rotation = candidate_steps[part_index, 2, :3, :3]
        base_offset = candidate_steps[part_index, 2, :3, 3]
        for candidate_index, alpha in ((1, 0.5), (3, 1.25)):
            candidate = candidate_steps[part_index, candidate_index]
            systems.append(
                (identity - candidate[:3, :3])
                - alpha * (identity - base_rotation)
            )
            targets.append(candidate[:3, 3] - alpha * base_offset)
    centroid, _, _, _ = np.linalg.lstsq(
        np.concatenate(systems, axis=0), np.concatenate(targets, axis=0), rcond=None
    )
    centered = all_points.astype(np.float64) - centroid[None]
    scale = float(np.linalg.norm(centered, axis=1).max())
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("Invalid reconstructed scale")
    normalized = (centered / scale).astype(np.float32)
    return (
        current,
        normalized,
        all_normals,
        counts,
        np.asarray(bones, dtype=np.int64),
        diameters,
        np.asarray(centroid, dtype=np.float32),
        scale,
    )


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


def normalized_candidates(candidate_relative, current, centroid, scale):
    steps = np.stack(
        [
            np.asarray(candidate_relative[index], dtype=np.float64)
            @ np.linalg.inv(np.asarray(current[index], dtype=np.float64))
            for index in range(len(current))
        ]
    )
    initial = build_tinit(centroid, scale).astype(np.float64)
    initial_inverse = np.linalg.inv(initial)
    return np.einsum("ij,nkjl,lm->nkim", initial, steps, initial_inverse)


def make_batch(coords, normals, counts, bones, diameters, scale, device):
    padded_counts = np.zeros(MAX_PARTS, dtype=np.int32)
    padded_counts[: len(counts)] = counts
    padded_bones = np.zeros(MAX_PARTS, dtype=np.int64)
    padded_bones[: len(bones)] = bones
    padded_diameters = np.zeros(MAX_PARTS, dtype=np.float32)
    padded_diameters[: len(diameters)] = diameters
    return {
        "pointclouds": torch.from_numpy(coords).to(device),
        "pointclouds_normals": torch.from_numpy(normals).to(device),
        "points_per_part": torch.from_numpy(padded_counts).unsqueeze(0).to(device),
        "bonetype": torch.from_numpy(padded_bones).unsqueeze(0).to(device),
        "fragment_diameter_mm": torch.from_numpy(padded_diameters).unsqueeze(0).to(device),
        "norm_scale": torch.tensor([scale], dtype=torch.float32, device=device),
    }


def score_candidates(model, data_dict, matrices, source_alphas, self_control=False):
    device = data_dict["pointclouds"].device
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
    ):
        backbone = model.transformer_model(
            input_coords=data_dict["pointclouds"],
            input_normals=data_dict["pointclouds_normals"],
            points_per_part=data_dict["points_per_part"],
            return_features=True,
        )
        input_coords = data_dict["pointclouds"].float()
        pred_coords = backbone["pred_coords"].float()
        part_ids = backbone["part_batch_ids"].long()
        rotation = torch.from_numpy(matrices[..., :3, :3].astype(np.float32)).to(device)
        translation = torch.from_numpy(matrices[..., :3, 3].astype(np.float32)).to(device)
        quaternion = matrix_to_quaternion(rotation)
        candidate_points = torch.stack(
            [
                model._apply_part_pose(
                    input_coords, quaternion[:, candidate], translation[:, candidate], part_ids
                )
                for candidate in range(4)
            ],
            dim=1,
        )
        valid_parts = data_dict["points_per_part"] > 0
        part_to_case = torch.nonzero(valid_parts, as_tuple=False)[:, 0].long()
        original_alphas = model.candidate_alphas
        model.candidate_alphas = tuple(float(value) for value in source_alphas)
        try:
            geometry, bone = model._geometry_features(
                data_dict,
                candidate_points,
                quaternion,
                translation,
                part_ids,
                pred_coords,
                part_to_case,
            )
        finally:
            model.candidate_alphas = original_alphas

        num_parts = geometry.shape[0]
        part_feature = model.backbone_projection(backbone["part_features"].float())
        token = part_feature[:, None, :] + model.geometry_projection(geometry)
        token = token + model.bone_embedding(bone)[:, None, :]
        candidate_ids = torch.arange(4, device=device)
        token = token + model.candidate_embedding(candidate_ids)[None, :, :]
        padded = token.new_zeros(1, MAX_PARTS, 4, token.shape[-1])
        padded[0, :num_parts] = token
        valid = torch.zeros(1, MAX_PARTS, dtype=torch.bool, device=device)
        valid[0, :num_parts] = True
        slot_ids = torch.arange(MAX_PARTS, device=device).clamp_max(model.max_parts - 1)
        padded = padded + model.fragment_slot_embedding(slot_ids)[None, :, None, :]
        flat = padded.reshape(1, MAX_PARTS * 4, -1)
        padding_mask = (~valid[:, :, None].expand(-1, -1, 4)).reshape(1, -1)
        encoded = model.context_encoder(flat, src_key_padding_mask=padding_mask)
        encoded_valid = encoded.reshape(1, MAX_PARTS, 4, -1)[0, :num_parts]
        scores = model.score_head(encoded_valid).squeeze(-1)
        metric_prediction = torch.expm1(
            F.softplus(model.metric_head(encoded_valid)).float()
        ).clamp_min(0.0)
        severe_probability = torch.sigmoid(model.severe_head(encoded_valid).squeeze(-1).float())
        native = None
        if self_control:
            original_builder = model._build_candidates

            def fixed_builder(control_data, control_backbone):
                control_part_ids = control_backbone["part_batch_ids"].long()
                control_points = control_data["pointclouds"].float()
                fixed_points = torch.stack(
                    [
                        model._apply_part_pose(
                            control_points,
                            quaternion[:, candidate],
                            translation[:, candidate],
                            control_part_ids,
                        )
                        for candidate in range(4)
                    ],
                    dim=1,
                )
                return (
                    fixed_points,
                    quaternion,
                    translation,
                    control_part_ids,
                    control_backbone["pred_coords"].float(),
                )

            model._build_candidates = fixed_builder
            try:
                native = model(data_dict)
            finally:
                model._build_candidates = original_builder
    result = {
        "geometry": geometry.detach().float().cpu().numpy(),
        "encoded": encoded_valid.detach().float().cpu().numpy(),
        "part_feature": part_feature.detach().float().cpu().numpy(),
        "scores": scores.detach().float().cpu().numpy(),
        "metrics": metric_prediction.detach().float().cpu().numpy(),
        "severe": severe_probability.detach().float().cpu().numpy(),
        "candidate_quaternion": quaternion.detach().float().cpu().numpy(),
        "candidate_translation": translation.detach().float().cpu().numpy(),
    }
    if native is not None:
        native_metrics = torch.expm1(native["metric_prediction"].float()).clamp_min(0.0)
        native_severe = torch.sigmoid(native["severe_logits"].float())
        result["self_control"] = {
            "candidate_quaternion": float(
                (quaternion.float() - native["candidate_quat"].float()).abs().max()
            ),
            "candidate_translation": float(
                (translation.float() - native["candidate_trans"].float()).abs().max()
            ),
            "scores": float((scores.float() - native["scores"].float()).abs().max()),
            "metrics": float((metric_prediction - native_metrics).abs().max()),
            "severe": float((severe_probability - native_severe).abs().max()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-dir", type=Path, required=True)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--canonical-trace-dir", type=Path, required=True)
    parser.add_argument("--candidate-trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-alphas", type=float, nargs=4, default=(0.0, 0.5, 1.0, 1.25))
    parser.add_argument("--reference-score-trace-dir", type=Path)
    parser.add_argument("--self-control", action="store_true")
    parser.add_argument("--expected-cases", type=int, default=170)
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str(args.app_dir.resolve()))
    sys.path.insert(1, str(Path(__file__).resolve().parent))
    import process
    from area64_cache import discover_cache_cases, load_cached_fragments

    process.configure_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = process._load_model(args.model_checkpoint, device)
    cases = discover_cache_cases(args.cache_dir)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    elif len(cases) != args.expected_cases:
        raise RuntimeError(f"Expected {args.expected_cases} cache cases, got {len(cases)}")

    feature_dir = args.output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    max_score_error = 0.0
    max_metric_error = 0.0
    max_severe_error = 0.0
    self_control_max = {
        "candidate_quaternion": 0.0,
        "candidate_translation": 0.0,
        "scores": 0.0,
        "metrics": 0.0,
        "severe": 0.0,
    }
    records = []
    for case_index, case_dir in enumerate(cases, 1):
        started = time.time()
        fragment_data, bones, names = load_cached_fragments(case_dir)
        canonical_path = args.canonical_trace_dir / f"{case_dir.name}.npz"
        candidate_path = args.candidate_trace_dir / f"{case_dir.name}.npz"
        with np.load(canonical_path, allow_pickle=False) as canonical, np.load(
            candidate_path, allow_pickle=False
        ) as candidate:
            candidate_names = [str(value) for value in candidate["fragment_names"].tolist()]
            if names != candidate_names:
                raise RuntimeError(f"Fragment order mismatch for {case_dir.name}")
            if not np.array_equal(canonical["iteration"], candidate["iteration"]):
                raise RuntimeError(f"Iteration mismatch for {case_dir.name}")
            if not np.allclose(
                canonical["current_relative"], candidate["current_relative"], atol=1e-8
            ):
                raise RuntimeError(f"Visited state mismatch for {case_dir.name}")
            iterations = []
            for iteration in range(len(canonical["iteration"])):
                (
                    current,
                    coords,
                    normals,
                    counts,
                    state_bones,
                    diameters,
                    centroid,
                    scale,
                ) = reconstruct_state(
                    fragment_data, bones, canonical, iteration, names.index("1")
                )
                matrices = normalized_candidates(
                    candidate["candidate_relative"][iteration],
                    current,
                    centroid,
                    scale,
                )
                batch = make_batch(
                    coords, normals, counts, state_bones, diameters, scale, device
                )
                scored = score_candidates(
                    model,
                    batch,
                    matrices,
                    args.source_alphas,
                    self_control=args.self_control,
                )
                control = scored.pop("self_control", None)
                if control is not None:
                    for key, value in control.items():
                        self_control_max[key] = max(self_control_max[key], value)
                scored.update(
                    {
                        "iteration": np.asarray(canonical["iteration"][iteration]),
                        "diameters_mm": diameters,
                        "norm_scale": np.asarray(scale, dtype=np.float32),
                    }
                )
                iterations.append(scored)

            payload = {
                "fragment_names": np.asarray(names),
                "bone_types": np.asarray(bones, dtype=np.int64),
                "source_alphas": np.asarray(args.source_alphas, dtype=np.float32),
            }
            for key in iterations[0]:
                payload[key] = np.stack([row[key] for row in iterations])
            atomic_npz(feature_dir / f"{case_dir.name}.npz", **payload)

        if args.reference_score_trace_dir is not None:
            with np.load(
                args.reference_score_trace_dir / f"{case_dir.name}.npz", allow_pickle=False
            ) as reference, np.load(
                feature_dir / f"{case_dir.name}.npz", allow_pickle=False
            ) as features:
                max_score_error = max(
                    max_score_error,
                    float(np.max(np.abs(features["scores"] - reference["scores"]))),
                )
                max_metric_error = max(
                    max_metric_error,
                    float(np.max(np.abs(features["metrics"] - reference["metrics"]))),
                )
                max_severe_error = max(
                    max_severe_error,
                    float(np.max(np.abs(features["severe"] - reference["severe"]))),
                )
        elapsed = time.time() - started
        records.append({"case": case_dir.name, "elapsed_s": elapsed})
        print(
            f"ranker_features progress={case_index}/{len(cases)} "
            f"case={case_dir.name} elapsed_s={elapsed:.3f}",
            flush=True,
        )

    manifest = {
        "status": "complete",
        "cases": len(records),
        "model_checkpoint": str(args.model_checkpoint),
        "model_sha256": sha256(args.model_checkpoint),
        "canonical_trace_dir": str(args.canonical_trace_dir),
        "candidate_trace_dir": str(args.candidate_trace_dir),
        "source_alphas": list(args.source_alphas),
        "reference_score_trace_dir": (
            str(args.reference_score_trace_dir)
            if args.reference_score_trace_dir is not None
            else None
        ),
        "control_max_abs": {
            "scores": max_score_error,
            "metrics": max_metric_error,
            "severe": max_severe_error,
        },
        "self_control_max_abs": self_control_max,
        "records": records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "status",
                    "cases",
                    "control_max_abs",
                    "self_control_max_abs",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
