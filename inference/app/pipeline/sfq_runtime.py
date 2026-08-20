#!/usr/bin/env python3
"""Evaluate SFQ four-candidate proposals under a fixed Area64 replay policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from datasets.transform import recenter_pc, rescale_pc
from models.AssemblyNet.sfq import SmallFragmentQueryModule, _matrix_to_quaternion
from models.AssemblyNet.transformer import AssemblyTransformer
from models.AssemblyNet.utils import quaternion_to_matrix
from .area64_cache import discover_cache_cases, load_cached_fragments


CANDIDATE_ALPHAS = (0.0, 0.5, 1.0, 1.25)
MAX_PARTS = 50


def configure_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "42")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_tinit(centroid: np.ndarray, scale: float) -> np.ndarray:
    # Match the verified submission's float32 normalization transform exactly.
    # Promoting this matrix to float64 changes the first pose by only microns,
    # but the iterative backbone can amplify that perturbation in later rounds.
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.eye(3, dtype=np.float32) / scale
    transform[:3, 3] = -np.asarray(centroid).reshape(3) / scale
    return transform


def pose_to_original(pose: np.ndarray, centroid: np.ndarray, scale: float) -> np.ndarray:
    initial = build_tinit(centroid, scale)
    return np.linalg.inv(initial) @ pose @ initial


def apply_pose(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return points @ pose[:3, :3].T + pose[:3, 3]


def scaled_quaternion(quaternion: torch.Tensor, alpha: float) -> torch.Tensor:
    quaternion = F.normalize(quaternion.float(), dim=-1)
    quaternion = torch.where(quaternion[:, :1] < 0.0, -quaternion, quaternion)
    half_angle = torch.acos(quaternion[:, :1].clamp(-1.0 + 1e-7, 1.0 - 1e-7))
    sin_half = torch.sin(half_angle)
    fallback = torch.zeros_like(quaternion[:, 1:])
    fallback[:, 0] = 1.0
    axis = torch.where(
        sin_half > 1e-6,
        quaternion[:, 1:] / sin_half.clamp_min(1e-6),
        fallback,
    )
    scaled_half = half_angle * float(alpha)
    result = torch.cat((torch.cos(scaled_half), axis * torch.sin(scaled_half)), dim=-1)
    return F.normalize(result, dim=-1)


def load_model(args, device: torch.device) -> SmallFragmentQueryModule:
    backbone = AssemblyTransformer(
        embed_dim=384,
        num_layers=12,
        num_heads=8,
        dropout_rate=0.0,
        max_parts=MAX_PARTS,
        output_type="coords",
    )
    model = SmallFragmentQueryModule(
        transformer_model=backbone,
        optimizer=partial(torch.optim.AdamW, lr=2e-4, weight_decay=0.01),
        checkpoint=str(args.base_checkpoint),
        variant=args.variant,
        hidden_dim=384,
        max_parts=MAX_PARTS,
        num_heads=8,
        context_layers=2,
        patches_per_fragment=16,
        local_points=128,
        max_rotation_deg=args.max_rotation_deg,
        max_translation_mm=args.max_translation_mm,
    )
    if args.sfq_checkpoint:
        checkpoint = torch.load(args.sfq_checkpoint, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
        model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def run_model(
    model, coords, normals, counts, bones, diameters, scale, device,
    base_quaternion=None, base_translation=None,
):
    padded_counts = np.zeros(MAX_PARTS, dtype=np.int32)
    padded_counts[: len(counts)] = counts
    padded_bones = np.zeros(MAX_PARTS, dtype=np.int64)
    padded_bones[: len(bones)] = bones
    padded_diameters = np.zeros(MAX_PARTS, dtype=np.float32)
    padded_diameters[: len(diameters)] = diameters
    batch = {
        "pointclouds": torch.from_numpy(coords).to(device),
        "pointclouds_normals": torch.from_numpy(normals).to(device),
        "points_per_part": torch.from_numpy(padded_counts).unsqueeze(0).to(device),
        "bonetype": torch.from_numpy(padded_bones).unsqueeze(0).to(device),
        "fragment_diameter_mm": torch.from_numpy(padded_diameters).unsqueeze(0).to(device),
        "norm_scale": torch.tensor([scale], dtype=torch.float32, device=device),
    }
    if base_quaternion is not None or base_translation is not None:
        if base_quaternion is None or base_translation is None:
            raise RuntimeError("Incomplete fixed-state base pose override")
        batch["base_quaternion_override"] = torch.from_numpy(
            np.asarray(base_quaternion, dtype=np.float32)
        ).to(device)
        batch["base_translation_override"] = torch.from_numpy(
            np.asarray(base_translation, dtype=np.float32)
        ).to(device)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
    ):
        return model(batch)


def candidate_steps(output, centroid, scale, canonical_steps_original=None):
    quaternion = output["quat"].float()
    translation = output["trans"].float()
    if canonical_steps_original is not None:
        residual_rotation = output["residual_rotation"].float()
        residual_translation = output["residual_translation_mm"].float()
        identity = torch.eye(3, device=residual_rotation.device)[None]
        reference_rotation = output["reference_base_rotation"].float()
        reference_translation = output["reference_base_translation"].float()
        base_correction_rotation = (
            output["base_rotation"].float() @ reference_rotation.transpose(-1, -2)
        )
        base_correction_translation = output["base_translation"].float() - torch.bmm(
            base_correction_rotation, reference_translation[:, :, None]
        ).squeeze(-1)
        if (
            float((residual_rotation - identity).abs().max()) <= 1e-8
            and float(residual_translation.abs().max()) <= 1e-8
            and float((base_correction_rotation - identity).abs().max()) <= 1e-8
            and float(base_correction_translation.abs().max()) <= 1e-8
        ):
            return np.asarray(canonical_steps_original, dtype=np.float64).copy()
        base_quaternion = _matrix_to_quaternion(reference_rotation)
        base_translation = reference_translation
        initial = build_tinit(centroid, scale).astype(np.float64)
        initial_inverse = np.linalg.inv(initial)
    steps = []
    for candidate_index, alpha in enumerate(CANDIDATE_ALPHAS):
        quat = scaled_quaternion(quaternion, alpha)
        rotation = quaternion_to_matrix(quat).detach().cpu().numpy()
        trans = (translation * alpha).detach().cpu().numpy()
        if canonical_steps_original is not None:
            base_quat = scaled_quaternion(base_quaternion, alpha)
            base_rotation = quaternion_to_matrix(base_quat).detach().cpu().numpy()
            base_trans = (base_translation * alpha).detach().cpu().numpy()
        local = []
        for part_index in range(rotation.shape[0]):
            normalized = np.eye(4, dtype=np.float64)
            normalized[:3, :3] = rotation[part_index]
            normalized[:3, 3] = trans[part_index]
            if canonical_steps_original is not None:
                base_normalized = np.eye(4, dtype=np.float64)
                base_normalized[:3, :3] = base_rotation[part_index]
                base_normalized[:3, 3] = base_trans[part_index]
                canonical_normalized = (
                    initial
                    @ canonical_steps_original[part_index, candidate_index]
                    @ initial_inverse
                )
                normalized = (
                    normalized @ np.linalg.inv(base_normalized) @ canonical_normalized
                )
            local.append(pose_to_original(normalized, centroid, scale))
        steps.append(np.stack(local))
    return np.stack(steps, axis=1)


def normalize_candidates(candidate_cumulative, selected_next, anchor_index):
    anchor_inverse = np.linalg.inv(selected_next[anchor_index])
    return np.einsum("ij,nkjl->nkil", anchor_inverse, candidate_cumulative)


def policy_value(policy, key, iteration, fallback):
    if key not in policy:
        return fallback
    return np.asarray(policy[key][iteration])


def reconstruct_canonical_state(
    original_coords, original_normals, bones, policy, iteration, anchor_index
):
    """Build a deterministic model input around an original canonical trace state.

    The canonical Area64 run retained poses and candidates but not its sampled
    surface points.  We therefore transform the shared deterministic Area64
    cache to the exact canonical visited pose, while injecting the canonical
    alpha=1 base step.  A zero SFQ residual then preserves the original four
    candidate pool independently of a repeated backbone/Kabsch SVD.
    """
    current = np.asarray(policy["current_relative"][iteration], dtype=np.float64)
    selected = np.asarray(policy["selected"][iteration], dtype=np.int64)
    if int(selected[anchor_index]) != 0:
        raise RuntimeError("Canonical reconstruction requires candidate 0 for the anchor")
    if not np.allclose(current[anchor_index], np.eye(4), atol=1e-8):
        raise RuntimeError("Canonical policy state is not anchor-normalized")

    posed_coords = [
        apply_pose(points.astype(np.float64), pose).astype(np.float32)
        for points, pose in zip(original_coords, current)
    ]
    posed_normals = [
        (normals.astype(np.float64) @ pose[:3, :3].T).astype(np.float32)
        for normals, pose in zip(original_normals, current)
    ]
    all_points = np.concatenate(posed_coords, axis=0).astype(np.float32)
    all_normals = np.concatenate(posed_normals, axis=0).astype(np.float32)
    counts = np.asarray([len(item) for item in posed_coords], dtype=np.int32)
    fallback_diameters = np.asarray(
        [np.linalg.norm(item.max(axis=0) - item.min(axis=0)) for item in posed_coords],
        dtype=np.float32,
    )
    diameters = policy_value(
        policy, "diameters_mm", iteration, fallback_diameters
    ).astype(np.float32)

    canonical_candidates = np.asarray(
        policy["candidate_relative"][iteration], dtype=np.float64
    )
    if canonical_candidates.shape[:2] != (len(current), len(CANDIDATE_ALPHAS)):
        raise RuntimeError(
            "Unexpected canonical candidate shape: "
            f"{canonical_candidates.shape}"
        )
    candidate_steps_original = np.stack(
        [
            canonical_candidates[index] @ np.linalg.inv(current[index])
            for index in range(len(current))
        ]
    )
    base_original = candidate_steps_original[:, 2]

    # Fractional candidate translations identify the original run's shared
    # bounding-box centre even though its randomly sampled points were not saved:
    #   t(a) - a*t(1) = ((I-R(a)) - a*(I-R(1))) * centre.
    systems = []
    targets = []
    identity = np.eye(3, dtype=np.float64)
    for part_index in range(len(current)):
        base_rotation = base_original[part_index, :3, :3]
        base_offset = base_original[part_index, :3, 3]
        for candidate_index in (1, 3):
            alpha = CANDIDATE_ALPHAS[candidate_index]
            candidate = candidate_steps_original[part_index, candidate_index]
            systems.append(
                (identity - candidate[:3, :3])
                - alpha * (identity - base_rotation)
            )
            targets.append(candidate[:3, 3] - alpha * base_offset)
    system = np.concatenate(systems, axis=0)
    target = np.concatenate(targets, axis=0)
    centroid, _, _, _ = np.linalg.lstsq(system, target, rcond=None)
    centered = all_points.astype(np.float64) - centroid[None]
    scale = float(np.linalg.norm(centered, axis=1).max())
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError(f"Invalid reconstructed canonical scale: {scale}")
    normalized = (centered / scale).astype(np.float32)
    initial = build_tinit(centroid, scale).astype(np.float64)
    initial_inverse = np.linalg.inv(initial)
    base_normalized = np.stack(
        [initial @ pose @ initial_inverse for pose in base_original]
    )
    rotations = torch.from_numpy(base_normalized[:, :3, :3].astype(np.float32))
    base_quaternion = _matrix_to_quaternion(rotations).cpu().numpy().astype(np.float32)
    base_translation = base_normalized[:, :3, 3].astype(np.float32)
    return (
        current,
        normalized,
        all_normals,
        counts,
        np.asarray(bones, dtype=np.int64),
        diameters,
        np.asarray(centroid, dtype=np.float32),
        float(scale),
        base_quaternion,
        base_translation,
        candidate_steps_original,
    )


def write_trace(path: Path, names, bones, records):
    payload = {
        "fragment_names": np.asarray(names),
        "bone_types": np.asarray(bones, dtype=np.int64),
        "iteration": np.asarray([row["iteration"] for row in records], dtype=np.int64),
        "policy_source": np.asarray("baseline_area64_ranker_c2_replay"),
    }
    array_keys = (
        "current_relative", "candidate_relative", "selected_relative", "selected",
        "raw_selected", "scores", "metrics", "utility", "severe", "utility_gain",
        "severe_delta", "fragment_accepted", "diameters_mm",
    )
    scalar_keys = ("patient_gain", "patient_accepted", "max_translation_mm")
    for key in array_keys:
        payload[key] = np.stack([row[key] for row in records])
    for key in scalar_keys:
        payload[key] = np.asarray([row[key] for row in records])
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_prediction(path: Path, names, cumulative):
    anchor = names.index("1")
    anchor_inverse = np.linalg.inv(cumulative[anchor])
    result = [
        {
            "fragment_id": name,
            "transformation": (anchor_inverse @ pose).tolist(),
        }
        for name, pose in zip(names, cumulative)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def run_case(model, fragment_data, bones, names, policy, states, device, state_mode):
    count = len(fragment_data)
    policy_names = [str(value) for value in policy["fragment_names"].tolist()]
    if names != policy_names:
        raise RuntimeError(f"Fragment order mismatch: cache={names}, policy={policy_names}")
    cumulative = np.repeat(np.eye(4, dtype=np.float64)[None], count, axis=0)
    original_coords = [item[0].copy() for item in fragment_data]
    original_normals = [item[1].copy() for item in fragment_data]
    coords = [item.copy() for item in original_coords]
    normals = [item.copy() for item in original_normals]
    anchor_index = names.index("1")
    records = []
    for iteration in range(len(policy["iteration"])):
        base_quaternion = None
        base_translation = None
        canonical_steps_original = None
        if state_mode == "fixed_policy":
            if states is None:
                raise RuntimeError("fixed_policy mode requires captured model inputs")
            selected_policy = np.asarray(policy["selected"][iteration], dtype=np.int64)
            if int(selected_policy[anchor_index]) != 0:
                raise RuntimeError(
                    "Fixed-state replay requires candidate 0 for the anchor"
                )
            current = np.asarray(
                policy["current_relative"][iteration], dtype=np.float64
            )
            if not np.allclose(current[anchor_index], np.eye(4), atol=1e-8):
                raise RuntimeError("Policy state is not anchor-normalized")
            cumulative = current.copy()
            normalized = np.asarray(states["pointclouds"][iteration], dtype=np.float32)
            all_normals = np.asarray(states["normals"][iteration], dtype=np.float32)
            counts = np.asarray(states["points_per_part"][iteration], dtype=np.int32)
            state_bones = np.asarray(states["bone_types"][iteration], dtype=np.int64)
            if not np.array_equal(state_bones, bones):
                raise RuntimeError("Captured bone order differs from cache")
            diameters = np.asarray(states["diameters_mm"][iteration], dtype=np.float32)
            centroid = np.asarray(states["centroid"][iteration], dtype=np.float32)
            scale = float(np.asarray(states["norm_scale"][iteration]).reshape(-1)[0])
            if "base_quaternion" in states and "base_translation" in states:
                base_quaternion = np.asarray(
                    states["base_quaternion"][iteration], dtype=np.float32
                )
                base_translation = np.asarray(
                    states["base_translation"][iteration], dtype=np.float32
                )
        elif state_mode == "canonical_reconstruct":
            (
                current,
                normalized,
                all_normals,
                counts,
                state_bones,
                diameters,
                centroid,
                scale,
                base_quaternion,
                base_translation,
                canonical_steps_original,
            ) = reconstruct_canonical_state(
                original_coords,
                original_normals,
                bones,
                policy,
                iteration,
                anchor_index,
            )
            if not np.array_equal(state_bones, bones):
                raise RuntimeError("Canonical reconstruction bone order differs from cache")
            cumulative = current.copy()
        else:
            current = cumulative.copy()
            all_points = np.concatenate(coords, axis=0).astype(np.float32)
            normalized, centroid = recenter_pc(all_points)
            normalized, scale = rescale_pc(normalized)
            all_normals = np.concatenate(normals, axis=0).astype(np.float32)
            counts = np.asarray([len(item) for item in coords], dtype=np.int32)
            diameters = np.asarray(
                [np.linalg.norm(item.max(axis=0) - item.min(axis=0)) for item in coords],
                dtype=np.float32,
            )
        output = run_model(
            model, normalized, all_normals, counts, bones, diameters, scale, device,
            base_quaternion=base_quaternion,
            base_translation=base_translation,
        )
        steps = candidate_steps(
            output,
            centroid,
            scale,
            canonical_steps_original=canonical_steps_original,
        )
        selected = np.asarray(policy["selected"][iteration], dtype=np.int64)
        if selected.shape != (count,) or np.any((selected < 0) | (selected >= len(CANDIDATE_ALPHAS))):
            raise RuntimeError(f"Invalid replay selection at iteration {iteration}")
        candidate_cumulative = np.einsum("nkij,njl->nkil", steps, current)
        selected_next = np.stack(
            [candidate_cumulative[index, selected[index]] for index in range(count)]
        )
        candidate_relative = normalize_candidates(
            candidate_cumulative, selected_next, anchor_index
        )
        current_relative = np.einsum(
            "ij,njl->nil", np.linalg.inv(current[anchor_index]), current
        )
        selected_relative = np.einsum(
            "ij,njl->nil", np.linalg.inv(selected_next[anchor_index]), selected_next
        )
        max_shift = 0.0
        for index in range(count):
            step = steps[index, selected[index]]
            max_shift = max(max_shift, float(np.linalg.norm(step[:3, 3])))
            if state_mode == "recurrent_replay":
                cumulative[index] = step @ cumulative[index]
                # The verified process keeps recurrent geometry in float64 and
                # casts only the concatenated model input each iteration.
                coords[index] = apply_pose(coords[index], step)
                normals[index] = normals[index] @ step[:3, :3].T
        if state_mode in ("fixed_policy", "canonical_reconstruct"):
            cumulative = selected_next
        candidate_count = len(CANDIDATE_ALPHAS)
        zero_scores = np.zeros((count, candidate_count), dtype=np.float64)
        zero_metrics = np.zeros((count, candidate_count, 4), dtype=np.float64)
        zero_candidate = np.zeros((count, candidate_count), dtype=np.float64)
        raw_selected = policy_value(policy, "raw_selected", iteration, selected).astype(np.int64)
        records.append(
            {
                "iteration": int(policy["iteration"][iteration]),
                "current_relative": current_relative,
                "candidate_relative": candidate_relative,
                "selected_relative": selected_relative,
                "selected": selected,
                "raw_selected": raw_selected,
                "scores": policy_value(policy, "scores", iteration, zero_scores),
                "metrics": policy_value(policy, "metrics", iteration, zero_metrics),
                "utility": policy_value(policy, "utility", iteration, zero_candidate),
                "severe": policy_value(policy, "severe", iteration, zero_candidate),
                "utility_gain": policy_value(policy, "utility_gain", iteration, np.zeros(count)),
                "severe_delta": policy_value(policy, "severe_delta", iteration, np.zeros(count)),
                "fragment_accepted": policy_value(
                    policy, "fragment_accepted", iteration, selected != 0
                ),
                "diameters_mm": diameters,
                "patient_gain": float(policy_value(policy, "patient_gain", iteration, 0.0)),
                "patient_accepted": int(policy_value(policy, "patient_accepted", iteration, 1)),
                "max_translation_mm": max_shift,
            }
        )
    return cumulative, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--policy-trace-dir", type=Path, required=True)
    parser.add_argument("--policy-state-dir", type=Path)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--sfq-checkpoint", type=Path)
    parser.add_argument(
        "--variant",
        choices=("r0", "f", "x", "fx", "l", "xl", "fxg", "fxh", "fxgh"),
        default="r0",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=170)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-rotation-deg", type=float, default=30.0)
    parser.add_argument("--max-translation-mm", type=float, default=30.0)
    parser.add_argument(
        "--state-mode",
        choices=("fixed_policy", "canonical_reconstruct", "recurrent_replay"),
        default="fixed_policy",
    )
    args = parser.parse_args()

    configure_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    trace_dir = args.output_dir / "traces"
    prediction_dir = args.output_dir / "cases"
    trace_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    cases = discover_cache_cases(args.cache_dir)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    records = []
    for index, case_dir in enumerate(cases, 1):
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        policy_path = args.policy_trace_dir / f"{case_dir.name}.npz"
        if not policy_path.is_file():
            raise FileNotFoundError(f"Missing replay policy: {policy_path}")
        fragments, bones, names = load_cached_fragments(case_dir)
        started = time.time()
        with np.load(policy_path, allow_pickle=False) as policy_payload:
            policy = {key: policy_payload[key] for key in policy_payload.files}
        states = None
        if args.state_mode == "fixed_policy":
            if args.policy_state_dir is None:
                raise ValueError("--policy-state-dir is required for fixed_policy mode")
            state_path = args.policy_state_dir / f"{case_dir.name}.npz"
            if not state_path.is_file():
                raise FileNotFoundError(f"Missing captured state input: {state_path}")
            with np.load(state_path, allow_pickle=False) as state_payload:
                states = {key: state_payload[key] for key in state_payload.files}
            state_names = [str(value) for value in states["fragment_names"].tolist()]
            if state_names != names:
                raise RuntimeError("Captured state fragment order differs from cache")
        cumulative, trace_records = run_case(
            model, fragments, bones, names, policy, states, device, args.state_mode
        )
        write_prediction(prediction_dir / f"{case_dir.name}.json", names, cumulative)
        write_trace(trace_dir / f"{case_dir.name}.npz", names, bones, trace_records)
        elapsed = time.time() - started
        records.append({"case": case_dir.name, "elapsed_s": elapsed})
        print(
            f"sfq_replay variant={args.variant} progress={index}/{len(cases)} "
            f"case={case_dir.name} elapsed_s={elapsed:.2f}",
            flush=True,
        )
    if args.max_cases <= 0 and len(records) != args.expected_cases:
        raise RuntimeError(f"Expected {args.expected_cases} cases, got {len(records)}")
    manifest = {
        "status": "complete",
        "protocol": "Area64 fixed-cache, baseline C2 visited-state candidate evaluation",
        "state_mode": args.state_mode,
        "variant": args.variant,
        "cases": len(records),
        "base_checkpoint": str(args.base_checkpoint),
        "base_sha256": sha256(args.base_checkpoint),
        "sfq_checkpoint": str(args.sfq_checkpoint) if args.sfq_checkpoint else None,
        "sfq_sha256": sha256(args.sfq_checkpoint) if args.sfq_checkpoint else None,
        "candidate_alphas": CANDIDATE_ALPHAS,
        "records": records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
