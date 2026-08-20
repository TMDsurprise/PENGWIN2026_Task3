"""Deterministic e27 four-backbone conservative inference for PENGWIN Task 3."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import trimesh

from datasets.fractures.eval import load_obj_fragments
from datasets.transform import recenter_pc, rescale_pc
from models.AssemblyNet import AssemblyTransformer, FragmentConfidenceModule
from models.AssemblyNet.sfq import SmallFragmentQueryModule
from models.AssemblyNet.utils import quaternion_to_matrix
from pipeline import delta_calibration, e27_pool, ranker_features, sfq_runtime
from pipeline import stage2_recurrent


INPUT_PATH = Path(
    os.environ.get("INPUT_PATH", "/input/peripelvic-fracture-fragments-meshes.obj")
)
OUTPUT_PATH = Path(
    os.environ.get("OUTPUT_PATH", "/output/reduction-poses-matrices.json")
)
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/opt/ml/model"))

PRIMARY_NAME = "e25_e3_original_c2_state.ckpt"
SFQ_F_NAME = "e25_sfq_f_state.ckpt"
SFQ_FX_NAME = "e25_sfq_fx_state.ckpt"
B1_NAME = "e26_b1_epoch1_state.ckpt"
B2_NAME = "e26_b2_epoch2_state.ckpt"
B3_NAME = "e26_b3_epoch6_state.ckpt"
E27_NAME = "e27_exact_lr2e5_seed42_state.ckpt"
CALIBRATION_NAME = "e25_stage2_exact_full_calibration.npz"
NPOINTS = 5000
MIN_POINTS = 64
MAX_PARTS = 50
MAX_ITERS = 10
CONVERGENCE_THRESHOLD_MM = 2.0
METRIC_SCALES = (2.53, 2.76, 2.67, 3.37)
STAGE2_SOURCE_ALPHAS = (0.0, 1.0, 1.25, 1.25)
E27_SOURCE_ALPHAS = (0.0, 1.0, 1.0, 1.0)

# Original E2 C2 gate retained by the Clinical170 five-fold OOF comparison.
METRIC_GATE_MARGIN = 0.05
SEVERE_GATE_TOLERANCE = 0.0
SEVERE_GATE_THRESHOLD = 0.2

# Conservative deadbands prevent tiny backend differences from changing a
# discrete candidate decision and then diverging over later iterations.
SCORE_TIE_EPS = float(os.environ.get("E2_PRIMARY_SCORE_TIE_EPS", "1e-5"))
DECISION_EPS = float(os.environ.get("E2_PRIMARY_DECISION_EPS", "0"))
LEGACY_DECISION = os.environ.get("E2_PRIMARY_LEGACY_DECISION", "0") == "1"
USE_AUTOCAST = os.environ.get(
    "E2_PRIMARY_AUTOCAST", "1"
) == "1"


def _new_model():
    transformer = AssemblyTransformer(
        embed_dim=384,
        num_layers=12,
        num_heads=8,
        dropout_rate=0.0,
        max_parts=MAX_PARTS,
        output_type="coords",
    )
    return FragmentConfidenceModule(
        transformer_model=transformer,
        optimizer=partial(torch.optim.AdamW, lr=2e-4, weight_decay=0.01),
        checkpoint=None,
        candidate_alphas=(0.0, 0.5, 1.0, 1.25),
        axis_offsets_deg=(),
        hidden_dim=256,
        num_context_layers=3,
        num_heads=8,
        dropout=0.1,
        max_parts=MAX_PARTS,
    )


def _load_model(path: Path, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(f"Missing model checkpoint: {path}")
    model = _new_model()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"Invalid tensor state in {path}")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def _new_sfq_model(variant: str):
    transformer = AssemblyTransformer(
        embed_dim=384,
        num_layers=12,
        num_heads=8,
        dropout_rate=0.0,
        max_parts=MAX_PARTS,
        output_type="coords",
    )
    return SmallFragmentQueryModule(
        transformer_model=transformer,
        optimizer=partial(torch.optim.AdamW, lr=2e-4, weight_decay=0.01),
        checkpoint=None,
        variant=variant,
        hidden_dim=384,
        max_parts=MAX_PARTS,
        num_heads=8,
        context_layers=2,
        patches_per_fragment=16,
        local_points=128,
        max_rotation_deg=30.0,
        max_translation_mm=30.0,
    )


def _load_sfq_model(path: Path, variant: str, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(f"Missing SFQ checkpoint: {path}")
    model = _new_sfq_model(variant)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"Invalid SFQ tensor state in {path}")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _load_calibration(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing Stage-2 calibration: {path}")
    calibration = stage2_recurrent.load_calibration(path)
    calibration["c2_margin"] = 0.15
    calibration["c2_threshold"] = 0.20
    calibration["c2_tolerance"] = 0.0
    return calibration


def init_models():
    configure_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    primary = _load_model(MODEL_DIR / PRIMARY_NAME, device)
    models = {
        "device": device,
        "primary": primary,
        "sfq_f": _load_sfq_model(MODEL_DIR / SFQ_F_NAME, "f", device),
        "sfq_fx": _load_sfq_model(MODEL_DIR / SFQ_FX_NAME, "fx", device),
        "b1": e27_pool.load_coordinate_backbone(
            MODEL_DIR / B1_NAME, device, point_reliability=False
        ),
        "b2": e27_pool.load_coordinate_backbone(
            MODEL_DIR / B2_NAME, device, point_reliability=True
        ),
        "b3": e27_pool.load_coordinate_backbone(
            MODEL_DIR / B3_NAME, device, point_reliability=True
        ),
        "selector": _load_model(MODEL_DIR / E27_NAME, device),
        "calibration": _load_calibration(MODEL_DIR / CALIBRATION_NAME),
    }
    print("models_loaded=primary,sfq_f,sfq_fx,b1,b2,b3,e27_selector", flush=True)
    return models


def configure_determinism():
    if LEGACY_DECISION:
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=True)
    force_math_sdpa = os.environ.get("E2_FORCE_MATH_SDPA", "0") == "1"
    if torch.cuda.is_available() and force_math_sdpa:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def fragment_names_and_bones(meshdict):
    names = []
    bones = []
    for bone_idx, bone in enumerate(("SA", "LI", "RI")):
        for name in sorted(meshdict.get(bone, {}).keys()):
            names.append(str(name))
            bones.append(bone_idx)
    return names, np.asarray(bones, dtype=np.int64)


def _sample_surface(mesh: trimesh.Trimesh, count: int):
    points, face_indices = trimesh.sample.sample_surface_even(mesh, count)
    normals = mesh.face_normals[face_indices]
    if len(points) < count:
        if len(points) == 0:
            raise RuntimeError("Surface sampling returned zero points")
        fill = np.random.choice(len(points), count - len(points), replace=True)
        points = np.concatenate((points, points[fill]), axis=0)
        normals = np.concatenate((normals, normals[fill]), axis=0)
    return (
        np.asarray(points[:count], dtype=np.float32),
        np.asarray(normals[:count], dtype=np.float32),
    )


def _area64_sub_sample(meshdict):
    sampled = {}
    for bone, fragments in meshdict.items():
        sampled[bone] = {}
        names = list(fragments)
        if not names:
            continue
        areas = np.asarray(
            [max(float(fragments[name]["mesh"].area), 1e-3) for name in names],
            dtype=np.float64,
        )
        raw = (areas / areas.sum() * NPOINTS).astype(np.int64)
        allocation = np.maximum(raw, MIN_POINTS)
        allocation[int(np.argmax(areas))] += NPOINTS - int(allocation.sum())
        if int(allocation.min()) < MIN_POINTS or int(allocation.sum()) != NPOINTS:
            raise RuntimeError(
                f"Invalid Area64 allocation for {bone}: {allocation.tolist()}"
            )
        for name, count in zip(names, allocation.tolist()):
            coords, normals = _sample_surface(fragments[name]["mesh"], count)
            sampled[bone][str(name)] = {"coords": coords, "normals": normals}
    return sampled


def build_fragment_data(meshdict):
    sampled = _area64_sub_sample(meshdict)
    fragment_data = []
    for bone in ("SA", "LI", "RI"):
        for name in sorted(sampled.get(bone, {}).keys()):
            coords = sampled[bone][name]["coords"]
            normals = sampled[bone][name]["normals"]
            if len(coords):
                fragment_data.append((coords, normals))
    return fragment_data


def build_initial_transform(centroid, scale):
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.eye(3, dtype=np.float32) / scale
    transform[:3, 3] = -np.asarray(centroid).reshape(3) / scale
    return transform


def transform_pose_to_original_space(transform, centroid, scale):
    initial = build_initial_transform(centroid, scale)
    return np.linalg.inv(initial) @ transform @ initial


def apply_transform(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def apply_rotation(normals, transform):
    return normals @ transform[:3, :3].T


def run_inference_pass(
    model,
    points,
    normals,
    points_per_part,
    bone_types,
    diameters_mm,
    norm_scale,
    device,
):
    padded_counts = np.zeros(MAX_PARTS, dtype=np.int32)
    padded_counts[: len(points_per_part)] = points_per_part
    padded_bones = np.zeros(MAX_PARTS, dtype=np.int64)
    padded_bones[: len(bone_types)] = bone_types
    padded_diameters = np.zeros(MAX_PARTS, dtype=np.float32)
    padded_diameters[: len(diameters_mm)] = diameters_mm
    batch = {
        "pointclouds": torch.from_numpy(points).to(device),
        "pointclouds_normals": torch.from_numpy(normals).to(device),
        "points_per_part": torch.from_numpy(padded_counts).unsqueeze(0).to(device),
        "bonetype": torch.from_numpy(padded_bones).unsqueeze(0).to(device),
        "fragment_diameter_mm": torch.from_numpy(padded_diameters).unsqueeze(0).to(device),
        "norm_scale": torch.tensor([norm_scale], dtype=torch.float32, device=device),
    }
    use_amp = device.type == "cuda" and USE_AUTOCAST
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=use_amp,
    ):
        return model(batch)


def predicted_quality(output):
    metrics = torch.expm1(output["metric_prediction"].float()).clamp_min(0.0)
    scales = metrics.new_tensor(METRIC_SCALES)
    utility = (metrics / scales).mean(dim=-1)
    severe = torch.sigmoid(output["severe_logits"].float())
    return metrics, utility, severe


def stable_argmax(scores):
    scores = scores.detach().to(device="cpu", dtype=torch.float64)
    maximum = scores.max(dim=1, keepdim=True).values
    tied = scores >= maximum - SCORE_TIE_EPS
    candidate_ids = torch.arange(scores.shape[1], dtype=torch.long)[None, :]
    sentinel = torch.full_like(candidate_ids, scores.shape[1])
    return torch.where(tied, candidate_ids, sentinel).min(dim=1).values


def select_c2_candidates(output, return_details=False):
    metrics, utility, severe = predicted_quality(output)
    if LEGACY_DECISION:
        raw_selected = output["scores"].argmax(dim=1).detach().cpu()
    else:
        raw_selected = stable_argmax(output["scores"])
    metrics = metrics.detach().to(device="cpu", dtype=torch.float64)
    utility = utility.detach().to(device="cpu", dtype=torch.float64)
    severe = severe.detach().to(device="cpu", dtype=torch.float64)
    rows = torch.arange(raw_selected.numel())
    proposed_utility = utility[rows, raw_selected]
    current_utility = utility[:, 0]
    proposed_severe = severe[rows, raw_selected]
    current_severe = severe[:, 0]

    utility_gain = current_utility - proposed_utility
    severe_delta = proposed_severe - current_severe
    if LEGACY_DECISION:
        improves = proposed_utility + METRIC_GATE_MARGIN < current_utility
        severe_safe = (
            (proposed_severe <= current_severe + SEVERE_GATE_TOLERANCE)
            & (
                (proposed_severe <= SEVERE_GATE_THRESHOLD)
                | (proposed_severe < current_severe)
            )
        )
    else:
        improves = utility_gain > METRIC_GATE_MARGIN + DECISION_EPS
        not_worse = severe_delta <= SEVERE_GATE_TOLERANCE + DECISION_EPS
        low_risk = proposed_severe <= SEVERE_GATE_THRESHOLD + DECISION_EPS
        meaningfully_better = severe_delta < -DECISION_EPS
        severe_safe = (
            not_worse & (low_risk | meaningfully_better)
        )
    accepted = (raw_selected == 0) | (improves & severe_safe)
    selected = torch.where(accepted, raw_selected, torch.zeros_like(raw_selected))

    gated_utility = utility[rows, selected]
    patient_gain = current_utility.mean() - gated_utility.mean()
    if LEGACY_DECISION:
        patient_accepted = bool(gated_utility.mean() < current_utility.mean())
    else:
        patient_accepted = bool(patient_gain > DECISION_EPS)
    if not patient_accepted:
        selected = torch.zeros_like(selected)
    if not return_details:
        return selected
    return selected, {
        "raw_selected": raw_selected,
        "metrics": metrics,
        "utility": utility,
        "severe": severe,
        "utility_gain": utility_gain,
        "severe_delta": severe_delta,
        "fragment_accepted": accepted,
        "patient_gain": patient_gain,
        "patient_accepted": patient_accepted,
        "scores": output["scores"].detach().to(device="cpu", dtype=torch.float64),
    }


def candidate_steps_in_original_space(output, centroid, scale):
    quaternions = output["candidate_quat"].float().detach().cpu()
    translations = output["candidate_trans"].float().detach().cpu()
    rotations = quaternion_to_matrix(quaternions.reshape(-1, 4)).numpy().reshape(
        quaternions.shape[0], quaternions.shape[1], 3, 3
    )
    steps = np.repeat(
        np.eye(4, dtype=np.float64)[None, None],
        repeats=quaternions.shape[0] * quaternions.shape[1],
        axis=0,
    ).reshape(quaternions.shape[0], quaternions.shape[1], 4, 4)
    for part in range(quaternions.shape[0]):
        for candidate in range(quaternions.shape[1]):
            normalized_pose = np.eye(4, dtype=np.float64)
            normalized_pose[:3, :3] = rotations[part, candidate]
            normalized_pose[:3, 3] = translations[part, candidate].numpy()
            steps[part, candidate] = transform_pose_to_original_space(
                normalized_pose, centroid, scale
            )
    return steps


def normalize_candidate_poses(candidate_cumulative, selected_next, anchor_index):
    anchor_inverse = np.linalg.inv(selected_next[anchor_index])
    return np.einsum("ij,nkjl->nkil", anchor_inverse, candidate_cumulative)


def reduce_fragments(models, fragment_data, bone_types, fragment_names):
    device = models["device"]
    count = len(fragment_data)
    cumulative = [np.eye(4, dtype=np.float64) for _ in range(count)]
    coords = [item[0].copy() for item in fragment_data]
    normals = [item[1].copy() for item in fragment_data]
    anchor_index = fragment_names.index("1")
    trace_records = []

    for iteration in range(MAX_ITERS):
        all_points = np.concatenate(coords, axis=0).astype(np.float32)
        normalized, centroid = recenter_pc(all_points)
        normalized, scale = rescale_pc(normalized)
        all_normals = np.concatenate(normals, axis=0).astype(np.float32)
        counts = np.asarray([len(item) for item in coords], dtype=np.int32)
        diameters = np.asarray(
            [np.linalg.norm(item.max(axis=0) - item.min(axis=0)) for item in coords],
            dtype=np.float32,
        )
        primary = run_inference_pass(
            models["primary"], normalized, all_normals, counts, bone_types,
            diameters, scale, device
        )
        selected, decision = select_c2_candidates(primary, return_details=True)
        candidate_steps = candidate_steps_in_original_space(primary, centroid, scale)
        current = np.stack(cumulative)
        candidate_cumulative = np.einsum(
            "nkij,njl->nkil", candidate_steps, current
        )
        selected_array = selected.numpy()
        selected_next = np.stack(
            [candidate_cumulative[index, selected_array[index]] for index in range(count)]
        )
        candidate_relative = normalize_candidate_poses(
            candidate_cumulative, selected_next, anchor_index
        )
        current_anchor_inverse = np.linalg.inv(current[anchor_index])
        current_relative = np.einsum("ij,njl->nil", current_anchor_inverse, current)
        selected_anchor_inverse = np.linalg.inv(selected_next[anchor_index])
        selected_relative = np.einsum(
            "ij,njl->nil", selected_anchor_inverse, selected_next
        )

        max_shift = 0.0
        for index in range(count):
            step = candidate_steps[index, selected_array[index]]
            cumulative[index] = step @ cumulative[index]
            coords[index] = apply_transform(coords[index], step)
            normals[index] = apply_rotation(normals[index], step)
            max_shift = max(max_shift, float(np.linalg.norm(step[:3, 3])))

        trace_records.append(
            {
                "iteration": iteration + 1,
                "current_relative": current_relative,
                "candidate_relative": candidate_relative,
                "selected_relative": selected_relative,
                "selected": selected_array,
                "raw_selected": decision["raw_selected"].numpy(),
                "scores": decision["scores"].numpy(),
                "metrics": decision["metrics"].numpy(),
                "utility": decision["utility"].numpy(),
                "severe": decision["severe"].numpy(),
                "utility_gain": decision["utility_gain"].numpy(),
                "severe_delta": decision["severe_delta"].numpy(),
                "fragment_accepted": decision["fragment_accepted"].numpy(),
                "patient_gain": float(decision["patient_gain"]),
                "patient_accepted": int(decision["patient_accepted"]),
                "diameters_mm": diameters,
                "max_translation_mm": max_shift,
            }
        )
        print(
            f"iteration={iteration + 1} max_translation_mm={max_shift:.6f} "
            f"selected_nonzero={int((selected != 0).sum())} "
            f"patient_accepted={decision['patient_accepted']} "
            f"patient_gain={float(decision['patient_gain']):.8f}",
            flush=True,
        )
        if max_shift < CONVERGENCE_THRESHOLD_MM or bool((selected == 0).all()):
            break
    return cumulative, trace_records


def anchor_normalize(transforms, anchor_index):
    transforms = np.asarray(transforms, dtype=np.float64)
    anchor_inverse = np.linalg.inv(transforms[anchor_index])
    normalized = np.einsum("ij,njk->nik", anchor_inverse, transforms)
    normalized[anchor_index] = np.eye(4, dtype=np.float64)
    return normalized


def stage2_helpers():
    return {
        "process": sys.modules[__name__],
        "make_batch": ranker_features.make_batch,
        "score_candidates": ranker_features.score_candidates,
        "feature_block": delta_calibration.feature_block,
        "run_model": sfq_runtime.run_model,
        "sfq_candidate_steps": sfq_runtime.candidate_steps,
    }


def run_canonical_baseline(models, fragment_data, bone_types, fragment_names):
    return stage2_recurrent.run_case(
        models["primary"],
        models["sfq_f"],
        models["sfq_fx"],
        fragment_data,
        bone_types,
        fragment_names,
        models["device"],
        stage2_helpers(),
        "baseline",
        "original",
        models["calibration"],
        MAX_ITERS,
        CONVERGENCE_THRESHOLD_MM,
        STAGE2_SOURCE_ALPHAS,
    )


def run_exact_full_candidate(models, fragment_data, bone_types, fragment_names):
    cumulative, records = stage2_recurrent.run_case(
        models["primary"],
        models["sfq_f"],
        models["sfq_fx"],
        fragment_data,
        bone_types,
        fragment_names,
        models["device"],
        stage2_helpers(),
        "safety",
        "rescue",
        models["calibration"],
        MAX_ITERS,
        CONVERGENCE_THRESHOLD_MM,
        STAGE2_SOURCE_ALPHAS,
    )
    return anchor_normalize(cumulative, fragment_names.index("1")), records


def run_direct_candidate(model, fragment_data, bone_types, device, anchor_index):
    points = [item[0] for item in fragment_data]
    normals = [item[1] for item in fragment_data]
    diameters = np.asarray(
        [np.linalg.norm(item.max(axis=0) - item.min(axis=0)) for item in points],
        dtype=np.float32,
    )
    cumulative, iterations = e27_pool.rollout_final_pose(
        model,
        points,
        normals,
        bone_types,
        diameters,
        1.0,
        device,
        ranker=False,
        max_iters=MAX_ITERS,
        convergence_mm=CONVERGENCE_THRESHOLD_MM,
    )
    return anchor_normalize(cumulative, anchor_index), iterations


def stack_baseline_policy(records):
    if not records:
        raise RuntimeError("The canonical baseline produced no iteration records")
    keys = ("current_relative", "candidate_relative", "selected", "diameters_mm")
    return {key: np.stack([record[key] for record in records]) for key in keys}


def apply_e27_original_c2(scored):
    scores = np.asarray(scored["scores"], dtype=np.float64)
    metrics = np.asarray(scored["metrics"], dtype=np.float64)
    severe = np.asarray(scored["severe"], dtype=np.float64)
    scales = np.asarray(METRIC_SCALES, dtype=np.float64)
    utility = (metrics / scales[None, None]).mean(axis=-1)

    maximum = scores.max(axis=-1, keepdims=True)
    raw = (scores >= maximum - SCORE_TIE_EPS).argmax(axis=-1)
    rows = np.arange(len(raw))
    proposed_utility = utility[rows, raw]
    current_utility = utility[:, 0]
    proposed_severe = severe[rows, raw]
    current_severe = severe[:, 0]
    improves = proposed_utility + METRIC_GATE_MARGIN < current_utility
    severe_safe = (
        proposed_severe <= current_severe + SEVERE_GATE_TOLERANCE
    ) & (
        (proposed_severe <= SEVERE_GATE_THRESHOLD)
        | (proposed_severe < current_severe)
    )
    accepted = (raw == 0) | (improves & severe_safe)
    selected = np.where(accepted, raw, 0)
    patient_accepted = bool(
        utility[rows, selected].mean() < current_utility.mean()
    )
    if not patient_accepted:
        selected.fill(0)
        accepted.fill(False)
    return {
        "raw": raw,
        "selected": selected,
        "scores": scores,
        "metrics": metrics,
        "utility": utility,
        "severe": severe,
        "fragment_accepted": accepted,
        "patient_accepted": patient_accepted,
        "patient_gain": float(
            current_utility.mean() - utility[rows, selected].mean()
        ),
    }


def select_final_candidate(
    models,
    fragment_data,
    bone_types,
    fragment_names,
    baseline_records,
    candidates,
):
    anchor_index = fragment_names.index("1")
    policy = stack_baseline_policy(baseline_records)
    iteration = len(baseline_records) - 1
    (
        current,
        coords,
        normals,
        counts,
        state_bones,
        diameters,
        centroid,
        scale,
    ) = ranker_features.reconstruct_state(
        fragment_data, bone_types, policy, iteration, anchor_index
    )
    candidate_relative = np.stack(candidates, axis=1).astype(np.float64)
    matrices = ranker_features.normalized_candidates(
        candidate_relative, current, centroid, scale
    ).astype(np.float32)
    batch = ranker_features.make_batch(
        coords,
        normals,
        counts,
        state_bones,
        diameters,
        scale,
        models["device"],
    )
    scored = ranker_features.score_candidates(
        models["selector"],
        batch,
        matrices,
        E27_SOURCE_ALPHAS,
        self_control=False,
    )
    decision = apply_e27_original_c2(scored)
    selected = decision["selected"]
    final = np.stack(
        [candidate_relative[index, selected[index]] for index in range(len(selected))]
    )
    final = anchor_normalize(final, anchor_index)
    return final, decision, current, candidate_relative


def write_e27_trace(
    fragment_names,
    bone_types,
    baseline_records,
    exact_full_records,
    current,
    candidate_relative,
    decision,
    direct_iterations,
):
    trace_dir = os.environ.get("E27_TRACE_DIR")
    if not trace_dir:
        return
    output_dir = Path(trace_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_id = os.environ.get("E27_CASE_ID", INPUT_PATH.parent.name)
    payload = {
        "fragment_names": np.asarray(fragment_names),
        "bone_types": np.asarray(bone_types, dtype=np.int64),
        "canonical_current_relative": np.asarray(current, dtype=np.float64),
        "candidate_relative": np.asarray(candidate_relative, dtype=np.float64),
        "raw_selected": np.asarray(decision["raw"], dtype=np.int64),
        "selected": np.asarray(decision["selected"], dtype=np.int64),
        "scores": np.asarray(decision["scores"], dtype=np.float32),
        "metrics": np.asarray(decision["metrics"], dtype=np.float32),
        "utility": np.asarray(decision["utility"], dtype=np.float32),
        "severe": np.asarray(decision["severe"], dtype=np.float32),
        "fragment_accepted": np.asarray(
            decision["fragment_accepted"], dtype=np.int8
        ),
        "patient_accepted": np.asarray(
            int(decision["patient_accepted"]), dtype=np.int8
        ),
        "patient_gain": np.asarray(decision["patient_gain"], dtype=np.float64),
        "baseline_iterations": np.asarray(len(baseline_records), dtype=np.int64),
        "exact_full_iterations": np.asarray(len(exact_full_records), dtype=np.int64),
        "direct_iterations": np.asarray(direct_iterations, dtype=np.int64),
    }
    path = output_dir / f"{case_id}.npz"
    with tempfile.NamedTemporaryFile(
        dir=output_dir, prefix=f".{case_id}-", suffix=".npz", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_and_write(fragment_names, transforms):
    if len(fragment_names) != len(transforms):
        raise RuntimeError("Fragment/output count mismatch")
    if len(set(fragment_names)) != len(fragment_names):
        raise RuntimeError("Duplicate fragment IDs")
    results = []
    for fragment_id, transform in zip(fragment_names, transforms):
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise RuntimeError(f"Invalid transform for fragment {fragment_id}")
        if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-5):
            raise RuntimeError(f"Invalid homogeneous row for fragment {fragment_id}")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
            raise RuntimeError(f"Non-orthogonal rotation for fragment {fragment_id}")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-3):
            raise RuntimeError(f"Invalid rotation determinant for fragment {fragment_id}")
        results.append(
            {
                "fragment_id": str(fragment_id),
                "transformation": transform.astype(float).tolist(),
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=OUTPUT_PATH.parent,
        prefix=".reduction-poses-matrices-",
        suffix=".tmp",
        delete=False,
    ) as output_file:
        json.dump(results, output_file, indent=2)
        output_file.flush()
        os.fsync(output_file.fileno())
        temporary = Path(output_file.name)
    os.replace(temporary, OUTPUT_PATH)


def write_trace(fragment_names, bone_types, records):
    trace_dir = os.environ.get("E2_PRIMARY_TRACE_DIR")
    if not trace_dir or not records:
        return
    output_dir = Path(trace_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_id = os.environ.get("E2_PRIMARY_CASE_ID", INPUT_PATH.parent.name)
    payload = {
        "fragment_names": np.asarray(fragment_names),
        "bone_types": np.asarray(bone_types, dtype=np.int64),
        "iteration": np.asarray([row["iteration"] for row in records], dtype=np.int64),
    }
    array_keys = (
        "current_relative", "candidate_relative", "selected_relative", "selected",
        "raw_selected", "scores", "metrics", "utility", "severe", "utility_gain",
        "severe_delta", "fragment_accepted", "diameters_mm",
    )
    scalar_keys = (
        "patient_gain", "patient_accepted", "max_translation_mm",
    )
    for key in array_keys:
        payload[key] = np.stack([row[key] for row in records])
    for key in scalar_keys:
        payload[key] = np.asarray([row[key] for row in records])
    path = output_dir / f"{case_id}.npz"
    with tempfile.NamedTemporaryFile(
        dir=output_dir, prefix=f".{case_id}-", suffix=".npz", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(models):
    started = time.time()
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Missing input: {INPUT_PATH}")
    OUTPUT_PATH.unlink(missing_ok=True)

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats()

    meshdict = load_obj_fragments(str(INPUT_PATH), verbose=False)
    fragment_names, bone_types = fragment_names_and_bones(meshdict)
    fragment_data = build_fragment_data(meshdict)
    if not fragment_data:
        raise RuntimeError("No valid fragments found in input OBJ")
    if len(fragment_data) != len(fragment_names):
        raise RuntimeError("Sampling changed the fragment count")
    if len(fragment_data) > MAX_PARTS:
        raise RuntimeError(
            f"Input has {len(fragment_data)} fragments; maximum is {MAX_PARTS}"
        )
    if "1" not in fragment_names:
        raise RuntimeError("Input has no sacrum anchor fragment ID 1")

    anchor_index = fragment_names.index("1")
    if anchor_index != 0:
        raise RuntimeError(
            "The frozen e26 direct backbones require sacrum fragment 1 at index 0"
        )

    _, baseline_records = run_canonical_baseline(
        models, fragment_data, bone_types, fragment_names
    )
    exact_full, exact_full_records = run_exact_full_candidate(
        models, fragment_data, bone_types, fragment_names
    )
    direct_candidates = []
    direct_iterations = []
    for key in ("b1", "b2", "b3"):
        candidate, iterations = run_direct_candidate(
            models[key], fragment_data, bone_types, models["device"], anchor_index
        )
        direct_candidates.append(candidate)
        direct_iterations.append(iterations)
    final, decision, current, candidate_relative = select_final_candidate(
        models,
        fragment_data,
        bone_types,
        fragment_names,
        baseline_records,
        [exact_full, *direct_candidates],
    )
    validate_and_write(fragment_names, final)
    write_trace(fragment_names, bone_types, baseline_records)
    write_e27_trace(
        fragment_names,
        bone_types,
        baseline_records,
        exact_full_records,
        current,
        candidate_relative,
        decision,
        direct_iterations,
    )

    peak_allocated = 0
    peak_reserved = 0
    if torch.cuda.is_available():
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
    print(
        f"wrote={OUTPUT_PATH} fragments={len(fragment_names)} "
        f"e27_selected_nonzero={int(np.count_nonzero(decision['selected']))} "
        f"e27_patient_accepted={decision['patient_accepted']} "
        f"elapsed_s={time.time() - started:.2f} "
        f"peak_cuda_allocated={peak_allocated} peak_cuda_reserved={peak_reserved}",
        flush=True,
    )


def main():
    models = init_models()
    run(models)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FATAL: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
