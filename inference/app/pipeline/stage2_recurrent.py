#!/usr/bin/env python3
"""Run the frozen B/F/FX candidate pool in the true recurrent inference loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


METRIC_SCALES = np.asarray([2.53, 2.76, 2.67, 3.37], dtype=np.float64)
MAX_PARTS = 50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fold_for_case(case_id: str) -> int:
    digest = hashlib.sha256(f"20260811:{case_id}".encode()).digest()
    return int.from_bytes(digest[:4], "little") % 5


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".json", mode="w", encoding="utf-8", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def stable_argmax(scores: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    maximum = scores.max(axis=1, keepdims=True)
    tied = scores >= maximum - float(epsilon)
    ids = np.broadcast_to(np.arange(scores.shape[1]), scores.shape)
    return np.where(tied, ids, scores.shape[1]).min(axis=1)


def build_tinit(centroid: np.ndarray, scale: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.eye(3, dtype=np.float32) / float(scale)
    transform[:3, 3] = -np.asarray(centroid, dtype=np.float32).reshape(3) / float(scale)
    return transform


def normalized_steps(candidate_cumulative, current, centroid, scale):
    steps = np.stack(
        [
            candidate_cumulative[index]
            @ np.linalg.inv(np.asarray(current[index], dtype=np.float64))
            for index in range(len(current))
        ]
    )
    initial = build_tinit(centroid, scale).astype(np.float64)
    inverse = np.linalg.inv(initial)
    return np.einsum("ij,nkjl,lm->nkim", initial, steps, inverse)


def candidate_pool(b_cumulative, f_cumulative, fx_cumulative, bones, diameters):
    """Build the provisional all-RI pool exactly as the Stage-1 trace builders."""
    result = b_cumulative.copy()
    li = bones == 1
    ri = bones == 2

    # LI: B0/B1/B2/FX3.
    result[li, 3] = fx_cumulative[li, 3]

    # RI: B0/B3/F3/Fused3. Fused3 uses B2 rotation + FX3 translation
    # below 150 mm and falls back to B3 for larger fragments.
    result[ri, 1] = b_cumulative[ri, 3]
    result[ri, 2] = f_cumulative[ri, 3]
    small_ri = ri & (diameters <= 150.0)
    result[small_ri, 3, :3, :3] = b_cumulative[small_ri, 2, :3, :3]
    result[small_ri, 3, :3, 3] = fx_cumulative[small_ri, 3, :3, 3]
    large_ri = ri & ~small_ri
    result[large_ri, 3] = b_cumulative[large_ri, 3]
    return result


def safety_candidate_pool(
    b_cumulative,
    f_cumulative,
    fx_cumulative,
    bones,
    diameters,
    baseline_selected,
):
    hybrid = candidate_pool(
        b_cumulative, f_cumulative, fx_cumulative, bones, diameters
    )
    result = b_cumulative.copy()
    rows = np.arange(len(bones))
    result[:, 0] = b_cumulative[:, 0]
    result[:, 1] = b_cumulative[rows, baseline_selected]
    li = bones == 1
    ri = bones == 2
    result[li, 2] = b_cumulative[li, 3]
    result[li, 3] = hybrid[li, 3]
    result[ri, 2] = hybrid[ri, 2]
    result[ri, 3] = hybrid[ri, 3]
    return result


def predict_calibrated(feature_block_fn, scored, iteration, bones, diameters, model):
    # Reuse the exact feature function used during nested patient-level OOF.
    feature_holder = {}
    for key in ("geometry", "scores", "metrics", "severe"):
        shape = (iteration + 1,) + tuple(scored[key].shape)
        feature_holder[key] = np.zeros(shape, dtype=scored[key].dtype)
        feature_holder[key][iteration] = scored[key]
    rows = []
    owners = []
    candidates = []
    for part in range(len(bones)):
        block, _ = feature_block_fn(
            feature_holder, iteration, part, int(bones[part]), float(diameters[part])
        )
        for offset in range(3):
            rows.append(block[offset])
            owners.append(part)
            candidates.append(offset + 1)
    x = np.asarray(rows, dtype=np.float64)
    z = np.concatenate(
        ((x - model["mean"]) / model["scale"], np.ones((len(x), 1))), axis=1
    )
    row_prediction = z @ model["coefficient"]
    prediction = np.zeros((len(bones), 4), dtype=np.float64)
    prediction[np.asarray(owners), np.asarray(candidates)] = row_prediction
    return prediction


def apply_original_c2(scored, anchor_index):
    scores = np.asarray(scored["scores"], dtype=np.float64)
    metrics = np.asarray(scored["metrics"], dtype=np.float64)
    severe = np.asarray(scored["severe"], dtype=np.float64)
    utility = (metrics / METRIC_SCALES[None, None]).mean(axis=-1)
    raw = stable_argmax(scores)
    raw[anchor_index] = 0
    row = np.arange(len(raw))
    gain = utility[:, 0] - utility[row, raw]
    severe_delta = severe[row, raw] - severe[:, 0]
    safe = (severe_delta <= 0.0) & ((severe[row, raw] <= 0.2) | (severe_delta < 0.0))
    accepted = (raw == 0) | ((gain > 0.05) & safe)
    selected = np.where(accepted, raw, 0)
    selected[anchor_index] = 0
    non_anchor = np.arange(len(raw)) != anchor_index
    patient_gain = float(np.mean(utility[non_anchor, 0] - utility[non_anchor, selected[non_anchor]]))
    patient_accepted = patient_gain > 0.0
    if not patient_accepted:
        selected[:] = 0
    return {
        "raw": raw,
        "selected": selected,
        "scores": scores,
        "metrics": metrics,
        "utility": utility,
        "severe": severe,
        "utility_gain": gain,
        "severe_delta": severe_delta,
        "fragment_accepted": accepted,
        "patient_gain": patient_gain,
        "patient_accepted": patient_accepted,
    }


def apply_calibrated_c2(
    feature_block_fn,
    scored,
    iteration,
    bones,
    diameters,
    anchor_index,
    model,
    fallback_slot=0,
):
    prediction = predict_calibrated(
        feature_block_fn, scored, iteration, bones, diameters, model
    )
    severe = np.asarray(scored["severe"], dtype=np.float64)
    raw = np.argmin(prediction, axis=1)
    raw[anchor_index] = 0
    row = np.arange(len(raw))
    fallback = np.full(len(raw), int(fallback_slot), dtype=np.int64)
    fallback[anchor_index] = 0
    proposed_gain = prediction[row, fallback] - prediction[row, raw]
    severe_delta = severe[row, raw] - severe[row, fallback]
    safe = (severe_delta <= model["c2_tolerance"]) & (
        (severe[row, raw] <= model["c2_threshold"]) | (severe_delta < 0.0)
    )
    accepted = (raw == fallback) | ((proposed_gain > model["c2_margin"]) & safe)
    selected = np.where(accepted, raw, fallback)
    selected[anchor_index] = 0
    non_anchor = np.arange(len(raw)) != anchor_index
    patient_gain = float(
        np.mean(
            prediction[non_anchor, fallback[non_anchor]]
            - prediction[non_anchor, selected[non_anchor]]
        )
    )
    patient_accepted = patient_gain > 0.0
    if not patient_accepted:
        selected[:] = fallback
        selected[anchor_index] = 0
    metrics = np.asarray(scored["metrics"], dtype=np.float64)
    utility = (metrics / METRIC_SCALES[None, None]).mean(axis=-1)
    return {
        "raw": raw,
        "selected": selected,
        "scores": -prediction,
        "metrics": metrics,
        "utility": utility,
        "severe": severe,
        "utility_gain": proposed_gain,
        "severe_delta": severe_delta,
        "fragment_accepted": accepted,
        "patient_gain": patient_gain,
        "patient_accepted": patient_accepted,
    }


def load_calibration(path: Path):
    with np.load(path, allow_pickle=False) as payload:
        return {
            "mean": payload["mean"].astype(np.float64),
            "scale": payload["scale"].astype(np.float64),
            "coefficient": payload["coefficient"].astype(np.float64),
            "ridge": float(payload["ridge"]),
            "c2_id": str(payload["c2_id"]),
            "c2_margin": float(payload["c2_margin"]),
            "c2_threshold": float(payload["c2_threshold"]),
            "c2_tolerance": float(payload["c2_tolerance"]),
        }


def write_trace(path: Path, names, bones, records, policy_source):
    payload = {
        "fragment_names": np.asarray(names),
        "bone_types": np.asarray(bones, dtype=np.int64),
        "iteration": np.asarray([row["iteration"] for row in records], dtype=np.int64),
        "policy_source": np.asarray(policy_source),
    }
    array_keys = (
        "current_relative", "candidate_relative", "selected_relative", "selected",
        "raw_selected", "geometry", "scores", "metrics", "utility", "severe", "utility_gain",
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
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def run_case(
    ranker,
    f_model,
    fx_model,
    fragment_data,
    bones,
    names,
    device,
    helpers,
    pool,
    selection,
    calibration,
    max_iters,
    convergence_mm,
    source_alphas,
):
    process = helpers["process"]
    run_model = helpers["run_model"]
    sfq_candidate_steps = helpers["sfq_candidate_steps"]
    score_candidates = helpers["score_candidates"]
    feature_block_fn = helpers["feature_block"]
    count = len(fragment_data)
    cumulative = np.repeat(np.eye(4, dtype=np.float64)[None], count, axis=0)
    coords = [item[0].copy() for item in fragment_data]
    normals = [item[1].copy() for item in fragment_data]
    anchor_index = names.index("1")
    records = []

    for iteration in range(max_iters):
        all_points = np.concatenate(coords, axis=0).astype(np.float32)
        normalized, centroid = process.recenter_pc(all_points)
        normalized, scale = process.rescale_pc(normalized)
        all_normals = np.concatenate(normals, axis=0).astype(np.float32)
        counts = np.asarray([len(item) for item in coords], dtype=np.int32)
        diameters = np.asarray(
            [np.linalg.norm(item.max(axis=0) - item.min(axis=0)) for item in coords],
            dtype=np.float32,
        )
        batch = helpers["make_batch"](
            normalized, all_normals, counts, bones, diameters, scale, device
        )
        direct = process.run_inference_pass(
            ranker, normalized, all_normals, counts, bones, diameters, scale, device
        )
        b_steps = process.candidate_steps_in_original_space(direct, centroid, scale)
        current = cumulative.copy()
        b_cumulative = np.einsum("nkij,njl->nkil", b_steps, current)
        direct_scored = {
            "scores": direct["scores"].detach().float().cpu().numpy(),
            "metrics": torch.expm1(direct["metric_prediction"].float()).clamp_min(0.0).detach().cpu().numpy(),
            "severe": torch.sigmoid(direct["severe_logits"].float()).detach().cpu().numpy(),
        }
        baseline_decision = apply_original_c2(direct_scored, anchor_index)

        if pool == "baseline":
            hybrid_cumulative = b_cumulative
            scored = direct_scored
        else:
            f_output = run_model(
                f_model, normalized, all_normals, counts, bones, diameters, scale, device
            )
            fx_output = run_model(
                fx_model, normalized, all_normals, counts, bones, diameters, scale, device
            )
            f_steps = sfq_candidate_steps(f_output, centroid, scale)
            fx_steps = sfq_candidate_steps(fx_output, centroid, scale)
            f_cumulative = np.einsum("nkij,njl->nkil", f_steps, current)
            fx_cumulative = np.einsum("nkij,njl->nkil", fx_steps, current)
            if pool == "safety":
                hybrid_cumulative = safety_candidate_pool(
                    b_cumulative,
                    f_cumulative,
                    fx_cumulative,
                    bones,
                    diameters,
                    baseline_decision["selected"],
                )
            else:
                hybrid_cumulative = candidate_pool(
                    b_cumulative, f_cumulative, fx_cumulative, bones, diameters
                )
            matrices = normalized_steps(
                hybrid_cumulative, current, centroid, scale
            ).astype(np.float32)
            scored = score_candidates(
                ranker, batch, matrices, source_alphas, self_control=False
            )

        if selection == "original":
            decision = apply_original_c2(scored, anchor_index)
        elif selection == "calibrated":
            decision = apply_calibrated_c2(
                feature_block_fn,
                scored,
                iteration,
                bones,
                diameters,
                anchor_index,
                calibration,
            )
        else:
            decision = apply_calibrated_c2(
                feature_block_fn,
                scored,
                iteration,
                bones,
                diameters,
                anchor_index,
                calibration,
                fallback_slot=1,
            )

        selected = decision["selected"].astype(np.int64)
        selected_next = np.stack(
            [hybrid_cumulative[index, selected[index]] for index in range(count)]
        )
        anchor_inverse = np.linalg.inv(selected_next[anchor_index])
        candidate_relative = np.einsum(
            "ij,nkjl->nkil", anchor_inverse, hybrid_cumulative
        )
        current_relative = np.einsum(
            "ij,njl->nil", np.linalg.inv(current[anchor_index]), current
        )
        selected_relative = np.einsum("ij,njl->nil", anchor_inverse, selected_next)

        max_shift = 0.0
        for index in range(count):
            step = hybrid_cumulative[index, selected[index]] @ np.linalg.inv(current[index])
            cumulative[index] = step @ cumulative[index]
            coords[index] = process.apply_transform(coords[index], step)
            normals[index] = process.apply_rotation(normals[index], step)
            max_shift = max(max_shift, float(np.linalg.norm(step[:3, 3])))

        records.append(
            {
                "iteration": iteration + 1,
                "current_relative": current_relative,
                "candidate_relative": candidate_relative,
                "selected_relative": selected_relative,
                "selected": selected,
                "raw_selected": decision["raw"],
                "geometry": np.asarray(
                    scored.get(
                        "geometry",
                        np.zeros((count, 4, 27), dtype=np.float32),
                    ),
                    dtype=np.float32,
                ),
                "scores": decision["scores"],
                "metrics": decision["metrics"],
                "utility": decision["utility"],
                "severe": decision["severe"],
                "utility_gain": decision["utility_gain"],
                "severe_delta": decision["severe_delta"],
                "fragment_accepted": decision["fragment_accepted"],
                "patient_gain": decision["patient_gain"],
                "patient_accepted": int(decision["patient_accepted"]),
                "diameters_mm": diameters,
                "max_translation_mm": max_shift,
            }
        )
        if max_shift < convergence_mm or np.all(selected == 0):
            break
    return cumulative, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--app-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ranker-checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--f-checkpoint", type=Path)
    parser.add_argument("--fx-checkpoint", type=Path)
    parser.add_argument("--calibration-model", type=Path)
    parser.add_argument("--calibration-model-dir", type=Path)
    parser.add_argument("--c2-margin", type=float)
    parser.add_argument("--c2-threshold", type=float)
    parser.add_argument("--c2-tolerance", type=float)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction-name", required=True)
    parser.add_argument(
        "--pool", choices=("baseline", "hybrid", "safety"), default="hybrid"
    )
    parser.add_argument(
        "--selection",
        choices=("original", "calibrated", "rescue"),
        default="calibrated",
    )
    parser.add_argument(
        "--source-alphas", type=float, nargs=4, default=(0.0, 0.5, 1.0, 1.25)
    )
    parser.add_argument("--max-iters", type=int, default=10)
    parser.add_argument("--convergence-mm", type=float, default=2.0)
    parser.add_argument("--expected-cases", type=int, default=170)
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()

    if args.pool in ("hybrid", "safety") and (
        args.f_checkpoint is None or args.fx_checkpoint is None
    ):
        raise ValueError("Hybrid/safety pool requires both SFQ checkpoints")
    if args.calibration_model is not None and args.calibration_model_dir is not None:
        raise ValueError("Use either --calibration-model or --calibration-model-dir")
    if args.selection in ("calibrated", "rescue") and (
        args.calibration_model is None and args.calibration_model_dir is None
    ):
        raise ValueError(
            "Calibrated/rescue selection requires a calibration model or model directory"
        )
    if args.selection == "rescue" and args.pool != "safety":
        raise ValueError("Rescue selection requires the safety pool")

    # Keep the verified app first so the frozen Ranker uses its exact module
    # implementations. Only append the experiment directory to AssemblyNet's
    # package search path for the isolated SFQ module that is absent from app.
    sys.path.insert(0, str(args.code_dir.resolve()))
    sys.path.insert(0, str(args.app_dir.resolve()))
    import process
    import models.AssemblyNet as assemblynet

    local_assemblynet = str(
        (args.code_dir / "models" / "AssemblyNet").resolve()
    )
    if local_assemblynet not in assemblynet.__path__:
        assemblynet.__path__.append(local_assemblynet)
    from area64_cache import discover_cache_cases, load_cached_fragments
    from extract_stage2_ranker_features import make_batch, score_candidates
    from fit_stage2_delta_ranker_oof import feature_block
    from run_sfq_area64_replay import (
        candidate_steps as sfq_candidate_steps,
        load_model as load_sfq_model,
        run_model,
        write_prediction,
    )

    process.configure_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ranker = process._load_model(args.ranker_checkpoint, device)
    f_model = None
    fx_model = None
    if args.pool in ("hybrid", "safety"):
        common = {
            "base_checkpoint": args.base_checkpoint,
            "max_rotation_deg": 30.0,
            "max_translation_mm": 30.0,
        }
        f_model = load_sfq_model(
            SimpleNamespace(**common, sfq_checkpoint=args.f_checkpoint, variant="f"), device
        )
        fx_model = load_sfq_model(
            SimpleNamespace(**common, sfq_checkpoint=args.fx_checkpoint, variant="fx"), device
        )
    calibration = None
    fold_calibrations = None
    if args.selection in ("calibrated", "rescue"):
        if args.calibration_model_dir is not None:
            fold_calibrations = {
                fold: load_calibration(args.calibration_model_dir / f"fold{fold}.npz")
                for fold in range(5)
            }
        else:
            calibration = load_calibration(args.calibration_model)
    all_calibrations = (
        list(fold_calibrations.values())
        if fold_calibrations is not None
        else ([calibration] if calibration is not None else [])
    )
    for calibration_item in all_calibrations:
        for argument, key in (
            (args.c2_margin, "c2_margin"),
            (args.c2_threshold, "c2_threshold"),
            (args.c2_tolerance, "c2_tolerance"),
        ):
            if argument is not None:
                calibration_item[key] = float(argument)
    helpers = {
        "process": process,
        "make_batch": make_batch,
        "score_candidates": score_candidates,
        "feature_block": feature_block,
        "run_model": run_model,
        "sfq_candidate_steps": sfq_candidate_steps,
    }

    cases = discover_cache_cases(args.cache_dir)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    elif len(cases) != args.expected_cases:
        raise RuntimeError(f"Expected {args.expected_cases} cases, got {len(cases)}")

    trace_dir = args.output_dir / "traces"
    prediction_dir = args.output_dir / "predictions"
    records = []
    for case_index, case_dir in enumerate(cases, 1):
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        fragment_data, bones, names = load_cached_fragments(case_dir)
        case_calibration = (
            fold_calibrations[fold_for_case(case_dir.name)]
            if fold_calibrations is not None
            else calibration
        )
        started = time.time()
        cumulative, trace = run_case(
            ranker,
            f_model,
            fx_model,
            fragment_data,
            np.asarray(bones, dtype=np.int64),
            names,
            device,
            helpers,
            args.pool,
            args.selection,
            case_calibration,
            args.max_iters,
            args.convergence_mm,
            tuple(float(value) for value in args.source_alphas),
        )
        prediction_path = prediction_dir / f"{case_dir.name}.json"
        write_prediction(prediction_path, names, cumulative)
        dataset_prediction = args.data_dir / case_dir.name / args.prediction_name
        dataset_prediction.write_text(prediction_path.read_text(encoding="utf-8"), encoding="utf-8")
        write_trace(
            trace_dir / f"{case_dir.name}.npz",
            names,
            bones,
            trace,
            f"stage2_recurrent_{args.pool}_{args.selection}",
        )
        elapsed = time.time() - started
        records.append(
            {
                "case": case_dir.name,
                "elapsed_s": elapsed,
                "iterations": len(trace),
                "final_nonzero": int(np.count_nonzero(trace[-1]["selected"])),
            }
        )
        print(
            f"stage2_recurrent progress={case_index}/{len(cases)} case={case_dir.name} "
            f"iters={len(trace)} elapsed_s={elapsed:.3f}",
            flush=True,
        )

    manifest = {
        "status": "complete",
        "protocol": "Area64 true recurrent Clinical170",
        "pool": args.pool,
        "selection": args.selection,
        "cases": len(records),
        "max_iters": args.max_iters,
        "convergence_mm": args.convergence_mm,
        "ranker_checkpoint": str(args.ranker_checkpoint),
        "ranker_sha256": sha256(args.ranker_checkpoint),
        "base_checkpoint": str(args.base_checkpoint),
        "base_sha256": sha256(args.base_checkpoint),
        "f_checkpoint": str(args.f_checkpoint) if args.f_checkpoint else None,
        "f_sha256": sha256(args.f_checkpoint) if args.f_checkpoint else None,
        "fx_checkpoint": str(args.fx_checkpoint) if args.fx_checkpoint else None,
        "fx_sha256": sha256(args.fx_checkpoint) if args.fx_checkpoint else None,
        "calibration_model": str(args.calibration_model) if args.calibration_model else None,
        "calibration_sha256": sha256(args.calibration_model) if args.calibration_model else None,
        "calibration_model_dir": (
            str(args.calibration_model_dir) if args.calibration_model_dir else None
        ),
        "calibration_fold_sha256": (
            {
                str(fold): sha256(args.calibration_model_dir / f"fold{fold}.npz")
                for fold in range(5)
            }
            if args.calibration_model_dir is not None
            else None
        ),
        "c2_policy": (
            {
                str(fold): {
                    key: fold_calibrations[fold][key]
                    for key in ("c2_id", "c2_margin", "c2_threshold", "c2_tolerance")
                }
                for fold in range(5)
            }
            if fold_calibrations is not None
            else {
                key: calibration[key]
                for key in ("c2_id", "c2_margin", "c2_threshold", "c2_tolerance")
            }
            if calibration is not None
            else "original"
        ),
        "prediction_name": args.prediction_name,
        "source_alphas": list(args.source_alphas),
        "records": records,
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("status", "pool", "selection", "cases")}, indent=2))


if __name__ == "__main__":
    main()
