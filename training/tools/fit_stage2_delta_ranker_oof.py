#!/usr/bin/env python3
"""Fit a low-capacity delta Ranker with nested patient-level OOF."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


METRIC_SCALES = np.asarray([2.53, 2.76, 2.67, 3.37], dtype=np.float64)
SIZE_BOUNDS = np.asarray([83.80764770507812, 122.42963409423828, 260.1748046875])
RIDGES = (0.1, 1.0, 10.0, 100.0, 1000.0)
C2_CONFIGS = (
    ("original", 0.050, 0.200, 0.000),
    ("margin000", 0.000, 0.200, 0.000),
    ("margin025", 0.025, 0.200, 0.000),
    ("margin075", 0.075, 0.200, 0.000),
    ("margin100", 0.100, 0.200, 0.000),
    ("risk100", 0.050, 0.100, 0.000),
    ("risk350", 0.050, 0.350, 0.000),
    ("margin025_risk350", 0.025, 0.350, 0.000),
    ("margin075_risk100", 0.075, 0.100, 0.000),
    ("tol025", 0.050, 0.200, 0.025),
)


def fold_for_case(case_id: str) -> int:
    digest = hashlib.sha256(f"20260811:{case_id}".encode()).digest()
    return int.from_bytes(digest[:4], "little") % 5


def atomic_npz(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def feature_block(features, iteration, part, bone, diameter):
    geometry = np.asarray(features["geometry"][iteration, part], dtype=np.float64)
    scores = np.asarray(features["scores"][iteration, part], dtype=np.float64)
    metrics = np.asarray(features["metrics"][iteration, part], dtype=np.float64)
    severe = np.asarray(features["severe"][iteration, part], dtype=np.float64)
    predicted_utility = (metrics / METRIC_SCALES[None]).mean(axis=1)
    size_bin = int(np.searchsorted(SIZE_BOUNDS, diameter, side="right"))
    rows = []
    for candidate in range(1, 4):
        slot = np.zeros(3, dtype=np.float64)
        slot[candidate - 1] = 1.0
        bone_slot = np.zeros(9, dtype=np.float64)
        bone_slot[bone * 3 + candidate - 1] = 1.0
        size_slot = np.zeros(12, dtype=np.float64)
        size_slot[size_bin * 3 + candidate - 1] = 1.0
        iteration_slot = slot * min(10.0, float(iteration + 1)) / 10.0
        rows.append(
            np.concatenate(
                (
                    geometry[candidate] - geometry[0],
                    [scores[candidate] - scores[0]],
                    [predicted_utility[candidate] - predicted_utility[0]],
                    [severe[candidate] - severe[0]],
                    slot,
                    bone_slot,
                    size_slot,
                    iteration_slot,
                )
            )
        )
    return np.stack(rows), severe


def load_dataset(feature_dir: Path, target_dir: Path):
    cases = sorted(path.stem for path in feature_dir.glob("*.npz"))
    if not cases:
        raise RuntimeError("No Ranker feature files found")
    x_rows = []
    y_rows = []
    case_rows = []
    fold_rows = []
    decision_rows = []
    candidate_rows = []
    utilities = []
    severe_predictions = []
    metric_predictions = []
    metadata = []
    case_payload = {}
    decision_id = 0
    for case_id in cases:
        target_path = target_dir / f"{case_id}.npz"
        if not target_path.is_file():
            raise FileNotFoundError(target_path)
        with np.load(feature_dir / f"{case_id}.npz", allow_pickle=False) as features, np.load(
            target_path, allow_pickle=False
        ) as targets:
            if not np.array_equal(features["iteration"], targets["iteration"]):
                raise RuntimeError(f"Iteration mismatch for {case_id}")
            if not np.array_equal(features["fragment_names"], targets["fragment_names"]):
                raise RuntimeError(f"Fragment mismatch for {case_id}")
            valid = np.asarray(targets["valid"], dtype=bool)
            case_payload[case_id] = {
                "shape": valid.shape,
                "valid": valid.copy(),
                "iteration": np.asarray(targets["iteration"], dtype=np.int64).copy(),
                "fragment_names": np.asarray(targets["fragment_names"]).copy(),
            }
            for iteration, part in np.argwhere(valid):
                bone = int(targets["bone_types"][part])
                diameter = float(targets["diameters_mm"][iteration, part])
                block, predicted_severe = feature_block(
                    features, iteration, part, bone, diameter
                )
                true_utility = np.asarray(
                    targets["utility"][iteration, part], dtype=np.float64
                )
                transformed_delta = np.arcsinh(true_utility[1:] - true_utility[0])
                for local_candidate in range(3):
                    x_rows.append(block[local_candidate])
                    y_rows.append(transformed_delta[local_candidate])
                    case_rows.append(case_id)
                    fold_rows.append(fold_for_case(case_id))
                    decision_rows.append(decision_id)
                    candidate_rows.append(local_candidate + 1)
                utilities.append(true_utility)
                severe_predictions.append(predicted_severe)
                metric_predictions.append(
                    np.asarray(features["metrics"][iteration, part], dtype=np.float64)
                )
                metadata.append(
                    {
                        "case": case_id,
                        "iteration_index": int(iteration),
                        "part": int(part),
                    }
                )
                decision_id += 1
    return {
        "x": np.asarray(x_rows, dtype=np.float64),
        "y": np.asarray(y_rows, dtype=np.float64),
        "case": np.asarray(case_rows),
        "fold": np.asarray(fold_rows, dtype=np.int64),
        "decision": np.asarray(decision_rows, dtype=np.int64),
        "candidate": np.asarray(candidate_rows, dtype=np.int64),
        "utility": np.asarray(utilities, dtype=np.float64),
        "severe_prediction": np.asarray(severe_predictions, dtype=np.float64),
        "metric_prediction": np.asarray(metric_predictions, dtype=np.float64),
        "metadata": metadata,
        "case_payload": case_payload,
        "cases": cases,
    }


def patient_weights(case_ids: np.ndarray) -> np.ndarray:
    counts = collections.Counter(case_ids.tolist())
    weights = np.asarray([1.0 / counts[value] for value in case_ids], dtype=np.float64)
    return weights / weights.mean()


def fit_ridge(x, y, cases, ridge, huber=0.5, iterations=4):
    base_weight = patient_weights(cases)
    mean = np.average(x, axis=0, weights=base_weight)
    variance = np.average((x - mean) ** 2, axis=0, weights=base_weight)
    scale = np.sqrt(variance).clip(min=1e-6)
    z = np.concatenate(((x - mean) / scale, np.ones((len(x), 1))), axis=1)
    coefficient = np.zeros(z.shape[1], dtype=np.float64)
    robust = np.ones(len(x), dtype=np.float64)
    penalty = np.eye(z.shape[1], dtype=np.float64) * float(ridge)
    penalty[-1, -1] = 0.0
    for _ in range(iterations):
        weight = base_weight * robust
        lhs = z.T @ (weight[:, None] * z) + penalty
        rhs = z.T @ (weight * y)
        coefficient = np.linalg.solve(lhs, rhs)
        residual = y - z @ coefficient
        robust = np.minimum(1.0, huber / np.abs(residual).clip(min=1e-8))
    return {"mean": mean, "scale": scale, "coefficient": coefficient, "ridge": ridge}


def predict(model, x):
    z = np.concatenate(
        ((x - model["mean"]) / model["scale"], np.ones((len(x), 1))), axis=1
    )
    return z @ model["coefficient"]


def decision_prediction(dataset, row_prediction):
    prediction = np.zeros((len(dataset["utility"]), 4), dtype=np.float64)
    prediction[dataset["decision"], dataset["candidate"]] = row_prediction
    return prediction


def selected_regret(dataset, prediction, decision_mask, selected=None):
    utility = dataset["utility"]
    if selected is None:
        selected = np.argmin(prediction, axis=1)
    oracle = np.argmin(utility, axis=1)
    rows = np.arange(len(utility))
    regret = utility[rows, selected] - utility[rows, oracle]
    cases = np.asarray([row["case"] for row in dataset["metadata"]])
    case_means = []
    for case_id in np.unique(cases[decision_mask]):
        local = decision_mask & (cases == case_id)
        case_means.append(float(regret[local].mean()))
    return {
        "patient_mean_regret": float(np.mean(case_means)),
        "top1_accuracy": float(np.mean(selected[decision_mask] == oracle[decision_mask])),
        "decision_mean_regret": float(np.mean(regret[decision_mask])),
    }


def apply_c2(dataset, prediction, config, decision_mask, fallback_slot=0):
    _, margin, threshold, tolerance = config
    severe = dataset["severe_prediction"]
    raw = np.argmin(prediction, axis=1)
    rows = np.arange(len(raw))
    fallback = np.full(len(raw), int(fallback_slot), dtype=np.int64)
    proposed_delta = prediction[rows, raw] - prediction[rows, fallback]
    severe_delta = severe[rows, raw] - severe[rows, fallback]
    improves = -proposed_delta > margin
    severe_safe = (severe_delta <= tolerance) & (
        (severe[rows, raw] <= threshold) | (severe_delta < 0.0)
    )
    selected = np.where((raw == fallback) | (improves & severe_safe), raw, fallback)
    groups = collections.defaultdict(list)
    for decision, row in enumerate(dataset["metadata"]):
        if decision_mask[decision]:
            groups[(row["case"], row["iteration_index"])].append(decision)
    patient_accepted = {}
    for key, local in groups.items():
        local = np.asarray(local, dtype=np.int64)
        gain = float(
            np.mean(
                prediction[local, fallback[local]]
                - prediction[local, selected[local]]
            )
        )
        accepted = gain > 0.0
        patient_accepted[key] = (accepted, gain)
        if not accepted:
            selected[local] = fallback[local]
    return raw, selected, patient_accepted


def choose_ridge_nested(dataset, outer_fold):
    train_decision = np.asarray(
        [fold_for_case(row["case"]) != outer_fold for row in dataset["metadata"]]
    )
    train_row = dataset["fold"] != outer_fold
    fold_rows = []
    candidates = []
    for ridge in RIDGES:
        inner_prediction = np.zeros(len(dataset["x"]), dtype=np.float64)
        inner_valid = np.zeros(len(dataset["x"]), dtype=bool)
        improvements = 0
        for inner_fold in range(5):
            if inner_fold == outer_fold:
                continue
            fit_mask = train_row & (dataset["fold"] != inner_fold)
            test_mask = dataset["fold"] == inner_fold
            model = fit_ridge(
                dataset["x"][fit_mask],
                dataset["y"][fit_mask],
                dataset["case"][fit_mask],
                ridge,
            )
            inner_prediction[test_mask] = predict(model, dataset["x"][test_mask])
            inner_valid[test_mask] = True
        decision_prediction_all = decision_prediction(dataset, inner_prediction)
        metrics = selected_regret(
            dataset, decision_prediction_all, train_decision
        )
        baseline_prediction = np.zeros_like(decision_prediction_all)
        score_delta = dataset["x"][:, 27]
        baseline_prediction[
            dataset["decision"], dataset["candidate"]
        ] = -score_delta
        baseline = selected_regret(dataset, baseline_prediction, train_decision)
        for inner_fold in range(5):
            if inner_fold == outer_fold:
                continue
            fold_mask = np.asarray(
                [fold_for_case(row["case"]) == inner_fold for row in dataset["metadata"]]
            )
            candidate_fold = selected_regret(dataset, decision_prediction_all, fold_mask)
            baseline_fold = selected_regret(dataset, baseline_prediction, fold_mask)
            improvements += int(
                candidate_fold["patient_mean_regret"]
                < baseline_fold["patient_mean_regret"]
            )
        candidates.append(
            {
                "ridge": ridge,
                "metrics": metrics,
                "baseline": baseline,
                "improved_inner_folds": improvements,
                "row_prediction": inner_prediction,
                "row_valid": inner_valid,
            }
        )
    eligible = [item for item in candidates if item["improved_inner_folds"] >= 3]
    chosen = min(
        eligible or candidates,
        key=lambda item: (
            item["metrics"]["patient_mean_regret"],
            -item["improved_inner_folds"],
            -item["ridge"],
        ),
    )
    fold_rows.extend(
        {
            "ridge": item["ridge"],
            "patient_mean_regret": item["metrics"]["patient_mean_regret"],
            "top1_accuracy": item["metrics"]["top1_accuracy"],
            "improved_inner_folds": item["improved_inner_folds"],
        }
        for item in candidates
    )
    return chosen, fold_rows, train_row, train_decision


def choose_c2(dataset, prediction, train_decision, outer_fold, fallback_slot=0):
    candidates = []
    for config in C2_CONFIGS:
        _, selected, _ = apply_c2(
            dataset, prediction, config, train_decision, fallback_slot
        )
        metrics = selected_regret(dataset, prediction, train_decision, selected=selected)
        fold_wins = 0
        for inner_fold in range(5):
            if inner_fold == outer_fold:
                continue
            fold_mask = np.asarray(
                [fold_for_case(row["case"]) == inner_fold for row in dataset["metadata"]]
            )
            _, fold_selected, _ = apply_c2(
                dataset, prediction, config, fold_mask, fallback_slot
            )
            candidate = selected_regret(
                dataset, prediction, fold_mask, selected=fold_selected
            )
            _, base_selected, _ = apply_c2(
                dataset, prediction, C2_CONFIGS[0], fold_mask, fallback_slot
            )
            baseline = selected_regret(
                dataset, prediction, fold_mask, selected=base_selected
            )
            fold_wins += int(
                candidate["patient_mean_regret"]
                <= baseline["patient_mean_regret"] + 1e-12
            )
        candidates.append(
            {
                "config": config,
                "metrics": metrics,
                "nonworse_inner_folds": fold_wins,
            }
        )
    eligible = [item for item in candidates if item["nonworse_inner_folds"] >= 3]
    return min(
        eligible or candidates,
        key=lambda item: (
            item["metrics"]["patient_mean_regret"],
            -item["nonworse_inner_folds"],
            item["config"][0] != "original",
        ),
    ), candidates


def save_model(path, model, c2, fallback_slot=0):
    atomic_npz(
        path,
        mean=model["mean"],
        scale=model["scale"],
        coefficient=model["coefficient"],
        ridge=np.asarray(model["ridge"], dtype=np.float64),
        c2_id=np.asarray(c2[0]),
        c2_margin=np.asarray(c2[1], dtype=np.float64),
        c2_threshold=np.asarray(c2[2], dtype=np.float64),
        c2_tolerance=np.asarray(c2[3], dtype=np.float64),
        fallback_slot=np.asarray(fallback_slot, dtype=np.int64),
    )


def materialize_oof_traces(
    dataset, trace_dir, output_dir, prediction, raw, selected, accepted, fallback_slot=0
):
    output_dir.mkdir(parents=True, exist_ok=True)
    per_case_decisions = collections.defaultdict(list)
    for decision, row in enumerate(dataset["metadata"]):
        per_case_decisions[row["case"]].append((decision, row))
    for case_id in dataset["cases"]:
        with np.load(trace_dir / f"{case_id}.npz", allow_pickle=False) as trace:
            arrays = {key: trace[key].copy() for key in trace.files}
        arrays["ranker_calibrated_delta"] = np.zeros_like(arrays["scores"], dtype=np.float64)
        for decision, row in per_case_decisions[case_id]:
            iteration = row["iteration_index"]
            part = row["part"]
            arrays["raw_selected"][iteration, part] = raw[decision]
            arrays["selected"][iteration, part] = selected[decision]
            arrays["scores"][iteration, part] = -prediction[decision]
            arrays["ranker_calibrated_delta"][iteration, part] = prediction[decision]
            arrays["metrics"][iteration, part] = dataset["metric_prediction"][decision]
            arrays["utility"][iteration, part] = (
                dataset["metric_prediction"][decision] / METRIC_SCALES[None]
            ).mean(axis=1)
            arrays["selected_relative"][iteration, part] = arrays["candidate_relative"][
                iteration, part, selected[decision]
            ]
            arrays["utility_gain"][iteration, part] = (
                prediction[decision, fallback_slot]
                - prediction[decision, selected[decision]]
            )
            severe = dataset["severe_prediction"][decision]
            arrays["severe"][iteration, part] = severe
            arrays["severe_delta"][iteration, part] = (
                severe[selected[decision]] - severe[fallback_slot]
            )
            arrays["fragment_accepted"][iteration, part] = (
                selected[decision] != fallback_slot
            )
        for iteration_index in range(len(arrays["iteration"])):
            key = (case_id, iteration_index)
            is_accepted, gain = accepted.get(key, (False, 0.0))
            arrays["patient_accepted"][iteration_index] = int(is_accepted)
            arrays["patient_gain"][iteration_index] = gain
        atomic_npz(output_dir / f"{case_id}.npz", **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fallback-slot", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.feature_dir, args.target_dir)
    decisions = len(dataset["utility"])
    oof_prediction = np.zeros((decisions, 4), dtype=np.float64)
    oof_raw = np.zeros(decisions, dtype=np.int64)
    oof_selected = np.zeros(decisions, dtype=np.int64)
    oof_accepted = {}
    outer_rows = []
    ridge_votes = []
    c2_votes = []
    for outer_fold in range(5):
        chosen_ridge, ridge_grid, train_row, train_decision = choose_ridge_nested(
            dataset, outer_fold
        )
        model = fit_ridge(
            dataset["x"][train_row],
            dataset["y"][train_row],
            dataset["case"][train_row],
            chosen_ridge["ridge"],
        )
        inner_decision_prediction = decision_prediction(
            dataset, chosen_ridge["row_prediction"]
        )
        chosen_c2, c2_grid = choose_c2(
            dataset,
            inner_decision_prediction,
            train_decision,
            outer_fold,
            args.fallback_slot,
        )
        test_row = dataset["fold"] == outer_fold
        test_decision = np.asarray(
            [fold_for_case(row["case"]) == outer_fold for row in dataset["metadata"]]
        )
        row_prediction = np.zeros(len(dataset["x"]), dtype=np.float64)
        row_prediction[test_row] = predict(model, dataset["x"][test_row])
        decision_pred = decision_prediction(dataset, row_prediction)
        raw, selected, accepted = apply_c2(
            dataset,
            decision_pred,
            chosen_c2["config"],
            test_decision,
            args.fallback_slot,
        )
        oof_prediction[test_decision] = decision_pred[test_decision]
        oof_raw[test_decision] = raw[test_decision]
        oof_selected[test_decision] = selected[test_decision]
        oof_accepted.update(accepted)
        metrics = selected_regret(
            dataset, decision_pred, test_decision, selected=selected
        )
        save_model(
            args.output_dir / "models" / f"fold{outer_fold}.npz",
            model,
            chosen_c2["config"],
            args.fallback_slot,
        )
        ridge_votes.append(chosen_ridge["ridge"])
        c2_votes.append(chosen_c2["config"][0])
        outer_rows.append(
            {
                "fold": outer_fold,
                "ridge": chosen_ridge["ridge"],
                "c2": chosen_c2["config"][0],
                "heldout": metrics,
                "ridge_grid": ridge_grid,
                "c2_grid": [
                    {
                        "id": item["config"][0],
                        "patient_mean_regret": item["metrics"]["patient_mean_regret"],
                        "top1_accuracy": item["metrics"]["top1_accuracy"],
                        "nonworse_inner_folds": item["nonworse_inner_folds"],
                    }
                    for item in c2_grid
                ],
            }
        )
        print(
            f"outer_fold={outer_fold} ridge={chosen_ridge['ridge']} "
            f"c2={chosen_c2['config'][0]} regret={metrics['patient_mean_regret']:.6f}",
            flush=True,
        )

    all_decision = np.ones(decisions, dtype=bool)
    oof_metrics = selected_regret(
        dataset, oof_prediction, all_decision, selected=oof_selected
    )
    raw_metrics = selected_regret(
        dataset, oof_prediction, all_decision, selected=oof_raw
    )
    materialize_oof_traces(
        dataset,
        args.trace_dir,
        args.output_dir / "oof_traces",
        oof_prediction,
        oof_raw,
        oof_selected,
        oof_accepted,
        args.fallback_slot,
    )
    ridge_choice = collections.Counter(ridge_votes).most_common(1)[0][0]
    c2_choice_id = collections.Counter(c2_votes).most_common(1)[0][0]
    c2_choice = next(config for config in C2_CONFIGS if config[0] == c2_choice_id)
    full_model = fit_ridge(
        dataset["x"], dataset["y"], dataset["case"], ridge_choice
    )
    save_model(
        args.output_dir / "models" / "full.npz",
        full_model,
        c2_choice,
        args.fallback_slot,
    )
    manifest = {
        "status": "complete",
        "protocol": "nested five-fold patient OOF, Huber ridge delta Ranker",
        "fold_rule": "sha256('20260811:'+case_id) modulo 5",
        "cases": len(dataset["cases"]),
        "decisions": decisions,
        "feature_dim": int(dataset["x"].shape[1]),
        "fallback_slot": args.fallback_slot,
        "oof_raw": raw_metrics,
        "oof_c2": oof_metrics,
        "ridge_votes": ridge_votes,
        "c2_votes": c2_votes,
        "full_model_ridge": ridge_choice,
        "full_model_c2": c2_choice_id,
        "outer_folds": outer_rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in ("oof_raw", "oof_c2", "ridge_votes", "c2_votes")}, indent=2))


if __name__ == "__main__":
    main()
