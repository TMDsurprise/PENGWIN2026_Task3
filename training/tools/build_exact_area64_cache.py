#!/usr/bin/env python3
"""Rebuild the exact source-space Area64 samples used by locked inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh


BONES = ("SA", "LI", "RI")


def allocate(weights: list[float], total: int, minimum: int) -> list[int]:
    raw = [int(value / sum(weights) * total) for value in weights]
    allocation = [max(value, minimum) for value in raw]
    largest = int(np.argmax(weights))
    allocation[largest] += total - sum(allocation)
    if min(allocation) < minimum or sum(allocation) != total:
        raise RuntimeError(
            f"Invalid Area64 allocation: total={sum(allocation)} min={min(allocation)}"
        )
    return allocation


def sample_mesh(mesh: trimesh.Trimesh, count: int) -> tuple[np.ndarray, np.ndarray]:
    points, face_indices = trimesh.sample.sample_surface_even(mesh, count)
    normals = mesh.face_normals[face_indices]
    if len(points) < count:
        if len(points) == 0:
            points = np.zeros((count, 3), dtype=np.float64)
            normals = np.zeros((count, 3), dtype=np.float64)
        else:
            selected = np.random.choice(len(points), count - len(points), replace=True)
            points = np.vstack((points, points[selected]))
            normals = np.vstack((normals, normals[selected]))
    elif len(points) > count:
        points = points[:count]
        normals = normals[:count]
    return points.astype(np.float32), normals.astype(np.float32)


def process_case(process, case_dir: Path, output_dir: Path, total: int, minimum: int):
    input_path = case_dir / "peripelvic-fracture-fragments.obj"
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    np.random.seed(42)
    meshdict = process.load_obj_fragments(str(input_path), verbose=False)
    sampled: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    area_lookup: dict[str, dict[str, float]] = {}
    allocation_record = {}
    for bone, fragments in meshdict.items():
        names = list(fragments.keys())
        if not names:
            continue
        weights = []
        for name in names:
            value = float(fragments[name]["mesh"].area)
            if not np.isfinite(value) or value <= 0:
                value = 0.001
            weights.append(value)
        counts = allocate(weights, total, minimum)
        sampled[bone] = {}
        area_lookup[bone] = {
            str(name): float(weight) for name, weight in zip(names, weights)
        }
        for name, count in zip(names, counts):
            sampled[bone][str(name)] = sample_mesh(fragments[name]["mesh"], count)
        allocation_record[bone] = {
            "sampling_order": [str(name) for name in names],
            "allocations": counts,
            "areas": weights,
        }

    destination = output_dir / case_dir.name
    destination.mkdir(parents=True, exist_ok=True)
    for bone in BONES:
        names = sorted(sampled.get(bone, {}))
        if not names:
            continue
        point_blocks = [sampled[bone][name][0] for name in names]
        normal_blocks = [sampled[bone][name][1] for name in names]
        counts = [len(points) for points in point_blocks]
        offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        np.savez_compressed(
            destination / f"{bone}_{len(names)}.npz",
            points=np.concatenate(point_blocks, axis=0),
            normals=np.concatenate(normal_blocks, axis=0),
            offsets=offsets,
            per_frag_T_inv=np.repeat(
                np.eye(4, dtype=np.float32)[None], len(names), axis=0
            ),
            fragment_ids=np.asarray(names),
            fragment_areas=np.asarray(
                [area_lookup[bone][name] for name in names],
                dtype=np.float64,
            ),
            sampling_mode=np.asarray("area64_exact_seed42_source"),
        )
    return {"case": case_dir.name, "bones": allocation_record}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--npoints", type=int, default=5000)
    parser.add_argument("--min-points", type=int, default=64)
    parser.add_argument("--expected-cases", type=int, default=170)
    args = parser.parse_args()
    sys.path.insert(0, str(args.app_dir.resolve()))
    import process

    args.output.mkdir(parents=True, exist_ok=True)
    case_dirs = [
        path
        for path in sorted(args.input.iterdir())
        if path.is_dir() and (path / "peripelvic-fracture-fragments.obj").is_file()
    ]
    if len(case_dirs) != args.expected_cases:
        raise RuntimeError(f"Expected {args.expected_cases} cases, got {len(case_dirs)}")
    records = []
    for index, case_dir in enumerate(case_dirs, 1):
        records.append(
            process_case(process, case_dir, args.output, args.npoints, args.min_points)
        )
        print(
            f"exact_area64_cache progress={index}/{len(case_dirs)} case={case_dir.name}",
            flush=True,
        )
    manifest = {
        "status": "complete",
        "sampling": "locked run_weight_variant Area64 source-space samples",
        "seed_reset_per_case": 42,
        "npoints_per_bone": args.npoints,
        "min_points_per_fragment": args.min_points,
        "cases": len(records),
        "records": records,
    }
    temporary = args.output / "cache_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output / "cache_manifest.json")


if __name__ == "__main__":
    main()
