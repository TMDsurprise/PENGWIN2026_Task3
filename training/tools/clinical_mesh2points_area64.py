#!/usr/bin/env python3
"""Build deterministic Clinical170 caches with Area64 fragment allocation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh


BONE_RANGES = {"SA": (1, 100), "LI": (101, 200), "RI": (201, 300)}


def load_poses(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {
            str(row["fragment_id"]): np.asarray(row["transformation"], dtype=np.float64)
            for row in payload
        }
    return {str(key): np.asarray(value, dtype=np.float64) for key, value in payload.items()}


def apply_pose(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return points @ pose[:3, :3].T + pose[:3, 3]


def allocate_area64(
    fragments: list[tuple[str, trimesh.Trimesh]], total: int, minimum: int
) -> list[int]:
    areas = np.asarray(
        [max(float(mesh.area), 1e-3) if np.isfinite(mesh.area) else 1e-3 for _, mesh in fragments],
        dtype=np.float64,
    )
    allocations = np.maximum((areas / areas.sum() * total).astype(np.int64), minimum)
    allocations[int(np.argmax(areas))] += total - int(allocations.sum())
    if np.any(allocations < minimum) or int(allocations.sum()) != total:
        raise RuntimeError(
            f"Invalid Area64 allocation: total={allocations.sum()} min={allocations.min()}"
        )
    return allocations.tolist()


def sample_surface(
    mesh: trimesh.Trimesh, count: int, rng: np.random.RandomState
) -> tuple[np.ndarray, np.ndarray]:
    points, face_indices = trimesh.sample.sample_surface_even(mesh, count)
    normals = mesh.face_normals[face_indices]
    if len(points) < count:
        if len(points) == 0:
            raise RuntimeError("sample_surface_even returned no points")
        selected = rng.choice(len(points), count - len(points), replace=True)
        points = np.concatenate([points, points[selected]], axis=0)
        normals = np.concatenate([normals, normals[selected]], axis=0)
    return points[:count].astype(np.float32), normals[:count].astype(np.float32)


def case_seed(seed: int, case_id: str, bone: str) -> int:
    digest = hashlib.sha256(f"{seed}:{case_id}:{bone}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def process_case(
    case_dir: Path, output_root: Path, npoints: int, minimum: int, seed: int
) -> dict:
    obj_files = sorted(case_dir.glob("*.obj"))
    pose_path = case_dir / "plan_pl_gt.json"
    if not obj_files or not pose_path.is_file():
        raise FileNotFoundError(f"Missing OBJ or plan_pl_gt.json in {case_dir}")
    scene = trimesh.load(obj_files[0], split_object=True, process=False)
    poses = load_poses(pose_path)
    grouped: dict[str, list[tuple[str, trimesh.Trimesh]]] = {
        bone: [] for bone in BONE_RANGES
    }
    for key, geometry in scene.geometry.items():
        fragment_id = str(key)
        try:
            numeric_id = int(fragment_id)
        except ValueError:
            continue
        if fragment_id not in poses or not isinstance(geometry, trimesh.Trimesh):
            continue
        for bone, (lower, upper) in BONE_RANGES.items():
            if lower <= numeric_id <= upper:
                grouped[bone].append((fragment_id, geometry))
                break

    destination = output_root / case_dir.name
    destination.mkdir(parents=True, exist_ok=True)
    case_record = {"case": case_dir.name, "bones": {}}
    for bone in ("SA", "LI", "RI"):
        fragments = sorted(grouped[bone], key=lambda item: item[0])
        if not fragments:
            continue
        allocations = allocate_area64(fragments, npoints, minimum)
        local_seed = case_seed(seed, case_dir.name, bone)
        np.random.seed(local_seed)
        rng = np.random.RandomState(local_seed)
        points_gt, normals_gt, inverse_poses = [], [], []
        offsets = [0]
        fragment_ids = []
        areas = []
        for (fragment_id, mesh), count in zip(fragments, allocations):
            points, normals = sample_surface(mesh, count, rng)
            pose = poses[fragment_id]
            points_gt.append(apply_pose(points, pose).astype(np.float32))
            normals_gt.append((normals @ pose[:3, :3].T).astype(np.float32))
            inverse_poses.append(np.linalg.inv(pose).astype(np.float32))
            offsets.append(offsets[-1] + count)
            fragment_ids.append(fragment_id)
            areas.append(float(mesh.area))
        np.savez_compressed(
            destination / f"{bone}_{len(fragments)}.npz",
            points=np.concatenate(points_gt, axis=0),
            normals=np.concatenate(normals_gt, axis=0),
            offsets=np.asarray(offsets, dtype=np.int64),
            per_frag_T_inv=np.stack(inverse_poses),
            fragment_ids=np.asarray(fragment_ids),
            fragment_areas=np.asarray(areas, dtype=np.float64),
            sampling_mode=np.asarray("area64"),
        )
        case_record["bones"][bone] = {
            "fragments": fragment_ids,
            "allocations": allocations,
            "areas": areas,
        }
    if len(case_record["bones"]) < 2:
        raise RuntimeError(f"Expected at least two bones in {case_dir}")
    return case_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--npoints", type=int, default=5000)
    parser.add_argument("--min-points", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    if args.npoints <= 0 or args.min_points <= 0:
        raise ValueError("npoints and min-points must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    case_dirs = [path for path in sorted(args.input.iterdir()) if path.is_dir()]
    for index, case_dir in enumerate(case_dirs, 1):
        records.append(
            process_case(case_dir, args.output, args.npoints, args.min_points, args.seed)
        )
        print(f"Area64 cache {index}/{len(case_dirs)} case={case_dir.name}", flush=True)
    if len(records) != 170:
        raise RuntimeError(f"Expected 170 Clinical cases, got {len(records)}")
    manifest = {
        "status": "complete",
        "sampling": "surface-area proportional per bone",
        "npoints_per_bone": args.npoints,
        "min_points_per_fragment": args.min_points,
        "seed": args.seed,
        "cases": len(records),
        "records": records,
    }
    temporary = args.output / "cache_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output / "cache_manifest.json")


if __name__ == "__main__":
    main()
