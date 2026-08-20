"""Load the fixed Area64 Clinical cache back into fractured input space."""

from __future__ import annotations

from pathlib import Path

import numpy as np


BONES = ("SA", "LI", "RI")


def apply_pose(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return points @ pose[:3, :3].T + pose[:3, 3]


def load_cached_fragments(case_dir: Path):
    fragments = []
    bone_types = []
    fragment_names = []
    for bone_index, bone in enumerate(BONES):
        files = sorted(case_dir.glob(f"{bone}_*.npz"))
        if len(files) > 1:
            raise RuntimeError(f"Multiple {bone} cache files in {case_dir}")
        if not files:
            continue
        with np.load(files[0], allow_pickle=False) as payload:
            points = payload["points"].astype(np.float32)
            normals = payload["normals"].astype(np.float32)
            offsets = payload["offsets"].astype(np.int64)
            inverse_poses = payload["per_frag_T_inv"].astype(np.float64)
            names = [str(value) for value in payload["fragment_ids"].tolist()]
        if len(names) + 1 != len(offsets) or len(names) != len(inverse_poses):
            raise RuntimeError(f"Invalid cache layout in {files[0]}")
        for index, name in enumerate(names):
            start, stop = int(offsets[index]), int(offsets[index + 1])
            pose = inverse_poses[index]
            source_points = apply_pose(points[start:stop], pose).astype(np.float32)
            source_normals = (normals[start:stop] @ pose[:3, :3].T).astype(np.float32)
            fragments.append((source_points, source_normals))
            bone_types.append(bone_index)
            fragment_names.append(name)
    if not fragments:
        raise RuntimeError(f"No Area64 fragments found in {case_dir}")
    if "1" not in fragment_names:
        raise RuntimeError(f"Missing sacrum anchor fragment 1 in {case_dir}")
    return fragments, np.asarray(bone_types, dtype=np.int64), fragment_names


def discover_cache_cases(cache_dir: Path):
    return [
        path
        for path in sorted(cache_dir.iterdir())
        if path.is_dir() and any(path.glob("*.npz"))
    ]
