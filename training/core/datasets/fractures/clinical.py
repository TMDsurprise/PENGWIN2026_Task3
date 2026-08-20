"""Clinical (real) fracture dataset for point cloud assembly training.

Loads OBJ fragment meshes and GT reduction poses from `plan_pl_gt.json`.
Applies GT poses to obtain assembled ground-truth positions, then feeds
through the same augmentation pipeline as the simulation dataset.

Expected directory structure:
    data_root_real/
        sample_001/
            fragments.obj          (all fragments; keys 1-100=SA, 101-200=LI, 201-300=RI)
            plan_pl_gt.json        (GT 4x4 reduction matrices)
        sample_002/
            ...
"""

import os
import json
import hashlib
import random
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh
from torch.utils.data import Dataset

from .sample import FragmentSamples
from ..transform import recenter_pc


GT_JSON_NAME = "plan_pl_gt.json"
BONE_RANGES = {
    "SA": (1, 100),
    "LI": (101, 200),
    "RI": (201, 300),
}
BONE_TYPE_MAP = {"SA": 0, "LI": 1, "RI": 2}


def _load_gt_poses(json_path: str) -> dict:
    """Load GT 4x4 matrices from plan_pl_gt.json.

    Returns:
        dict mapping fragment_id (str) -> np.ndarray (4, 4)
    """
    with open(json_path) as f:
        data = json.load(f)

    poses = {}
    if isinstance(data, list):
        for entry in data:
            fid = str(entry["fragment_id"])
            poses[fid] = np.array(entry["transformation"], dtype=np.float64)
    elif isinstance(data, dict):
        for fid, mat in data.items():
            poses[str(fid)] = np.array(mat, dtype=np.float64)
    return poses


def _apply_T(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 4x4 homogeneous transform to (N, 3) points."""
    ones = np.ones((len(pts), 1), dtype=pts.dtype)
    return (T @ np.concatenate([pts, ones], axis=1).T).T[:, :3]


def _apply_T_normals(normals: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply rotation part of T to normals."""
    return (T[:3, :3] @ normals.T).T


def _sample_mesh_proportional(mesh: trimesh.Trimesh, n_points: int, min_points: int = 20) -> tuple:
    """Sample n_points from mesh surface, returning (coords, normals)."""
    n_points = max(n_points, min_points)
    pts, face_idx = trimesh.sample.sample_surface_even(mesh, n_points)
    normals = mesh.face_normals[face_idx]

    current = len(pts)
    if current < n_points:
        extra = np.random.choice(current, n_points - current, replace=True)
        pts = np.vstack([pts, pts[extra]])
        normals = np.vstack([normals, normals[extra]])
    elif current > n_points:
        pts = pts[:n_points]
        normals = normals[:n_points]

    return pts.astype(np.float32), normals.astype(np.float32)


class ClinicalFragmentSamples(FragmentSamples):
    """Clinical fracture dataset.

    Loads OBJ + plan_pl_gt.json pairs, applies GT poses to obtain assembled
    ground-truth positions, then runs the same augmentation pipeline as
    FragmentSamples (clinical pose variation, random rotation, etc.).
    """

    def __init__(
        self,
        split: Literal["train", "val", "test"] = "train",
        data_root_real: str = "",
        npoints_per_bone: int = 5000,
        min_parts: int = 2,
        max_parts: int = 50,
        max_bone: int = 3,
        random_anchor: bool = True,
        merge_prob: float = 0.15,
        frac_prob: float = 0.0,   # not used for clinical (no template)
        drop_prob: float = 0.1,
        region_dropout_prob: float = 0.0,
        point_jitter_std: float = 0.0,
    ):
        # Bypass FragmentDataBase.__init__ — we build our own index
        Dataset.__init__(self)

        self.data_root = data_root_real
        self.split = split
        self.npoints_per_bone = npoints_per_bone
        self.min_parts = min_parts
        self.max_parts = max_parts
        self.max_bone = max_bone
        self.random_anchor = random_anchor
        self.merge_prob = merge_prob
        self.frac_prob = frac_prob
        self.drop_prob = drop_prob
        self.region_dropout_prob = region_dropout_prob
        self.point_jitter_std = point_jitter_std
        self.BONE_TYPE_MAP = BONE_TYPE_MAP

        if not os.path.exists(data_root_real):
            if os.environ.get("LOCAL_RANK", "0") == "0":
                print(f"[Warning] Clinical data root does not exist: {data_root_real}")

        self.samples = self._discover_samples()
        if os.environ.get("LOCAL_RANK", "0") == "0":
            print(f"[Clinical] Split: {split}, Samples: {len(self.samples)}")

        # Build index_map (list of sample paths) — reshuffle randomises order
        self.index_map = list(self.samples)
        self.reshuffle()

    def _discover_samples(self) -> list:
        """Return list of (sample_name, obj_path, json_path) for valid samples."""
        root = Path(self.data_root)
        if not root.is_dir():
            return []

        all_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
        n = len(all_dirs)
        split_idx = int(n * 0.9)

        if self.split == "train":
            dirs = all_dirs[:split_idx]
        elif self.split == "val":
            dirs = all_dirs[split_idx:]
        else:
            dirs = all_dirs

        samples = []
        for d in dirs:
            obj_files = sorted(d.glob("*.obj"))
            json_path = d / GT_JSON_NAME
            if not obj_files or not json_path.exists():
                continue
            samples.append((d.name, str(obj_files[0]), str(json_path)))
        return samples

    def reshuffle(self):
        if self.split not in ["val", "test"]:
            random.shuffle(self.index_map)
        else:
            rng = random.Random(42)
            rng.shuffle(self.index_map)

    def _get_rng(self, key: str):
        if self.split in ["val", "test"]:
            seed = int(hashlib.md5(key.encode()).hexdigest(), 16) % (2 ** 32)
            return random.Random(seed), np.random.RandomState(seed)
        return random, np.random

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        sample_name, obj_path, json_path = self.index_map[index]
        rng, np_rng = self._get_rng(f"{sample_name}_{index}")

        # Load meshes and GT poses
        scene = trimesh.load(obj_path, split_object=True, process=False)
        gt_poses = _load_gt_poses(json_path)

        # Collect fragments per bone type
        bone_fragments: dict[str, list] = {bt: [] for bt in ["SA", "LI", "RI"]}
        for key, geom in scene.geometry.items():
            try:
                key_int = int(key)
            except (ValueError, TypeError):
                continue
            for bt, (lo, hi) in BONE_RANGES.items():
                if lo <= key_int <= hi:
                    if isinstance(geom, trimesh.Trimesh) and str(key) in gt_poses:
                        bone_fragments[bt].append((str(key), geom))
                    break

        # Build assembled GT point cloud
        all_points_gt, all_normals_gt = [], []
        all_bone_types, all_points_per_part, all_points_per_bone = [], [], []
        all_point_bone_ids = []
        num_parts = 0

        for bone_idx, bt in enumerate(["SA", "LI", "RI"]):
            frags = sorted(bone_fragments[bt], key=lambda x: int(x[0]))
            if not frags:
                continue

            n_frags = len(frags)
            pts_per_frag = max(self.npoints_per_bone // n_frags, 20)
            bone_total = 0

            for frag_key, mesh in frags:
                T_gt = gt_poses[frag_key]
                pts, normals = _sample_mesh_proportional(mesh, pts_per_frag)

                # Apply GT pose: scattered → assembled
                pts_gt = _apply_T(pts, T_gt).astype(np.float32)
                normals_gt = _apply_T_normals(normals, T_gt).astype(np.float32)

                all_points_gt.append(pts_gt)
                all_normals_gt.append(normals_gt)
                all_bone_types.append(BONE_TYPE_MAP[bt])
                all_points_per_part.append(len(pts_gt))
                all_point_bone_ids.append(np.full(len(pts_gt), bone_idx, dtype=np.int32))
                bone_total += len(pts_gt)
                num_parts += 1

            if bone_total > 0:
                all_points_per_bone.append(bone_total)

        if num_parts == 0:
            raise RuntimeError(f"No valid fragments found in {obj_path}")

        frac_dict = {
            "name": sample_name,
            "num_parts": num_parts,
            "pointclouds_gt": np.concatenate(all_points_gt, axis=0),
            "pointclouds_normals_gt": np.concatenate(all_normals_gt, axis=0),
            "boneType": all_bone_types,
            "points_per_part": all_points_per_part,
            "points_per_bone": all_points_per_bone,
            "point_bone_ids": np.concatenate(all_point_bone_ids, axis=0),
            "anon_bone_ids_per_part": [
                i for i, bt in enumerate(["SA", "LI", "RI"])
                for _ in range(len(bone_fragments[bt]))
                if bone_fragments[bt]
            ],
        }

        # Reuse simulation augmentation pipeline
        from .base import remove_extra_parts
        frac_dict = remove_extra_parts(frac_dict, max_parts=self.max_parts)
        return self.transform(frac_dict, rng=rng, np_rng=np_rng)
