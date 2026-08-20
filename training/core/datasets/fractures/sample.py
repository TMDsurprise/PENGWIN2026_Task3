from typing import List, Tuple
import random

import numpy as np
from .base import FragmentDataBase
from ..transform import *


class FragmentSamples(FragmentDataBase):
    """Fracture dataset with augmentation: centering, rotation, mirroring, merging.

    Sim2Real augmentations (applied to scattered input only, not GT):
        region_dropout_prob: probability of dropping a spherical region per fragment.
        point_jitter_std: std of Gaussian noise added to input coordinates.
    """

    def __init__(self, *args, region_dropout_prob: float = 0.0, point_jitter_std: float = 0.0, **kwargs):
        self.region_dropout_prob = region_dropout_prob
        self.point_jitter_std = point_jitter_std
        super().__init__(*args, **kwargs)

    def get_graph(self, pointclouds_gt_In: np.ndarray, points_per_part: np.ndarray, num_parts: int,
                  distance_threshold: float = 0.1, min_adjacent_points: int = 5) -> np.ndarray:
        """Build adjacency graph between fragments using KDTree query_ball_tree."""
        pointclouds_gt = pointclouds_gt_In.copy()
        pointclouds_gt, _ = rescale_pc(pointclouds_gt)
        graph = np.zeros((self.max_parts, self.max_parts), dtype=bool)

        if num_parts <= 1:
            return graph

        offsets = np.concatenate([[0], np.cumsum(points_per_part[:num_parts])])
        part_pointclouds = []

        for i in range(num_parts):
            start, end = offsets[i], offsets[i + 1]
            part_pointclouds.append(pointclouds_gt[start:end])

        from scipy.spatial import cKDTree

        trees = []
        for i in range(num_parts):
            if len(part_pointclouds[i]) > 0:
                trees.append(cKDTree(part_pointclouds[i]))
            else:
                trees.append(None)

        for i in range(num_parts):
            if trees[i] is None:
                continue

            for j in range(i + 1, num_parts):
                if trees[j] is None:
                    continue

                indices_i = trees[i].query_ball_tree(trees[j], distance_threshold)

                adjacent_pairs = 0
                for idx_list in indices_i:
                    adjacent_pairs += len(idx_list)

                if adjacent_pairs >= min_adjacent_points:
                    graph[i, j] = graph[j, i] = True

        return graph

    def _merge_adjacent_parts(self, pointclouds_gt, pointclouds_normals_gt, point_bone_ids,
                              points_per_part, num_parts, adj_graph, boneType, np_rng=None):
        """Merge adjacent fragments with 60% probability per adjacent pair."""
        if np_rng is None:
            np_rng = np.random

        adjacent_pairs = []
        for i in range(num_parts):
            for j in range(i + 1, num_parts):
                if adj_graph[i, j]:
                    adjacent_pairs.append((i, j))

        if not adjacent_pairs:
            return pointclouds_gt, pointclouds_normals_gt, point_bone_ids, \
                points_per_part, num_parts, adj_graph, boneType

        np_rng.shuffle(adjacent_pairs)
        merged_groups = [[i] for i in range(num_parts)]
        merged_indices = set()

        for i, j in adjacent_pairs:
            if i not in merged_indices and j not in merged_indices:
                if np_rng.rand() < 0.6:
                    group_i, group_j = None, None
                    for group in merged_groups:
                        if i in group: group_i = group
                        if j in group: group_j = group

                    if group_i is not None and group_j is not None and group_i != group_j:
                        merged_groups.remove(group_j)
                        group_i.extend(group_j)
                        merged_indices.add(i)
                        merged_indices.add(j)

        if len(merged_groups) == num_parts:
            return pointclouds_gt, pointclouds_normals_gt, point_bone_ids, \
                points_per_part, num_parts, adj_graph, boneType

        new_num_parts = len(merged_groups)
        new_points_per_part = np.zeros(new_num_parts, dtype=np.int32)

        new_pointclouds_gt, new_pointclouds_normals_gt = [], []
        new_point_bone_ids = []
        new_boneType = []

        original_offsets = np.concatenate([[0], np.cumsum(points_per_part)])

        for group_idx, group in enumerate(merged_groups):
            group_points, group_normals = [], []
            group_bids = []

            new_boneType.append(boneType[group[0]])

            for part_idx in group:
                start = original_offsets[part_idx]
                end = original_offsets[part_idx + 1]

                group_points.append(pointclouds_gt[start:end])
                group_normals.append(pointclouds_normals_gt[start:end])
                group_bids.append(point_bone_ids[start:end])

            merged_points = np.concatenate(group_points)
            merged_normals = np.concatenate(group_normals)
            merged_bids = np.concatenate(group_bids)

            new_points_per_part[group_idx] = len(merged_points)

            new_pointclouds_gt.append(merged_points)
            new_pointclouds_normals_gt.append(merged_normals)
            new_point_bone_ids.append(merged_bids)

        final_pointclouds_gt = np.concatenate(new_pointclouds_gt)
        final_pointclouds_normals_gt = np.concatenate(new_pointclouds_normals_gt)
        final_point_bone_ids = np.concatenate(new_point_bone_ids)

        new_adj_graph = np.zeros((self.max_parts, self.max_parts), dtype=bool)
        for i in range(new_num_parts):
            for j in range(i + 1, new_num_parts):
                group_i, group_j = merged_groups[i], merged_groups[j]
                is_adjacent = any(adj_graph[pi, pj] for pi in group_i for pj in group_j)
                if is_adjacent:
                    new_adj_graph[i, j] = new_adj_graph[j, i] = True

        return final_pointclouds_gt, final_pointclouds_normals_gt, final_point_bone_ids, \
            new_points_per_part, new_num_parts, new_adj_graph, new_boneType

    def anchor_reorder(self, pointclouds_gt, pointclouds_normals_gt, point_bone_ids, offsets,
                       num_parts, boneType, np_rng=None):
        """Put a reliable anchor first, then randomly order remaining parts."""
        if np_rng is None:
            np_rng = np.random
        points_per_part = offsets[1:] - offsets[:-1]

        parts, normals_parts, bids_parts = [], [], []
        for i in range(num_parts):
            s, e = offsets[i], offsets[i + 1]
            parts.append(pointclouds_gt[s:e])
            normals_parts.append(pointclouds_normals_gt[s:e])
            bids_parts.append(point_bone_ids[s:e])

        bone_array = np.asarray(boneType, dtype=np.int64)
        draw = float(np_rng.rand())
        sa_indices = np.flatnonzero(bone_array == 0)
        non_sa_indices = np.flatnonzero(bone_array != 0)
        if draw < 0.85 and len(sa_indices):
            anchor_idx = int(sa_indices[np.argmax(points_per_part[sa_indices])])
        elif draw < 0.95 and len(non_sa_indices):
            anchor_idx = int(non_sa_indices[np.argmax(points_per_part[non_sa_indices])])
        else:
            anchor_idx = int(np_rng.randint(0, num_parts))

        remaining = np.asarray([i for i in range(num_parts) if i != anchor_idx], dtype=np.int64)
        np_rng.shuffle(remaining)
        indices = np.concatenate([[anchor_idx], remaining])

        parts = [parts[i] for i in indices]
        normals_parts = [normals_parts[i] for i in indices]
        bids_parts = [bids_parts[i] for i in indices]
        points_per_part = points_per_part[indices]
        boneType = [boneType[i] for i in indices]

        pointclouds_gt = np.concatenate(parts, axis=0)
        pointclouds_normals_gt = np.concatenate(normals_parts, axis=0)
        point_bone_ids = np.concatenate(bids_parts, axis=0)

        offsets = np.concatenate([[0], np.cumsum(points_per_part[:num_parts])])

        return pointclouds_gt, pointclouds_normals_gt, point_bone_ids, offsets, points_per_part, boneType

    def swap_bone_types(self, bone_type):
        """Swap bone type labels: 1 <-> 2, 0 remains unchanged."""
        if isinstance(bone_type, (list, np.ndarray)):
            swapped = []
            for bt in bone_type:
                if bt == 1:
                    swapped.append(2)
                elif bt == 2:
                    swapped.append(1)
                else:
                    swapped.append(bt)
            return swapped
        else:
            if bone_type == 1:
                return 2
            elif bone_type == 2:
                return 1
            else:
                return bone_type

    def _pad_data(self, input_data: np.ndarray):
        """Pad data to shape [self.max_parts, ...]."""
        d = np.array(input_data)
        pad_shape = (self.max_parts,) + tuple(d.shape[1:])
        pad_data = np.zeros(pad_shape, dtype=np.float32)
        pad_data[: d.shape[0]] = d
        return pad_data

    def _pad_bone(self, input_data: np.ndarray):
        """Pad data to shape [self.max_bone, ...]."""
        d = np.array(input_data)
        pad_shape = (self.max_bone,) + tuple(d.shape[1:])
        pad_data = np.zeros(pad_shape, dtype=np.float32)
        pad_data[: d.shape[0]] = d
        return pad_data

    def _apply_region_dropout(self, pointclouds: np.ndarray, pointclouds_normals: np.ndarray,
                               pointclouds_gt: np.ndarray, pointclouds_normals_gt: np.ndarray,
                               point_bone_ids: np.ndarray,
                               offsets: np.ndarray, num_parts: int, np_rng) -> tuple:
        """Drop points within a random sphere for each fragment (train only).

        Radius is 0.15 in normalised coordinates. At least 10 points are kept.
        The same mask is applied to both scattered input and GT to keep them aligned.
        """
        result_pts, result_nrm = [], []
        result_gt, result_nrm_gt, result_bids = [], [], []
        new_offsets = [0]

        for i in range(num_parts):
            s, e = int(offsets[i]), int(offsets[i + 1])
            pts = pointclouds[s:e]
            nrm = pointclouds_normals[s:e]
            pts_gt = pointclouds_gt[s:e]
            nrm_gt = pointclouds_normals_gt[s:e]
            bids = point_bone_ids[s:e]

            if len(pts) > 10 and np_rng.rand() < self.region_dropout_prob:
                center = pts[np_rng.randint(0, len(pts))]
                dists = np.linalg.norm(pts - center, axis=1)
                keep = dists > 0.15
                if keep.sum() >= 10:
                    pts = pts[keep]
                    nrm = nrm[keep]
                    pts_gt = pts_gt[keep]
                    nrm_gt = nrm_gt[keep]
                    bids = bids[keep]

            result_pts.append(pts)
            result_nrm.append(nrm)
            result_gt.append(pts_gt)
            result_nrm_gt.append(nrm_gt)
            result_bids.append(bids)
            new_offsets.append(new_offsets[-1] + len(pts))

        return (np.concatenate(result_pts, axis=0),
                np.concatenate(result_nrm, axis=0),
                np.concatenate(result_gt, axis=0),
                np.concatenate(result_nrm_gt, axis=0),
                np.concatenate(result_bids, axis=0),
                np.array(new_offsets, dtype=np.int64))

    def transform(self, data, recon_tgt=None, rng=None, np_rng=None):
        """Apply data augmentation: centering, random rotation, mirroring, merging."""
        if rng is None: rng = random
        if np_rng is None: np_rng = np.random

        num_parts = data["num_parts"]
        pointclouds_gt = data["pointclouds_gt"]
        pointclouds_normals_gt = data["pointclouds_normals_gt"]
        points_per_part = data["points_per_part"]
        points_per_bone = data["points_per_bone"]
        boneType = data["boneType"]

        point_bone_ids = data.get("point_bone_ids", np.zeros(len(pointclouds_gt), dtype=np.int32))
        offsets = np.concatenate([[0], np.cumsum(points_per_part)])

        pointclouds_gt, translation_init = recenter_pc(pointclouds_gt)

        if self.split == 'train' and rng.random() < 0.5:
            pointclouds_gt, pointclouds_normals_gt, quat_flip = flip_x_axis(pointclouds_gt, pointclouds_normals_gt)
            boneType = self.swap_bone_types(boneType)
        else:
            quat_flip = None

        pointclouds_gt, pointclouds_normals_gt, init_rot, rot_mat = rotate_pc(
            pointclouds_gt,
            pointclouds_normals_gt,
            numpy_rng=np_rng
        )

        if num_parts > 2 and np_rng.rand() < self.merge_prob:
            adj_graph = self.get_graph(pointclouds_gt, points_per_part, num_parts)

            pointclouds_gt, pointclouds_normals_gt, point_bone_ids, \
                points_per_part, num_parts, adj_graph, boneType = self._merge_adjacent_parts(
                pointclouds_gt, pointclouds_normals_gt, point_bone_ids,
                points_per_part, num_parts, adj_graph, boneType, np_rng=np_rng
            )
            offsets = np.concatenate([[0], np.cumsum(points_per_part)])

        if self.random_anchor:
            pointclouds_gt, pointclouds_normals_gt, point_bone_ids, offsets, points_per_part, boneType = (
                self.anchor_reorder(pointclouds_gt, pointclouds_normals_gt, point_bone_ids, offsets, num_parts,
                                    boneType, np_rng=np_rng))

        fragment_diameters_mm = []
        for part_idx in range(num_parts):
            start, end = int(offsets[part_idx]), int(offsets[part_idx + 1])
            part = pointclouds_gt[start:end]
            fragment_diameters_mm.append(float(np.linalg.norm(part.max(axis=0) - part.min(axis=0))))
        fragment_diameters_mm = np.asarray(fragment_diameters_mm, dtype=np.float32)

        pose_mode = data.get("pose_mode", "baseline")

        (pointclouds_gt, pointclouds_normals_gt, point_bone_ids, pointclouds, pointclouds_normals,
         quaternions, translations, norm_center, norm_scale, gt_pose_rot_deg,
         gt_pose_translation_mm, qsmall_target_mask) = process_fragments_clinical_pose(
            pointclouds_gt,
            pointclouds_normals_gt,
            point_bone_ids,
            offsets,
            np_rng=np_rng,
            bone_types=boneType,
            fragment_diameters_mm=fragment_diameters_mm,
            sampler_mode=pose_mode,
        )

        # Sim2Real augmentations — applied to both input and GT (same mask keeps them aligned)
        if self.split == 'train':
            if self.region_dropout_prob > 0:
                pointclouds, pointclouds_normals, pointclouds_gt, pointclouds_normals_gt, point_bone_ids, new_offsets = \
                    self._apply_region_dropout(
                        pointclouds, pointclouds_normals,
                        pointclouds_gt, pointclouds_normals_gt, point_bone_ids,
                        offsets, num_parts, np_rng
                    )
                points_per_part = list(new_offsets[1:] - new_offsets[:-1])
            if self.point_jitter_std > 0:
                pointclouds = pointclouds + np_rng.normal(
                    0, self.point_jitter_std, pointclouds.shape
                ).astype(np.float32)

        points_per_part = self._pad_data(points_per_part).astype(np.int32)
        quaternions = self._pad_data(quaternions).astype(np.float32)
        translations = self._pad_data(translations).astype(np.float32)
        fragment_diameters_mm = self._pad_data(fragment_diameters_mm).astype(np.float32)
        gt_pose_rot_deg = self._pad_data(gt_pose_rot_deg).astype(np.float32)
        gt_pose_translation_mm = self._pad_data(gt_pose_translation_mm).astype(np.float32)
        qsmall_target_mask = self._pad_data(qsmall_target_mask).astype(np.float32)
        points_per_bone = self._pad_bone(points_per_bone).astype(np.int32)
        boneType = self._pad_data(boneType).astype(np.int32)

        last_name = data['name'].replace(".npz", "")

        data = {
            "name": last_name,
            "num_parts": num_parts,
            "pointclouds": pointclouds,
            "pointclouds_normals": pointclouds_normals.astype(np.float32),
            "points_per_part": points_per_part,
            "points_per_bone": points_per_bone,
            "bonetype": boneType,
            "point_bone_ids": point_bone_ids,
            "pointclouds_gt": pointclouds_gt.astype(np.float32),
            "quat_gt": quaternions,
            "trans_gt": translations,
            "norm_center": norm_center,
            "norm_scale": norm_scale,
            "fragment_diameter_mm": fragment_diameters_mm,
            "gt_pose_rot_deg": gt_pose_rot_deg,
            "gt_pose_translation_mm": gt_pose_translation_mm,
            "qsmall_target_mask": qsmall_target_mask,
            "sample_index": np.int64(data.get("sample_index", -1)),
            "augmentation_seed": np.int64(data.get("augmentation_seed", -1)),
            "pose_mode_id": np.int64({
                "baseline": 0,
                "natural": 1,
                "proposal": 2,
                "qsmall_hard": 3,
            }.get(pose_mode, 0)),
            "sample_source_id": np.int64(data.get("sample_source_id", 0)),
        }

        return data
