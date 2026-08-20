import random

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation as SciPyRot
from scipy.stats import truncnorm


CLINICAL_ROTATION_PROFILES = {
    # Smoothed patient-level clinical frequencies for
    # [0, 15), [15, 45), [45, 90), [90, 140] degrees.
    "small": {
        0: np.array([0.409, 0.384, 0.134, 0.073]),
        1: np.array([0.344, 0.420, 0.186, 0.050]),
        2: np.array([0.363, 0.432, 0.152, 0.053]),
    },
    "medium": {
        0: np.array([0.850, 0.136, 0.014, 0.000]),
        1: np.array([0.855, 0.130, 0.015, 0.000]),
        2: np.array([0.803, 0.188, 0.009, 0.000]),
    },
    "large": {
        0: np.array([0.986, 0.014, 0.000, 0.000]),
        1: np.array([0.918, 0.082, 0.000, 0.000]),
        2: np.array([0.915, 0.085, 0.000, 0.000]),
    },
}

PROPOSAL_ROTATION_PROFILES = {
    "small": {
        0: np.array([0.15, 0.30, 0.35, 0.20]),
        1: np.array([0.10, 0.25, 0.45, 0.20]),
        2: np.array([0.10, 0.25, 0.45, 0.20]),
    },
    "medium": {
        0: np.array([0.50, 0.35, 0.15, 0.00]),
        1: np.array([0.45, 0.35, 0.20, 0.00]),
        2: np.array([0.40, 0.35, 0.25, 0.00]),
    },
    "large": {
        0: np.array([0.85, 0.15, 0.00, 0.00]),
        1: np.array([0.70, 0.30, 0.00, 0.00]),
        2: np.array([0.70, 0.30, 0.00, 0.00]),
    },
}

ROTATION_INTERVALS_DEG = ((0.0, 15.0), (15.0, 45.0), (45.0, 90.0), (90.0, 140.0))


def _fragment_size_bin(diameter_mm):
    if diameter_mm < 60.0:
        return "small"
    if diameter_mm < 120.0:
        return "medium"
    return "large"


def _random_unit_vector(np_rng):
    direction = np_rng.normal(size=3)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return direction / norm


def _sample_conditioned_pose(pointcloud, bone_type, diameter_mm, sampler_mode, bone_context, np_rng):
    """Sample a size/bone-conditioned pose around a fragment centre.

    The proposal distribution deliberately has wider support than the natural
    clinical profile. Online hard mining decides which proposal samples deserve
    replay; a large angle alone does not make a sample hard.
    """
    size_bin = _fragment_size_bin(float(diameter_mm))
    bone_type = int(bone_type)
    profiles = PROPOSAL_ROTATION_PROFILES if sampler_mode == "proposal" else CLINICAL_ROTATION_PROFILES
    probs = profiles[size_bin].get(bone_type, profiles[size_bin][1]).astype(np.float64)
    probs /= probs.sum()

    interval_idx = int(np_rng.choice(len(ROTATION_INTERVALS_DEG), p=probs))
    low, high = ROTATION_INTERVALS_DEG[interval_idx]
    # Beta(1.5, 2.0) avoids piling samples exactly at interval boundaries.
    angle_deg = low + (high - low) * np_rng.beta(1.5, 2.0)

    shared_axis = bone_context[bone_type]["axis"]
    local_axis = _random_unit_vector(np_rng)
    axis = 0.65 * shared_axis + 0.35 * local_axis
    axis /= max(np.linalg.norm(axis), 1e-8)
    rot_mat = R.from_rotvec(np.deg2rad(angle_deg) * axis).as_matrix()

    shared_translation = bone_context[bone_type]["translation_dir"]
    local_translation = _random_unit_vector(np_rng)
    translation_dir = 0.60 * shared_translation + 0.40 * local_translation
    translation_dir /= max(np.linalg.norm(translation_dir), 1e-8)

    trans_std = min(18.0, 4.0 + 0.12 * angle_deg)
    trans_cap = min(80.0, 15.0 + 0.50 * angle_deg)
    translation_mag = min(abs(np_rng.normal(0.0, trans_std)), trans_cap)
    translation_rand = translation_dir * translation_mag

    center = np.mean(pointcloud, axis=0)
    pointcloud_transformed = (pointcloud - center) @ rot_mat.T + center + translation_rand
    translation_eq = -rot_mat @ center + center + translation_rand
    quat = R.from_matrix(rot_mat).as_quat()
    return pointcloud_transformed, quat, translation_eq, rot_mat, float(angle_deg), float(translation_mag)

def recenter_pc(pc):
    """Center point cloud by computing bounding box center."""
    centroid = (pc.max(axis=0) + pc.min(axis=0)) / 2
    return pc - centroid[None], centroid[None]


def rotate_pc(pc, normal=None, numpy_rng=None):
    """Apply random rotation to point cloud."""
    if numpy_rng is None:
        numpy_rng = np.random

    rot_mat = R.random(random_state=numpy_rng).as_matrix()
    rotated_pc = (rot_mat @ pc.T).T
    quat_gt = R.from_matrix(rot_mat.T).as_quat()
    quat_gt = quat_gt[[3, 0, 1, 2]]
    if normal is None:
        return rotated_pc, None, quat_gt, rot_mat

    rotated_normal = (rot_mat @ normal.T).T
    return rotated_pc, rotated_normal, quat_gt, rot_mat


def flip_x_axis(pc, normals=None):
    """Flip point cloud along x-axis (mirror transformation)."""
    flip_matrix = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    flipped_pc = pc @ flip_matrix.T
    quat_flip = np.array([0., 1., 0., 0.])

    if normals is None:
        return flipped_pc, None, quat_flip

    normals_flipped = normals @ flip_matrix.T
    return flipped_pc, normals_flipped, quat_flip


def rescale_pc(pc):
    """Rescale point cloud to unit sphere."""
    distances = np.linalg.norm(pc, axis=1)
    scale = np.max(distances)
    pc /= scale
    return pc, scale


def apply_clinical_pose_variation(pointcloud, max_angle_deg=15, max_translation_mm=15, seed=None, np_rng=None):
    """
    Apply random rotation around fragment center + random translation.
    Returns transformed point cloud and equivalent (R, t_eq).
    """
    if np_rng is None:
        np_rng = np.random

    center = np.mean(pointcloud, axis=0)

    def truncated_normal(mean, std, clip):
        a, b = -clip/std, clip/std
        return truncnorm.rvs(a, b, loc=mean, scale=std, random_state=np_rng)

    #angles = np.deg2rad(np_rng.uniform(-max_angle_deg, max_angle_deg, size=3))
    angles = np.deg2rad([truncated_normal(0, max_angle_deg/3 , max_angle_deg) for _ in range(3)])
    rot = R.from_euler('xyz', angles)
    rot_mat = rot.as_matrix()
    quat = rot.as_quat()

    translation_rand = np.array([truncated_normal(0, max_translation_mm/3, max_translation_mm) for _ in range(3)])
    #translation_rand = np_rng.uniform(-max_translation_mm, max_translation_mm, size=3)

    pc_centered = pointcloud - center
    pc_rot = (rot_mat @ pc_centered.T).T
    pc_rot_back = pc_rot + center
    pointcloud_transformed_seq = pc_rot_back + translation_rand

    translation_eq = -rot_mat @ center + center + translation_rand

    return pointcloud_transformed_seq, quat, translation_eq, rot_mat


def process_fragments_clinical_pose(
    pointclouds_gt,
    pointclouds_normals_gt,
    point_bone_ids,
    offset,
    np_rng=None,
    bone_types=None,
    fragment_diameters_mm=None,
    sampler_mode="baseline",
):
    """
    Apply clinical pose variation to fragments:
    1. Apply first fragment's pose to entire GT
    2. Compute recovery transforms for each fragment
    3. Apply centering and scaling globally
    """
    if np_rng is None:
        np_rng = np.random

    num_parts = len(offset) - 1

    if bone_types is None:
        bone_types = np.zeros(num_parts, dtype=np.int64)
    if fragment_diameters_mm is None:
        fragment_diameters_mm = np.full(num_parts, 120.0, dtype=np.float32)

    bone_context = {
        bone: {
            "axis": _random_unit_vector(np_rng),
            "translation_dir": _random_unit_vector(np_rng),
        }
        for bone in (0, 1, 2)
    }

    absolute_poses = []
    sampled_angles_deg = []
    sampled_translation_mm = []
    for part_idx in range(num_parts):
        start = offset[part_idx]
        end = offset[part_idx + 1]
        pc = pointclouds_gt[start:end]

        if sampler_mode in ("natural", "proposal"):
            if part_idx == 0:
                q_abs = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
                t_abs = np.zeros(3, dtype=np.float64)
                rot_abs = np.eye(3, dtype=np.float64)
                angle_deg = 0.0
                translation_mm = 0.0
            else:
                _, q_abs, t_abs, rot_abs, angle_deg, translation_mm = _sample_conditioned_pose(
                    pc,
                    bone_types[part_idx],
                    fragment_diameters_mm[part_idx],
                    sampler_mode,
                    bone_context,
                    np_rng,
                )
        else:
            _, q_abs, t_abs, rot_abs = apply_clinical_pose_variation(pc, np_rng=np_rng)
            angle_deg = float(np.rad2deg(R.from_matrix(rot_abs).magnitude()))
            center = np.mean(pc, axis=0)
            translation_mm = float(np.linalg.norm(rot_abs @ center + t_abs - center))
        absolute_poses.append({
            'rot_mat': rot_abs,
            't': t_abs
        })
        sampled_angles_deg.append(angle_deg)
        sampled_translation_mm.append(translation_mm)

    R0 = absolute_poses[0]['rot_mat']
    T0 = absolute_poses[0]['t']

    pointclouds_gt = pointclouds_gt @ R0.T + T0
    pointclouds_normals_gt = pointclouds_normals_gt @ R0.T

    pointclouds, pointclouds_normals = [], []
    forward_transforms = []

    for part_idx in range(num_parts):
        start = offset[part_idx]
        end = offset[part_idx + 1]

        pc_new_gt = pointclouds_gt[start:end]
        nm_new_gt = pointclouds_normals_gt[start:end]

        Ri = absolute_poses[part_idx]['rot_mat']
        Ti = absolute_poses[part_idx]['t']

        R_rel = Ri @ R0.T
        T_rel = Ti - (T0 @ R_rel.T)

        forward_transforms.append((R_rel, T_rel))

        pc_posed = pc_new_gt @ R_rel.T + T_rel
        nm_posed = nm_new_gt @ R_rel.T

        order = np_rng.permutation(len(pc_posed))

        pc_posed = pc_posed[order]
        nm_posed = nm_posed[order]
        pointclouds_gt[start:end] = pointclouds_gt[start:end][order]
        pointclouds_normals_gt[start:end] = pointclouds_normals_gt[start:end][order]
        point_bone_ids[start:end] = point_bone_ids[start:end][order]

        pointclouds.append(pc_posed)
        pointclouds_normals.append(nm_posed)

    pointclouds = np.concatenate(pointclouds).astype(np.float32)
    pointclouds_normals = np.concatenate(pointclouds_normals).astype(np.float32)

    pointclouds, translation_final = recenter_pc(pointclouds)
    pointclouds, scales = rescale_pc(pointclouds)
    pointclouds_gt = (pointclouds_gt - translation_final) / scales

    quaternions, translations_rec = [], []

    for R_rel, T_rel in forward_transforms:
        R_rec = R_rel.T
        T_rec = ((translation_final - T_rel) @ R_rel - translation_final) / scales

        q_xyzw = SciPyRot.from_matrix(R_rec).as_quat()
        q_rec = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)

        quaternions.append(q_rec)
        translations_rec.append(T_rec.flatten().astype(np.float32))

    quaternions = np.stack(quaternions).astype(np.float32)
    translations = np.stack(translations_rec).astype(np.float32)

    return (
        pointclouds_gt,
        pointclouds_normals_gt,
        point_bone_ids,
        pointclouds,
        pointclouds_normals,
        quaternions,
        translations,
        translation_final.reshape(3).astype(np.float32),
        np.float32(scales),
        np.asarray(sampled_angles_deg, dtype=np.float32),
        np.asarray(sampled_translation_mm, dtype=np.float32),
    )


def process_fragments_with_given_poses(
    pointclouds_gt, pointclouds_normals_gt, point_bone_ids, offset,
    per_frag_rot, per_frag_trans, np_rng=None,
):
    """Same as process_fragments_clinical_pose but uses given per-fragment poses
    instead of random ones.  Used to reconstruct real clinical fracture configurations.

    Args:
        per_frag_rot: (M, 3, 3) rotation matrices assembled→scattered per fragment.
        per_frag_trans: (M, 3) translation vectors assembled→scattered per fragment.
    """
    if np_rng is None:
        np_rng = np.random

    num_parts = len(offset) - 1

    # Build absolute poses from provided transforms
    absolute_poses = [
        {"rot_mat": per_frag_rot[i], "t": per_frag_trans[i]}
        for i in range(num_parts)
    ]

    # --- remainder is identical to process_fragments_clinical_pose ---
    R0 = absolute_poses[0]["rot_mat"]
    T0 = absolute_poses[0]["t"]

    pointclouds_gt = pointclouds_gt @ R0.T + T0
    pointclouds_normals_gt = pointclouds_normals_gt @ R0.T

    pointclouds, pointclouds_normals = [], []
    forward_transforms = []
    pose_angles_deg = []
    pose_translation_mm = []

    for part_idx in range(num_parts):
        start = offset[part_idx]
        end = offset[part_idx + 1]

        pc_new_gt = pointclouds_gt[start:end]
        nm_new_gt = pointclouds_normals_gt[start:end]

        Ri = absolute_poses[part_idx]["rot_mat"]
        Ti = absolute_poses[part_idx]["t"]

        R_rel = Ri @ R0.T
        T_rel = Ti - (T0 @ R_rel.T)

        forward_transforms.append((R_rel, T_rel))
        pose_angles_deg.append(float(np.rad2deg(SciPyRot.from_matrix(R_rel).magnitude())))
        center = np.mean(pc_new_gt, axis=0)
        moved_center = center @ R_rel.T + T_rel
        pose_translation_mm.append(float(np.linalg.norm(moved_center - center)))

        pc_posed = pc_new_gt @ R_rel.T + T_rel
        nm_posed = nm_new_gt @ R_rel.T

        order = np_rng.permutation(len(pc_posed))

        pc_posed = pc_posed[order]
        nm_posed = nm_posed[order]
        pointclouds_gt[start:end] = pointclouds_gt[start:end][order]
        pointclouds_normals_gt[start:end] = pointclouds_normals_gt[start:end][order]
        point_bone_ids[start:end] = point_bone_ids[start:end][order]

        pointclouds.append(pc_posed)
        pointclouds_normals.append(nm_posed)

    pointclouds = np.concatenate(pointclouds).astype(np.float32)
    pointclouds_normals = np.concatenate(pointclouds_normals).astype(np.float32)

    pointclouds, translation_final = recenter_pc(pointclouds)
    pointclouds, scales = rescale_pc(pointclouds)
    pointclouds_gt = (pointclouds_gt - translation_final) / scales

    quaternions, translations_rec = [], []

    for R_rel, T_rel in forward_transforms:
        R_rec = R_rel.T
        T_rec = ((translation_final - T_rel) @ R_rel - translation_final) / scales

        q_xyzw = SciPyRot.from_matrix(R_rec).as_quat()
        q_rec = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)

        quaternions.append(q_rec)
        translations_rec.append(T_rec.flatten().astype(np.float32))

    quaternions = np.stack(quaternions).astype(np.float32)
    translations = np.stack(translations_rec).astype(np.float32)

    return (
        pointclouds_gt,
        pointclouds_normals_gt,
        point_bone_ids,
        pointclouds,
        pointclouds_normals,
        quaternions,
        translations,
        translation_final.reshape(3).astype(np.float32),
        np.float32(scales),
        np.asarray(pose_angles_deg, dtype=np.float32),
        np.asarray(pose_translation_mm, dtype=np.float32),
    )
