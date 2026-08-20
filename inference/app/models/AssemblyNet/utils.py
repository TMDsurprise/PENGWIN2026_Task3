"""Transformation utility functions for 3D point clouds and poses."""
import torch
import torch.nn.functional as F
import os
import numpy as np
import trimesh

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Converts quaternions to rotation matrices."""
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1).clamp(min=1e-6)  # 1e-8 underflows in fp16
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def apply_pose_to_points(input_coords: torch.Tensor, trans: torch.Tensor, quat: torch.Tensor,
                         part_batch_ids: torch.Tensor) -> torch.Tensor:
    """Applies fragment-level 7D pose (translations and quaternions) to point-level coordinates."""
    point_trans = trans[part_batch_ids]
    point_quat = quat[part_batch_ids]

    point_quat = F.normalize(point_quat, p=2, dim=-1, eps=1e-6)
    rot_matrices = quaternion_to_matrix(point_quat)
    transformed_coords = torch.einsum('nij,nj->ni', rot_matrices, input_coords) + point_trans
    return transformed_coords


def enforce_rigid_transform_svd(src_points: torch.Tensor, pred_points: torch.Tensor,
                                part_batch_ids: torch.Tensor) -> torch.Tensor:
    """
    Uses Kabsch-Umeyama algorithm (SVD) to extract strictly rigid transformations
    from predicted coordinates and applies them back to source points.
    Ensures zero deformation when output_type == "coords".
    """
    orig_dtype = src_points.dtype
    device_type = src_points.device.type

    with torch.autocast(device_type=device_type, enabled=False):
        src_p_f32 = src_points.to(torch.float32)
        pred_p_f32 = pred_points.to(torch.float32)

        num_parts = part_batch_ids.max().item() + 1
        rigid_points_f32 = torch.zeros_like(src_p_f32)

        for i in range(num_parts):
            mask = part_batch_ids == i
            src_p = src_p_f32[mask]
            pred_p = pred_p_f32[mask]

            if src_p.shape[0] < 3:
                rigid_points_f32[mask] = pred_p
                continue

            src_mean = src_p.mean(dim=0, keepdim=True)
            pred_mean = pred_p.mean(dim=0, keepdim=True)

            src_c = src_p - src_mean
            pred_c = pred_p - pred_mean

            H = src_c.T @ pred_c

            U, S, V = torch.linalg.svd(H)
            R = V @ U.T

            if torch.det(R) < 0:
                V[:, 2] *= -1
                R = V @ U.T

            t = pred_mean.squeeze() - src_mean.squeeze() @ R.T

            rigid_points_f32[mask] = src_p @ R.T + t

    return rigid_points_f32.to(orig_dtype)


def output_results(pred_assemble, data_dict, output_dict, save_path, save_all=False, save_gt=True):
    """Save prediction and ground truth results to xyzlabel.txt format."""
    sample_name = data_dict['name'][0] if isinstance(data_dict['name'], (list, tuple)) else data_dict['name']

    if "points_per_sample" in data_dict:
        n_points = data_dict["points_per_sample"][0]
        if torch.is_tensor(n_points):
            n_points = n_points.item()
    else:
        n_points = data_dict['pointclouds'].shape[0]

    raw_labels = output_dict["part_batch_ids"][:n_points].detach().cpu().numpy()
    labels = (raw_labels - raw_labels.min()).reshape(-1, 1)

    pts_input = data_dict['pointclouds'][:n_points].detach().cpu().float().numpy()
    pts_gt = data_dict['pointclouds_gt'][:n_points].detach().cpu().float().numpy()
    pts_assemble = pred_assemble[:n_points].detach().cpu().float().numpy()

    data_input = np.hstack((pts_input, labels))
    data_assemble = np.hstack((pts_assemble, labels))
    data_gt = np.hstack((pts_gt, labels))

    fmt = "%.6f %.6f %.6f %d"
    np.savetxt(os.path.join(save_path, f"{sample_name}_input.txt"), data_input, fmt=fmt)
    np.savetxt(os.path.join(save_path, f"{sample_name}_pred_assemble.txt"), data_assemble, fmt=fmt)

    if save_gt:
        np.savetxt(os.path.join(save_path, f"{sample_name}_gt.txt"), data_gt, fmt=fmt)

    if "pred_coords" in output_dict:
        pts_pred = output_dict['pred_coords'][:n_points].detach().cpu().float().numpy()
        data_pred = np.hstack((pts_pred, labels))
        np.savetxt(os.path.join(save_path, f"{sample_name}_pred_coords.txt"), data_pred, fmt=fmt)


def check_visualization(coords1, part_batch_ids, coords2=None):
    """Visualization utility for point clouds with color-coded fragment IDs."""
    def to_numpy(data):
        if torch.is_tensor(data):
            return data.detach().cpu().numpy()
        return np.array(data)

    c1 = to_numpy(coords1)
    p_ids = to_numpy(part_batch_ids)

    scene = trimesh.Scene()

    if coords2 is None:
        unique_ids = np.unique(p_ids)
        colors = np.zeros((len(c1), 4), dtype=np.uint8)

        for uid in unique_ids:
            mask = (p_ids == uid)
            random_color = np.append(np.random.randint(50, 255, size=3), 255)
            colors[mask] = random_color

        pc1 = trimesh.PointCloud(c1, colors=colors)
        scene.add_geometry(pc1)
        print(f"Single point cloud visualization: {len(unique_ids)} fragments.")

    else:
        c2 = to_numpy(coords2)

        color1 = np.array([255, 50, 50, 255], dtype=np.uint8)
        colors_arr1 = np.tile(color1, (len(c1), 1))

        color2 = np.array([50, 255, 50, 255], dtype=np.uint8)
        colors_arr2 = np.tile(color2, (len(c2), 1))

        pc1 = trimesh.PointCloud(c1, colors=colors_arr1)
        pc2 = trimesh.PointCloud(c2, colors=colors_arr2)

        scene.add_geometry(pc1)
        scene.add_geometry(pc2)
        print("Dual point cloud comparison: Red = Input 1, Green = Input 2.")

    try:
        scene.show()
    except ImportError:
        print("[Visualization] trimesh viewer unavailable (headless server), skipping show()")

def debug_visualize_pred_assemble(pred_assemble, part_batch_ids):
    """Debug: visualize predicted assembled point cloud with different colors per fragment (trimesh)."""
    pts = pred_assemble.detach().cpu().numpy()
    ids = part_batch_ids.detach().cpu().numpy()

    # Dark10 palette (same as vis_func.py)
    dark10_colors = [
        [0.1216, 0.4667, 0.7059],
        [1.0000, 0.4980, 0.0549],
        [0.1725, 0.6275, 0.1725],
        [0.8392, 0.1529, 0.1569],
        [0.5804, 0.4039, 0.7412],
        [0.4980, 0.4980, 0.4980],
        [0.0902, 0.7451, 0.8118],
        [0.6902, 0.3490, 0.1922],
        [0.6510, 0.6510, 0.6510],
        [0.9686, 0.7137, 0.8235],
    ]

    scene = trimesh.Scene()
    unique_ids = np.unique(ids)

    for idx, part_id in enumerate(unique_ids):
        mask = ids == part_id
        part_pts = pts[mask]
        color = dark10_colors[idx % len(dark10_colors)]
        color_uint8 = (np.array(color) * 255).astype(np.uint8)
        pc = trimesh.PointCloud(part_pts, colors=np.tile(color_uint8, (len(part_pts), 1)))
        scene.add_geometry(pc, geom_name=f"part_{part_id}")

    try:
        scene.show()
    except ImportError:
        print("[Visualization] trimesh viewer unavailable (headless server), skipping show()")


def debug_visualize_coords_vs_pred(input_coords, pred_coords, part_batch_ids):
    """
    Debug: visualize input_coords (blue) and pred_coords (red) overlaid per fragment.

    Args:
        input_coords: (N, 3) tensor, original fragment point clouds
        pred_coords: (N, 3) tensor, predicted assembled point cloud
        part_batch_ids: (N,) tensor, fragment ID per point
    """
    inp = input_coords.detach().cpu().numpy()
    pred = pred_coords.detach().cpu().numpy()
    ids = part_batch_ids.detach().cpu().numpy()

    # Dark10 palette (same as vis_func.py)
    dark10_colors = [
        [0.1216, 0.4667, 0.7059],
        [1.0000, 0.4980, 0.0549],
        [0.1725, 0.6275, 0.1725],
        [0.8392, 0.1529, 0.1569],
        [0.5804, 0.4039, 0.7412],
        [0.4980, 0.4980, 0.4980],
        [0.0902, 0.7451, 0.8118],
        [0.6902, 0.3490, 0.1922],
        [0.6510, 0.6510, 0.6510],
        [0.9686, 0.7137, 0.8235],
    ]

    scene = trimesh.Scene()
    unique_ids = np.unique(ids)

    for idx, part_id in enumerate(unique_ids):
        mask = ids == part_id
        base_color = dark10_colors[idx % len(dark10_colors)]

        # Input = blue-ish tint
        inp_pts = inp[mask]
        inp_color = [int(c * 200) for c in base_color] + [255]
        inp_color[0] = min(255, int(base_color[0] * 1.5 * 255))  # boost blue channel
        pc_inp = trimesh.PointCloud(inp_pts, colors=np.tile(inp_color, (len(inp_pts), 1)))
        scene.add_geometry(pc_inp, geom_name=f"inp_part_{part_id}")

        # Pred = red-ish tint
        pred_pts = pred[mask]
        pred_color = [255, int(base_color[1] * 80), int(base_color[2] * 80), 255]
        pc_pred = trimesh.PointCloud(pred_pts, colors=np.tile(pred_color, (len(pred_pts), 1)))
        scene.add_geometry(pc_pred, geom_name=f"pred_part_{part_id}")

    try:
        scene.show()
    except ImportError:
        print("[Visualization] trimesh viewer unavailable (headless server), skipping show()")


def extract_poses_from_coords(input_coords, pred_coords, part_batch_ids):
    """
    Extract per-fragment rigid poses via SVD from matched point pairs.
    Uses Kabsch-Umeyama algorithm: finds R,t such that pred = R @ input + t.

    Args:
        input_coords: (N, 3) tensor, original fragment point clouds (in normalized space)
        pred_coords: (N, 3) tensor, predicted assembled coordinates (in normalized space)
        part_batch_ids: (N,) tensor, fragment ID per point

    Returns:
        quat: (num_valid_parts, 4) tensor, unit quaternions (w,x,y,z) per fragment
        trans: (num_valid_parts, 3) tensor, translations per fragment
    """
    inp = input_coords.detach().float()
    pred = pred_coords.detach().float()
    ids = part_batch_ids.detach()

    num_valid_parts = ids.max().item() + 1
    quat = torch.zeros(num_valid_parts, 4, dtype=torch.float32, device=input_coords.device)
    trans = torch.zeros(num_valid_parts, 3, dtype=torch.float32, device=input_coords.device)

    with torch.autocast(device_type=input_coords.device.type, enabled=False):
        for part_id in range(num_valid_parts):
            mask = ids == part_id
            src = inp[mask]
            dst = pred[mask]

            if src.shape[0] < 3:
                quat[part_id] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=input_coords.device)
                trans[part_id] = torch.zeros(3, device=input_coords.device)
                continue

            src_mean = src.mean(dim=0, keepdim=True)
            dst_mean = dst.mean(dim=0, keepdim=True)

            src_c = src - src_mean
            dst_c = dst - dst_mean

            H = src_c.T @ dst_c  # (3, 3)

            U, S, Vt = torch.linalg.svd(H)
            R = Vt.T @ U.T

            if torch.det(R) < 0:
                Vt = Vt.clone()
                Vt[2, :] *= -1
                R = Vt.T @ U.T

            t = dst_mean.squeeze() - src_mean.squeeze() @ R.T

            # Convert R -> quaternion (w, x, y, z)
            trace = R.trace()
            if trace > 0:
                s = 0.5 / torch.sqrt(trace + 1.0)
                w = 0.25 / s
                x = (R[2, 1] - R[1, 2]) * s
                y = (R[0, 2] - R[2, 0]) * s
                z = (R[1, 0] - R[0, 1]) * s
            elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s

            # Use torch.stack instead of torch.tensor([]) to preserve computation graph
            q = torch.stack([w, x, y, z])
            q = q / q.norm()
            quat[part_id] = q
            trans[part_id] = t

    return quat, trans


def extract_poses_from_coords_grad(input_coords, pred_coords, part_batch_ids):
    """Same as extract_poses_from_coords but keeps gradient on pred_coords for loss backprop."""
    inp = input_coords.detach().float()
    pred = pred_coords.float()
    ids = part_batch_ids.detach()

    num_valid_parts = ids.max().item() + 1
    quat = torch.zeros(num_valid_parts, 4, dtype=torch.float32, device=input_coords.device)
    trans = torch.zeros(num_valid_parts, 3, dtype=torch.float32, device=input_coords.device)

    with torch.autocast(device_type=input_coords.device.type, enabled=False):
        for part_id in range(num_valid_parts):
            mask = ids == part_id
            src = inp[mask]
            dst = pred[mask]

            if src.shape[0] < 3:
                quat[part_id] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=input_coords.device, dtype=torch.float32)
                trans[part_id] = torch.zeros(3, device=input_coords.device, dtype=torch.float32)
                continue

            src_mean = src.mean(dim=0, keepdim=True)
            dst_mean = dst.mean(dim=0, keepdim=True)

            src_c = src - src_mean
            dst_c = dst - dst_mean

            H = src_c.T @ dst_c

            U, S, Vt = torch.linalg.svd(H)
            R = Vt.T @ U.T

            if torch.det(R) < 0:
                Vt = Vt.clone()
                Vt[2, :] *= -1
                R = Vt.T @ U.T

            t = dst_mean.squeeze() - src_mean.squeeze() @ R.T

            trace = R.trace()
            if trace > 0:
                s = 0.5 / torch.sqrt(trace + 1.0)
                w = 0.25 / s
                x = (R[2, 1] - R[1, 2]) * s
                y = (R[0, 2] - R[2, 0]) * s
                z = (R[1, 0] - R[0, 1]) * s
            elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s

            q = torch.stack([w, x, y, z])
            q = q / q.norm()
            quat[part_id] = q
            trans[part_id] = t

    return quat, trans


def rot_geodesic_deg(quat_pred: torch.Tensor, quat_gt: torch.Tensor) -> torch.Tensor:
    """Geodesic rotation error in degrees. Both inputs (N,4) in [w,x,y,z] format."""
    quat_pred = F.normalize(quat_pred, p=2, dim=-1)
    quat_gt = F.normalize(quat_gt, p=2, dim=-1)
    R_pred = quaternion_to_matrix(quat_pred)
    R_gt = quaternion_to_matrix(quat_gt)
    R_rel = torch.bmm(R_pred, R_gt.transpose(1, 2))
    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    return torch.rad2deg(torch.acos(cos_angle))