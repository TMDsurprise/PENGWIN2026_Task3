import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import torch
import trimesh

def _get_id_colors(ids):
    """Generate a high-contrast color map for a set of IDs using matplotlib tab20."""
    unique_ids = np.unique(ids)
    cmap = plt.get_cmap("tab20")

    id_to_color = {uid: cmap(i % 20)[:3] for i, uid in enumerate(unique_ids)}

    colors = np.zeros((len(ids), 3))
    for uid in unique_ids:
        colors[ids == uid] = id_to_color[uid]

    return colors

# Dark 10-color palette for better visualization
_DARK10_COLORS = [
    [0.1216, 0.4667, 0.7059],  # Dark blue
    [1.0000, 0.4980, 0.0549],  # Orange
    [0.1725, 0.6275, 0.1725],  # Green
    [0.8392, 0.1529, 0.1569],  # Red
    [0.5804, 0.4039, 0.7412],  # Purple
    [0.4980, 0.4980, 0.4980],  # Gray
    [0.0902, 0.7451, 0.8118],  # Cyan
    [0.6902, 0.3490, 0.1922],  # Brown
    [0.6510, 0.6510, 0.6510],  # Light gray
    [0.9686, 0.7137, 0.8235],  # Pink
]


def visualize_fragments(meshdict, save_path=None):
    """
    Visualize mesh fragments using trimesh with different colors for each fragment.

    Args:
        meshdict: Dictionary with bone_type -> {frag_name: {'mesh': trimesh, 'volume': float}}
        save_path: Optional path to save the screenshot
    """
    scene = trimesh.Scene()

    color_idx = 0

    for bone_type in ["SA", "LI", "RI"]:
        fragments = meshdict.get(bone_type, {})
        for frag_name, frag_data in fragments.items():
            mesh = frag_data['mesh']

            # Use dark10 palette cyclically
            fragment_color = _DARK10_COLORS[color_idx % len(_DARK10_COLORS)]

            # Apply color to the mesh
            mesh.visual.vertex_colors[:, :3] = (np.array(fragment_color) * 255).astype(np.uint8)
            scene.add_geometry(mesh, geom_name=f"{bone_type}_{frag_name}")
            color_idx += 1

    if save_path:
        # Save screenshot
        scene.save_image(save_path)
        print(f"Saved visualization to {save_path}")

    scene.show()
    return scene


def visualize_transformed_fragments(meshdict, predictions, save_path=None):
    """
    Visualize mesh fragments after applying predicted 4x4 transformations.

    Each fragment is colored with the same Dark10 palette as visualize_fragments(),
    so before/after comparison is consistent.

    Args:
        meshdict: Dictionary with bone_type -> {frag_name: {'mesh': trimesh, 'volume': float}}
        predictions: List of dicts with 'fragment_id' and 'transformation' (4x4 matrix).
        save_path: Optional path to save the screenshot.
    """
    # Build lookup: fragment_id -> 4x4 matrix
    transform_lookup = {}
    for pred in predictions:
        transform_lookup[str(pred["fragment_id"])] = np.array(pred["transformation"])

    scene = trimesh.Scene()
    color_idx = 0

    for bone_type in ["SA", "LI", "RI"]:
        fragments = meshdict.get(bone_type, {})
        for frag_name, frag_data in fragments.items():
            mesh = frag_data['mesh'].copy()

            # Apply predicted transformation if available
            if frag_name in transform_lookup:
                mesh.apply_transform(transform_lookup[frag_name])

            fragment_color = _DARK10_COLORS[color_idx % len(_DARK10_COLORS)]
            mesh.visual.vertex_colors[:, :3] = (np.array(fragment_color) * 255).astype(np.uint8)
            scene.add_geometry(mesh, geom_name=f"{bone_type}_{frag_name}")
            color_idx += 1

    if save_path:
        scene.save_image(save_path)
        print(f"Saved visualization to {save_path}")

    scene.show()
    return scene


def visualize_pointcloud_by_part(pointclouds, points_per_part, title="Point Cloud"):
    """
    Visualize a flat point cloud colored by fragment using Dark10 palette.

    Args:
        pointclouds: (N, 3) numpy array of point coordinates.
        points_per_part: list/array of point counts per fragment.
        title: window title string.
    """
    scene = trimesh.Scene()
    offset = 0

    for i, count in enumerate(points_per_part):
        if count <= 0:
            continue
        pts = pointclouds[offset:offset + count]
        color = _DARK10_COLORS[i % len(_DARK10_COLORS)]
        color_uint8 = (np.array(color) * 255).astype(np.uint8)
        pc = trimesh.PointCloud(pts, colors=np.tile(color_uint8, (len(pts), 1)))
        scene.add_geometry(pc, geom_name=f"part_{i}")
        offset += count

    scene.show()
    return scene


def visualize_pointcloud_fragments(sampled_data, bone_type_to_id=None):
    """
    Visualize sampled point cloud fragments with different colors using trimesh.

    Args:
        sampled_data: Dictionary from sub_sample with 'coords' and 'normals' per fragment
        bone_type_to_id: Optional dict mapping bone type to ID, defaults to {"SA": 0, "LI": 1, "RI": 2}
    """
    if bone_type_to_id is None:
        bone_type_to_id = {"SA": 0, "LI": 1, "RI": 2}
    id_to_bone = {v: k for k, v in bone_type_to_id.items()}

    scene = trimesh.Scene()

    color_idx = 0

    for bone_type, fragments in sampled_data.items():
        for frag_name, frag_data in fragments.items():
            coords = frag_data['coords']

            if len(coords) == 0:
                continue

            # Use dark10 palette cyclically
            fragment_color = _DARK10_COLORS[color_idx % len(_DARK10_COLORS)]
            point_cloud = trimesh.PointCloud(coords, colors=(np.array(fragment_color) * 255).astype(np.uint8))
            scene.add_geometry(point_cloud, geom_name=f"{bone_type}_{frag_name}")
            color_idx += 1

    scene.show()
    return scene


def visualize_pointclouds_by_bone(pointclouds_gt, point_bone_ids, recon_tgt=None):
    """
    Visualize point cloud colored by bone type.

    All points belonging to the same bone (even across different fragments)
    share the same color.

    Args:
        pointclouds_gt: Ground-truth point cloud [N, 3]
        point_bone_ids: Bone ID per point [N,]
        recon_tgt: Reconstructed point cloud [N, 3] (optional)
    """
    if hasattr(pointclouds_gt, "cpu"): pointclouds_gt = pointclouds_gt.detach().cpu().numpy()
    if hasattr(point_bone_ids, "cpu"): point_bone_ids = point_bone_ids.detach().cpu().numpy()
    if recon_tgt is not None and hasattr(recon_tgt, "cpu"): recon_tgt = recon_tgt.detach().cpu().numpy()

    colors = _get_id_colors(point_bone_ids)
    all_vis_object = []

    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(pointclouds_gt)
    pcd_gt.colors = o3d.utility.Vector3dVector(colors)
    all_vis_object.append(pcd_gt)

    if recon_tgt is not None:
        pcd_recon = o3d.geometry.PointCloud()

        # Shift Reconstruction to the right by bounding-box width to avoid Z-fighting
        shift_x = np.max(pointclouds_gt[:, 0]) - np.min(pointclouds_gt[:, 0]) + 0.1
        shifted_recon = recon_tgt.copy()
        shifted_recon[:, 0] += shift_x * 1.2

        pcd_recon.points = o3d.utility.Vector3dVector(shifted_recon)
        pcd_recon.colors = o3d.utility.Vector3dVector(colors)
        all_vis_object.append(pcd_recon)

    o3d.visualization.draw_geometries(all_vis_object,
                                      window_name="Point Cloud (by Bone) - Left: GT, Right: Recon",
                                      width=1600, height=800)


def visualize_pointclouds_by_part(pointclouds_gt, points_per_part, recon_tgt=None):
    """
    Visualize point cloud colored by fragment (part).

    Points within the same fragment share a color; different fragments use distinct colors.

    Args:
        pointclouds_gt: Ground-truth point cloud [N, 3]
        points_per_part: Point counts per fragment (list or array)
        recon_tgt: Reconstructed point cloud [N, 3] (optional)
    """
    if hasattr(pointclouds_gt, "cpu"): pointclouds_gt = pointclouds_gt.detach().cpu().numpy()
    if hasattr(points_per_part, "cpu"): points_per_part = points_per_part.detach().cpu().numpy()
    if recon_tgt is not None and hasattr(recon_tgt, "cpu"): recon_tgt = recon_tgt.detach().cpu().numpy()

    valid_counts = [int(c) for c in points_per_part if c > 0]
    part_ids = np.concatenate([np.full(c, i, dtype=np.int32) for i, c in enumerate(valid_counts)])

    colors = _get_id_colors(part_ids)
    all_vis_object = []

    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(pointclouds_gt)
    pcd_gt.colors = o3d.utility.Vector3dVector(colors)
    all_vis_object.append(pcd_gt)

    if recon_tgt is not None:
        pcd_recon = o3d.geometry.PointCloud()

        shift_x = np.max(pointclouds_gt[:, 0]) - np.min(pointclouds_gt[:, 0]) + 0.1
        shifted_recon = recon_tgt.copy()
        shifted_recon[:, 0] += shift_x * 1.2

        pcd_recon.points = o3d.utility.Vector3dVector(shifted_recon)
        pcd_recon.colors = o3d.utility.Vector3dVector(colors)
        all_vis_object.append(pcd_recon)

    o3d.visualization.draw_geometries(all_vis_object,
                                      window_name="Point Cloud (by Part) - Left: GT, Right: Recon",
                                      width=1600, height=800)


from matplotlib import cm
def visualize_fracture_graph(pointclouds_gt: np.ndarray, points_per_part: np.ndarray,
                             graph: np.ndarray, num_parts: int,
                             window_name: str = "Fracture Graph Visualization"):
    """
    Visualize fracture fragments and their adjacency graph.

    Args:
        pointclouds_gt: Concatenated point clouds of all fragments [N, 3]
        points_per_part: Point counts per fragment [P]
        graph: Adjacency matrix [max_parts, max_parts]
        num_parts: Actual number of fragments
    """
    if isinstance(num_parts, (np.ndarray, np.generic)):
        if num_parts.size == 1:
            num_parts = int(num_parts.item())
        else:
            num_parts = int(num_parts[0])
    elif hasattr(num_parts, 'item'):  # Handle torch.Tensor
        num_parts = int(num_parts.item())
    else:
        num_parts = int(num_parts)

    if not isinstance(points_per_part, np.ndarray):
        points_per_part = np.array(points_per_part)

    if num_parts > len(points_per_part):
        print(f"[Warn] num_parts ({num_parts}) > len(points_per_part) ({len(points_per_part)}). Clipping.")
        num_parts = len(points_per_part)

    offsets = np.concatenate([[0], np.cumsum(points_per_part[:num_parts])])
    part_pointclouds = []

    for i in range(num_parts):
        start, end = offsets[i], offsets[i + 1]
        part_pointclouds.append(pointclouds_gt[start:end])

    colors = cm.get_cmap('tab10', num_parts)
    geometries = []

    # Draw each fragment as a colored point cloud
    for i in range(num_parts):
        if len(part_pointclouds[i]) == 0:
            continue

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(part_pointclouds[i])

        color = colors(i)[:3]
        pcd.paint_uniform_color(color)
        geometries.append(pcd)

        # Add fragment label
        centroid = np.mean(part_pointclouds[i], axis=0)
        text_mesh = _create_text_mesh(f"Part {i}", centroid, color)
        if text_mesh:
            geometries.append(text_mesh)

    # Draw adjacency edges
    line_points = []
    line_indices = []
    line_colors = []

    line_count = 0
    for i in range(num_parts):
        for j in range(i + 1, num_parts):
            if graph[i, j]:
                centroid_i = np.mean(part_pointclouds[i], axis=0)
                centroid_j = np.mean(part_pointclouds[j], axis=0)

                line_points.append(centroid_i)
                line_points.append(centroid_j)

                line_indices.append([line_count * 2, line_count * 2 + 1])
                line_colors.append([1.0, 0.0, 0.0])

                line_count += 1

    if line_points:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_points)
        line_set.lines = o3d.utility.Vector2iVector(line_indices)
        line_set.colors = o3d.utility.Vector3dVector(line_colors)
        geometries.append(line_set)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1200, height=800)

    for geometry in geometries:
        vis.add_geometry(geometry)

    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([0.1, 0.1, 0.1])

    view_control = vis.get_view_control()
    view_control.set_front([0, 0, 1])
    view_control.set_up([0, 1, 0])
    view_control.set_lookat([0, 0, 0])

    vis.run()
    vis.destroy_window()

def _create_text_mesh(text: str, position: np.ndarray, color: np.ndarray):
    """
    Create a text label mesh (Open3D does not support text directly; uses a sphere as a proxy).
    """
    try:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(position)
        sphere.paint_uniform_color(color)
        return sphere
    except:
        return None


def visualize_pointclouds_comparison(pointclouds_gt, recon_tgt=None, colors=None):
    """
    Visualize a side-by-side comparison of two point clouds using Open3D.

    Args:
        pointclouds_gt: Ground-truth point cloud [N, 3]
        recon_tgt: Reconstructed point cloud [N, 3]
        colors: Optional (color_gt, color_recon) pair; defaults to green/red
    """
    if colors is None:
        color_gt = [0, 1, 0]    # green = ground truth
        color_recon = [1, 0, 0]  # red = reconstruction
    else:
        color_gt, color_recon = colors

    all_vis_object = []
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(pointclouds_gt)
    pcd_gt.paint_uniform_color(color_gt)
    all_vis_object.append(pcd_gt)

    if recon_tgt is not None:
        pcd_recon = o3d.geometry.PointCloud()
        pcd_recon.points = o3d.utility.Vector3dVector(recon_tgt)
        pcd_recon.paint_uniform_color(color_recon)
        all_vis_object.append(pcd_recon)

    o3d.visualization.draw_geometries(all_vis_object,
                                      window_name="Point Cloud Comparison: Green=GT, Red=Recon",
                                      width=1200, height=800)

def visualize_pointclouds(pointclouds_gt, pointclouds_normals_gt, fracture_surface):
    """
    Display two views:
    1. Point cloud with fracture surface highlighted (gray + red)
    2. (Optionally) Point cloud with normal vector arrows

    Args:
        pointclouds_gt: [N,3] numpy array
        pointclouds_normals_gt: [N,3] numpy array
        fracture_surface: [N,] bool or 0/1 mask for fracture surface points
    """
    # ---- View 1: Gray + Red (fracture surface highlighted) ----
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pointclouds_gt)

    colors = np.tile(np.array([0.7, 0.7, 0.7]), (len(pointclouds_gt), 1))
    colors[fracture_surface.astype(bool)] = np.array([1.0, 0.0, 0.0])
    pcd1.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd1], window_name="Fracture Surface (Gray + Red)")

def visualize_pointclouds_and_fractures(pointclouds_gt, pointclouds_normals_gt, fracture_surface):
    """
    Visualize point cloud with fracture surface highlights and normal vectors.

    Features:
    - Auto-scales arrow length based on object bounding-box diagonal.
    - Normalises normal vectors to prevent length variation.
    - Adaptive sub-sampling for clean display.
    """
    def sanitize(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if x.dtype != np.float64 and x.dtype != np.int32 and x.dtype != bool:
            x = x.astype(np.float64)
        return np.ascontiguousarray(x)

    pts = sanitize(pointclouds_gt)
    nrms = sanitize(pointclouds_normals_gt)
    labels = sanitize(fracture_surface).flatten()

    if pts.ndim == 3:
        pts = pts.reshape(-1, 3)
        nrms = nrms.reshape(-1, 3)
        labels = labels.reshape(-1)

    print(f"[Visualizer] Points: {pts.shape}")

    # Auto-scale arrow length from object size
    bbox_min = pts.min(0)
    bbox_max = pts.max(0)
    object_scale = np.linalg.norm(bbox_max - bbox_min)

    arrow_len = object_scale * 0.03
    print(f"[Auto-Scale] Object Size: {object_scale:.2f}, Arrow Length: {arrow_len:.4f}")

    # ---- Window 1: Fracture Surface ----
    pcd_surf = o3d.geometry.PointCloud()
    pcd_surf.points = o3d.utility.Vector3dVector(pts)

    colors = np.zeros_like(pts)
    colors[:] = [0.7, 0.7, 0.7]
    mask = (labels > 0.5)
    colors[mask] = [1.0, 0.0, 0.0]
    pcd_surf.colors = o3d.utility.Vector3dVector(colors)

    # ---- Window 2: Normals (arrows) ----
    vis_objs = []

    pcd_norm = o3d.geometry.PointCloud()
    pcd_norm.points = o3d.utility.Vector3dVector(pts)
    pcd_norm.paint_uniform_color([0.6, 0.6, 0.6])
    pcd_norm.normals = o3d.utility.Vector3dVector(nrms)
    vis_objs.append(pcd_norm)

    # Adaptive sub-sampling for arrows
    target_arrow_count = 10000
    step = max(1, len(pts) // target_arrow_count)

    sample_pts = pts[::step]
    sample_nrms = nrms[::step]

    starts = sample_pts
    ends = sample_pts + sample_nrms * arrow_len

    line_points = np.concatenate([starts, ends], axis=0)
    N = len(starts)
    lines = np.stack([np.arange(N), np.arange(N) + N], axis=1)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(lines.astype(np.int32))
    line_set.paint_uniform_color([1.0, 0.2, 0.2])

    vis_objs.append(line_set)

    print(f">> Displaying Window. Press 'q' to close.")
    o3d.visualization.draw_geometries([pcd_surf], window_name="1. Fracture Surface (Red)", width=800, height=600,
                                      left=50, top=50)
    o3d.visualization.draw_geometries(vis_objs, window_name="2. Normals (Arrows)", width=800, height=600, left=900,
                                      top=50)
