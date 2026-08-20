"""Evaluation and inference data loading utilities."""

import numpy as np
import trimesh

from datasets.transform import recenter_pc, rescale_pc


NPOINTS = 5000
MIN_POINTS = 20


def sub_sample(meshdict, npoints=5000, min_points=20, vis=False):
    """
    Sample points from mesh fragments based on area proportions.

    Args:
        meshdict: Dictionary with bone_type -> {frag_name: {'mesh': trimesh, 'volume': float}}
        npoints: Target number of points per bone
        min_points: Minimum points per fragment
        vis: Whether to visualize
    """
    sampled_data = {}

    for bone_type, fragments in meshdict.items():
        sampled_data[bone_type] = {}

        frag_keys = list(fragments.keys())
        if not frag_keys:
            continue

        volumes = []
        for k in frag_keys:
            volumes.append(fragments[k]['volume'])

        total_volume = sum(volumes)
        if total_volume <= 0:
            total_volume = 1.0

        allocations = []
        for vol in volumes:
            ratio = vol / total_volume
            count = int(ratio * npoints)
            count = max(count, min_points)
            allocations.append(count)

        current_total = sum(allocations)
        diff = npoints - current_total

        if diff != 0:
            max_vol_idx = np.argmax(volumes)
            if diff > 0:
                allocations[max_vol_idx] += diff
            else:
                allocations[max_vol_idx] += diff
                if allocations[max_vol_idx] < min_points:
                    allocations[max_vol_idx] = min_points

        for i, frag_name in enumerate(frag_keys):
            frag_data = fragments[frag_name]
            mesh = frag_data['mesh']
            target_count = allocations[i]

            if target_count <= 0:
                sampled_data[bone_type][frag_name] = {
                    'coords': np.zeros((0, 3)),
                    'normals': np.zeros((0, 3))
                }
                continue

            points, face_indices = trimesh.sample.sample_surface_even(mesh, target_count)
            normals = mesh.face_normals[face_indices]

            current_count = len(points)

            if current_count < target_count:
                needed = target_count - current_count
                if current_count > 0:
                    indices = np.random.choice(current_count, needed, replace=True)
                    add_points = points[indices]
                    add_normals = normals[indices]
                    points = np.vstack([points, add_points])
                    normals = np.vstack([normals, add_normals])
                else:
                    points = np.zeros((target_count, 3))
                    normals = np.zeros((target_count, 3))

            elif current_count > target_count:
                points = points[:target_count]
                normals = normals[:target_count]

            sampled_data[bone_type][frag_name] = {
                'coords': np.array(points, dtype=np.float32),
                'normals': np.array(normals, dtype=np.float32)
            }

    return sampled_data


def _parse_obj_groups(obj_path):
    """Read g-lines from an OBJ to get the real fragment group names."""
    names = []
    with open(obj_path) as f:
        for line in f:
            if line.startswith("g "):
                name = line[2:].strip()
                if name not in names:
                    names.append(name)
    return names


def load_obj_fragments(obj_path, verbose=True):
    """
    Load fragments from OBJ file using trimesh.

    Keys 1-100: SA (Sacrum)
    Keys 101-200: LI (Left Ilium)
    Keys 201-300: RI (Right Ilium)
    """
    scene = trimesh.load(obj_path, split_object=True)

    # Both local and Docker pin trimesh==4.8.2, which preserves OBJ group
    # names as scene keys.  Group insertion order varies but names are correct.
    if isinstance(scene, trimesh.Scene) and verbose:
        print(f"Loaded scene with {len(scene.geometry)} geometry objects")

    meshdict = {"SA": {}, "LI": {}, "RI": {}}

    for key, geom in scene.geometry.items():
        try:
            key_int = int(key)
        except (ValueError, TypeError):
            if verbose:
                print(f"Skipping invalid key: {key}")
            continue

        if 1 <= key_int <= 100:
            bone_type = "SA"
        elif 101 <= key_int <= 200:
            bone_type = "LI"
        elif 201 <= key_int <= 300:
            bone_type = "RI"
        else:
            if verbose:
                print(f"Skipping key {key} (outside valid ranges)")
            continue

        if isinstance(geom, trimesh.Trimesh):
            mesh = geom
        elif hasattr(geom, 'merged_vertices'):
            mesh = geom.merged_vertices
        else:
            continue

        volume = mesh.volume if hasattr(mesh, 'volume') and mesh.volume > 0 else 0.001

        meshdict[bone_type][str(key)] = {
            'mesh': mesh,
            'volume': volume
        }

        if verbose:
            print(f"  Key: {key} -> Bone: {bone_type}, Vertices: {len(mesh.vertices)}, Volume: {volume:.4f}")

    return meshdict


def prepare_network_input(meshdict, npoints=5000):
    """
    Convert meshdict to network input format.

    Returns:
        Dictionary with:
            - pointclouds: (N, 3) all point coordinates
            - pointclouds_normals: (N, 3) all point normals
            - points_per_part: list of point counts per part
            - points_per_bone: list of point counts per bone
            - bonetype: list of bone type IDs per part
            - point_bone_ids: (N,) bone ID for each point
            - num_parts: number of parts
            - offsets: cumulative offsets for parts
    """
    sampled = sub_sample(meshdict, npoints=npoints, min_points=MIN_POINTS)

    all_points = []
    all_normals = []
    points_per_part = []
    points_per_bone = []
    bonetype = []
    point_bone_ids = []
    fragment_names = []

    bone_type_to_id = {"SA": 0, "LI": 1, "RI": 2}

    offsets = [0]
    part_idx = 0

    for bone_type in ["SA", "LI", "RI"]:
        bone_id = bone_type_to_id[bone_type]
        fragments = sampled[bone_type]

        bone_start_idx = part_idx

        for frag_name, frag_data in fragments.items():
            coords = frag_data['coords']
            normals = frag_data['normals']

            if len(coords) == 0:
                continue

            all_points.append(coords)
            all_normals.append(normals)
            points_per_part.append(len(coords))
            point_bone_ids.append(np.full(len(coords), bone_id, dtype=np.int32))

            bonetype.append(bone_id)
            fragment_names.append(frag_name)
            part_idx += 1

        if part_idx > bone_start_idx:
            points_per_bone.append(sum(points_per_part[bone_start_idx:part_idx]))

    all_points = np.concatenate(all_points, axis=0) if all_points else np.zeros((0, 3))
    all_normals = np.concatenate(all_normals, axis=0) if all_normals else np.zeros((0, 3))
    point_bone_ids = np.concatenate(point_bone_ids, axis=0) if point_bone_ids else np.zeros(0, dtype=np.int32)

    # Recenter and rescale to unit sphere (matching training preprocessing)
    all_points, centroid = recenter_pc(all_points)
    all_points, scale = rescale_pc(all_points)
    # Normals are direction vectors - not affected by translation/uniform scaling

    offsets = [0] + list(np.cumsum(points_per_part))
    num_parts = len(points_per_part)

    result = {
        "pointclouds": all_points.astype(np.float32),
        "pointclouds_normals": all_normals.astype(np.float32),
        "points_per_part": np.array(points_per_part, dtype=np.int32),
        "points_per_bone": np.array(points_per_bone, dtype=np.int32),
        "bonetype": np.array(bonetype, dtype=np.int32),
        "point_bone_ids": point_bone_ids,
        "num_parts": num_parts,
        "offsets": np.array(offsets, dtype=np.int32),
        "fragment_names": fragment_names,
        "centroid": centroid.flatten().astype(np.float32),  # (3,)
        "scale": float(scale),
    }

    return result


def pad_data(data, max_parts=50, max_bone=3):
    """Pad data to match network input shape requirements."""
    padded = {}

    points_per_part = np.zeros(max_parts, dtype=np.int32)
    points_per_part[:len(data["points_per_part"])] = data["points_per_part"]
    padded["points_per_part"] = points_per_part

    points_per_bone = np.zeros(max_bone, dtype=np.int32)
    points_per_bone[:len(data["points_per_bone"])] = data["points_per_bone"]
    padded["points_per_bone"] = points_per_bone

    bonetype = np.zeros(max_parts, dtype=np.int32)
    bonetype[:len(data["bonetype"])] = data["bonetype"]
    padded["bonetype"] = bonetype

    padded["pointclouds"] = data["pointclouds"]
    padded["pointclouds_normals"] = data["pointclouds_normals"]
    padded["point_bone_ids"] = data["point_bone_ids"]
    padded["num_parts"] = data["num_parts"]
    padded["centroid"] = data.get("centroid", np.zeros(3, dtype=np.float32))
    padded["scale"] = data.get("scale", 1.0)

    return padded