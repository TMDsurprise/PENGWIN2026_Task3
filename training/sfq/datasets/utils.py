import trimesh
import numpy as np


def are_meshes_connected(
    mesh_a: trimesh.Trimesh,
    mesh_b: trimesh.Trimesh,
    decimals: int = 5,
):
    """
    Check if two meshes are connected.

    Args:
        mesh_a (trimesh.Trimesh): The first mesh.
        mesh_b (trimesh.Trimesh): The second mesh.
        decimals (int, optional): The number of decimal places to round the vertices to. Defaults to 5.

    Returns:
        bool: True if the meshes are connected, False otherwise.
    """
    vertices_a = mesh_a.vertices
    vertices_b = mesh_b.vertices
    faces_a = mesh_a.faces
    faces_b = mesh_b.faces

    shared_faces_a = np.zeros(len(faces_a), dtype=bool)
    shared_faces_b = np.zeros(len(faces_b), dtype=bool)

    vertices_a = vertices_a.round(decimals=decimals)
    vertices_b = vertices_b.round(decimals=decimals)

    common_vertices = set(map(tuple, vertices_a)).intersection(map(tuple, vertices_b))

    # calculate common faces
    if len(common_vertices) > 0:
        for i, face_a in enumerate(faces_a):
            if all([tuple(vertices_a[vertex]) in common_vertices for vertex in face_a]):
                shared_faces_a[i] = True
        for i, face_b in enumerate(faces_b):
            if all([tuple(vertices_b[vertex]) in common_vertices for vertex in face_b]):
                shared_faces_b[i] = True

    return len(common_vertices) > 0, shared_faces_a, shared_faces_b
