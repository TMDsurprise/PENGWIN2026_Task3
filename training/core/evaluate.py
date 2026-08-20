"""
PENGWIN 2026 Task 3 — Official Evaluation Script
=================================================

Evaluates fragment reduction predictions against ground-truth rigid transforms,
producing five competition metrics.

Metrics
-------
Competition:

1.  Translation Error (mm)
    Per-fragment Euclidean distance between predicted and GT centroid positions.

2.  Rotation Error (geodesic deg)
    Per-fragment geodesic rotation angle:  arccos((trace(R_pred @ R_gt.T) - 1) / 2).

3.  TRE — Target Registration Error (mm)
    Mean Euclidean distance between paired points across the full assembly.

4.  CD_raw (mm)
    Euclidean Chamfer distance between full assembled point cloud and GT in
    physical mm.  Primary continuous quality metric.

5.  Part Accuracy — PA  (fraction)
    Fraction of fragments with CD_sum < threshold in globally-normalised space.
    Default threshold = 0.05 (~4 mm for 157 mm pelvis), calibrated to the
    orthopedic "good reduction" standard (2–5 mm residual displacement).

Normalisation
-------------
Clinical CT data is NOT pre-normalised.  We normalise at evaluation time:

    assembly_center = mean( GT_assembly_points )
    assembly_radius = max( ||GT_assembly_points - assembly_center|| )
    pts_norm = (pts - assembly_center) / assembly_radius

All normalised metrics (PA, CD_euc, CD_sq) operate in this unit-sphere space.
Physical metrics (Trans, Rot, TRE) are computed in raw mm space.

PA Threshold
------------
Default PA_THRESHOLD = 0.05 in CD_sum form.

Clinical interpretation: for a typical pelvis (assembly radius ~157 mm),
    CD_sum < 0.05  ⇔  average per-point surface error < ~4 mm

This aligns with the orthopedic "good reduction" standard (2–5 mm residual
displacement).  The original ECCV 2020 threshold (CD_avg < 0.05, equivalent to
CD_sum < 0.10, i.e. ~8 mm) is too lenient for clinical data — it produces
PA 97%+ with no method discrimination (ceiling effect).

    ECCV 2020 original (PartNet, furniture assembly):
        CD_avg < 0.05  ⇔  ~8 mm average error

    This script (clinical fracture reduction):
        CD_sum < 0.05  ⇔  ~4 mm average error  (2× stricter, clinically motivated)

Methodology follows ECCV 2020 (Euclidean CD, global normalisation).  Only the
threshold is calibrated to clinical precision requirements.

Usage
-----
    python evaluate.py
    python evaluate.py --methods coords pose
    python evaluate.py --data_dir /path/to/test_set --save_csv results.csv
    python evaluate.py --pa_threshold 0.02   # stricter: CD_sum < 0.02
"""

import csv
import json
import argparse
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
from scipy.spatial import KDTree

# ============================================================
# Settings
# ============================================================
DATA_DIR     = "./data/mesh"  # datasets folders
PA_THRESHOLD = 0.05    # CD_sum threshold in globally-normalised space (~4mm avg surface error for typical pelvis)
NPOINTS_BONE = 5000    # points per BONE for Sample CD and TRE
NPOINTS_FRAG = 1000    # points per FRAGMENT for PA
# ============================================================

PRED_FILES = {
    "baseline": "reduction-poses-matrices.json",
}
GT_FILE = "plan_pl_gt.json"

BONE_ORDER = ["SA", "LI", "RI"]


def _bone_key_for_fragment(fid: int) -> Optional[str]:
    """Map numeric fragment ID to bone region."""
    if 1 <= fid <= 100:
        return "SA"
    elif 101 <= fid <= 200:
        return "LI"
    elif 201 <= fid <= 300:
        return "RI"
    return None


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def load_obj_meshes(obj_path: str) -> dict:
    scene = trimesh.load(obj_path, split_object=True, process=False)
    meshes = {}
    if isinstance(scene, trimesh.Scene):
        for key, geom in scene.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                meshes[str(key)] = geom
    elif isinstance(scene, trimesh.Trimesh):
        meshes["1"] = scene
    return meshes


def meshes_to_bone_dict(meshes: dict) -> dict:
    bone_dict = {"SA": {}, "LI": {}, "RI": {}}
    for key, mesh in meshes.items():
        try:
            k = int(key)
        except ValueError:
            continue
        bone = _bone_key_for_fragment(k)
        if bone:
            bone_dict[bone][key] = mesh
    return bone_dict


def apply_T(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    ones = np.ones((len(pts), 1), dtype=np.float64)
    return (T @ np.concatenate([pts, ones], axis=1).T).T[:, :3]


def sample_surface_area_proportional(mesh: trimesh.Trimesh, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(mesh.vertices) == 0:
        return np.zeros((n, 3), dtype=np.float64)
    if len(mesh.faces) > 0:
        pts, _ = trimesh.sample.sample_surface(mesh, n)
    else:
        idx = rng.integers(0, len(mesh.vertices), size=n)
        pts = mesh.vertices[idx]
    return np.array(pts, dtype=np.float64)



# ---------------------------------------------------------------------------
# Chamfer Distance
# ---------------------------------------------------------------------------

def chamfer_distance_sq(P, Q):
    """
    Squared Euclidean Chamfer distance (DGL/Breaking Bad convention).
    CD_sq = mean(d²_P->Q) + mean(d²_Q->P)

    KDTree.query returns Euclidean distances, so we square them here.
    """
    tree_P = KDTree(P)
    tree_Q = KDTree(Q)
    dist_P_to_Q, _ = tree_Q.query(P)
    dist_Q_to_P, _ = tree_P.query(Q)
    return float(np.mean(dist_P_to_Q ** 2) + np.mean(dist_Q_to_P ** 2))


def chamfer_distance_euc(P, Q):
    """Regular Euclidean Chamfer distance."""
    tree_P = KDTree(P)
    tree_Q = KDTree(Q)
    dist_P_to_Q, _ = tree_Q.query(P)
    dist_Q_to_P, _ = tree_P.query(Q)
    return float(np.mean(dist_P_to_Q) + np.mean(dist_Q_to_P))



# ---------------------------------------------------------------------------
# Pose JSON loading
# ---------------------------------------------------------------------------

def load_poses_json(json_path: str) -> dict:
    with open(json_path) as f:
        data = json.load(f)
    result = {}
    if isinstance(data, list):
        for entry in data:
            fid = str(entry["fragment_id"])
            result[fid] = np.array(entry["transformation"], dtype=np.float64)
    elif isinstance(data, dict):
        for fid, T in data.items():
            result[str(fid)] = np.array(T, dtype=np.float64)
    else:
        raise ValueError(f"Unexpected pose JSON format in {json_path}")
    return result


# ---------------------------------------------------------------------------
# Anchor normalisation
# ---------------------------------------------------------------------------

def get_anchor_id(poses: dict, bone_dict: dict) -> str:
    sa_keys = sorted(bone_dict.get("SA", {}).keys(), key=lambda x: int(x))
    if sa_keys:
        return sa_keys[0]
    all_keys = sorted(poses.keys(), key=lambda x: int(x))
    return all_keys[0]


def normalise_poses(poses: dict, anchor_id: str) -> dict:
    T_anchor = poses[anchor_id]
    T_anchor_inv = np.linalg.inv(T_anchor)
    return {fid: (T_anchor_inv @ T) for fid, T in poses.items()}


# ---------------------------------------------------------------------------
# Rotation error
# ---------------------------------------------------------------------------

def rotation_geodesic_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    R_rel = R1 @ R2.T
    cos_angle = float(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


# ---------------------------------------------------------------------------
# Per-sample evaluation (ECCV 2020 global-normalisation style)
# ---------------------------------------------------------------------------

def evaluate_sample(
    meshes: dict,
    gt_poses_raw: dict,
    pred_poses_raw: dict,
    pa_threshold: float,
    npoints_bone: int = 5000,
    npoints_frag: int = 1000,
) -> Optional[dict]:
    """
    Evaluate one sample using ECCV 2020 PA convention with global normalisation.

    Key properties:
      - Global assembly-level normalisation (divide by GT assembly radius)
      - Euclidean Chamfer distance (matching ECCV 2020 sqrt=True)
      - PA: NO per-fragment centering, NO per-fragment diameter normalisation
      - PA threshold applied in globally normalised space
    """
    bone_dict = meshes_to_bone_dict(meshes)

    common_ids = sorted(
        set(gt_poses_raw.keys()) & set(pred_poses_raw.keys()) & set(meshes.keys()),
        key=lambda x: int(x)
    )
    if not common_ids:
        return None

    # ---- Anchor normalisation ----
    anchor_id = get_anchor_id(gt_poses_raw, bone_dict)
    if anchor_id not in gt_poses_raw or anchor_id not in pred_poses_raw:
        anchor_id = common_ids[0]

    gt_poses   = normalise_poses(gt_poses_raw,   anchor_id)
    pred_poses = normalise_poses(pred_poses_raw, anchor_id)

    rng = np.random.default_rng(42)

    # ---- GT assembly bounding sphere (for GLOBAL normalisation) ----
    gt_all_verts = []
    for fid in common_ids:
        verts = apply_T(meshes[fid].vertices, gt_poses[fid])
        gt_all_verts.append(verts)
    gt_all_verts = np.concatenate(gt_all_verts, axis=0)
    assembly_center = gt_all_verts.mean(axis=0)
    assembly_radius = float(np.max(np.linalg.norm(gt_all_verts - assembly_center, axis=1)))
    if assembly_radius < 1e-8:
        assembly_radius = 1.0
    assembly_diam = 2.0 * assembly_radius

    # ---- Per-fragment metrics ----
    trans_errors  = []
    rot_errors    = []
    frag_diams    = []   # per-fragment diameter in mm (GT space, for size analysis)
    frag_pa_pass   = []
    frag_cd_euc_norm = []  # kept internally for PA computation
    frag_cd_euc_raw  = []

    bone_pts = {b: {"gt": [], "pred": []} for b in BONE_ORDER}

    for fid in common_ids:
        T_gt   = gt_poses[fid]
        T_pred = pred_poses[fid]

        # ----- Translation error (mm) -----
        verts_orig = meshes[fid].vertices.astype(np.float64)
        centroid_gt   = apply_T(verts_orig, T_gt).mean(axis=0)
        centroid_pred = apply_T(verts_orig, T_pred).mean(axis=0)
        trans_errors.append(float(np.linalg.norm(centroid_pred - centroid_gt)))

        # ----- Rotation error (deg) -----
        rot_errors.append(rotation_geodesic_deg(T_pred[:3, :3], T_gt[:3, :3]))

        # ----- PA: ECCV 2020 style (global norm, Euclidean CD) -----
        pts_orig = sample_surface_area_proportional(meshes[fid], npoints_frag, rng)
        pts_gt   = apply_T(pts_orig, T_gt)
        pts_pred = apply_T(pts_orig, T_pred)

        # Global assembly-level normalisation
        pts_gt_norm   = (pts_gt   - assembly_center) / assembly_radius
        pts_pred_norm = (pts_pred - assembly_center) / assembly_radius

        cd_euc_norm = chamfer_distance_euc(pts_pred_norm, pts_gt_norm)
        frag_cd_euc_norm.append(cd_euc_norm)
        frag_pa_pass.append(cd_euc_norm < pa_threshold)

        cd_euc_raw = chamfer_distance_euc(pts_pred, pts_gt)
        frag_cd_euc_raw.append(cd_euc_raw)

        # Fragment diameter in mm (GT space, for size analysis)
        frag_center = pts_gt.mean(axis=0)
        frag_diam = float(2.0 * np.max(np.linalg.norm(pts_gt - frag_center, axis=1)))
        frag_diams.append(max(frag_diam, 1e-8))

        # ----- Collect per-bone points for Sample CD / TRE -----
        bone = _bone_key_for_fragment(int(fid))
        if bone is None:
            continue
        bone_pts[bone]["gt"].append(pts_gt)
        bone_pts[bone]["pred"].append(pts_pred)

    # ----- Sample-level CD and TRE -----
    gt_assembly_pts   = []
    pred_assembly_pts = []

    for bone_key in BONE_ORDER:
        if not bone_pts[bone_key]["gt"]:
            continue
        bone_frag_ids = sorted(
            [k for k in common_ids if _bone_key_for_fragment(int(k)) == bone_key],
            key=lambda x: int(x)
        )
        if not bone_frag_ids:
            continue

        frag_areas = []
        for fid in bone_frag_ids:
            try:
                a = meshes[fid].area
            except Exception:
                a = 1e-6
            frag_areas.append(max(a, 1e-6))
        total_a = sum(frag_areas)

        pts_gt_bone   = []
        pts_pred_bone = []
        allocated = 0
        for i, fid in enumerate(bone_frag_ids):
            if i < len(bone_frag_ids) - 1:
                c = max(1, int(round(frag_areas[i] / total_a * npoints_bone)))
            else:
                c = max(1, npoints_bone - allocated)
            allocated += c
            pts_frag = sample_surface_area_proportional(meshes[fid], c, rng)
            pts_gt_bone.append(apply_T(pts_frag, gt_poses[fid]))
            pts_pred_bone.append(apply_T(pts_frag, pred_poses[fid]))

        gt_assembly_pts.append(np.concatenate(pts_gt_bone, axis=0))
        pred_assembly_pts.append(np.concatenate(pts_pred_bone, axis=0))

    if not gt_assembly_pts:
        return None

    gt_all   = np.concatenate(gt_assembly_pts, axis=0)
    pred_all = np.concatenate(pred_assembly_pts, axis=0)

    # ---- Assembly-level metrics ----
    gt_all_norm   = (gt_all   - assembly_center) / assembly_radius
    pred_all_norm = (pred_all - assembly_center) / assembly_radius
    # sample_cd_euc_norm = chamfer_distance_euc(pred_all_norm, gt_all_norm)   # Reference only
    # sample_cd_sq_norm  = chamfer_distance_sq(pred_all_norm, gt_all_norm)    # Reference only
    sample_cd_raw = chamfer_distance_euc(pred_all, gt_all)

    tre = float(np.mean(np.linalg.norm(pred_all - gt_all, axis=1)))

    # ---- Aggregate ----
    n_frags = len(common_ids)
    pa      = float(np.mean(frag_pa_pass))
    # success = bool(all(frag_pa_pass))  # reference metric, commented out

    return {
        "n_fragments":         n_frags,
        "anchor_id":           anchor_id,
        "trans_mean_mm":       float(np.mean(trans_errors)),
        "trans_errors":        trans_errors,
        "rot_mean_deg":        float(np.mean(rot_errors)),
        "rot_errors":          rot_errors,
        "frag_diams":          frag_diams,
        "pa":                  pa,
        # "pa_success":          success,
        "frag_pa_pass":        frag_pa_pass,
        "frag_cd_euc_norm":    frag_cd_euc_norm,
        "frag_cd_euc_raw":     frag_cd_euc_raw,
        # "sample_cd_euc_norm":  sample_cd_euc_norm,
        # "sample_cd_sq_norm":   sample_cd_sq_norm,
        "sample_cd_raw_mm":    sample_cd_raw,
        "tre_mm":              tre,
        "assembly_radius":     assembly_radius,
        "assembly_diam":       assembly_diam,
    }


# ---------------------------------------------------------------------------
# Dataset scan
# ---------------------------------------------------------------------------

def discover_samples(data_dir: str) -> list:
    root = Path(data_dir)
    if not root.is_dir():
        print(f"Error: data directory not found: {data_dir}")
        print("Specify with --data_dir /path/to/test_set")
        return []
    samples = []
    for sample_dir in sorted(root.iterdir()):
        if not sample_dir.is_dir():
            continue
        if not (sample_dir / GT_FILE).exists():
            continue
        obj_files = sorted(sample_dir.glob("*.obj"))
        if not obj_files:
            continue
        samples.append((sample_dir.name, str(sample_dir), str(obj_files[0])))
    return samples


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(method: str, results: list, pa_threshold: float):
    valid = [r for r in results if r is not None]
    n_total = len(results)
    n_valid = len(valid)
    n_skip  = n_total - n_valid

    print(f"\n  [{method.upper()}]  samples={n_valid}/{n_total}  skipped={n_skip}")
    if not valid:
        return

    trans  = [r["trans_mean_mm"] for r in valid]
    rot    = [r["rot_mean_deg"]  for r in valid]
    tre    = [r["tre_mm"]     for r in valid]
    cd_raw = [r["sample_cd_raw_mm"] for r in valid]
    # cd_euc = [r["sample_cd_euc_norm"] for r in valid]
    # cd_sq  = [r["sample_cd_sq_norm"] for r in valid]
    pa     = [r["pa"]         for r in valid]
    # suc    = [r["pa_success"] for r in valid]
    def fmt(arr, decimals=4):
        return f"{np.mean(arr):.{decimals}f} ± {np.std(arr):.{decimals}f}"

    print(f"    --- Competition Metrics ---")
    print(f"    Trans error  (mm)    : {fmt(trans, 3)}")
    print(f"    Rot error   (deg)    : {fmt(rot, 2)}")
    print(f"    TRE          (mm)    : {fmt(tre, 3)}")
    print(f"    CD_raw       (mm)    : {fmt(cd_raw, 4)}")
    print(f"    PA           (%)     : {np.mean(pa)*100:.1f}%  (CD_sum < {pa_threshold}, ~{pa_threshold * 157:.0f} mm)")
    # ---- Reference metrics (commented out for competition release) ----
    # print(f"    --- Reference (supplementary) ---")
    # print(f"    PA success   (%)     : {np.mean(suc)*100:.1f}%  (all fragments pass)")
    # print(f"    CD_euc (norm)        : {fmt(cd_euc, 6)}  (ECCV 2020)")
    # print(f"    CD_sq  (norm)        : {fmt(cd_sq, 6)}  (DGL / Breaking Bad)")
    # all_cd_euc = []
    # for r in valid:
    #     all_cd_euc.extend(r["frag_cd_euc_norm"])
    # if all_cd_euc:
    #     print(f"    Per-frag CD_euc_norm : {np.mean(all_cd_euc):.6f} ± {np.std(all_cd_euc):.6f}  "
    #           f"min={np.min(all_cd_euc):.6f}  max={np.max(all_cd_euc):.6f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate assembly prediction quality (ECCV 2020 style PA, global normalisation)")
    parser.add_argument("--data_dir",     type=str,   default=None)
    parser.add_argument("--pa_threshold", type=float, default=None,
                        help=f"Euclidean CD sum threshold for PA in normalised space (default: {PA_THRESHOLD})")
    parser.add_argument("--npoints_bone", type=int,   default=None)
    parser.add_argument("--npoints_frag", type=int,   default=None)
    parser.add_argument("--methods",      type=str,   nargs="+",
                        default=["baseline"])
    parser.add_argument("--save_csv",     type=str,   default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir     = args.data_dir     or DATA_DIR
    pa_threshold = args.pa_threshold if args.pa_threshold is not None else PA_THRESHOLD
    npoints_bone = args.npoints_bone if args.npoints_bone is not None else NPOINTS_BONE
    npoints_frag = args.npoints_frag if args.npoints_frag is not None else NPOINTS_FRAG
    methods      = args.methods

    print(f"=== PENGWIN 2026 Task 3 — Official Evaluation ===")
    print(f"Data dir      : {data_dir}")
    print(f"PA threshold  : {pa_threshold}  (Euclidean CD_sum, assembly-radius-normalised)")
    print(f"Points/bone   : {npoints_bone}  (for Sample CD & TRE)")
    print(f"Points/frag   : {npoints_frag}  (for PA)")
    print(f"Methods       : {methods}\n")

    samples = discover_samples(data_dir)
    print(f"Found {len(samples)} samples with GT poses.\n")
    if not samples:
        print("No samples found.")
        return

    all_results = {m: [] for m in methods}
    csv_rows    = []

    for idx, (sample_name, sample_dir, obj_path) in enumerate(samples):
        print(f"[{idx+1:3d}/{len(samples)}] {sample_name}", end="  ")

        try:
            gt_poses_raw = load_poses_json(str(Path(sample_dir) / GT_FILE))
        except Exception as e:
            print(f"[SKIP — GT: {e}]")
            for m in methods:
                all_results[m].append(None)
            continue

        try:
            meshes = load_obj_meshes(obj_path)
        except Exception as e:
            print(f"[SKIP — OBJ: {e}]")
            for m in methods:
                all_results[m].append(None)
            continue

        row = {"sample": sample_name}
        method_strs = []

        for method in methods:
            pred_json = Path(sample_dir) / PRED_FILES[method]
            if not pred_json.exists():
                all_results[method].append(None)
                method_strs.append(f"{method}=N/A")
                continue

            try:
                pred_poses_raw = load_poses_json(str(pred_json))
                metrics = evaluate_sample(
                    meshes, gt_poses_raw, pred_poses_raw,
                    pa_threshold, npoints_bone, npoints_frag,
                )
                all_results[method].append(metrics)
            except Exception as e:
                print(f"\n  [WARN] {method} eval error: {e}")
                traceback.print_exc()
                all_results[method].append(None)
                method_strs.append(f"{method}=ERR")
                continue

            if metrics is not None:
                method_strs.append(
                    f"{method}: T={metrics['trans_mean_mm']:.2f}mm "
                    f"R={metrics['rot_mean_deg']:.1f}° "
                    f"TRE={metrics['tre_mm']:.2f}mm "
                    f"CD={metrics['sample_cd_raw_mm']:.1f}mm "
                    f"PA={metrics['pa']*100:.0f}%"
                )
                row.update({
                    f"{method}_trans_mm":        metrics["trans_mean_mm"],
                    f"{method}_rot_deg":         metrics["rot_mean_deg"],
                    f"{method}_tre_mm":          metrics["tre_mm"],
                    # f"{method}_cd_euc_norm":     metrics["sample_cd_euc_norm"],
                    # f"{method}_cd_sq_norm":      metrics["sample_cd_sq_norm"],
                    f"{method}_cd_raw_mm":       metrics["sample_cd_raw_mm"],
                    f"{method}_pa":              metrics["pa"],
                    # f"{method}_pa_success":      int(metrics["pa_success"]),
                    f"{method}_assembly_radius": metrics["assembly_radius"],
                })

        print("  |  ".join(method_strs))
        csv_rows.append(row)

    # ---- Summary ----
    print(f"\n{'=' * 65}")
    print("SUMMARY")
    print(f"{'=' * 65}")
    for method in methods:
        print_summary(method, all_results[method], pa_threshold)

    # ---- CSV export ----
    if args.save_csv and csv_rows:
        import csv
        fieldnames = ["sample"] + sorted(k for k in csv_rows[0] if k != "sample")
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nPer-sample CSV saved to: {args.save_csv}")

    print(f"\nDone. Evaluated {len(samples)} samples.")


if __name__ == "__main__":
    main()
