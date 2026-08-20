"""
Inference script for bone fracture reduction.

Usage:
    python inference.py                                 # default
    python inference.py --config configs/test_pose.yaml # pose mode
    python inference.py --debug_vis --max_iters 3
"""

import os
import json
import time
from pathlib import Path
from functools import partial

import numpy as np
import torch
from omegaconf import OmegaConf

from models.AssemblyNet.transformer import AssemblyTransformer
from models.AssemblyNet.lightning_module import AssemblyLightningModule
from models.AssemblyNet.fragment_confidence import FragmentConfidenceModule
from models.AssemblyNet.utils import quaternion_to_matrix
from datasets.fractures.eval import load_obj_fragments, sub_sample
from datasets.transform import recenter_pc, rescale_pc

# ============================================================
# Default config - all settings can be overridden via CLI
# ============================================================
CONFIG_PATH = "configs/test.yaml"
# ============================================================

def parse_args():
    """Parse command-line arguments (all optional, defaults come from the YAML config)."""
    import argparse
    parser = argparse.ArgumentParser(description="Bone fracture reduction inference")
    parser.add_argument("--config", type=str, default=None,
                        help=f"Config YAML path (default: {CONFIG_PATH})")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Override input_dir from config")
    parser.add_argument("--npoints", type=int, default=None,
                        help="Override npoints from config")
    parser.add_argument("--max_iters", type=int, default=None,
                        help="Override max_iters from config")
    parser.add_argument("--debug_vis", action="store_true",
                        help="Enable trimesh debug visualizations")
    return parser.parse_args()


def load_config_and_model(config_path, device, debug_vis=None):
    """
    Load test config, build model, and load weights.

    Test config provides: input_dir, npoints, max_iters, checkpoints, experiment_name
    Model config (model_config path) provides: output_type, transformer_model

    Returns:
        model: AssemblyLightningModule in eval mode
        cfg: merged OmegaConf dict
    """
    test_cfg = OmegaConf.load(config_path)

    # Load model-specific config (has output_type and transformer_model)
    model_cfg_path = Path(config_path).parent / f"{test_cfg.model_config}.yaml"
    model_cfg = OmegaConf.load(model_cfg_path)

    # Merge: test config values override model config values
    cfg = OmegaConf.merge(model_cfg, test_cfg)

    output_type = cfg.output_type
    tm = cfg.transformer_model
    experiment_name = cfg.experiment_name

    transformer = AssemblyTransformer(
        embed_dim=tm.embed_dim,
        num_layers=tm.num_layers,
        num_heads=tm.num_heads,
        dropout_rate=tm.dropout_rate,
        max_parts=tm.max_parts,
        output_type=output_type,
        fragment_context_enabled=tm.get("fragment_context_enabled", False),
        fragment_context_start_layer=tm.get("fragment_context_start_layer", 8),
        fragment_context_dim=tm.get("fragment_context_dim", 192),
        fragment_context_heads=tm.get("fragment_context_heads", 4),
        point_reliability_enabled=tm.get("point_reliability_enabled", False),
    )

    # debug_vis from CLI override, else from config
    vis = cfg.get("debug_vis", False)

    training_mode = cfg.get("training_mode", "full")
    lora_rank = cfg.get("lora_rank", 16)
    lora_alpha = cfg.get("lora_alpha", 32.0)
    # Resolve checkpoints: explicit path or auto-discover latest epoch
    if cfg.get("checkpoint"):
        checkpoint_path = str(Path(cfg.checkpoint))
    else:
        exp_dir = Path("output") / experiment_name
        checkpoints = sorted(exp_dir.glob("epoch-epoch=*.ckpt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {exp_dir}")
        checkpoint_path = str(checkpoints[-1])

    inference_model = cfg.get("inference_model", "baseline")
    if inference_model == "fragment_confidence":
        model = FragmentConfidenceModule(
            transformer_model=transformer,
            optimizer=partial(torch.optim.AdamW, lr=2e-4, weight_decay=0.01),
            checkpoint=checkpoint_path,
            axis_offsets_deg=cfg.get("axis_offsets_deg", ()),
        )
    else:
        model = AssemblyLightningModule(
            transformer_model=transformer,
            optimizer=partial(torch.optim.AdamW, lr=1e-4, weight_decay=0.01),
            output_type=output_type,
            training_mode=training_mode,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            debug_vis=vis,
        )

    print(f"  Loading checkpoint: {checkpoint_path} ({inference_model})")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model = model.to(device)
    model.eval()
    return model, cfg


def configure_deterministic_inference(enabled):
    if not enabled:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=True)


def stable_argmax(scores, epsilon=1e-5):
    scores_cpu = scores.detach().to(device="cpu", dtype=torch.float64)
    maximum = scores_cpu.max(dim=1, keepdim=True).values
    tied = scores_cpu >= maximum - float(epsilon)
    candidate_ids = torch.arange(scores_cpu.shape[1], dtype=torch.long)[None, :]
    sentinel = torch.full_like(candidate_ids, scores_cpu.shape[1])
    return torch.where(tied, candidate_ids, sentinel).min(dim=1).values


def discover_samples(input_dir):
    """
    Find all patient sample directories containing OBJ files.
    Expected structure: input_dir/sample_name/xxx.obj
    Returns: List of (sample_name, sample_dir, obj_path) tuples, sorted by name.
    """
    samples = []
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    for sample_dir in sorted(input_path.iterdir()):
        if not sample_dir.is_dir():
            continue
        obj_files = list(sample_dir.glob("*.obj"))
        if len(obj_files) == 0:
            print(f"  [WARN] No OBJ file in {sample_dir.name}, skipping.")
            continue
        if len(obj_files) > 1:
            print(f"  [WARN] Multiple OBJ files in {sample_dir.name}, using: {obj_files[0].name}")
        samples.append((sample_dir.name, str(sample_dir), str(obj_files[0])))
    return samples


def build_Tinit(centroid, scale):
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -centroid / scale
    T[:3, :3] = np.diag(1.0 / np.array([scale, scale, scale], dtype=np.float32))
    return T


def transform_pose_to_original_space(T_pred, centroid, scale):
    T_init = build_Tinit(centroid, scale)
    T_init_inv = np.linalg.inv(T_init)
    T_final = T_init_inv @ T_pred @ T_init
    return T_final


def apply_transform_to_fragment_coords(fragment_coords, T):
    ones = np.ones((fragment_coords.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([fragment_coords, ones], axis=1)
    transformed = (T @ pts_h.T).T
    return transformed[:, :3]


def apply_rotation_to_fragment_normals(fragment_normals, T):
    R = T[:3, :3]
    return (R @ fragment_normals.T).T


def build_fragment_data_from_meshdict(
        meshdict,
        npoints,
        min_points=20,
        allocation_mode="volume",
):
    sampled = sub_sample(
        meshdict,
        npoints=npoints,
        min_points=min_points,
        allocation_mode=allocation_mode,
    )
    fragment_data = []
    for bone_type in ["SA", "LI", "RI"]:
        fragments = sampled.get(bone_type, {})
        for frag_name in sorted(fragments.keys()):
            coords = fragments[frag_name]["coords"]
            normals = fragments[frag_name]["normals"]
            fragment_data.append((coords, normals))
    return fragment_data


def run_inference_pass(model, all_points, all_normals, points_per_part, bone_types,
                       fragment_diameters_mm, norm_scale, device, max_parts):
    points_per_part_padded = np.zeros(max_parts, dtype=np.int32)
    points_per_part_padded[:len(points_per_part)] = points_per_part

    tensor_dict = {
        "pointclouds": torch.from_numpy(all_points).to(device),
        "pointclouds_normals": torch.from_numpy(all_normals).to(device),
        "points_per_part": torch.from_numpy(points_per_part_padded).unsqueeze(0).to(device),
    }
    if isinstance(model, FragmentConfidenceModule):
        bone_types_padded = np.zeros(max_parts, dtype=np.int64)
        bone_types_padded[:len(bone_types)] = bone_types
        diameters_padded = np.zeros(max_parts, dtype=np.float32)
        diameters_padded[:len(fragment_diameters_mm)] = fragment_diameters_mm
        tensor_dict.update({
            "bonetype": torch.from_numpy(bone_types_padded).unsqueeze(0).to(device),
            "fragment_diameter_mm": torch.from_numpy(diameters_padded).unsqueeze(0).to(device),
            "norm_scale": torch.tensor([norm_scale], dtype=torch.float32, device=device),
        })

    use_amp = (device == "cuda")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        output = model(tensor_dict)

    # Free input tensors from GPU immediately — only need quat/trans from output
    del tensor_dict
    torch.cuda.empty_cache()

    return output


def select_confidence_candidates(output, cfg):
    """Apply learned metric/severe-risk gates to the ranker's proposal."""
    deterministic = bool(cfg.get("deterministic_inference", False))
    if deterministic:
        raw_selected = stable_argmax(
            output["scores"], cfg.get("score_tie_epsilon", 1e-5)
        )
    else:
        raw_selected = output["scores"].argmax(dim=1)
    if cfg.get("selection_policy", "score") == "score":
        return raw_selected.to(output["scores"].device), {
            "accepted": int(raw_selected.numel()), "patient_accepted": True
        }

    metric_scales = output["metric_prediction"].new_tensor([2.53, 2.76, 2.67, 3.37])
    predicted_metrics = torch.expm1(output["metric_prediction"].float()).clamp_min(0.0)
    predicted_utility = (predicted_metrics / metric_scales).mean(dim=-1)
    severe_probability = torch.sigmoid(output["severe_logits"].float())
    if deterministic:
        predicted_utility = predicted_utility.detach().to(device="cpu", dtype=torch.float64)
        severe_probability = severe_probability.detach().to(device="cpu", dtype=torch.float64)

    part_ids = torch.arange(raw_selected.numel(), device=raw_selected.device)
    selected_utility = predicted_utility[part_ids, raw_selected]
    current_utility = predicted_utility[:, 0]
    selected_severe = severe_probability[part_ids, raw_selected]
    current_severe = severe_probability[:, 0]

    utility_margin = float(cfg.get("metric_gate_margin", 0.0))
    severe_tolerance = float(cfg.get("severe_gate_tolerance", 0.05))
    severe_threshold = float(cfg.get("severe_gate_threshold", 0.5))
    utility_improves = selected_utility + utility_margin < current_utility
    severe_safe = (
        (selected_severe <= current_severe + severe_tolerance)
        & ((selected_severe <= severe_threshold) | (selected_severe < current_severe))
    )
    accepted = (raw_selected == 0) | (utility_improves & severe_safe)
    selected = torch.where(accepted, raw_selected, torch.zeros_like(raw_selected))

    patient_accepted = True
    if cfg.get("patient_level_gate", False):
        gated_utility = predicted_utility[part_ids, selected]
        patient_margin = float(cfg.get("patient_gate_margin", 0.0))
        patient_accepted = bool(
            gated_utility.mean() + patient_margin < current_utility.mean()
        )
        if not patient_accepted:
            selected = torch.zeros_like(selected)
            accepted = torch.zeros_like(accepted)

    return selected.to(output["scores"].device), {
        "accepted": int((selected != 0).sum().item()),
        "proposed": int((raw_selected != 0).sum().item()),
        "patient_accepted": patient_accepted,
    }


def run_iterative_inference(model, fragment_data, bone_types, device, output_type, max_parts,
                            max_iters=5, convergence_threshold=1e-4, cfg=None):
    num_parts = len(fragment_data)
    cumulative_transforms = [np.eye(4, dtype=np.float32) for _ in range(num_parts)]
    iter_deltas = []

    current_coords = [fd[0].copy() for fd in fragment_data]
    current_normals = [fd[1].copy() for fd in fragment_data]

    print(f"  Starting iterative inference ({max_iters} max iters, mode={output_type})...")

    for iter_idx in range(max_iters):
        all_points = np.concatenate(current_coords, axis=0).astype(np.float32)
        all_points, centroid = recenter_pc(all_points)
        all_points, scale = rescale_pc(all_points)

        offsets = [0]
        for fc in current_coords:
            offsets.append(offsets[-1] + len(fc))
        norm_coords = [all_points[offsets[i]:offsets[i+1]] for i in range(num_parts)]

        all_normals = np.concatenate(current_normals, axis=0).astype(np.float32)
        points_per_part = np.array([len(nc) for nc in norm_coords], dtype=np.int32)
        fragment_diameters_mm = np.array([
            np.linalg.norm(coords.max(axis=0) - coords.min(axis=0))
            for coords in current_coords
        ], dtype=np.float32)

        output = run_inference_pass(
            model, all_points, all_normals, points_per_part, bone_types,
            fragment_diameters_mm, scale, device, max_parts
        )

        # Immediately move to CPU and delete GPU tensors
        if isinstance(model, FragmentConfidenceModule):
            selected, selection_stats = select_confidence_candidates(output, cfg)
            part_ids = torch.arange(selected.numel(), device=selected.device)
            quat = output["candidate_quat"][part_ids, selected].float().detach().cpu()
            trans = output["candidate_trans"][part_ids, selected].float().detach().cpu()
            print(
                f"  Gate: accepted={selection_stats['accepted']}/"
                f"{selection_stats['proposed']} proposed, "
                f"patient_accepted={selection_stats['patient_accepted']}"
            )
        else:
            quat = output["quat"].float().detach().cpu()
            trans = output["trans"].float().detach().cpu()
        rot = quaternion_to_matrix(quat)
        del output
        torch.cuda.empty_cache()

        T_preds = []
        for i in range(num_parts):
            T_pred = np.eye(4, dtype=np.float32)
            T_pred[:3, :3] = rot[i].numpy()
            T_pred[:3, 3] = trans[i].numpy()
            T_preds.append(T_pred)

        T_is = [transform_pose_to_original_space(T_preds[i], centroid, scale) for i in range(num_parts)]

        max_delta = 0.0
        for i in range(num_parts):
            cumulative_transforms[i] = T_is[i] @ cumulative_transforms[i]
            current_coords[i] = apply_transform_to_fragment_coords(current_coords[i], T_is[i])
            current_normals[i] = apply_rotation_to_fragment_normals(current_normals[i], T_is[i])
            delta = np.linalg.norm(T_is[i][:3, 3])
            max_delta = max(max_delta, delta)

        iter_deltas.append(max_delta)
        print(f"  Iter {iter_idx + 1}: max_trans_delta={max_delta:.6f}")

        if max_delta < convergence_threshold:
            print(f"  Converged at iteration {iter_idx + 1}!")
            return cumulative_transforms, iter_idx + 1, True, iter_deltas

    print(f"  Reached max iterations ({max_iters}), not converged.")
    return cumulative_transforms, max_iters, False, iter_deltas


def normalize_transforms_by_first_sa(cumulative_transforms, meshdict):
    """
    Re-express all transforms relative to the first SA fragment.
    The first SA fragment gets identity (stays in place); others are relative to it.
    """
    if not meshdict.get("SA"):
        return cumulative_transforms
    # SA is always iterated first, so index 0 is the first SA fragment
    T_sa0_inv = np.linalg.inv(cumulative_transforms[0])
    return [T_sa0_inv @ T for T in cumulative_transforms]


def build_final_results(cumulative_transforms, fragment_names):
    results = []
    for i, frag_name in enumerate(fragment_names):
        results.append({
            "fragment_id": frag_name,
            "transformation": cumulative_transforms[i].tolist(),
        })
    return results


def save_prediction(sample_dir, results, output_type, training_mode="full", result_suffix=""):
    if result_suffix:
        suffix = f"_{result_suffix}"
    elif output_type != "coords":
        suffix = f"_{output_type}"
    elif training_mode != "full":
        suffix = f"_{training_mode}"
    else:
        suffix = ""
    filename = f"reduction-poses-matrices{suffix}.json"
    json_path = os.path.join(sample_dir, filename)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    return json_path


def main():
    args = parse_args()

    config_path = args.config if args.config is not None else CONFIG_PATH
    initial_cfg = OmegaConf.load(config_path)
    configure_deterministic_inference(
        bool(initial_cfg.get("deterministic_inference", False))
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading config: {config_path}")
    model, cfg = load_config_and_model(config_path, device, debug_vis=args.debug_vis)
    output_type = cfg.output_type
    training_mode = cfg.get("training_mode", "full")
    max_parts = cfg.transformer_model.max_parts

    input_dir = args.input_dir if args.input_dir is not None else cfg.input_dir
    npoints = args.npoints if args.npoints is not None else cfg.get("npoints", 5000)
    min_points = int(cfg.get("min_points", 20))
    allocation_mode = str(cfg.get("allocation_mode", "volume"))
    max_iters = args.max_iters if args.max_iters is not None else cfg.get("max_iters", 5)
    convergence_threshold = cfg.get("convergence_threshold", 1e-4)
    debug_vis = args.debug_vis if args.debug_vis else cfg.get("debug_vis", False)

    print(f"  output_type={output_type}, training_mode={training_mode}, max_parts={max_parts}")
    print(
        f"  debug_vis={debug_vis}, max_iters={max_iters}, npoints={npoints}, "
        f"min_points={min_points}, allocation_mode={allocation_mode}"
    )
    print(f"  Model loaded.\n")

    print(f"Scanning: {input_dir}")
    samples = discover_samples(input_dir)
    print(f"Found {len(samples)} sample(s).\n")

    if not samples:
        print("No samples found.")
        return

    num_success = 0
    num_failed = 0

    for idx, (sample_name, sample_dir, obj_path) in enumerate(samples):
        if cfg.get("deterministic_inference", False):
            np.random.seed(42)
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
        print(f"\n[{idx + 1}/{len(samples)}] {sample_name}")
        t_start = time.time()

        try:
            meshdict = load_obj_fragments(obj_path, verbose=False)
            fragment_data = build_fragment_data_from_meshdict(
                meshdict,
                npoints=npoints,
                min_points=min_points,
                allocation_mode=allocation_mode,
            )
            bone_types = []
            for bone_id, bone_type in enumerate(["SA", "LI", "RI"]):
                bone_types.extend([bone_id] * len(meshdict.get(bone_type, {})))
            num_parts = len(fragment_data)
            print(f"  Fragments: {num_parts}")

            if debug_vis:
                print("  Visualizing input fragments...")
                from datasets.vis_func import visualize_fragments; visualize_fragments(meshdict)

            cumulative_transforms, iter_used, converged, iter_deltas = run_iterative_inference(
                model, fragment_data, bone_types, device,
                output_type=output_type,
                max_parts=max_parts,
                max_iters=max_iters,
                convergence_threshold=convergence_threshold,
                cfg=cfg,
            )

            fragment_names = []
            for bone_type in ["SA", "LI", "RI"]:
                fragments = meshdict.get(bone_type, {})
                for frag_name in sorted(fragments.keys()):
                    fragment_names.append(frag_name)

            normalized_transforms = normalize_transforms_by_first_sa(cumulative_transforms, meshdict)
            results = build_final_results(normalized_transforms, fragment_names)
            result_suffix = cfg.get("result_suffix", "")
            json_path = save_prediction(sample_dir, results, output_type, training_mode, result_suffix)

            if debug_vis:
                print("  Visualizing predicted assembly...")
                from datasets.vis_func import visualize_transformed_fragments; visualize_transformed_fragments(meshdict, results)

            elapsed = time.time() - t_start
            num_success += 1

            delta_str = ", ".join([f"{d:.6f}" for d in iter_deltas])
            print(f"  Iters: {iter_used}/{max_iters}, converged={converged}")
            print(f"  Deltas: [{delta_str}]")
            print(f"  Saved: {json_path}  ({elapsed:.2f}s)")

        except Exception as e:
            import traceback
            elapsed = time.time() - t_start
            num_failed += 1
            print(f"  FAILED: {e}")
            traceback.print_exc()

    print(f"\n{'=' * 50}")
    print(f"Total: {len(samples)}  Success: {num_success}  Failed: {num_failed}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
