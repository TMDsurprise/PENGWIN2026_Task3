#!/usr/bin/env python3
"""Build deterministic simulation caches for the e27 four-backbone pose pool."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from datasets.fractures.sample import FragmentSamples
from tools.e27_sim_pool_common import (
    build_cached_ranker_record,
    deterministic_seed,
    load_coordinate_backbone,
    load_e3_ranker,
    rollout_final_pose,
    seed_sample,
    split_fragments,
    summarize_oracle,
)


def atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_dataset(args: argparse.Namespace) -> FragmentSamples:
    is_train = args.split == "train"
    return FragmentSamples(
        split=args.split,
        data_root=str(args.sim_root),
        data_category="all",
        min_parts=2,
        max_parts=50,
        max_bone=3,
        random_anchor=True,
        merge_prob=0.15,
        frac_prob=0.70,
        drop_prob=0.10,
        sim2clinic_enabled=True,
        proposal_ratio=0.20 if is_train else 0.0,
        qsmall_hard_ratio=0.0,
        replay_ratio=0.0,
        region_dropout_prob=0.10 if is_train else 0.0,
        point_jitter_std=0.002 if is_train else 0.0,
    )


def task_list(dataset_size: int, repeats: int, max_total: int) -> list[tuple[int, int]]:
    tasks = [(repeat, index) for repeat in range(repeats) for index in range(dataset_size)]
    if max_total > 0:
        tasks = tasks[:max_total]
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-root", type=Path, required=True)
    parser.add_argument("--e3-checkpoint", type=Path, required=True)
    parser.add_argument("--b1-checkpoint", type=Path, required=True)
    parser.add_argument("--b2-checkpoint", type=Path, required=True)
    parser.add_argument("--b3-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-total", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=10)
    parser.add_argument("--convergence-mm", type=float, default=2.0)
    parser.add_argument(
        "--source-alphas", type=float, nargs=4, default=(0.0, 1.0, 1.0, 1.0)
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    if args.repeats < 1:
        raise ValueError("repeats must be positive")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Cache construction requires CUDA")

    dataset = build_dataset(args)
    tasks = task_list(len(dataset), args.repeats, args.max_total)
    local_tasks = [task for ordinal, task in enumerate(tasks) if ordinal % args.shard_count == args.shard_index]
    if not local_tasks:
        raise RuntimeError("No tasks assigned to this shard")

    print(
        json.dumps(
            {
                "event": "load_models",
                "device": str(device),
                "split": args.split,
                "dataset_size": len(dataset),
                "global_tasks": len(tasks),
                "local_tasks": len(local_tasks),
                "shard": [args.shard_index, args.shard_count],
            }
        ),
        flush=True,
    )
    e3 = load_e3_ranker(args.e3_checkpoint, device)
    b1 = load_coordinate_backbone(args.b1_checkpoint, device, point_reliability=False)
    b2 = load_coordinate_backbone(args.b2_checkpoint, device, point_reliability=True)
    b3 = load_coordinate_backbone(args.b3_checkpoint, device, point_reliability=True)
    models = ((e3, True), (b1, False), (b2, False), (b3, False))

    cache_dir = args.output_dir / args.split
    cache_dir.mkdir(parents=True, exist_ok=True)
    records = []
    errors = []
    started_all = time.time()
    for local_ordinal, (repeat, index) in enumerate(local_tasks, 1):
        output_path = cache_dir / f"repeat{repeat:03d}_index{index:06d}.npz"
        if output_path.exists() and not args.overwrite:
            with np.load(output_path, allow_pickle=False) as cached:
                records.append({key: cached[key] for key in cached.files})
            continue

        started = time.time()
        seed = deterministic_seed("e27", args.split, repeat, index)
        try:
            seed_sample(seed)
            sample = dataset[index]
            points = split_fragments(sample, "pointclouds")
            normals = split_fragments(sample, "pointclouds_normals")
            num_parts = len(points)
            bones = np.asarray(sample["bonetype"][:num_parts], dtype=np.int64)
            diameters = np.asarray(
                sample["fragment_diameter_mm"][:num_parts], dtype=np.float32
            )
            physical_scale_mm = float(np.asarray(sample["norm_scale"]))
            final_poses = []
            rollout_iters = []
            for model, is_ranker in models:
                pose, iterations = rollout_final_pose(
                    model,
                    points,
                    normals,
                    bones,
                    diameters,
                    physical_scale_mm,
                    device,
                    ranker=is_ranker,
                    max_iters=args.max_iters,
                    convergence_mm=args.convergence_mm,
                )
                final_poses.append(pose)
                rollout_iters.append(iterations)
            record = build_cached_ranker_record(
                e3,
                sample,
                final_poses,
                args.source_alphas,
                device,
            )
            record.update(
                {
                    "name": np.asarray(str(sample["name"])),
                    "sample_index": np.asarray(index, dtype=np.int64),
                    "augmentation_seed": np.asarray(seed, dtype=np.int64),
                    "repeat": np.asarray(repeat, dtype=np.int16),
                    "rollout_iters": np.asarray(rollout_iters, dtype=np.int16),
                }
            )
            atomic_npz(output_path, record)
            records.append(record)
        except Exception as error:
            errors.append(
                {
                    "repeat": repeat,
                    "index": index,
                    "seed": seed,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(json.dumps({"event": "error", **errors[-1]}), flush=True)
            if len(errors) >= 3:
                raise
            continue

        elapsed = time.time() - started
        if local_ordinal <= 3 or local_ordinal % 10 == 0 or local_ordinal == len(local_tasks):
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "local": [local_ordinal, len(local_tasks)],
                        "repeat": repeat,
                        "index": index,
                        "parts": num_parts,
                        "rollout_iters": rollout_iters,
                        "elapsed_s": round(elapsed, 3),
                        "total_elapsed_s": round(time.time() - started_all, 3),
                    }
                ),
                flush=True,
            )

    summary = {
        "status": "complete" if not errors else "complete_with_errors",
        "split": args.split,
        "dataset_size": len(dataset),
        "global_tasks": len(tasks),
        "local_tasks": len(local_tasks),
        "written_or_loaded": len(records),
        "errors": errors,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "elapsed_s": time.time() - started_all,
        "source_alphas": list(args.source_alphas),
        "oracle": summarize_oracle(records),
    }
    manifest = args.output_dir / f"manifest_{args.split}_shard{args.shard_index:02d}.json"
    manifest.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
