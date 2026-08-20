#!/usr/bin/env python3
"""Train the complete e3 ranking branch on cached four-backbone candidates."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tools.e27_sim_pool_common import load_e3_ranker


class CachedPoolDataset(Dataset):
    def __init__(self, root: Path):
        self.paths = sorted(root.glob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No cache files under {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        with np.load(self.paths[index], allow_pickle=False) as data:
            return {
                "part_features": data["part_features"].astype(np.float32),
                "geometry": data["geometry"].astype(np.float32),
                "bone": data["bone"].astype(np.int64),
                "metrics": data["metrics"].astype(np.float32),
                "utility": data["utility"].astype(np.float32),
                "severe": data["severe"].astype(np.float32),
                "loss_mask": data["loss_mask"].astype(bool),
                "name": str(data["name"]),
            }


def collate_cached(rows: list[dict]) -> dict:
    batch_size = len(rows)
    max_parts = max(len(row["bone"]) for row in rows)
    candidates = rows[0]["geometry"].shape[1]
    feature_dim = rows[0]["part_features"].shape[-1]
    geometry_dim = rows[0]["geometry"].shape[-1]
    tensors = {
        "part_features": torch.zeros(batch_size, max_parts, feature_dim),
        "geometry": torch.zeros(batch_size, max_parts, candidates, geometry_dim),
        "bone": torch.zeros(batch_size, max_parts, dtype=torch.long),
        "metrics": torch.zeros(batch_size, max_parts, candidates, 4),
        "utility": torch.zeros(batch_size, max_parts, candidates),
        "severe": torch.zeros(batch_size, max_parts, candidates),
        "valid": torch.zeros(batch_size, max_parts, dtype=torch.bool),
        "loss_mask": torch.zeros(batch_size, max_parts, dtype=torch.bool),
        "names": [row["name"] for row in rows],
    }
    for batch_index, row in enumerate(rows):
        count = len(row["bone"])
        tensors["part_features"][batch_index, :count] = torch.from_numpy(row["part_features"])
        tensors["geometry"][batch_index, :count] = torch.from_numpy(row["geometry"])
        tensors["bone"][batch_index, :count] = torch.from_numpy(row["bone"])
        tensors["metrics"][batch_index, :count] = torch.from_numpy(row["metrics"])
        tensors["utility"][batch_index, :count] = torch.from_numpy(row["utility"])
        tensors["severe"][batch_index, :count] = torch.from_numpy(row["severe"])
        tensors["valid"][batch_index, :count] = True
        tensors["loss_mask"][batch_index, :count] = torch.from_numpy(row["loss_mask"])
    return tensors


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def cached_forward(model, batch: dict) -> dict:
    part_feature = model.backbone_projection(batch["part_features"].float())
    token = part_feature[:, :, None, :] + model.geometry_projection(batch["geometry"].float())
    token = token + model.bone_embedding(batch["bone"].clamp(0, 2))[:, :, None, :]
    candidate_ids = torch.arange(model.num_candidates, device=token.device)
    token = token + model.candidate_embedding(candidate_ids)[None, None, :, :]
    slots = torch.arange(token.shape[1], device=token.device).clamp_max(model.max_parts - 1)
    token = token + model.fragment_slot_embedding(slots)[None, :, None, :]
    flat = token.reshape(token.shape[0], token.shape[1] * token.shape[2], token.shape[3])
    padding = (~batch["valid"][:, :, None].expand(-1, -1, model.num_candidates)).reshape(
        token.shape[0], -1
    )
    encoded = model.context_encoder(flat, src_key_padding_mask=padding)
    encoded = encoded.reshape(token.shape)
    return {
        "scores_padded": model.score_head(encoded).squeeze(-1),
        "metric_prediction_padded": F.softplus(model.metric_head(encoded)),
        "severe_logits_padded": model.severe_head(encoded).squeeze(-1),
    }


def loss_output(batch: dict, output: dict) -> dict:
    mask = batch["loss_mask"] & batch["valid"]
    return {
        "scores": output["scores_padded"][mask],
        "metric_prediction": output["metric_prediction_padded"][mask],
        "severe_logits": output["severe_logits_padded"][mask],
        "metrics": batch["metrics"][mask],
        "utility": batch["utility"][mask],
        "severe": batch["severe"][mask],
    }


def c2_select(model, batch: dict, output: dict) -> torch.Tensor:
    valid = batch["valid"]
    flattened = {
        "scores": output["scores_padded"][valid],
        "metric_prediction": output["metric_prediction_padded"][valid],
        "severe_logits": output["severe_logits_padded"][valid],
    }
    points_per_part = valid.long()
    return model._c2_select(flattened, points_per_part)


def batch_statistics(model, batch: dict, output: dict) -> dict[str, float]:
    valid = batch["valid"]
    eval_mask = batch["loss_mask"] & valid
    scores = output["scores_padded"][valid]
    utility = batch["utility"][valid]
    metrics = batch["metrics"][valid]
    severe = batch["severe"][valid]
    flat_eval = batch["loss_mask"][valid]
    raw = scores.argmax(dim=1)
    gated = c2_select(model, batch, output)
    oracle = utility.argmin(dim=1)
    rows = torch.arange(len(raw), device=raw.device)

    raw = raw[flat_eval]
    gated = gated[flat_eval]
    oracle = oracle[flat_eval]
    utility = utility[flat_eval]
    metrics = metrics[flat_eval]
    severe = severe[flat_eval]
    rows = torch.arange(len(raw), device=raw.device)
    raw_utility = utility[rows, raw]
    gated_utility = utility[rows, gated]
    oracle_utility = utility[rows, oracle]
    current_utility = utility[:, 0]
    raw_metrics = metrics[rows, raw]
    gated_metrics = metrics[rows, gated]
    oracle_metrics = metrics[rows, oracle]
    current_metrics = metrics[:, 0]
    severe_current = severe[:, 0] > 0.5
    raw_recovered = severe_current & (severe[rows, raw] < 0.5)
    gated_recovered = severe_current & (severe[rows, gated] < 0.5)
    pairwise, spearman = model._ranking_diagnostics(scores[flat_eval], utility)
    count = float(len(raw))
    result = {
        "count": count,
        "raw_regret": float((raw_utility - oracle_utility).sum()),
        "c2_regret": float((gated_utility - oracle_utility).sum()),
        "raw_correct": float((raw == oracle).sum()),
        "c2_correct": float((gated == oracle).sum()),
        "raw_improvement": float((current_utility - raw_utility).sum()),
        "c2_improvement": float((current_utility - gated_utility).sum()),
        "current_utility": float(current_utility.sum()),
        "raw_utility": float(raw_utility.sum()),
        "c2_utility": float(gated_utility.sum()),
        "oracle_utility": float(oracle_utility.sum()),
        "raw_preserve": float((raw == 0).sum()),
        "c2_preserve": float((gated == 0).sum()),
        "severe_count": float(severe_current.sum()),
        "raw_recovered": float(raw_recovered.sum()),
        "c2_recovered": float(gated_recovered.sum()),
        "pairwise": float(pairwise) * count,
        "spearman": float(spearman) * count,
    }
    for prefix, values in (
        ("current", current_metrics),
        ("raw", raw_metrics),
        ("c2", gated_metrics),
        ("oracle", oracle_metrics),
    ):
        for metric_index, metric_name in enumerate(("tre", "trans", "rot", "cd")):
            result[f"{prefix}_{metric_name}"] = float(values[:, metric_index].sum())
    return result


def merge_statistics(rows: list[dict[str, float]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for row in rows:
        for key, value in row.items():
            totals[key] = totals.get(key, 0.0) + float(value)
    count = max(totals.pop("count"), 1.0)
    severe_count = max(totals.pop("severe_count"), 1.0)
    result = {key: value / count for key, value in totals.items()}
    result["raw_severe_coverage"] = totals["raw_recovered"] / severe_count
    result["c2_severe_coverage"] = totals["c2_recovered"] / severe_count
    result["fragments"] = int(count)
    return result


@torch.inference_mode()
def validate(model, loader: DataLoader, device: torch.device, use_amp: bool) -> dict:
    model.eval()
    statistics = []
    loss_totals: dict[str, float] = {}
    batches = 0
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            output = cached_forward(model, batch)
            losses = model._losses(loss_output(batch, output))
        for key, value in losses.items():
            loss_totals[key] = loss_totals.get(key, 0.0) + float(value)
        statistics.append(batch_statistics(model, batch, output))
        batches += 1
    result = merge_statistics(statistics)
    result.update({f"loss_{key}": value / max(batches, 1) for key, value in loss_totals.items()})
    return result


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: Path, model, epoch: int, metrics: dict, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    safe_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    torch.save(
        {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "validation": metrics,
            "e27_config": safe_config,
        },
        temporary,
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=5e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--scope", choices=("all", "heads"), default="all")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Ranker training requires CUDA")
    use_amp = not args.no_amp

    train_dataset = CachedPoolDataset(args.cache_root / "train")
    val_dataset = CachedPoolDataset(args.cache_root / "val")
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate_cached,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = load_e3_ranker(args.init_checkpoint, device)
    model._apply_finetune_scope(args.scope)
    model.transformer_model.cpu()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, threshold=args.min_delta, threshold_mode="abs"
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "validation.jsonl"
    summary_path = args.output_dir / "summary.json"
    print(
        json.dumps(
            {
                "event": "start",
                "device": str(device),
                "train_cases": len(train_dataset),
                "val_cases": len(val_dataset),
                "trainable_parameters": trainable_count,
                "scope": args.scope,
                "seed": args.seed,
            }
        ),
        flush=True,
    )

    best_regret = math.inf
    best_epoch = -1
    wait = 0
    history = []
    started = time.time()
    for epoch in range(args.max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        train_batches = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                output = cached_forward(model, batch)
                losses = model._losses(loss_output(batch, output))
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            train_loss += float(losses["loss"].detach())
            train_batches += 1

        validation = validate(model, val_loader, device, use_amp)
        validation.update(
            {
                "epoch": epoch,
                "train_loss": train_loss / max(train_batches, 1),
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_s": time.time() - started,
            }
        )
        scheduler.step(validation["raw_regret"])
        history.append(validation)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(validation, sort_keys=True) + "\n")
        print(json.dumps({"event": "validation", **validation}), flush=True)

        improved = validation["raw_regret"] < best_regret - args.min_delta
        if improved:
            best_regret = validation["raw_regret"]
            best_epoch = epoch
            wait = 0
            save_checkpoint(args.output_dir / "best.ckpt", model, epoch, validation, args)
        else:
            wait += 1
        save_checkpoint(args.output_dir / "last.ckpt", model, epoch, validation, args)
        if wait >= args.patience:
            print(json.dumps({"event": "early_stop", "epoch": epoch, "wait": wait}), flush=True)
            break

    summary = {
        "status": "complete",
        "seed": args.seed,
        "scope": args.scope,
        "trainable_parameters": trainable_count,
        "best_epoch": best_epoch,
        "best_raw_regret": best_regret,
        "best": history[best_epoch] if best_epoch >= 0 else None,
        "epochs_run": len(history),
        "elapsed_s": time.time() - started,
        "history": history,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
