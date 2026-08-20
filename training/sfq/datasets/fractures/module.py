"""Lightning DataModule for fragment assembly training.

Supports four data modes via `data_type`:
    'simu'         — simulation NPZ data only (data_root_sim)
    'real'         — clinical data only (data_root_real), auto-detects OBJ vs NPZ
    'mix'          — both sources fully concatenated
    'mix_balanced' — each epoch: randomly sample len(real) from sim, concat with
                      real so every epoch has equal sim/real counts

Typical workflow:
    1. Pre-train on simulation:  data_type: simu
    2. Fine-tune on clinical:    data_type: real  (with training_mode: lora)
    3. Joint training:           data_type: mix_balanced
"""

import random
from typing import Literal, List, Optional, Union

import lightning as L
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torch.utils.data._utils.collate import default_collate

from pathlib import Path

from .sample import FragmentSamples
from .clinical import ClinicalFragmentSamples
from .clinical_npz import ClinicalFragmentSamplesCached


class BalancedMixDataset(Dataset):
    """Each epoch randomly samples sim_multiplier * len(real) from sim.

    Pass seed=None for training (random per epoch), seed=42 for val/test (deterministic).
    """

    def __init__(self, sim_dataset: Dataset, real_dataset: Dataset, seed: Optional[int] = None,
                 sim_multiplier: int = 1):
        super().__init__()
        self.sim_dataset = sim_dataset
        self.real_dataset = real_dataset
        self.split = "train" if seed is None else "val"
        self._rng = random.Random(seed)
        self.sim_multiplier = sim_multiplier
        self._resample()

    def _resample(self):
        n_real = len(self.real_dataset)
        n_sim_target = min(self.sim_multiplier * n_real, len(self.sim_dataset))
        self._sim_indices = self._rng.sample(range(len(self.sim_dataset)), n_sim_target)

    def reshuffle(self):
        if self.split != "train":
            return
        if hasattr(self.sim_dataset, "reshuffle"):
            self.sim_dataset.reshuffle()
        if hasattr(self.real_dataset, "reshuffle"):
            self.real_dataset.reshuffle()
        self._resample()

    def __len__(self):
        return len(self.real_dataset) + len(self._sim_indices)

    def __getitem__(self, idx):
        n_real = len(self.real_dataset)
        if idx < n_real:
            return self.real_dataset[idx]
        else:
            return self.sim_dataset[self._sim_indices[idx - n_real]]


class FragmentDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_root_sim: str = "",
        data_root_real: str = "",
        data_type: Literal["simu", "real", "mix", "mix_balanced"] = "simu",
        val_data_type: Optional[Literal["simu", "real", "mix", "mix_balanced"]] = None,
        min_parts: int = 2,
        max_parts: int = 50,
        batch_size: int = 5,
        num_workers: int = 5,
        random_anchor: bool = True,
        merge_prob: float = 0.15,
        frac_prob: float = 0.70,
        drop_prob: float = 0.1,
        region_dropout_prob: float = 0.0,
        point_jitter_std: float = 0.0,
        npoints_per_bone: int = 5000,
        real_pose_ratio: float = 0.2,
        real_pose_flip_prob: float = 0.0,
        sim2clinic_enabled: bool = False,
        proposal_ratio: float = 0.2,
        qsmall_hard_ratio: float = 0.0,
        replay_ratio: float = 0.1,
        hard_pool_size: int = 20000,
        hard_pool_per_bucket: int = 256,
        hard_pool_age_decay: float = 0.08,
        initial_hard_pool_paths=None,
        fixed_train_seed_base=None,
        fixed_train_pose_mode: str = "natural",
        train_subset_size: int = 0,
        val_subset_size: int = 0,
        val_from_train_subset: bool = False,
    ):
        super().__init__()
        self.data_root_sim = data_root_sim
        self.data_root_real = data_root_real
        self.data_type = data_type
        self.val_data_type = val_data_type
        self.min_parts = min_parts
        self.max_parts = max_parts
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_anchor = random_anchor
        self.merge_prob = merge_prob
        self.frac_prob = frac_prob
        self.drop_prob = drop_prob
        self.region_dropout_prob = region_dropout_prob
        self.point_jitter_std = point_jitter_std
        self.npoints_per_bone = npoints_per_bone
        self.real_pose_ratio = real_pose_ratio
        self.real_pose_flip_prob = real_pose_flip_prob
        self.sim2clinic_enabled = sim2clinic_enabled
        self.proposal_ratio = proposal_ratio
        self.qsmall_hard_ratio = qsmall_hard_ratio
        self.replay_ratio = replay_ratio
        self.hard_pool_size = hard_pool_size
        self.hard_pool_per_bucket = hard_pool_per_bucket
        self.hard_pool_age_decay = hard_pool_age_decay
        self.initial_hard_pool_paths = list(initial_hard_pool_paths or [])
        self.fixed_train_seed_base = fixed_train_seed_base
        self.fixed_train_pose_mode = str(fixed_train_pose_mode)
        self.train_subset_size = int(train_subset_size)
        self.val_subset_size = int(val_subset_size)
        self.val_from_train_subset = bool(val_from_train_subset)

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def _make_sim_dataset(self, split: str) -> FragmentSamples:
        return FragmentSamples(
            split=split,
            data_root=self.data_root_sim,
            min_parts=self.min_parts,
            max_parts=self.max_parts,
            random_anchor=self.random_anchor,
            merge_prob=self.merge_prob,
            frac_prob=self.frac_prob,
            drop_prob=self.drop_prob,
            region_dropout_prob=self.region_dropout_prob,
            point_jitter_std=self.point_jitter_std,
            sim2clinic_enabled=self.sim2clinic_enabled,
            proposal_ratio=self.proposal_ratio,
            qsmall_hard_ratio=self.qsmall_hard_ratio,
            replay_ratio=self.replay_ratio,
            hard_pool_size=self.hard_pool_size,
            hard_pool_per_bucket=self.hard_pool_per_bucket,
            hard_pool_age_decay=self.hard_pool_age_decay,
            initial_hard_pool_paths=(
                self.initial_hard_pool_paths if split == "train" else []
            ),
            fixed_train_seed_base=(
                self.fixed_train_seed_base if split == "train" else None
            ),
            fixed_train_pose_mode=self.fixed_train_pose_mode,
        )

    def _make_real_dataset(self, split: str) -> Dataset:
        # Auto-detect: .npz files → cached NPZ loader, otherwise → OBJ loader
        root = Path(self.data_root_real)
        has_npz = root.is_dir() and any(root.rglob("*.npz"))
        if has_npz:
            return ClinicalFragmentSamplesCached(
                split=split,
                data_root_real=self.data_root_real,
                npoints_per_bone=self.npoints_per_bone,
                min_parts=self.min_parts,
                max_parts=self.max_parts,
                random_anchor=self.random_anchor,
                merge_prob=self.merge_prob,
                drop_prob=self.drop_prob,
                region_dropout_prob=self.region_dropout_prob,
                point_jitter_std=self.point_jitter_std,
                real_pose_ratio=self.real_pose_ratio,
                real_pose_flip_prob=self.real_pose_flip_prob,
            )
        else:
            return ClinicalFragmentSamples(
                split=split,
                data_root_real=self.data_root_real,
                npoints_per_bone=self.npoints_per_bone,
                min_parts=self.min_parts,
                max_parts=self.max_parts,
                random_anchor=self.random_anchor,
                merge_prob=self.merge_prob,
                drop_prob=self.drop_prob,
                region_dropout_prob=self.region_dropout_prob,
                point_jitter_std=self.point_jitter_std,
            )

    def _make_dataset(self, split: str) -> Dataset:
        data_type = self.val_data_type if split == "val" and self.val_data_type else self.data_type
        if data_type == "simu":
            return self._make_sim_dataset(split)
        elif data_type == "real":
            return self._make_real_dataset(split)
        elif data_type == "mix":
            datasets = []
            if self.data_root_sim:
                datasets.append(self._make_sim_dataset(split))
            if self.data_root_real:
                datasets.append(self._make_real_dataset(split))
            if not datasets:
                raise ValueError("data_type='mix' requires at least one of data_root_sim or data_root_real")
            return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
        elif data_type == "mix_balanced":
            sim = self._make_sim_dataset(split)
            real = self._make_real_dataset(split)
            if split == "train":
                return BalancedMixDataset(sim, real)
            else:
                return BalancedMixDataset(sim, real, seed=42)
        else:
            raise ValueError(f"Unknown data_type: {data_type!r}. "
                             f"Choose 'simu', 'real', 'mix', or 'mix_balanced'.")

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = self._make_dataset("train")
            self.val_dataset = self._make_dataset("val")
            if self.train_subset_size > 0:
                self.train_dataset = Subset(
                    self.train_dataset,
                    range(min(self.train_subset_size, len(self.train_dataset))),
                )
            if self.val_from_train_subset:
                self.val_dataset = self.train_dataset
            elif self.val_subset_size > 0:
                self.val_dataset = Subset(
                    self.val_dataset,
                    range(min(self.val_subset_size, len(self.val_dataset))),
                )
            self._log_dataset_info()
        elif stage == "validate":
            self.val_dataset = self._make_dataset("val")
        elif stage in ("test", "predict"):
            self.test_dataset = self._make_dataset("test")

    def _log_dataset_info(self):
        from lightning.pytorch.utilities.rank_zero import rank_zero_only

        @rank_zero_only
        def _print():
            n_train = len(self.train_dataset)
            n_val = len(self.val_dataset)
            aug = []
            if self.region_dropout_prob > 0:
                aug.append(f"region_dropout={self.region_dropout_prob}")
            if self.point_jitter_std > 0:
                aug.append(f"jitter={self.point_jitter_std}")
            if self.sim2clinic_enabled:
                aug.append(
                    f"sim2clinic(qsmall={self.qsmall_hard_ratio}, "
                    f"proposal={self.proposal_ratio}, replay={self.replay_ratio})"
                )
            aug_str = f"  augmentations: {', '.join(aug)}" if aug else ""
            val_mode = self.val_data_type or self.data_type
            print(
                f"[DataModule] data_type={self.data_type} val_data_type={val_mode} "
                f" train={n_train}  val={n_val}{aug_str}"
            )

        _print()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=False,
            pin_memory=True,
            collate_fn=variable_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=False,
            pin_memory=True,
            collate_fn=variable_collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            persistent_workers=False,
            collate_fn=variable_collate_fn,
        )


def variable_collate_fn(batch: list) -> dict:
    """Collate variable-length point cloud samples into a single batch dict."""
    if not batch:
        return {}

    batch_size = len(batch)
    has_keys = set()
    for sample in batch:
        has_keys.update(sample.keys())

    has_points = "pointclouds" in has_keys
    has_normals = "pointclouds_normals" in has_keys
    has_gt = "pointclouds_gt" in has_keys
    has_normals_gt = "pointclouds_normals_gt" in has_keys
    has_bone_ids = "point_bone_ids" in has_keys
    has_ppp = "points_per_part" in has_keys

    total_points = sum(len(s["pointclouds"]) for s in batch if "pointclouds" in s)
    total_points_gt = sum(len(s["pointclouds_gt"]) for s in batch if "pointclouds_gt" in s)

    if has_points:    batch_points = torch.empty((total_points, 3), dtype=torch.float32)
    if has_normals:   batch_normals = torch.empty((total_points, 3), dtype=torch.float32)
    if has_gt:        batch_gt = torch.empty((total_points_gt, 3), dtype=torch.float32)
    if has_normals_gt: batch_normals_gt = torch.empty((total_points_gt, 3), dtype=torch.float32)
    if has_bone_ids:
        batch_bone_ids = torch.empty((total_points_gt,), dtype=torch.int64)
        batch_global_bone_ids = torch.empty((total_points_gt,), dtype=torch.int64)
    if has_ppp:
        batch_part_ids = torch.empty((total_points,), dtype=torch.int64)

    points_per_sample = torch.empty(batch_size, dtype=torch.int32)
    other_keys: dict = {}
    start_idx = 0
    start_idx_gt = 0
    current_bone_offset = 0

    for i, sample in enumerate(batch):
        num_points = len(sample["pointclouds"])
        end_idx = start_idx + num_points

        if has_points:
            batch_points[start_idx:end_idx] = torch.from_numpy(sample["pointclouds"].astype(np.float32))
        if has_normals and "pointclouds_normals" in sample:
            batch_normals[start_idx:end_idx] = torch.from_numpy(sample["pointclouds_normals"].astype(np.float32))

        if has_gt and "pointclouds_gt" in sample:
            num_gt = len(sample["pointclouds_gt"])
            end_idx_gt = start_idx_gt + num_gt
            batch_gt[start_idx_gt:end_idx_gt] = torch.from_numpy(sample["pointclouds_gt"].astype(np.float32))
            if has_normals_gt and "pointclouds_normals_gt" in sample:
                batch_normals_gt[start_idx_gt:end_idx_gt] = torch.from_numpy(sample["pointclouds_normals_gt"].astype(np.float32))
            if has_bone_ids and "point_bone_ids" in sample:
                p_ids = sample["point_bone_ids"]
                if isinstance(p_ids, np.ndarray):
                    p_ids = torch.from_numpy(p_ids.astype(np.int64)).view(-1)
                batch_bone_ids[start_idx_gt:end_idx_gt] = p_ids
                batch_global_bone_ids[start_idx_gt:end_idx_gt] = p_ids + current_bone_offset
                current_bone_offset += p_ids.max().item() + 1
            start_idx_gt = end_idx_gt

        if has_ppp and "points_per_part" in sample:
            ppp = sample["points_per_part"]
            if isinstance(ppp, (list, np.ndarray)):
                ppp = torch.tensor(ppp)
            ppp_valid = ppp[ppp > 0]
            p_part_ids = torch.repeat_interleave(
                torch.arange(len(ppp_valid), dtype=torch.int64), ppp_valid.to(torch.int64)
            )
            batch_part_ids[start_idx:end_idx] = p_part_ids

        points_per_sample[i] = num_points

        for k, v in sample.items():
            if k not in {"pointclouds", "pointclouds_normals", "pointclouds_gt",
                         "pointclouds_normals_gt", "point_bone_ids"}:
                other_keys.setdefault(k, []).append(v)

        start_idx = end_idx

    batch_out = {"points_per_sample": points_per_sample}
    if has_points:     batch_out["pointclouds"] = batch_points
    if has_normals:    batch_out["pointclouds_normals"] = batch_normals
    if has_gt:         batch_out["pointclouds_gt"] = batch_gt
    if has_normals_gt: batch_out["pointclouds_normals_gt"] = batch_normals_gt
    if has_bone_ids:
        batch_out["point_bone_ids"] = batch_bone_ids
        batch_out["global_bone_ids"] = batch_global_bone_ids
    if has_ppp:
        batch_out["point_part_ids"] = batch_part_ids
        batch_out["points_per_part"] = default_collate(
            other_keys.pop("points_per_part")) if "points_per_part" in other_keys else None

    for k, v_list in other_keys.items():
        try:
            batch_out[k] = default_collate(v_list)
        except Exception:
            batch_out[k] = v_list

    return batch_out
