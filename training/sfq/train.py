"""Training entry point for AssembleNet.

Usage:
    python train.py                                    # training
    python train.py --config-name train_pose           # pose training
    python train.py --config-name finetune      # LoRA fine-tune
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HYDRA_FULL_ERROR"] = "1"

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import lightning as L
from typing import List

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

from callbacks.trusted_checkpoint_io import TrustedTorchCheckpointIO

@hydra.main(version_base=None, config_path="../../configs", config_name="sfq/train_sfq_screen")
def main(cfg: DictConfig):
    L.seed_everything(cfg.get("seed", 42), workers=True)

    datamodule: L.LightningDataModule = instantiate(cfg.data)
    model: L.LightningModule = instantiate(cfg.model)

    init_checkpoint = cfg.get("init_checkpoint", None)
    if init_checkpoint:
        checkpoint = torch.load(init_checkpoint, map_location="cpu", weights_only=True)
        state = checkpoint.get("state_dict", checkpoint)
        init_strict = bool(cfg.get("init_checkpoint_strict", True))
        missing, unexpected = model.load_state_dict(state, strict=init_strict)
        if unexpected or (init_strict and missing):
            raise RuntimeError(
                f"Warm-start state mismatch: missing={missing}, unexpected={unexpected}"
            )
        print(
            f"[train] tensor-only warm start: {init_checkpoint} "
            f"strict={init_strict} missing={len(missing)} unexpected={len(unexpected)}"
        )

    callbacks: List[L.Callback] = [
        instantiate(cb) for cb in cfg.get("callbacks", {}).values()
    ]

    trainer_kwargs = {"callbacks": callbacks}
    if cfg.get("trusted_full_state_resume", False):
        trainer_kwargs["plugins"] = [TrustedTorchCheckpointIO()]
    trainer: L.Trainer = instantiate(cfg.trainer, **trainer_kwargs)
    ckpt_path = cfg.get("ckpt_path", None)
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
    # # watch -n 0.1 -d nvidia-smi
    # CUDA_VISIBLE_DEVICES=2,3 // 0,1
