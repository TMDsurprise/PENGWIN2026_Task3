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


class RankerOnlyCheckpoint(L.Callback):
    """Persist each Ranker epoch without duplicating the frozen coordinate backbone."""

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        if not hasattr(pl_module, "candidate_alphas"):
            return
        output_dir = os.path.join(trainer.default_root_dir, "ranker_only_checkpoints")
        os.makedirs(output_dir, exist_ok=True)
        state = {
            key: value.detach().cpu()
            for key, value in pl_module.state_dict().items()
            if not key.startswith("transformer_model.")
        }
        path = os.path.join(output_dir, f"epoch-{trainer.current_epoch:03d}.pt")
        torch.save(
            {
                "epoch": int(trainer.current_epoch),
                "global_step": int(trainer.global_step),
                "state_dict": state,
                "candidate_alphas": list(pl_module.candidate_alphas),
                "finetune_scope": pl_module.finetune_scope,
            },
            path,
        )

@hydra.main(version_base=None, config_path="../../configs", config_name="backbone/train_rotation_aware_e659_v2")
def main(cfg: DictConfig):
    L.seed_everything(cfg.get("seed", 42), workers=True)

    datamodule: L.LightningDataModule = instantiate(cfg.data)
    model: L.LightningModule = instantiate(cfg.model)

    callbacks: List[L.Callback] = [
        instantiate(cb) for cb in cfg.get("callbacks", {}).values() if cb is not None
    ]
    callbacks.append(RankerOnlyCheckpoint())

    trainer: L.Trainer = instantiate(cfg.trainer, callbacks=callbacks)
    ckpt_path = cfg.get("ckpt_path", None)
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
    # # watch -n 0.1 -d nvidia-smi
    # CUDA_VISIBLE_DEVICES=2,3 // 0,1
