"""Exponential moving average weights for validation checkpoints."""

from __future__ import annotations

import lightning as L
import torch


class EMAWeights(L.Callback):
    def __init__(self, decay: float = 0.999, update_after_step: int = 50,
                 validate_with_ema: bool = True):
        self.decay = float(decay)
        self.update_after_step = int(update_after_step)
        self.validate_with_ema = bool(validate_with_ema)
        self.shadow = {}
        self.backup = {}
        self._swapped = False
        self._restored_from_checkpoint = False

    def state_dict(self):
        return {
            "shadow": {name: value.detach().cpu() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state_dict):
        shadow = state_dict.get("shadow", {})
        self.shadow = {name: value.detach().clone() for name, value in shadow.items()}
        self._restored_from_checkpoint = bool(self.shadow)

    @staticmethod
    def _parameters(module):
        for name, parameter in module.named_parameters():
            if parameter.requires_grad and parameter.is_floating_point():
                yield name, parameter

    def on_fit_start(self, trainer, pl_module):
        parameters = dict(self._parameters(pl_module))
        if self._restored_from_checkpoint and self.shadow.keys() == parameters.keys():
            self.shadow = {
                name: self.shadow[name].to(device=parameter.device, dtype=parameter.dtype)
                for name, parameter in parameters.items()
            }
        else:
            self.shadow = {
                name: parameter.detach().clone()
                for name, parameter in parameters.items()
            }
            self._restored_from_checkpoint = False
        if trainer.is_global_zero:
            count = sum(value.numel() for value in self.shadow.values())
            source = "checkpoint" if self._restored_from_checkpoint else "model"
            print(
                f"[EMA] tracking {count / 1e6:.1f}M parameters, "
                f"decay={self.decay}, initialized_from={source}"
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        with torch.no_grad():
            for name, parameter in self._parameters(pl_module):
                if trainer.global_step < self.update_after_step:
                    self.shadow[name].copy_(parameter.detach())
                else:
                    self.shadow[name].mul_(self.decay).add_(
                        parameter.detach(), alpha=1.0 - self.decay
                    )

    def on_validation_start(self, trainer, pl_module):
        if not self.validate_with_ema or self._swapped:
            return
        self.backup = {}
        with torch.no_grad():
            for name, parameter in self._parameters(pl_module):
                self.backup[name] = parameter.detach().clone()
                parameter.copy_(self.shadow[name])
        self._swapped = True

    def _restore(self, pl_module):
        if not self._swapped:
            return
        with torch.no_grad():
            for name, parameter in self._parameters(pl_module):
                parameter.copy_(self.backup[name])
        self.backup = {}
        self._swapped = False

    def on_validation_end(self, trainer, pl_module):
        self._restore(pl_module)

    def on_exception(self, trainer, pl_module, exception):
        self._restore(pl_module)
