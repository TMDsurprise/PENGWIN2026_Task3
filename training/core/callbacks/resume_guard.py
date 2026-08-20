from __future__ import annotations

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping


class ResumeEarlyStoppingGuard(L.Callback):
    """Give a resumed run a fresh patience window while retaining its best score."""

    def __init__(self, patience: int):
        self.patience = int(patience)
        self._applied = False

    def on_train_start(self, trainer, pl_module):
        if self._applied:
            return
        for callback in trainer.callbacks:
            if isinstance(callback, EarlyStopping):
                callback.patience = self.patience
                callback.wait_count = 0
                callback.stopped_epoch = 0
                callback._check_on_train_epoch_end = False
                if trainer.is_global_zero:
                    print(
                        "[ResumeEarlyStoppingGuard] "
                        f"fresh patience={self.patience}, retained best_score={callback.best_score}"
                    )
                self._applied = True
                return
