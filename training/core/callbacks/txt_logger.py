"""Callback that writes per-epoch metrics to a human-readable txt log file."""

from __future__ import annotations

import os
from datetime import datetime

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping


_METRIC_ORDER = [
    "val/loss",
    "val/rot_err_deg",
    "val/trans_err",
    "val/tre",
    "val/qsmall_rot_deg",
    "val/qsmall_trans_mm",
    "val/qsmall_tre_mm",
    "val/qsmall_rot_gt30_rate",
    "train/loss",
    "lr",
]

_HEADER = (
    "# {:>5s}  {:>12s}  {:>12s}  {:>12s}  {:>12s}  {:>12s}  {:>12s}  {:>12s}  "
    "{:>12s}  {:>12s}  {:>10s}  {:>8s}  {:>8s}"
).format(
    "epoch", "val/loss", "rot_deg", "trans", "tre", "q_rot_deg", "q_trans_mm",
    "q_tre_mm", "q_rot30", "train/loss", "lr", "best", "patience"
)


def _find_early_stopping(trainer) -> EarlyStopping | None:
    for cb in trainer.callbacks:
        if isinstance(cb, EarlyStopping):
            return cb
    return None


class TrainingLogCallback(L.Callback):
    """Writes a training_log.txt in log_dir (defaults to trainer.default_root_dir).

    Each row = one epoch, written after validation finishes so both train and val
    metrics are current.  Final block records why training stopped.
    """

    def __init__(self, log_dir: str = ""):
        self.log_dir = log_dir
        self._path: str | None = None
        self._best_val: float | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_fit_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        root = self.log_dir or trainer.default_root_dir
        os.makedirs(root, exist_ok=True)
        self._path = os.path.join(root, "training_log.txt")

        # Find EarlyStopping callback now so we can report its config
        es = _find_early_stopping(trainer)

        with open(self._path, "w") as f:
            f.write(f"# Training log — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# experiment : {trainer.default_root_dir}\n")
            if es is not None:
                f.write(f"# early_stop : monitor={es.monitor}  patience={es.patience}  "
                        f"min_delta={es.min_delta}  mode={es.mode}\n")
            f.write("#\n")
            f.write(_HEADER + "\n")
            f.write("# " + "-" * (len(_HEADER) - 2) + "\n")

    def on_validation_epoch_end(self, trainer, pl_module):
        """Write after validation so val metrics for this epoch are available."""
        if not trainer.is_global_zero:
            return
        self._write_epoch(trainer)

    def on_fit_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        if self._path is None:
            return
        es = _find_early_stopping(trainer)

        lines: list[str] = []
        lines.append("")
        lines.append("# " + "=" * 50)

        if es is not None and es.stopped_epoch != 0:
            lines.append(f"# STOPPED — EarlyStopping triggered")
            lines.append(f"#   monitored   : {es.monitor}")
            lines.append(f"#   best_epoch  : {es.stopped_epoch - es.wait_count}")
            lines.append(f"#   best_score  : {es.best_score:.6f}")
            lines.append(f"#   stopped_at  : {es.stopped_epoch}")
            lines.append(f"#   patience    : {es.patience}  (waited {es.wait_count} epochs with no improvement)")
        elif trainer.current_epoch >= (trainer.max_epochs or 0) - 1:
            lines.append(f"# STOPPED — Reached max_epochs={trainer.max_epochs}")
        else:
            lines.append(f"# STOPPED — epoch {trainer.current_epoch}")

        lines.append("")
        with open(self._path, "a") as f:
            f.write("\n".join(lines) + "\n")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _write_epoch(self, trainer):
        if self._path is None:
            return

        metrics = {
            k: v.item() if hasattr(v, "item") else v
            for k, v in trainer.callback_metrics.items()
        }
        epoch = trainer.current_epoch
        es = _find_early_stopping(trainer)

        # Build the line
        parts = [f"  {epoch:>5d}"]

        for key in _METRIC_ORDER:
            if key == "epoch":
                continue
            val = metrics.get(key)
            if key == "lr" and val is None:
                val = next(
                    (value for name, value in metrics.items() if name.startswith("lr-")),
                    None,
                )
            if val is not None:
                parts.append(f"  {float(val):>12.6f}")
            else:
                parts.append(f"  {'-':>12s}")

        # Best val/loss so far
        current_val = metrics.get("val/loss")
        if current_val is not None:
            current_val = float(current_val)
            if self._best_val is None or current_val < self._best_val:
                self._best_val = current_val
            parts.append(f"  {self._best_val:>10.6f}")
        else:
            parts.append(f"  {'-':>10s}")

        # EarlyStopping wait count
        if es is not None:
            parts.append(f"  {es.wait_count:>8d}")
        else:
            parts.append(f"  {'-':>8s}")

        with open(self._path, "a") as f:
            f.write("".join(parts) + "\n")
