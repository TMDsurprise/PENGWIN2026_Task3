from __future__ import annotations

from lightning.fabric.plugins.io.torch_io import TorchCheckpointIO


class TrustedTorchCheckpointIO(TorchCheckpointIO):
    """Load a locally generated Lightning checkpoint with full trainer state."""

    def load_checkpoint(self, path, map_location=lambda storage, loc: storage, weights_only=None):
        return super().load_checkpoint(
            path, map_location=map_location, weights_only=False
        )
