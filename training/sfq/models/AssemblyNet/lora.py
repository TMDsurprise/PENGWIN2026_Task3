"""LoRA (Low-Rank Adaptation) for AssemblyTransformer fine-tuning.

Freeze everything except LoRA adapters + output head.
Strategy: keep pretrained feature-extraction weights frozen (encoding, FFN, norms)
to avoid overfitting on small/noisy clinical GT; only train low-rank attention
residuals and the task head.

Usage:
    from models.AssemblyNet.lora import apply_lora_to_transformer

    apply_lora_to_transformer(model.transformer_model, rank=16, alpha=32.0)
    # Only LoRA adapter weights + head are trainable; everything else is frozen.
"""

import os
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a low-rank adapter.

    The original weight is frozen. Only lora_A and lora_B are trained.
    Output = W(x) + scale * lora_B(lora_A(x))
    """

    def __init__(self, linear: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.scale = alpha / rank

        in_features = linear.in_features
        out_features = linear.out_features

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.scale * self.lora_B(self.lora_A(x))

    def extra_repr(self) -> str:
        return (f"in={self.linear.in_features}, out={self.linear.out_features}, "
                f"rank={self.rank}, scale={self.scale:.3f}")


def apply_lora_to_transformer(transformer: nn.Module, rank: int = 16, alpha: float = 32.0) -> nn.Module:
    """Wrap attention projections with LoRA and freeze everything except LoRA + head.

    LoRA wrapped: intra_qkv_proj, intra_out_proj, inter_qkv_proj, inter_out_proj.
    Frozen: encoding_manager, LayerNorms, MultiHeadRMSNorms, FeedForward MLPs.
    Trainable: LoRA lora_A/lora_B matrices, output head.

    Args:
        transformer: AssemblyTransformer instance.
        rank: LoRA rank (typical: 8-64).
        alpha: LoRA scaling factor (typical: rank or 2*rank).

    Returns:
        The same transformer with LoRA adapters applied in-place.
    """
    # Wrap attention projections with LoRA
    for layer in transformer.transformer_layers:
        layer.intra_qkv_proj = LoRALinear(layer.intra_qkv_proj, rank, alpha)
        layer.intra_out_proj = LoRALinear(layer.intra_out_proj, rank, alpha)
        layer.inter_qkv_proj = LoRALinear(layer.inter_qkv_proj, rank, alpha)
        layer.inter_out_proj = LoRALinear(layer.inter_out_proj, rank, alpha)

    # Freeze everything ...
    for p in transformer.parameters():
        p.requires_grad = False

    # ... then selectively unfreeze LoRA adapters
    for layer in transformer.transformer_layers:
        for attr in ("intra_qkv_proj", "intra_out_proj", "inter_qkv_proj", "inter_out_proj"):
            wrapped = getattr(layer, attr)
            for p in wrapped.lora_A.parameters():
                p.requires_grad = True
            for p in wrapped.lora_B.parameters():
                p.requires_grad = True

    # ... and the output head
    for p in transformer.head.parameters():
        p.requires_grad = True

    if os.environ.get("LOCAL_RANK", "0") == "0":
        trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
        total = sum(p.numel() for p in transformer.parameters())
        print(f"[LoRA] Trainable params: {trainable:,} / {total:,} ({100.0 * trainable / total:.2f}%)")

    return transformer
