"""Export a Lightning checkpoint to a minimal inference-only checkpoint.

Usage:
    python tools/export_inference_ckpt.py checkpoints/assemble_coord_baseline_854.ckpt
    python tools/export_inference_ckpt.py checkpoints/assemble_coord_baseline_854.ckpt --fp16
"""

import argparse
import json
import os
import torch


def main():
    parser = argparse.ArgumentParser(description="Export inference checkpoint")
    parser.add_argument("input", help="Path to Lightning .ckpt file")
    parser.add_argument("--fp16", action="store_true", help="Also cast to FP16")
    parser.add_argument("-o", "--output", help="Output path (default: input_stem_infer.pt)")
    args = parser.parse_args()

    input_size = os.path.getsize(args.input) / 1e6

    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt["model_state_dict"]

    # Strip Lightning wrapper prefix + filter internal keys
    skipped = []
    cleaned = {}
    for k, v in state_dict.items():
        if any(k.startswith(p) for p in ("optimizer", "lr_schedul", "callbacks", "loops")):
            skipped.append(k)
            continue
        if k.startswith("transformer_model."):
            k = k[len("transformer_model."):]
        if args.fp16 and v.dtype == torch.float32:
            v = v.half()
        cleaned[k] = v

    if not args.output:
        stem = os.path.splitext(os.path.basename(args.input))[0]
        suffix = "_infer_fp16" if args.fp16 else "_infer"
        args.output = os.path.join(os.path.dirname(args.input), f"{stem}{suffix}.pt")

    # Infer model config from state_dict shapes
    config = _infer_config(cleaned)

    torch.save({"model_state_dict": cleaned, "config": config}, args.output)
    output_size = os.path.getsize(args.output) / 1e6

    # Also write a standalone config.json for easy inspection
    config_path = os.path.splitext(args.output)[0] + ".json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Input:  {args.input}  ({input_size:.0f} MB, {len(state_dict)} keys)")
    print(f"Output: {args.output}  ({output_size:.0f} MB, {len(cleaned)} keys)")
    print(f"Config: {config_path}")
    print(f"Ratio:  {output_size / input_size * 100:.0f}%")
    if skipped:
        print(f"Skipped: {skipped}")


def _infer_config(state_dict):
    """Infer model architecture from state_dict tensor shapes."""
    # Embed dim from encoding or first transformer layer
    embed_dim = None
    for k, v in state_dict.items():
        if "attn.qkv.weight" in k or "attn.out_proj.weight" in k:
            embed_dim = v.shape[0]
            break
    if embed_dim is None:
        embed_dim = 384  # fallback

    # Number of layers from transformer layer count
    num_layers = 0
    for k in state_dict:
        if k.startswith("transformer_layers.") and k.endswith(".attn.qkv.weight"):
            num_layers += 1
    if num_layers == 0:
        num_layers = 12  # fallback

    # Output type from head
    output_type = "coords"
    head_weight = state_dict.get("head.2.weight")
    if head_weight is not None:
        if head_weight.shape[0] == 7:
            output_type = "pose"
        elif head_weight.shape[0] == 3:
            output_type = "coords"

    # Number of heads from qkv weight
    num_heads = 8
    for k, v in state_dict.items():
        if "attn.qkv.weight" in k:
            qkv_dim = v.shape[0]
            # qkv_dim = num_heads * (head_dim * 3)
            # Usually head_dim = embed_dim // num_heads
            # qkv_dim = num_heads * (embed_dim // num_heads * 3) = embed_dim * 3
            # So num_heads can't be inferred from qkv alone...
            # Try to infer from per-head dimension
            break

    # Check for LoRA adapters
    has_lora = any("lora" in k.lower() for k in state_dict)
    lora_rank = 32
    for k, v in state_dict.items():
        if "lora_A" in k:
            lora_rank = v.shape[0]  # rank is first dim of lora_A
            break

    # Max parts from encoding manager
    max_parts = 50
    for k, v in state_dict.items():
        if "part_id_embedding" in k or "encoding_manager" in k:
            if len(v.shape) >= 1:
                # e.g., part embedding table
                pass
        if "frag_embed" in k and len(v.shape) == 2:
            max_parts = v.shape[0]

    return {
        "output_type": output_type,
        "embed_dim": embed_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "max_parts": max_parts,
        "training_mode": "lora" if has_lora else "full",
        "lora_rank": lora_rank if has_lora else 32,
        "lora_alpha": 64.0 if has_lora else 32.0,
    }


if __name__ == "__main__":
    main()
