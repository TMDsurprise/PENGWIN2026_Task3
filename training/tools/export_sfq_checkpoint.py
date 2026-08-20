#!/usr/bin/env python3
"""Export tensor-only raw or materialized-EMA SFQ checkpoints for inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_ema_shadow(checkpoint: dict) -> dict[str, torch.Tensor]:
    matches = []
    for callback_name, callback_state in checkpoint.get("callbacks", {}).items():
        if isinstance(callback_state, dict) and isinstance(callback_state.get("shadow"), dict):
            matches.append((str(callback_name), callback_state["shadow"]))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one EMA shadow, found {len(matches)}")
    return matches[0][1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("raw", "ema"), required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    source_state = checkpoint.get("state_dict", checkpoint)
    state = {
        name: value.detach().cpu().clone()
        for name, value in source_state.items()
        if torch.is_tensor(value)
    }
    replaced = []
    if args.mode == "ema":
        shadow = find_ema_shadow(checkpoint)
        missing = sorted(set(shadow) - set(state))
        if missing:
            raise RuntimeError(f"EMA tensors missing from state_dict: {missing[:10]}")
        for name, value in shadow.items():
            if state[name].shape != value.shape:
                raise RuntimeError(
                    f"EMA shape mismatch for {name}: {tuple(value.shape)} vs "
                    f"{tuple(state[name].shape)}"
                )
            state[name] = value.detach().cpu().to(dtype=state[name].dtype).clone()
            replaced.append(name)

    payload = {
        "state_dict": state,
        "epoch": int(checkpoint.get("epoch", -1)),
        "global_step": int(checkpoint.get("global_step", -1)),
        "export_mode": args.mode,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    verified = torch.load(args.output, map_location="cpu", weights_only=True)
    if verified["state_dict"].keys() != state.keys():
        raise RuntimeError("Tensor-only checkpoint round-trip changed state keys")
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "mode": args.mode,
        "epoch": payload["epoch"],
        "global_step": payload["global_step"],
        "state_tensors": len(state),
        "ema_tensors_replaced": len(replaced),
        "sha256": sha256(args.output),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
