"""Regression test for multi-fragment candidate geometry on the active device."""

from __future__ import annotations

from functools import partial
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.AssemblyNet import FragmentConfidenceModule


class DummyTransformer(torch.nn.Module):
    embed_dim = 384


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer = DummyTransformer()
    module = FragmentConfidenceModule(
        transformer_model=transformer,
        optimizer=partial(torch.optim.AdamW, lr=2e-4),
        checkpoint=None,
        candidate_alphas=(0.0, 0.5, 1.0, 1.25),
        axis_offsets_deg=(),
        max_parts=50,
    ).to(device)

    num_parts = 12
    points_per_part = 5
    num_points = num_parts * points_per_part
    part_ids = torch.arange(num_parts, device=device).repeat_interleave(points_per_part)
    candidate_points = torch.randn(num_points, 4, 3, device=device)
    candidate_quat = torch.randn(num_parts, 4, 4, device=device)
    candidate_quat = torch.nn.functional.normalize(candidate_quat, dim=-1)
    candidate_trans = torch.randn(num_parts, 4, 3, device=device)
    pred_coords = torch.randn(num_points, 3, device=device)
    data = {
        "points_per_part": torch.tensor(
            [[points_per_part] * num_parts + [0] * (50 - num_parts)], device=device
        ),
        "norm_scale": torch.tensor([100.0], device=device),
        "bonetype": torch.tensor(
            [[index % 3 for index in range(num_parts)] + [0] * (50 - num_parts)],
            device=device,
        ),
        "fragment_diameter_mm": torch.tensor(
            [[30.0] * num_parts + [0.0] * (50 - num_parts)], device=device
        ),
    }
    part_to_case = torch.zeros(num_parts, dtype=torch.long, device=device)

    geometry, bone = module._geometry_features(
        data,
        candidate_points,
        candidate_quat,
        candidate_trans,
        part_ids,
        pred_coords,
        part_to_case,
    )
    assert geometry.shape == (num_parts, 4, 27), geometry.shape
    assert bone.shape == (num_parts,), bone.shape
    assert torch.isfinite(geometry).all()
    print(f"fragment_geometry_regression=PASS device={device} shape={tuple(geometry.shape)}")


if __name__ == "__main__":
    main()
