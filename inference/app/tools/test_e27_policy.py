"""Regression tests for the frozen e27 final selector and Area64 sampler."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import process


def test_original_c2() -> None:
    scores = np.asarray([[0.0, 3.0, 1.0, 0.0], [0.0, 1.0, 4.0, 0.0]])
    metrics = np.ones((2, 4, 4), dtype=np.float64) * 10.0
    metrics[0, 1] = 1.0
    metrics[1, 2] = 1.0
    severe = np.ones((2, 4), dtype=np.float64) * 0.1
    decision = process.apply_e27_original_c2(
        {"scores": scores, "metrics": metrics, "severe": severe}
    )
    assert decision["raw"].tolist() == [1, 2]
    assert decision["selected"].tolist() == [1, 2]
    assert decision["patient_accepted"]

    severe[1, 2] = 0.9
    decision = process.apply_e27_original_c2(
        {"scores": scores, "metrics": metrics, "severe": severe}
    )
    assert decision["selected"].tolist() == [1, 0]


def test_area64() -> None:
    first = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    second = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    meshdict = {
        "SA": {
            "1": {"mesh": first},
            "2": {"mesh": second},
        }
    }
    np.random.seed(42)
    sampled = process._area64_sub_sample(meshdict)
    counts = [len(sampled["SA"][key]["coords"]) for key in ("1", "2")]
    assert sum(counts) == process.NPOINTS, counts
    assert min(counts) >= process.MIN_POINTS, counts
    assert all(np.isfinite(sampled["SA"][key]["coords"]).all() for key in ("1", "2"))


def test_anchor_normalization() -> None:
    transforms = np.repeat(np.eye(4)[None], 3, axis=0)
    transforms[0, :3, 3] = [4.0, -2.0, 1.0]
    transforms[1, :3, 3] = [5.0, 3.0, 1.0]
    normalized = process.anchor_normalize(transforms, 0)
    assert np.allclose(normalized[0], np.eye(4))
    assert np.allclose(normalized[1, :3, 3], [1.0, 5.0, 0.0])


def main() -> None:
    test_original_c2()
    test_area64()
    test_anchor_normalization()
    print("e27_policy_regression=PASS")


if __name__ == "__main__":
    main()
