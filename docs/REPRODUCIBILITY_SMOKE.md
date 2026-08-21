# Reproducibility smoke test

Date: 2026-08-21

The public tree was exercised on one NVIDIA RTX A6000 using challenge-provided simulation data. Each
training smoke used one training batch, one validation batch, batch size one, and one epoch. These
runs verify code, tensor, gradient, checkpoint, and stage-to-stage interfaces; their metrics are not
model-quality measurements.

## Passed stages

1. Clean coordinate backbone training and validation.
2. Rotation-aware continuation from the Clean Lightning checkpoint.
3. Anti-forgetting OHM continuation and hard-pool JSON generation.
4. SFQ-F screen, tensor-only export, and continuation.
5. e26 B1, B2, and B3 sequential continuation, including B3 hard replay.
6. Original four-candidate ranker warm-start and low-learning-rate continuation.
7. e27 four-backbone train/validation cache construction.
8. e27 cached ranker training and checkpoint export.
9. Area64/C2 policy and candidate-geometry regression tests on CUDA.

No NaN, CUDA out-of-memory error, or abnormal process exit occurred in the successful runs. The audit
also found and fixed three portability issues: machine-specific dataset/checkpoint paths, PyTorch 2.1
tuple-dimension compatibility, and loading native Lightning checkpoints between training stages.

## Scope

The smoke test proves that the published pipeline is executable. It does not certify convergence or
reproduce the frozen Clinical170 metrics from one batch. Exact numerical reproduction additionally
requires the challenge data, the documented training duration and hardware, deterministic data
preparation, and the historical ranker initialization used by the submitted model. A random-init
ranker configuration is provided for a fully weight-free executable route.
