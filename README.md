# PENGWIN 2026 Task 3 - OBL Lab

Code release for OBL Lab's PENGWIN 2026 Task 3 pelvic fracture reduction method:
**Small-Fragment-Aware Multi-Candidate Assembly Transformer with Conservative Risk Gating**.

The method predicts a rigid `4 x 4` reduction matrix for every SA/LI/RI fracture fragment. It extends
the official Task 3 Assembly Transformer baseline with rotation-aware coordinate supervision,
anti-forgetting online hard-example mining, Area64 surface sampling, Small-Fragment Query branches,
heterogeneous full-pelvis candidates, a global candidate ranker, and conservative C2 rollback.

## Method at a glance

```text
fragment meshes
  -> Area64 surface sampling
  -> e25 paired-coordinate backbone + per-fragment Kabsch/Horn
  -> original four alpha candidates + e3 ranker/C2 safety pose
  -> SFQ-F/SFQ-FX exact-full candidate
  -> e26 B1/B2/B3 complementary candidates
  -> e27 full-pelvis candidate ranker
  -> conservative outer C2 and patient rollback
  -> per-fragment 4 x 4 matrices
```

The coordinate-backbone lineage is Clean e659 (660 epochs), rotation-aware e19 (20 epochs), and
anti-forgetting OHM e24 (25 epochs), for 705 sequential epochs. The original candidate-ranker lineage
is historical e119 (120 epochs), e25 strict recalibration (30 epochs), and low-LR continuation (4 epochs).
See [the full algorithm description](docs/ALGORITHM_DESCRIPTION_CN.md) for the exact training chain.

## Repository layout

- `inference/app/`: frozen Grand Challenge inference implementation.
- `training/core/`: shared training, data, and model implementation.
- `training/sfq/`: SFQ training implementation.
- `training/tools/`: Area64, calibration, cache, export, and e27 training tools.
- `configs/`: final backbone, SFQ, e26, and ranker configurations.
- `tests/`: deterministic geometry and policy tests.
- `docs/`: algorithm description, data layout, and exact training lineage.

## Data

No external dataset or externally pretrained model was used. Training uses the datasets supplied by
PENGWIN 2026. Challenge data are not redistributed here. Set `PENGWIN_ROOT` to a local workspace and
place the official data according to [docs/DATA.md](docs/DATA.md).

## Environment

Training was performed with PyTorch/Lightning on RTX 5090 GPUs. The frozen submission targets the
Grand Challenge T4 runtime. A minimal setup is:

```bash
conda create -n pengwin-task3 python=3.10 -y
conda activate pengwin-task3
pip install -r requirements-train.txt
```

FlashAttention is optional; the implementation falls back to standard PyTorch attention.

## Training

The published YAML files preserve the selected settings while replacing server-specific roots with
`${oc.env:PENGWIN_ROOT,.}`. Representative stages are:

```bash
export PENGWIN_ROOT=/path/to/PENGWIN2026
cd training/core
python train.py --config-name backbone/train_clean_baseline
python train.py --config-name backbone/train_rotation_aware_e659_v2
python train.py --config-name backbone/train_rotation_antiforget_ohm

# Recalibrate the original four-candidate ranker on e25.
python train.py --config-name ranker/train_e25_e2_strict_warmstart_5090
python train.py --config-name ranker/train_e29_continue_earlystop_5090

# Train the complementary e26 candidates.
python train.py --config-name e26/train_e26_b1_fragment_context
python train.py --config-name e26/train_e26_b2_reliability
python train.py --config-name e26/train_e26_b3_hard

# Train SFQ in its dedicated source tree.
cd ../sfq
python train.py --config-name sfq/train_sfq_screen model.variant=f
python train.py --config-name sfq/train_sfq_screen model.variant=fx
```

Later stages require the checkpoint named by the preceding stage. Full commands and selected epochs
are documented in [docs/TRAINING_LINEAGE.md](docs/TRAINING_LINEAGE.md).

The final e27 ranker is trained from cached four-candidate simulation rollouts:

```bash
python training/tools/build_e27_sim_pool_cache.py --help
python training/tools/train_e27_cached_ranker.py \
  --cache-root /path/to/cache \
  --init-checkpoint /path/to/e25_e3_ranker.ckpt \
  --output-dir /path/to/e27_run \
  --seed 42 --scope all --batch-size 64 --num-workers 5 \
  --lr 2e-5 --max-epochs 30 --patience 8 --min-delta 0.0005
```

The exact checkpoint paths are environment variables rather than server-specific paths. Set
`E25_CHECKPOINT` and the stage-specific checkpoint variables before launching each continuation.

## Inference and evaluation

`training/core/inference.py` and `evaluate.py` provide the research inference/evaluation path. The
frozen multi-candidate implementation is under `inference/app/`. Trained weights are not published;
readers should train the stages above and export tensor-only checkpoints with
`training/tools/export_inference_ckpt.py`.

## Frozen Clinical170 replay

| TRE (mm) | Trans (mm) | Rot (deg) | PA | CD (mm) |
|---:|---:|---:|---:|---:|
| 3.151657 | 3.857754 | 5.167839 | 0.812782 | 3.681537 |

Clinical170 was challenge-provided training data and was used for low-capacity calibration and
candidate ablation. This is a frozen training-domain replay, not hidden-test performance.

## Ablation summary

- Area64 protects small fragments from under-sampling while keeping a fixed per-bone point budget.
- Rotation-aware continuation improves rotation observability through centered vectors, long-baseline
  pairs, cross-covariance, and differentiable Horn/Kabsch losses.
- Anti-forgetting OHM retains natural-case performance better than unrestricted hard-only training.
- SFQ-F/FX improves candidate-pool coverage but requires conservative calibration to protect correct poses.
- e26 B1/B2/B3 are complementary proposals rather than direct replacements for e25.
- The final e27 ranker compares complete pelvis combinations; C2 accepts only sufficiently large,
  low-risk predicted improvements.

## Upstream and licensing

This work is derived from the
[PENGWIN2026 Task 3 Reduction Baseline](https://github.com/Sutuk/PENGWIN2026_Task3_Reduction_Baseline).
The upstream repository did not expose a license when this release was prepared. Accordingly, this
repository does not assert a new license over upstream-derived files. See [NOTICE.md](NOTICE.md).

## Authors

Zhengliang Li, Tianyun Gu, Nan Zheng, Chunjie Xia, Wanxian Yu, and Yangyang Yang.

Affiliations: School of Biomedical Engineering, Shanghai Jiao Tong University; School of Intelligent
Sports Engineering, Shanghai University of Sport.
