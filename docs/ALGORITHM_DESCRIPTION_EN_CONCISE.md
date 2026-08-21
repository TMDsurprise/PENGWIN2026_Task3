# [Algorithm Description] Task 3 Team OBL Lab

> This description corresponds to our final frozen submission, OBL Reduction V4.

## 1. Task

Task 3: PENGWIN-Reduction (pelvic fracture fragment reduction).

## 2. Team name

OBL Lab

## 3. Authors

Zhengliang Li, Tianyun Gu, Nan Zheng, Chunjie Xia, Wanxin Yu, and Yangyang Yang

## 4. Affiliations

1. School of Biomedical Engineering, Shanghai Jiao Tong University, Shanghai, China
2. School of Intelligent Sports Engineering, Shanghai University of Sport, Shanghai, China

Zhengliang Li, Tianyun Gu, Nan Zheng, Chunjie Xia, and Wanxin Yu are affiliated with institution 1.
Yangyang Yang is affiliated with institution 2.

## 5. Contact author and email address

Zhengliang Li, lizhengliang@sjtu.edu.cn

## 6. Algorithm name or title

**OBL Reduction V4: Small-Fragment-Aware Multi-Candidate Assembly Transformer with Conservative Risk Gating**

## 7. Method description

We formulate pelvic reduction as paired-coordinate regression followed by per-fragment rigid-pose
recovery. Given the SA, LI, and RI fracture-fragment meshes of one patient, an Assembly Transformer
jointly processes all sampled points, surface normals, fragment identifiers, and bone types. Global
point-level attention predicts a paired coordinate in the reduced pelvis for every input point. An
SVD-based Kabsch/Horn solver then converts each fragment's source-to-target correspondences into a
rotation and translation. Pose updates are iterated for at most 10 steps and terminate early when the
coordinate update is below 2 mm.

The final e25 coordinate backbone was trained sequentially in three stages: a Clean Baseline trained
from scratch for 660 epochs; a 20-epoch RA continuation with centered-vector, long-baseline-pair,
cross-covariance, and differentiable Horn/Kabsch supervision; and a 25-epoch anti-forgetting OHM
continuation using hard-example pools, replay, and selective rotation-teacher consistency. The final
backbone therefore has a genuine 705-epoch weight lineage.

The post-backbone system generates multiple complete-pelvis hypotheses. An original four-candidate
ranker provides a stable safety pose. SFQ-F uses equal-weight fragment tokens and fragment-level
self-attention, while SFQ-FX additionally lets each fragment query complete-pelvis patch memory.
Both branches predict zero-initialized, gated `SE(3)` residual corrections and are accepted only by a
calibrated internal C2 gate. Three complementary e26 models provide additional complete-pelvis
candidates: B1 adds equal-fragment context, B2 adds point-reliability supervision, and B3 adds
small-fragment hard sampling, replay, and OHM.

The final e27 ranker jointly compares candidate 0 = the SFQ-calibrated exact-full safety candidate,
candidate 1 = B1, candidate 2 = B2, and candidate 3 = B3. It uses fragment features, candidate
geometry, bone/candidate/fragment embeddings, and a context Transformer to predict candidate scores,
TRE/Trans/Rot/CD proxies, and severe-failure risk. A fixed outer C2 gate and patient-level rollback
retain candidate 0 whenever the predicted gain is insufficient or risk increases.

## 8. Main technical contributions and/or novel components

1. **Area64 sampling:** allocates each bone's point budget by surface area while retaining at least 64
   points per fragment.
2. **Rotation-aware paired-coordinate learning:** sends centered-vector, long-baseline,
   cross-covariance, and differentiable rigid-pose gradients directly into the coordinate head.
3. **Anti-forgetting OHM:** stratifies hard examples by bone, fragment size, and failure mode while
   preserving natural-case performance through replay and teacher consistency.
4. **Small-Fragment Query:** gives each fragment an equal-weight token and complete-pelvis query
   capability, followed by a conservative rigid residual.
5. **Heterogeneous complete-pelvis proposals:** combines exact-full, fragment-context,
   point-reliability, and hard-replay candidates.
6. **Complete-pelvis ranking and two-level risk gating:** ranks whole SA/LI/RI assemblies and retains
   explicit fragment- and patient-level fallbacks.

## 9. Complete pipeline

1. Parse SA/LI/RI meshes, fragment IDs, points, and surface normals from the input OBJ files.
2. Apply Area64 sampling: 5,000 points per bone, distributed by surface area with at least 64 points
   per fragment.
3. Center and isotropically normalize the complete case.
4. Predict paired reduced coordinates with e25 and recover iterative fragment poses using Kabsch/Horn.
5. Obtain the original ranker/C2 safety pose and SFQ-F/SFQ-FX correction candidates.
6. Use calibrated internal C2 to form exact-full candidate 0.
7. Independently generate e26 B1, B2, and B3 candidates 1-3.
8. Map all complete-pelvis candidates into one canonical state and rank them with e27.
9. Apply outer C2 and patient-level rollback.
10. Anchor-normalize using SA fragment 1 and output one `4 x 4` matrix per fragment in JSON format.

## 10. Use of external data

No external dataset was used. We used only PENGWIN 2026-provided simulation and clinical training
data; the hidden test set and labels were never accessed.

## 11. Use of externally pretrained models

No externally pretrained model or external weight was used. All checkpoints were trained by our team
using challenge-provided simulation data.

## 12. Preprocessing techniques

We parse bone and fragment identifiers from OBJ meshes, sample surface points and normals, apply
Area64 allocation, center each complete case, and perform isotropic scale normalization. Fragment
diameter, centroid, covariance, count, and bone-type metadata are retained. NumPy and PyTorch seeds are
fixed to 42 per case, and the final matrices are normalized using SA fragment 1 as the rigid anchor.

## 13. Data augmentation techniques

Training uses online simulated fragment combinations and random `SE(3)` transformations, random
anchors, fragment merging, variable fracture patterns and fragment counts, point dropping, region
dropout, surface resampling, and coordinate jitter. OHM additionally uses bone-, size-, and
failure-mode-stratified sampling, small-fragment hard cases, and Clean/e25 replay.

## 14. Training and validation strategy

The submitted coordinate backbone follows Clean e659 (660 epochs) -> RA e19 (20 epochs) -> OHM e24
(25 epochs), trained using two RTX 5090 GPUs with DDP and mixed precision. The original ranker was
initialized from a 120-epoch historical simulation ranker, strictly recalibrated for 30 epochs with
the e25 backbone frozen, and continued for four low-LR epochs, giving a 154-epoch Ranker parameter
lineage. Each SFQ branch used a three-epoch
screen followed by a nine-epoch continuation. B1, B2, and B3 ran for 3, 3, and 8 epochs, with e1, e2,
and e6 selected, respectively. These post-backbone modules were independent experiments and their
epochs are not added to the 705 backbone epochs.

The e27 ranker was trained on 8,617 simulation training samples and 958 patient-held-out validation
samples, with up to 10 cached rollout iterations from each candidate source. Validation was performed
every epoch, the maximum was 30 epochs, and early-stopping patience was eight. The final model used
`LR=2e-5`, seed 42, and `epoch=11`, i.e., the 12th epoch rather than the 30-epoch maximum.

Clinical calibration used patient-level five-fold OOF predictions. A patient's OOF prediction was
always generated without fitting or gate selection on that patient. The official-scale frozen replay
on 170 challenge-provided clinical training cases (1,014 fragments) was:

| TRE (mm) | Trans (mm) | Rot (deg) | PA | CD (mm) |
|---:|---:|---:|---:|---:|
| 3.151657 | 3.857754 | 5.167839 | 0.812782 | 3.681537 |

Because Clinical170 was used for low-capacity calibration and candidate ablation, these values are a
training-domain replay and are not hidden-test performance.

## 15. Loss function(s)

The coordinate backbone uses:

```text
L_coord = 0.75 * L_global_point_MSE + 0.25 * L_fragment_balanced_MSE

L_RA = L_coord + s(epoch) * (
    0.010 * L_centered_vector + 0.004 * L_long_baseline_pair
  + 0.005 * L_cross_covariance + 0.010 * L_Horn_rotation
  + 0.0015 * L_Horn_translation)
```

OHM retains the rotation auxiliaries and mixes `L_coord` with difficulty-weighted hard loss using a
maximum coefficient of 0.30. Selective teacher consistency is applied every four steps with weight
0.020.

SFQ uses `2.0 * rotation_chordal + 0.5 * translation_SmoothL1 + 0.5 * paired_coordinate_SmoothL1 +
0.2 * preserve + 0.02 * residual_regularization`, with increased weight for small fragments. The
submitted F/FX variants did not enable the optional confidence heads.

The candidate ranker uses `1.0 * hard_oracle_CE + 0.5 * pairwise_ranking + 0.25 * metric_regression +
0.2 * severe_BCE`. Metric regression predicts log-transformed TRE/Trans/Rot/CD proxies. PA is not part
of the learned utility. A severe target is defined as `Rot > 30 degrees` or `Trans > 20 mm`.

## 16. Base network architecture

The coordinate backbone is a 12-layer, 384-dimensional, eight-head Assembly Transformer followed by
per-fragment SVD Kabsch/Horn. SFQ uses 384-dimensional fragment tokens, a two-layer eight-head
fragment Transformer, optional pelvis-patch cross-attention, and a continuous 6D-rotation plus
3D-translation residual head. The e27 ranker uses a 27-dimensional candidate-geometry descriptor,
bone/candidate/fragment embeddings, a three-layer hidden-dimension-256 eight-head context
Transformer, and score, metric, and severe-risk heads.

## 17. Ensembling strategies used during inference

We do not average model weights or pose matrices. The system is a candidate-selection ensemble over
exact-full, B1, B2, and B3 complete-pelvis predictions. Candidate 0 is the explicit safety fallback.
The internal C2 uses `margin=0.15`; the outer C2 uses `margin=0.05`, zero severe-risk tolerance, and a
severe-probability threshold of 0.2, followed by patient-level rollback.

## 18. Public code repository

<https://github.com/TMDsurprise/PENGWIN2026_Task3>

The source code is publicly available at the repository above. Trained weights and
challenge-provided data are not redistributed.

## 19. Relevant references

1. Sutuk. *PENGWIN2026 Task 3 Reduction Baseline*.
   https://github.com/Sutuk/PENGWIN2026_Task3_Reduction_Baseline
