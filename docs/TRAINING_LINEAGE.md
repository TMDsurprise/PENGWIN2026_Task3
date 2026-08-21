# Training lineage

Checkpoint epoch labels are zero-based: `epoch=24` means 25 completed epochs.

## Coordinate backbone

| Stage | Initializer | Selected checkpoint | Effective epochs | Hardware |
|---|---|---:|---:|---|
| Clean Baseline | random | e659 | 660 | 2x RTX 5090 DDP |
| Rotation-aware | Clean e659 | e19 | 20 | 2x RTX 5090 DDP |
| Anti-forgetting OHM | RA e19 | e24 | 25 | 2x RTX 5090 DDP |

The final e25 coordinate backbone therefore inherits 705 sequential epochs.

## Original four-candidate ranker

| Stage | Selected checkpoint | Effective epochs | Hardware |
|---|---:|---:|---|
| Historical E2 ranker | e119 | 120 | RTX 5090 training |
| e25 strict recalibration | e29 | 30 | 2x RTX 5090 DDP |
| Low-LR continuation | e3 | 4 | 2x RTX 5090 DDP |

Only ranker tensors were imported from e119; all historical coordinate-backbone tensors were excluded.
The final e3 ranker has a 154-epoch optimization lineage.

## Small-fragment and outer candidates

| Component | Run | Selected | Hardware |
|---|---:|---:|---|
| SFQ-F | 3-epoch screen + 9-epoch continuation | final seed42 | one 5090 per parallel job |
| SFQ-FX | 3 + 9 epochs | final seed42 | one 5090 per parallel job |
| e26 B1 | 3 epochs | e1 | one 5090 |
| e26 B2 | 3 epochs | e2 | one 5090 |
| e26 B3 | 8 epochs | e6 | one 5090 |
| e27 full ranker | max 30, patience 8 | e11 (12th epoch) | one 5090 per parallel run |

For each submitted SFQ branch, the selected 3-epoch screen checkpoint was exported as raw,
tensor-only weights and used to initialize `configs/sfq/train_sfq_continuation9.yaml`. The
continuation used seed 42, learning rate `2e-5`, nine epochs, and one RTX 5090 per F/FX job.

Area64, Huber-ridge full calibration, inner C2, and outer C2 do not have neural-network epochs. SFQ,
e26, and e27 are independent post-backbone modules and must not be added to the 705 backbone epochs as
if the complete system were trained end-to-end.
