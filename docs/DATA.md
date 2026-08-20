# Data layout

Download the official PENGWIN 2026 Task 3 data under the challenge's access and usage terms. Data are
not included in this repository.

Set:

```bash
export PENGWIN_ROOT=/path/to/PENGWIN2026
```

The released configurations expect the simulation point data at:

```text
${PENGWIN_ROOT}/Dataset/PENGWIN26_task3_simulation_fractures_train/points
```

Clinical mesh paths should be supplied explicitly for evaluation. Do not commit patient data, generated
point caches, candidate traces, predictions, or ground-truth matrices to Git.

Patient-level train/validation splits must be used. A fragment from one patient must never be placed in
a different split from the remaining fragments of that patient.
