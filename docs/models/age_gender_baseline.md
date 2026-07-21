# Age/gender baseline classifiers (RV-004)

## Setup

Two independent `yolov8n-cls` models trained from scratch on UTKFace
(`data/utkface/processed/age/`, `data/utkface/processed/gender/`), using
Ultralytics default hyperparameters — 100 epochs, `imgsz=224` (the
classification-resolution default; 640 is a detection-mode default and
doesn't apply here), `lr0=0.01`, seed 42, on Apple Silicon MPS. See
`scripts/age_gender_baseline/constants.py` for the exact values and
`docs/datasets/utkface.md` for why age/gender are trained as two separate
classifiers instead of one multi-head model.

Total wall time: ~3.9h (age) + ~5.25h (gender) ≈ 9.1h combined.

YOLOv8 classification mode reports top1/top5 accuracy rather than mAP@0.5
(a detection-mode metric); per-class precision/recall/F1 are computed
separately via scikit-learn on the held-out test split, per
`scripts/age_gender_baseline/evaluate.py`. Full numbers in
`models/age_gender/baseline_report.json`.

## Gender — strong baseline, not overfit

| Metric | Value |
|---|---|
| top1 accuracy | 95.78% |
| top5 accuracy | 100% |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Female | 0.953 | 0.963 | 0.958 | 2,508 |
| Male | 0.963 | 0.953 | 0.958 | 2,516 |

Balanced across both classes, consistent with the ~50/50 split documented
in `docs/datasets/utkface.md`. Val loss decreases monotonically across all
100 epochs (0.384 → 0.139) and val top1 accuracy climbs the entire run
(83.6% → 95.8%) — no sign of overfitting; the model was likely still
improving at epoch 100.

## Age — mild overfitting after ~epoch 70-80, weak on middle brackets

| Metric | Value |
|---|---|
| top1 accuracy | 84.43% |
| top5 accuracy | 100% |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0-17 | 0.974 | 0.949 | 0.962 | 1,225 |
| 18-30 | 0.806 | 0.838 | 0.822 | 1,544 |
| 31-50 | 0.720 | 0.715 | 0.718 | 1,222 |
| 51+ | 0.901 | 0.882 | 0.891 | 1,033 |

The two extreme brackets (`0-17`, `51+`) are strong; the two middle
brackets (`18-30`, `31-50`) are the weak point, consistent with age being a
continuum rather than hard categories — the boundary between "young adult"
and "middle-aged" is the most visually ambiguous split in the label set.

Train loss falls monotonically for the full run (1.00 → 0.185), but val
loss bottoms out around epoch 61-71 (~0.424) and rises back to 0.445 by
epoch 100; val top1 accuracy peaks at epoch 91 (83.75%) and ticks down
slightly by epoch 100 (83.59%). This is a mild overfitting signature — the
model kept fitting the training set past ~epoch 70 without further
generalization gains.

## Implications for RV-005 (fine-tuning)

- **Gender**: near production-ready as-is; low priority for RV-005 effort.
- **Age**: the model to focus on. Candidates for RV-005:
  - Earlier stopping (`patience` ~20-30 instead of 100), or restore the
    checkpoint around epoch 70-90 rather than the final epoch.
  - Target the `18-30`/`31-50` confusion specifically — stronger
    augmentation, class-weighted loss, or reconsidering whether that
    boundary should be redrawn.
  - Per `docs/datasets/utkface.md`, still owed: per-race breakdown on the
    test set, since race is unbalanced (White ~5.5x overrepresented) and
    aggregate accuracy can hide group-level gaps.

## Artifacts

- Weights: `models/age_gender/baseline_age.pt`, `models/age_gender/baseline_gender.pt`
  (gitignored, like all `*.pt`).
- Full metrics: `models/age_gender/baseline_report.json`
- Loss/accuracy curves: `models/age_gender/baseline_age_loss_curves.png`,
  `models/age_gender/baseline_gender_loss_curves.png`
- Raw per-epoch logs: `runs/age_gender_baseline/{age,gender}/results.csv`

Not yet integrated into the live pipeline (`pipeline_demo.py`) — these are
standalone classifiers, wiring them into the detection loop is a separate,
later task.