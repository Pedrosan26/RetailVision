# Age/gender results — asymmetric 6-bin scheme (RET-31)

## Setup

Both the baseline and fine-tuned classifiers were retrained from scratch
against the asymmetric 6-bin age scheme (`0-5`, `6-12`, `13-17`, `18-40`,
`41-64`, `65+`) adopted after RV-008's findings — see
`docs/datasets/utkface.md` ("Why these bins") and
`docs/models/age_rebinning_investigation.md` for the reasoning: keep every
bin that RV-008 proved reliable exactly as fine as it was, and merge the
entire adult range that plateaued regardless of tuning into one bucket.

Same methodology as RV-004/RV-005 throughout — baseline uses Ultralytics
defaults (100 epochs, `imgsz=224`), fine-tune retrains from `yolov8n-cls.pt`
with augmentation and adjusted hyperparameters (60 epochs, batch 32,
`optimizer="SGD"`, `lr0=0.005`, `patience=15`). Gender is unaffected by any
of this (bins are an age-only concept) and is included only because both
scripts always train/evaluate both tasks together.

Both stages ran back-to-back overnight; total wall time ~11.6h.

## Results vs. every scheme tried so far

| Scheme | Age accuracy | Gender accuracy | Age threshold (75%) |
|---|---|---|---|
| RV-005 — uniform 4-bin | 84.87% | 96.40% | PASS |
| RV-008 — uniform 10-bin (abandoned) | 69.36% | 96.81% | FAIL |
| **RET-31 — asymmetric 6-bin** | **88.77%** | **96.34%** | **PASS** |

The 6-bin scheme beats the original 4-bin classifier outright — not just
avoiding the 10-bin's collapse, but improving on the scheme that never had
a granularity problem to begin with. Gender is flat across all three
schemes, as expected (bins are an age-only concept).

## Per-class F1, baseline → fine-tuned

| Age bin | Baseline F1 | Fine-tuned F1 | Support |
|---|---|---|---|
| 0-5 | 0.969 | 0.971 | 701 |
| 6-12 | 0.837 | 0.866 | 300 |
| 13-17 | 0.749 | 0.779 | 223 |
| 18-40 | 0.920 | 0.919 | 2,346 |
| 41-64 | 0.785 | 0.790 | 992 |
| 65+ | 0.843 | 0.872 | 460 |

Every class lands at F1 ≥ 0.78 — there is no weak class left. For
comparison, the 10-bin scheme's `26-32` and `33-40` (which fed into what is
now `18-40`) were stuck at F1 0.51-0.63 and refused to move under
fine-tuning; merged, that same age range is now the **strongest** class at
0.919 — expected, since it's both the easiest distinction the model has to
make ("not a child, not older") and by far the most data (2,346 of 5,022
test images, ~47% of the set). `41-64` (0.790) also clears every individual
bin that fed into it in the 10-bin attempt (41-50: 0.56, 51-60: 0.65,
61-70: 0.60).

Gender: Female F1 0.958 → 0.963, Male F1 0.959 → 0.964 — consistent with
every prior run, unaffected by the age rebinning.

## Training health — no overfitting reversal this time

**Age (fine-tuned)**: val loss bottoms around epoch 32-33 (0.377) and ticks
up slightly afterward (to ~0.42 by epoch 60) — but unlike every previous
age model (RV-004, RV-005, and RV-008's baseline/fine-tune), **validation
accuracy kept climbing anyway**, from 88.6% at epoch 41 to 88.95% by
epoch 51, rather than reversing once val loss passed its minimum. The
baseline (100-epoch run) shows the same pattern: val loss bottoms ~epoch 70
(0.368), accuracy still climbs to 87.98% by epoch 100.

**Gender (fine-tuned)**: val loss flat and low from ~epoch 30 onward
(~0.127-0.130), accuracy still inching up at the end (96.3%+) — consistent
with every prior gender run, no overfitting concern.

The healthier curves likely reflect the coarser, more separable classes
(6-bin vs. 10-bin) giving the model an easier optimization landscape overall
— less capacity spent trying to draw boundaries the data doesn't support.

## Artifacts

- Weights: `models/age_gender/baseline_age.pt`, `baseline_gender.pt`,
  `final_age.pt`, `final_gender.pt`
- Full metrics: `models/age_gender/baseline_report.json`,
  `final_report.json`
- Loss curves: `models/age_gender/{baseline,final}_{age,gender}_loss_curves.png`
- Raw per-epoch logs: `runs/age_gender_{baseline,finetune}/{age,gender}/results.csv`

`final_age.pt` / `final_gender.pt` are the current production classifiers
on this branch, used for analytics-style demographic buckets. The
continuous age-regression model (RET-31, separate pipeline) covers the
finer-grained live-display estimate.