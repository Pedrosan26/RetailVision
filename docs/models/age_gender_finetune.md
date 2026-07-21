# Age/gender fine-tuned classifiers (RV-005)

## Setup

One additional training iteration on top of the RV-004 baseline
(`docs/models/age_gender_baseline.md`), retraining from the original
`yolov8n-cls.pt` checkpoint rather than continuing from the baseline
weights, to avoid compounding the mild overfitting already present in
`baseline_age.pt`. See `scripts/age_gender_finetune/constants.py` for exact
values.

Adjusted from baseline defaults:

| Hyperparameter | Baseline | Fine-tune |
|---|---|---|
| epochs | 100 | 60 |
| batch | 16 | 32 |
| optimizer | auto (AdamW) | SGD |
| lr0 | 0.01 (auto-overridden) | 0.005 |
| patience | 100 | 15 |

Augmentation enabled (none was used in the baseline, which needed to be a
faithful "defaults" floor): `fliplr=0.5` (horizontal flip), `hsv_s=0.3` /
`hsv_v=0.2` (brightness/contrast), `translate=0.1` / `scale=0.2` (random
crop/zoom jitter), `degrees=10`, `erasing=0.2`. Strategy and rationale
decided in `docs/datasets/utkface.md` (RV-003).

**Note on a bug caught during smoke-testing**: Ultralytics' default
`optimizer="auto"` silently ignores any explicitly-passed `lr0` and picks
its own (it logged `ignoring 'lr0=0.005' ... determining best 'optimizer',
'lr0' ... automatically`). `optimizer="SGD"` had to be set explicitly for
the tuned learning rate to actually take effect — confirmed via a 1-epoch
smoke test showing `optimizer: SGD(lr=0.005, momentum=0.937)` in the log
before the real run was launched.

Both classifiers ran the full 60 epochs — `patience=15` did not trigger
early stopping, meaning validation metrics kept improving or plateauing
within the patience window throughout.

## Results vs. RV-005 thresholds (top1 accuracy, held-out test split)

| Task | Threshold | Result | Status |
|---|---|---|---|
| Age | 75% | 84.87% | PASS |
| Gender | 85% | 96.40% | PASS |

Both tasks clear their required threshold. Full metrics in
`models/age_gender/final_report.json`.

## Gender — genuine improvement, no overfitting

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Female | 0.963 | 0.965 | 0.964 | 2,508 |
| Male | 0.965 | 0.963 | 0.964 | 2,516 |

top1 improved from 95.78% (baseline) to 96.40%. Val loss decreases
monotonically across all 60 epochs (0.400 → 0.138) with no rise at the
end — augmentation and the tuned hyperparameters clearly helped here
without introducing overfitting.

## Age — passes the threshold but did not meaningfully improve

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0-17 | 0.973 | 0.953 | 0.963 | 1,225 |
| 18-30 | 0.813 | 0.822 | 0.817 | 1,544 |
| 31-50 | 0.720 | 0.744 | 0.732 | 1,222 |
| 51+ | 0.918 | 0.890 | 0.904 | 1,033 |

top1 is essentially flat vs. baseline (84.87% vs. 84.43%), and per-class F1
for the confusable middle brackets is unchanged (`18-30`: 0.822 → 0.817,
`31-50`: 0.718 → 0.732). The augmentation and hyperparameter changes did
not fix the `18-30`/`31-50` confusion identified in RV-004.

More notably, the overfitting signature got **worse in timing**, not
better: val loss now bottoms out at epoch 22 (0.4247) and rises steadily
to 0.4834 by epoch 60, versus the baseline's val loss bottoming around
epoch 61-71. Train loss keeps falling the whole run (1.065 → 0.116) while
val loss climbs from epoch ~22 onward. The larger batch size and lower,
faster-decaying LR schedule likely caused the model to reach its best
generalization point earlier in training rather than later.

**Decision**: accepted as-is rather than spending a second iteration,
since the ticket's bar is clearing 75% (done) and running one additional
iteration with augmentation/tuned hyperparameters (done) — not requiring
strict improvement over the baseline. Documenting this as an explicit
finding rather than treating it as a hidden regression: the age model is
not meaningfully better than the RV-004 baseline, and if further work on
age classification is picked up later, the fix should target the
`18-30`/`31-50` boundary specifically (data quality, class-weighted loss,
or reconsidering the bin boundaries) rather than more of the same
hyperparameter tuning — and should stop training around epoch 20-25
(e.g. `patience=8`) rather than running the full budget.

No DeepFace fallback was needed — both tasks passed on the first
fine-tuning iteration.

## Artifacts

- Weights: `models/age_gender/final_age.pt`, `models/age_gender/final_gender.pt`
  (gitignored, like all `*.pt`).
- Full metrics: `models/age_gender/final_report.json`
- Loss/accuracy curves: `models/age_gender/final_age_loss_curves.png`,
  `models/age_gender/final_gender_loss_curves.png`
- Raw per-epoch logs: `runs/age_gender_finetune/{age,gender}/results.csv`

Not yet integrated into the live pipeline (`pipeline_demo.py`) — these
remain standalone classifiers.