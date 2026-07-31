# Face-detector fine-tune (WIDER FACE)

## Setup

Retrained from `models/yolov8n.pt` (not continued from
`models/face_detection/baseline.pt`), same from-scratch-retrain rationale
used for every other fine-tune in this project: a clean run with a lower
learning rate and tuned augmentation, instead of compounding whatever the
baseline already converged to.

Adjusted from the baseline: `lr0=0.005` (down from 0.01), `patience=15`
(early stopping — the baseline showed no overfitting even at epoch 100,
so this isn't fighting instability the way the classification
fine-tunes' shorter epoch budgets were, it's just a safety net),
`optimizer="SGD"`. `batch=8` and `cache="disk"` carry over unchanged from
the baseline — the memory-spike risk from WIDER FACE's dense crowd
images (docs/datasets/widerface.md) doesn't go away just because
augmentation changed. See `scripts/widerface_finetune/constants.py` for
exact values.

Augmentation is tuned specifically for the retail-camera domain gap —
see `docs/datasets/widerface.md`'s "Augmentation strategy" section for
the full rationale behind each change (`scale=0.9`, `mosaic=0.5`,
`degrees=10.0`).

## Results

Trained on the same RTX 3070 setup as the baseline — 100 epochs in 2.569
hours, no early stopping triggered (`patience=15` never fired).

| Metric | Baseline | Fine-tuned | Δ |
|---|---|---|---|
| mAP@0.5 (test split) | 76.23% | 74.63% | **-1.60pp** |
| mAP@0.5:0.95 (test split) | 41.75% | 40.42% | -1.33pp |
| Precision (test split) | 86.35% | 84.87% | -1.48pp |
| Recall (test split) | 67.54% | 66.43% | -1.11pp |

| Difficulty | Baseline recall | Fine-tuned recall | Δ |
|---|---|---|---|
| Easy | 94.18% | 93.70% | -0.48pp |
| Medium | 89.87% | 89.14% | -0.73pp |
| Hard | 71.10% | 68.79% | **-2.31pp** |

**Minimum acceptable Hard recall threshold: 71.10%** (the baseline's own
result) — **FAILED** (68.79%).

## Findings

**Every metric regressed, consistently, not just noise in one
direction.** This is a real result to explain, not dismiss.

**Working hypothesis: the regression is a direct, expected consequence
of deliberately shifting the training distribution away from WIDER
FACE's own domain — not a training failure.** The fine-tune's
augmentation changes (`docs/datasets/widerface.md`) intentionally pull
training data *away* from WIDER FACE's native crowd-scene framing
(reduced mosaic, wider zoom range) and toward single-subject,
close-range framing that resembles retail camera footage. But the
**test set is still WIDER FACE** — still crowd-scene photography. Moving
the training distribution toward a different target domain while
evaluating against the original domain's own benchmark should be
expected to cost some accuracy on that benchmark, by construction,
regardless of whether it helps on the actual target domain. The
regression being largest on Hard (-2.31pp) and smallest on Easy
(-0.48pp) is consistent with this: Hard is the tier most defined by
small/dense/crowd-like faces — exactly the framing the augmentation
was tuned to de-emphasize.

**This was a hypothesis, not yet a confirmed explanation, at the time
this fine-tune finished.** WIDER FACE metrics alone couldn't confirm or
refute it; only real retail-camera footage could. Real-world evaluation
against live footage — evaluating **both** `baseline.pt` and `final.pt`
against the same recordings, so they'd be compared fairly rather than on
two separate live takes — was run specifically to answer "did the
retail-tuned augmentation help" rather than assume it from a metric that
may be structurally biased against it. See `docs/model_evaluation.md`
for the result: the hypothesis held up. Detection rate on the exact
conditions this migration was about (non-frontal angles) jumped from
Haar cascade's documented 27%/46% to ~97-100% for both checkpoints, and
`final.pt` showed more resistance than `baseline.pt` to a recurring
false-positive case in matched-frame comparison.

**The Hard-recall threshold failure documented above is real and
stands** — the WIDER FACE regression genuinely happened, and the
explanation for it only became more than a hypothesis once real-world
results came in. `final.pt` is now the production `FaceDetector` on the
strength of that real-world result, not the WIDER FACE benchmark.

## Training health

![WIDER FACE fine-tune loss & mAP curves](../../models/face_detection/final_loss_curves.png)

Structurally healthy, same as the baseline — no overfitting signature,
`val/*` losses track `train/*` losses closely throughout (`cls_loss`
converges to ~0.6-0.65 on both), and `precision(B)`/`recall(B)`/`mAP50(B)`/
`mAP50-95(B)` all plateau smoothly around epoch 60-80 without regressing
through epoch 100. This matters for the finding above: a broken or
unstable training run would show noisy, diverging, or oscillating
curves. This one doesn't — it converged normally, just to a different
(lower, on WIDER FACE's own metrics) point, which is what a real
distribution shift looks like rather than a training bug.

## Artifacts

- Weights: `models/face_detection/final.pt`
- Full metrics: `models/face_detection/final_report.json`
- Loss/mAP curve: `models/face_detection/final_loss_curves.png`
- Raw per-epoch log: `runs/widerface_finetune/widerface/results.csv`

This checkpoint is now the production `FaceDetector`, replacing Haar
cascade — see `docs/model_evaluation.md` for the real-world evaluation
that decided it, and `docs/inference_pipeline.md` for the integration.
