# Face-detector baseline (WIDER FACE)

## Setup

A single `yolov8n` **detection**-mode model trained from scratch on WIDER
FACE (`data/widerface/processed/`, see `docs/datasets/widerface.md` for
dataset prep, filtering, and the known crowd-scene-vs-retail domain gap),
using Ultralytics default hyperparameters — 100 epochs, `imgsz=640` (the
actual detection-mode default; unlike the age/gender/emotion
*classification* baselines, which override down to `imgsz=224`, this one
keeps the framework default since it's the correct resolution for this
task, not a deviation from it), seed 42. See
`scripts/widerface_baseline/constants.py` for the exact values.

Two throughput/stability adjustments, neither affecting what's learned:
`cache="disk"` (WIDER FACE's source images are far larger than anything
else trained in this project, so caching decoded/resized images avoids
re-decoding from scratch every epoch), and `batch=8`, down from
Ultralytics' detection-mode default of 16 (WIDER FACE's crowd-scene
images can carry up to ~2,000 ground-truth boxes in a single frame,
which spikes label-assignment memory well past typical detection
datasets — a first attempt at batch=16 hit an out-of-memory condition
during training and crashed). See `scripts/widerface_baseline/train.py`
for the full rationale on both.

This is the project's first **detection**-mode YOLOv8 training run — the
age/gender and emotion models are all `yolov8n-cls` (classification
mode: one label per already-cropped face image). Detection mode predicts
bounding boxes directly on full scenes, so the training loss, output
head, and evaluation metrics all differ from the classification
baselines (mAP instead of top1/top5 accuracy).

### Evaluation methodology

Two separate evaluations are run, both against our own held-out test
split (1,613 of WIDER FACE's official 3,226-image val set — the official
test set has no public ground truth, see `docs/datasets/widerface.md`):

1. **Overall metrics** (mAP@0.5, mAP@0.5:0.95, precision, recall) via
   Ultralytics' own detection validator (`model.val()`).
2. **Official Easy/Medium/Hard recall** — WIDER FACE's real,
   author-defined difficulty partition, not an approximation. This
   required downloading the dataset's separate `eval_tools` package
   (`http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip`)
   for its ground-truth `.mat` files
   (`data/widerface/raw/wider_face_split/ground_truth/wider_{easy,medium,hard}_val.mat`),
   since the difficulty partition itself was computed by the original
   authors via a proposal-recall procedure and can't be reconstructed
   from the raw box annotations alone. Recall (not the full
   confidence-ranked AP curve WIDER FACE's official protocol computes) is
   measured via greedy IoU≥0.5 matching between each image's official
   keep-boxes and the model's predictions, restricted to the ~1,613
   official-val images that fall in our test split. See
   `scripts/widerface_baseline/official_eval.py` for the full
   implementation and rationale — notably, an earlier plan to substitute
   an ad-hoc face-count bucket (`0`, `1`, `2-5`, ..., `101+`, built for
   this dataset's own val/test split stratification) in place of the real
   Easy/Medium/Hard partition was reconsidered before writing evaluation
   code, in favor of the real official ground truth.

## Results

Trained on an RTX 3070 (CUDA) — 100 epochs in 2.636 hours. (An earlier
attempt on Apple Silicon MPS extrapolated to 40+ hours before a
memory-pressure crash forced moving to CUDA; see `train.py`'s docstring
for the `cache`/`batch` adjustments that came out of that.)

| Metric | Value |
|---|---|
| mAP@0.5 (test split) | 76.23% |
| mAP@0.5:0.95 (test split) | 41.75% |
| Precision (test split) | 86.35% |
| Recall (test split) | 67.54% |

| Difficulty | Recall | GT faces evaluated | Images evaluated |
|---|---|---|---|
| Easy | 94.18% | 3,538 | 1,308 |
| Medium | 89.87% | 6,527 | 1,517 |
| Hard | 71.10% | 15,766 | 1,609 |

## Findings

**Hard is meaningfully lower than Easy (71.10% vs. 94.18%) — the expected
shape, not a red flag.** This confirmed prediction (see `docs/datasets/widerface.md`'s
box-size percentiles: median annotated face is only 15px) is exactly why
the real official partition was worth downloading rather than trusting
an approximation: a face-count-bucket substitute couldn't have produced
this specific, checkable expectation ahead of time. Recall degrades
monotonically Easy → Medium → Hard, consistent with each tier including
progressively smaller/more occluded faces.

**Image coverage differs by difficulty tier** (Easy: 1,308/1,613 images,
Medium: 1,517, Hard: 1,609) because not every image contains a
difficulty-qualifying face — Hard's looser inclusion criteria means
almost every image (99.8%) contributes at least one Hard-tier face,
while only 81% contain an Easy-tier one. This isn't a data-quality gap,
it's the partition working as designed.

**mAP@0.5:0.95 (41.75%) is well below mAP@0.5 (76.23%) — expected, not
concerning.** mAP@0.5:0.95 averages precision across IoU thresholds from
0.5 to 0.95 in steps of 0.05, so it penalizes boxes that are
approximately-but-not-precisely located much more harshly. A large gap
between the two is normal for detection models generally, not specific
to this run.

**Test-split numbers are slightly below the val-split numbers logged
during training** (test mAP@0.5 76.23% vs. training's own final val-split
validation at 79.9%; similarly precision 86.35% vs. 88.6%, recall 67.54%
vs. 72.1%). This is exactly the expected direction: val gets used for
monitoring during training even though not for gradient updates, while
test stays genuinely untouched until this evaluation — same held-out
discipline as every other model in this project (e.g. FER-2013's
official test split, kept intact through the emotion model's entire
training history).

## Training health

![WIDER FACE baseline loss & mAP curves](../../models/face_detection/baseline_loss_curves.png)

No overfitting signature — unlike the age/gender and emotion
classification baselines, which both showed val loss reversing upward
after roughly epoch 40 while train loss kept falling. Here, `val/box_loss`,
`val/cls_loss`, and `val/dfl_loss` track their `train/*` counterparts
closely throughout, with `cls_loss` in particular converging to nearly
the same value (~0.6) on both train and val by epoch 100. `precision(B)`,
`recall(B)`, `mAP50(B)`, and `mAP50-95(B)` all rise smoothly and plateau
around epoch 60-80, holding flat rather than regressing through the
remaining epochs. Fine-tuning can lean on this: there's no early-stopping
pressure this baseline is already fighting against, unlike the
classification baselines' documented `patience` recommendation.

## Artifacts

- Weights: `models/face_detection/baseline.pt`
- Full metrics: `models/face_detection/baseline_report.json`
- Loss/mAP curve: `models/face_detection/baseline_loss_curves.png`
- Raw per-epoch log: `runs/widerface_baseline/widerface/results.csv`

This baseline itself was never integrated into the live pipeline — it
was a starting point for fine-tuning (`docs/models/widerface_finetune.md`),
whose output (`final.pt`) is what real-world evaluation validated and
what `FaceDetector` runs today. See `docs/model_evaluation.md` for the
real-world results that motivated the production swap.