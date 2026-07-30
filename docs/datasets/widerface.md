# WIDER FACE dataset — face-detector preparation

## Source

Downloaded manually from the official WIDER FACE release
([shuoyang1213.me/WIDERFACE](http://shuoyang1213.me/WIDERFACE/)) into
`data/widerface/raw/` (gitignored — not committed): `WIDER_train.zip`,
`WIDER_val.zip`, and `wider_face_split.zip` (the ground-truth annotation
files). `WIDER_test.zip` was downloaded but is unused — the official test
set ships with no public ground truth, so it isn't usable for training or
evaluation.

Verified against the documented WIDER FACE format: 12,880 train images and
3,226 val images, each with a matching entry in
`wider_face_split/wider_face_{train,val}_bbx_gt.txt`. Each entry is
`filename`, `face count`, then one `x y w h blur expression illumination
invalid occlusion pose` line per face — with a single placeholder
`0 0 0 0 0 0 0 0 0 0` line even when the count is 0, a documented quirk of
this release's format that the parser accounts for explicitly.

## Format decision: detection, not classification

Unlike UTKFace and FER-2013 ([[utkface]], FER-2013), WIDER FACE images are
**not** pre-cropped single-face chips — they're full scenes, often crowds,
with bounding boxes for every face. This is the first *detection*-mode
dataset in the project (the age/gender and emotion classifiers are all
`yolov8n-cls`, one label per image). WIDER FACE is laid out as an
Ultralytics YOLOv8 detection dataset: `images/` + matching `labels/`
(`class x_center y_center width height`, normalized 0-1), plus a
`widerface.yaml` dataset config, rather than folder-per-class.

## Layout

```
data/widerface/processed/
  images/{train,val,test}/*.jpg   (symlinks into raw/)
  labels/{train,val,test}/*.txt   (YOLO detection format, one line per retained box; empty for background images)
  widerface.yaml                  (Ultralytics dataset config: path, train/val/test, names)
  distribution_report.json
```

Images are symlinked rather than copied, same rationale as UTKFace/FER-2013:
avoids duplicating several GB of source images.

### Box normalization

The annotation files give absolute pixel `x, y, w, h` with no image
dimensions alongside them, so each image is opened once (via
`PIL.Image.open(...).size`, a header read, not a full decode) to compute
the width/height needed to normalize boxes into YOLO's required 0-1 range.

### Train/val/test split

The official train split (12,880 images) is kept as-is for training.
The official val split (3,226 images — the only other portion with public
ground truth, since the official test set has none) is divided 50/50 into
our own val and test sets, stratified by a face-count bucket (`0`, `1`,
`2-5`, `6-20`, `21-50`, `51-100`, `101+`) so neither split skews toward
sparse or crowd-dense images relative to the other, fixed seed (42). The
top bucket was initially just `21+`, but the data has a long tail up to
1,968 faces in a single image — collapsing 21 and 1,968 into one bucket
would have made the val/test split unreliable at exactly the dense end
that matters most for stress-testing a detector, so it's split further
into `21-50`, `51-100`, and `101+`.

| Split | Images | Faces (after filtering) |
|---|---|---|
| train | 12,880 | 132,326 |
| val | 1,613 | 16,553 |
| test | 1,613 | 16,043 |

## Filtering: dropped-invalid and dropped-tiny boxes

Of 199,128 total annotated boxes across train+val:

| Dropped as | Count | Share |
|---|---|---|
| `invalid == 1` (flagged invalid by the original annotators) | 2,984 | 1.5% |
| narrower or shorter than 8px (either dimension) | 31,222 | 15.7% |
| **Kept** | 164,922 | 82.8% |

Box size (min of width/height, pixels) is heavily right-skewed:

| Percentile | Train | Val |
|---|---|---|
| p10 | 6px | 6px |
| p25 | 9px | 9px |
| p50 (median) | 15px | 15px |
| p75 | 30px | 31px |
| p90 | 58px | 59px |
| max | 976px | 1,008px |

The **8px minimum** threshold is deliberately conservative — it only
removes genuinely degenerate/near-zero-signal annotations (down to 0px in
the raw data), not the broader "small face" population. It does not by
itself close the gap described below.

## Known domain gap: dense crowd scenes vs. single retail subject

WIDER FACE's images are drawn from event photography (parades, protests,
sports, etc.) — faces are frequently numerous, small, and partially
occluded within one frame (median face count per image is well above 1,
see `distribution_report.json`'s `face_count_distribution`). The retail
camera use case this detector is ultimately built for is close to the
opposite: typically one or a few prominent, front-facing subjects at
close-to-medium range, not tens of small faces in a crowd.

This is documented explicitly, the same way UTKFace's race imbalance and
FER-2013's Disgust scarcity are documented, so any accuracy gap between
WIDER FACE test-set metrics and real retail-camera performance is
explainable rather than a surprise. It's also precisely why RV-028 (real
world evaluation) validates directly against retail-style live-camera
footage rather than trusting WIDER FACE metrics alone before the
production swap in RV-029.

## Augmentation strategy

The baseline (RV-026) deliberately used pure Ultralytics defaults, no
augmentation tuning — it needed to be a faithful "defaults" floor to
fine-tune against, same rationale as every classification baseline in
this project. Tuning happened at fine-tuning time (RV-027,
`scripts/widerface_finetune/constants.py`) instead:

- **Horizontal flip (`fliplr=0.5`)**: safe — faces are roughly bilaterally
  symmetric.
- **No vertical flip (`flipud=0.0`)**: retail camera faces are never
  upside-down.
- **Mild rotation (`degrees=10.0`, up from the baseline's 0.0)**: robustness
  to camera angle / head tilt, same rationale as UTKFace's augmentation.
- **Scale jitter (`scale=0.9`, up from the detection default 0.5)**: WIDER
  FACE's face-size distribution is dominated by small faces (median
  15px); a wider random-zoom range exposes the model to more "zoomed in,
  large face" training crops than the raw data provides on its own.
- **Mosaic (`mosaic=0.5`, down from the detection default 1.0)**: revised
  from RV-025's original plan to keep mosaic at its default. Mosaic
  stitches 4 images into one training frame, so each sub-image only
  occupies roughly 1/4 of the frame — every face ends up looking
  smaller/more crowd-like than it actually is, compounding WIDER FACE's
  existing small-face bias in exactly the wrong direction for this use
  case. Lowering (not eliminating — mosaic still has real generalization
  value) the probability lets more batches train on full-frame,
  single-scene images that better resemble retail camera framing.
- **Mild HSV jitter (`hsv_h=0.015`, `hsv_s=0.4`, `hsv_v=0.3`)**: robustness
  to in-store lighting variation, close to Ultralytics' detection-mode
  defaults — unlike the classification models, there's no
  race-imbalance-driven reason to keep color jitter especially
  conservative for this task.