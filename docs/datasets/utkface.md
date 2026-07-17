# UTKFace dataset — age/gender preparation (RV-002)

## Source

Downloaded manually from Kaggle (`jangedoo/utkface-new`) into `data/utkface/raw/`
(gitignored — not committed). That Kaggle package ships three folders:

- `UTKFace/` — 23,708 images, the primary curated set.
- `crop_part1/` — 9,780 additional images, included for more training data.
- `utkface_aligned_cropped/` — **excluded**: verified to be an exact duplicate
  of `UTKFace/` + `crop_part1/` (identical file counts and checksums on
  sampled files), just repackaged under a nested folder layout.

`UTKFace/` + `crop_part1/` = 33,488 images. 7 were skipped for malformed
filenames (missing the race field, stray whitespace, or a missing `.` before
`jpg`) that don't match the `age_gender_race_date.jpg.chip.jpg` pattern —
listed in full in `data/utkface/processed/distribution_report.json`
(`skipped_filenames`). **33,481 images** were used.

## Format decision: classification, not detection

UTKFace images are already single pre-cropped face chips. Face *localization*
is handled by an earlier, separate pipeline stage (`FaceDetector`, currently
Haar cascade). So age/gender here is a classification problem — given a face
crop, predict its class — not a detection problem, and there's no meaningful
bounding box to draw beyond "the whole image." The dataset is laid out as a
YOLOv8 classification dataset (`yolov8n-cls`-compatible): folder-per-class,
no `.txt` annotation files.

## Layout

```
data/utkface/processed/
  age/{train,val,test}/{0-17,18-30,31-50,51+}/*.jpg   (symlinks into raw/)
  gender/{train,val,test}/{Male,Female}/*.jpg          (symlinks into raw/)
  distribution_report.json
```

Two independent classification trees (age, gender) are generated from the
same split so a future contributor can train `yolov8n-cls` directly against
either `data/utkface/processed/age/` or `.../gender/` with no further
conversion. Images are symlinked rather than copied to avoid duplicating the
~1-2GB of source images across two class trees.

Split is 70/15/15 (train/val/test), stratified by `(age_group, gender)` pair
with a fixed seed (42) so re-running the script reproduces the same split.

## Class distribution (33,481 images)

| Age group | Count | Share |
|---|---|---|
| 0–17 | 8,158 | 24.4% |
| 18–30 | 10,292 | 30.7% |
| 31–50 | 8,143 | 24.3% |
| 51+ | 6,888 | 20.6% |

Mild imbalance — largest class (18–30) is ~1.5x the smallest (51+). Not
severe enough to require oversampling, but worth watching in per-class
eval metrics; if 51+ recall lags noticeably, revisit with class weighting.

| Gender | Count | Share |
|---|---|---|
| Male | 16,761 | 50.06% |
| Female | 16,720 | 49.94% |

Effectively balanced — no correction needed.

| Race (not a training target — tracked for fairness auditing) | Count | Share |
|---|---|---|
| White | 15,342 | 45.8% |
| Indian | 5,427 | 16.2% |
| Asian | 4,987 | 14.9% |
| Black | 4,930 | 14.7% |
| Others | 2,795 | 8.3% |

This is the imbalance the ticket flagged as the real risk. White faces are
~5.5x overrepresented relative to "Others." Race is not a model output, but
because it isn't balanced, age/gender accuracy should be **evaluated
per-race on the test set**, not just in aggregate — an aggregate accuracy
figure can hide a model that performs much worse on underrepresented groups.
This is a known, documented UTKFace characteristic, not a preprocessing bug.

## Augmentation strategy (decided, applied at training time via YOLOv8 hyperparameters)

Recommended settings for the `yolov8n-cls` training run (RV-003):

- **Horizontal flip (`fliplr=0.5`)**: safe — faces are roughly bilaterally
  symmetric and flipping doesn't change age or gender.
- **No vertical flip (`flipud=0.0`)**: retail camera faces are never
  upside-down; would only add noise.
- **Mild rotation (`degrees≈10`)**: accounts for head tilt / camera angle
  without distorting features enough to look like a different age.
- **Mild translate/scale jitter (`translate≈0.1`, `scale≈0.2`)**: robustness
  to the upstream face detector's crop not being pixel-perfect — in
  production, boxes come from Haar cascade / a future YOLO face detector,
  not a hand-labeled dataset.
- **Conservative HSV jitter (`hsv_h≈0.01`, `hsv_s≈0.3`, `hsv_v≈0.2`)**:
  robustness to in-store lighting variation. Kept mild deliberately — skin
  tone is a real signal correlated with the race imbalance above, and
  aggressive color jitter risks distorting it in ways that could worsen
  accuracy for already-underrepresented groups rather than help.
- **Random erasing (`erasing≈0.2`)**: simulates partial occlusion (glasses,
  hands, masks), relevant to the Week 8 occlusion-robustness testing called
  for in the project plan.
- **No mosaic**: mosaic (stitching 4 images together) is a detection-time
  augmentation and doesn't apply to single-face classification crops.

Class imbalance itself (age groups, and race for fairness monitoring) is
handled by evaluation discipline rather than synthetic oversampling for this
phase: report per-class and per-race precision/recall on the held-out test
set after training, and only introduce class-weighted loss or oversampling
if a specific class/group shows a real accuracy gap.
