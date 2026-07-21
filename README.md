# RetailVision

AI-powered computer vision pipeline for real-time customer demographic detection, emotion recognition, foot traffic counting, and zone-engagement scoring in retail environments.

Single-process pipeline: one service ingests camera streams and runs detection, demographic/emotion recognition, foot traffic counting, and zone-engagement scoring, rather than splitting these into separate microservices. Built on OpenCV (capture/detection) and PyTorch/Ultralytics YOLOv8 (recognition/classification, trained from scratch on public datasets rather than a pretrained fallback).

## Setup

```
python3.12 -m venv venv
./venv/bin/pip install -r requirements.txt
```

The venv must be created with **Python 3.12** (via Homebrew's `python@3.12`), not the system `/usr/bin/python3` (3.9.6, too old) and not another project's venv that may be first on `PATH`.

`opencv-python` is pinned to `4.10.0.84` in `requirements.txt` — the current latest ships a broken build on macOS missing `cv2.CascadeClassifier` and the bundled Haar cascade XML files. Don't bump this pin without verifying `cv2.CascadeClassifier` and `cv2.data.haarcascades` still work.


## Running the live pipeline

```
./venv/bin/python3 -m src.retailvision.pipeline_demo   # live camera preview with face-detection boxes
./venv/bin/python3 -m src.retailvision.camera_test      # camera-only sanity check, no detection
```

Both open a live window; press `q` to quit. Currently reads only from the laptop's default camera (`cv2.VideoCapture(0)`); multi-camera support is planned but not yet implemented.

## Module layout

- `src/retailvision/camera_test.py` — minimal camera-open/read sanity check.
- `src/retailvision/detection.py` — `FaceDetector`, wraps OpenCV's bundled Haar cascade classifier. First stage of the pipeline.
- `src/retailvision/pipeline_demo.py` — wires capture → `FaceDetector` → live preview with drawn bounding boxes. Entry point later stages (demographics, emotion, tracking, zone scoring) get added onto.

## Dataset preparation

Datasets live under gitignored `data/<dataset>/raw/` (original downloads) and `data/<dataset>/processed/` (YOLOv8-classification-ready folder trees + a `distribution_report.json`). Human-readable documentation of sourcing, format decisions, and class distributions is tracked in `docs/datasets/`.

```
PYTHONPATH=scripts ./venv/bin/python3 scripts/prepare_utkface.py   # age/gender labels, see docs/datasets/utkface.md
PYTHONPATH=scripts ./venv/bin/python3 scripts/prepare_fer2013.py   # 7 emotion classes, see docs/datasets/fer2013.md
```

Both datasets consist of pre-cropped single-face images, so they're prepared as YOLOv8 **classification** datasets (folder-per-class), face localization is a separate upstream pipeline stage.

- **UTKFace** (`scripts/utkface_prep/`) — 33,481 images, age binned into `0-17`/`18-30`/`31-50`/`51+`, gender mapped to `Male`/`Female`, stratified 70/15/15 split. Known imbalance: White overrepresented ~5.5x across race labels (documented, not corrected).
- **FER-2013** (`scripts/fer2013_prep/`) — 35,887 images across 7 emotion classes. Official test split kept untouched; a stratified 10% validation split is carved out of train. Known imbalance: Disgust is severely underrepresented (1.5% of data); Fear is normal-sized but documented as noisy/confusable with Sad and Surprise.

## Model training

Baseline classifiers are trained from scratch on the prepared datasets using `yolov8n-cls`. Age and gender are trained as two independent classifiers (YOLOv8 classification mode is single-label per run), even though both come from UTKFace.

```
PYTHONPATH=scripts ./venv/bin/python3 scripts/train_age_gender_baseline.py     # trains age, then gender
PYTHONPATH=scripts ./venv/bin/python3 scripts/evaluate_age_gender_baseline.py  # per-class precision/recall/F1, loss curves, report
```

- `scripts/age_gender_baseline/` — training/evaluation package (`constants.py`, `train.py`, `evaluate.py`, `plotting.py`).
- Trained weights are saved to `models/age_gender/baseline_age.pt` and `baseline_gender.pt` (gitignored, like all `*.pt` files).
- Evaluation writes `models/age_gender/baseline_report.json` and a loss/accuracy curve PNG per task. YOLOv8 classification mode reports top1/top5 accuracy rather than mAP@0.5 (a detection-mode metric); per-class precision/recall/F1 are computed separately via scikit-learn on the held-out test split.

Fine-tuning builds on the baseline with augmentation and adjusted hyperparameters, retraining from `yolov8n-cls.pt` rather than continuing from the baseline weights (see `docs/models/age_gender_finetune.md` for why).

```
PYTHONPATH=scripts ./venv/bin/python3 scripts/finetune_age_gender.py           # trains age, then gender, with augmentation
PYTHONPATH=scripts ./venv/bin/python3 scripts/evaluate_age_gender_finetune.py  # per-class metrics, loss curves, threshold check, report
```

- `scripts/age_gender_finetune/` — fine-tune training package (`constants.py`, `train.py`); reuses `age_gender_baseline`'s evaluation/plotting helpers, which are generic.
- Final weights are saved to `models/age_gender/final_age.pt` and `final_gender.pt`.
- Evaluation writes `models/age_gender/final_report.json`, checking top1 accuracy against the required 75% (age) / 85% (gender) thresholds per task.
