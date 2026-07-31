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
PYTHONPATH=. ./venv/bin/python3 -m src.retailvision.pipeline_demo                      # live camera, full age/gender/emotion pipeline
PYTHONPATH=. ./venv/bin/python3 -m src.retailvision.pipeline_demo --source path/to.mp4 # pre-recorded video file instead of live camera
PYTHONPATH=. ./venv/bin/python3 -m src.retailvision.pipeline_demo --benchmark          # headless, prints average FPS on exit
./venv/bin/python3 -m src.retailvision.camera_test                                     # camera-only sanity check, no detection
```

The live/video modes open a preview window; press `q` to quit. Currently reads only from the laptop's default camera (`cv2.VideoCapture(0)`) or a local video file; multi-camera support is planned but not yet implemented. See `docs/inference_pipeline.md` for the pipeline's architecture, design decisions, and FPS results.

## Module layout

- `src/retailvision/camera_test.py` — minimal camera-open/read sanity check.
- `src/retailvision/detection.py` — `FaceDetector`, wraps OpenCV's bundled Haar cascade classifier. First stage of the pipeline.
- `src/retailvision/inference.py` — `InferencePipeline`, combines `FaceDetector` with the fine-tuned age/gender and emotion classifiers into one per-frame call. See `docs/inference_pipeline.md`.
- `src/retailvision/pipeline_demo.py` — wires capture (camera or video file) → `InferencePipeline` → live preview with drawn bounding boxes and predictions, or a headless FPS benchmark.

## Dataset preparation

Datasets live under gitignored `data/<dataset>/raw/` (original downloads) and `data/<dataset>/processed/` (prepared, training-ready layout + a `distribution_report.json`). Human-readable documentation of sourcing, format decisions, and class distributions is tracked in `docs/datasets/`.

```
PYTHONPATH=scripts ./venv/bin/python3 scripts/prepare_utkface.py    # age/gender labels, see docs/datasets/utkface.md
PYTHONPATH=scripts ./venv/bin/python3 scripts/prepare_fer2013.py    # 7 emotion classes, see docs/datasets/fer2013.md
PYTHONPATH=scripts ./venv/bin/python3 scripts/prepare_widerface.py  # face bounding boxes, see docs/datasets/widerface.md
```

UTKFace and FER-2013 consist of pre-cropped single-face images, so they're prepared as YOLOv8 **classification** datasets (folder-per-class); face localization is a separate upstream pipeline stage. WIDER FACE is the exception — it's full scenes with bounding-box annotations, prepared as a YOLOv8 **detection** dataset (`images/` + matching `labels/`) instead, for training a face detector to eventually replace the Haar cascade `FaceDetector`.

- **UTKFace** (`scripts/utkface_prep/`) — 33,481 images, age binned into `0-17`/`18-30`/`31-50`/`51+`, gender mapped to `Male`/`Female`, stratified 70/15/15 split. Known imbalance: White overrepresented ~5.5x across race labels (documented, not corrected).
- **FER-2013** (`scripts/fer2013_prep/`) — 35,887 images across 7 emotion classes. Official test split kept untouched; a stratified 10% validation split is carved out of train. Known imbalance: Disgust is severely underrepresented (1.5% of data); Fear is normal-sized but documented as noisy/confusable with Sad and Surprise.

## Model training

Baseline classifiers are trained from scratch on the prepared datasets using `yolov8n-cls`. Age and gender are trained as two independent classifiers (YOLOv8 classification mode is single-label per run), even though both come from UTKFace. For the full history of decisions, results, and graphics behind the age/gender model line (dataset → baseline → fine-tune → real-world eval → rebinning → regression), see [`docs/models/README.md`](docs/models/README.md).

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

### Emotion classifier

A single `yolov8n-cls` classifier trained from scratch on FER-2013 (7 emotion classes), same baseline methodology as age/gender: Ultralytics defaults, 100 epochs, `imgsz=224`.

```
PYTHONPATH=scripts ./venv/bin/python3 scripts/train_emotion_baseline.py     # trains the emotion classifier
PYTHONPATH=scripts ./venv/bin/python3 scripts/evaluate_emotion_baseline.py  # per-class precision/recall/F1, loss curves, report
```

- `scripts/emotion_baseline/` — training/evaluation package (`constants.py`, `train.py`, `evaluate.py`, `plotting.py`), mirroring `age_gender_baseline/`'s structure.
- Trained weights are saved to `models/emotion/baseline.pt`.
- Evaluation writes `models/emotion/baseline_report.json` and a loss/accuracy curve PNG. See `docs/models/emotion_baseline.md` for full results and findings (71.12% top1; Fear is the weakest class as expected, but Disgust — despite being the most underrepresented class — outperforms several more common classes).

Fine-tuning follows the same pattern as age/gender, plus a per-class recall bar (80% on Happy and Neutral specifically, not an aggregate threshold):

```
PYTHONPATH=scripts ./venv/bin/python3 scripts/finetune_emotion.py           # fine-tunes with augmentation
PYTHONPATH=scripts ./venv/bin/python3 scripts/evaluate_emotion_finetune.py  # per-class metrics, loss curves, threshold check, report
```

- `scripts/emotion_finetune/` — fine-tune training package; reuses `emotion_baseline`'s evaluation/plotting helpers.
- Final weights are saved to `models/emotion/final.pt`.
- Neutral failed its 80% recall threshold across two fine-tuning iterations (67.6%, 67.3%) with confusion-matrix evidence of a structural neutral/sad overlap at FER-2013's 48×48 resolution. DeepFace's pre-trained emotion model was evaluated as an alternative (`scripts/evaluate_emotion_deepface.py`) but performed worse on every class (56.4% vs. our 69.7% overall) and was rejected. Our fine-tuned classifier remains production; Neutral (alongside Fear/Disgust) is accepted as a documented limitation. Full investigation: `docs/models/emotion_finetune.md`.

### Face detector

A `yolov8n` **detection**-mode model trained from scratch on WIDER FACE — the project's first detection-mode YOLOv8 run (age/gender/emotion are all classification-mode `yolov8n-cls`), intended to eventually replace the Haar cascade `FaceDetector`.

```
PYTHONPATH=scripts ./venv/bin/python3 scripts/train_widerface_baseline.py     # trains the face-detection baseline
PYTHONPATH=scripts ./venv/bin/python3 scripts/evaluate_widerface_baseline.py  # mAP/precision/recall + official Easy/Medium/Hard recall, report
```

- `scripts/widerface_baseline/` — training/evaluation package; evaluation includes both Ultralytics' own detection metrics and WIDER FACE's real, author-defined Easy/Medium/Hard difficulty partition (downloaded separately from the dataset's `eval_tools` package — the partition can't be reconstructed from the raw box annotations alone).
- Trained weights are saved to `models/face_detection/baseline.pt`.
- Evaluation writes `models/face_detection/baseline_report.json` and a loss/mAP curve PNG. See `docs/models/widerface_baseline.md` for full results (76.23% mAP@0.5 on the held-out test split; recall degrades monotonically from 94.18% on Easy faces to 71.10% on Hard, the expected WIDER FACE pattern).

Fine-tuning retrains from `yolov8n.pt` with augmentation tuned for the retail-camera domain gap documented in `docs/datasets/widerface.md` (wider zoom range, reduced mosaic — mosaic shrinks every face to ~1/4 frame, compounding WIDER FACE's small-face bias in the wrong direction for this use case):

```
PYTHONPATH=scripts ./venv/bin/python3 scripts/finetune_widerface.py           # fine-tunes with retail-tuned augmentation
PYTHONPATH=scripts ./venv/bin/python3 scripts/evaluate_widerface_finetune.py  # metrics vs. baseline, threshold check, report
```

- `scripts/widerface_finetune/` — fine-tune training package; reuses `widerface_baseline`'s evaluation/plotting helpers, which are generic across any trained checkpoint.
- Final weights are saved to `models/face_detection/final.pt`.
- Evaluation checks Hard-difficulty recall against a minimum threshold (71.10% — the baseline's own result). The fine-tune **regressed on every WIDER FACE metric** (mAP@0.5 74.63% vs. baseline's 76.23%; Hard recall 68.79%, failing the threshold) — working hypothesis is that this is an expected cost of deliberately shifting training data toward retail-camera framing and away from WIDER FACE's own crowd-scene domain, not a training failure, but that's unconfirmed until RV-028 tests both checkpoints against real footage. See `docs/models/widerface_finetune.md` for the full analysis.
- Not yet integrated into the live pipeline — real-world evaluation (RV-028) is the actual gate for the production swap (RV-029), evaluating both `baseline.pt` and `final.pt` rather than assuming which is better from WIDER FACE metrics alone.

## Live demo

To just watch the current classifiers and the age-regression model run on your own webcam, with no ground truth or logging needed:

```
PYTHONPATH=.:scripts ./venv/bin/python3 scripts/live_demo.py
```

Opens a live preview with each detected face boxed and labeled with its predicted age bin, continuous age estimate, gender, and confidence — e.g. `18-40 (~26y, 0.91) / Male (0.97)`. Always reflects whatever weights currently sit at `models/age_gender/final_age.pt`/`final_gender.pt`/`regression_age.pt`. Press `q` to quit. For accuracy evaluation (logging predictions against a known ground truth across conditions), see Real-world evaluation below instead.

## Real-world evaluation

The fine-tuned classifiers are also validated against live webcam video, not just the static test sets, to check whether test-set accuracy holds up under real capture conditions. See `docs/model_evaluation.md` for full results.

**Age/gender** — conditions are lighting/occlusion/angle. Gender classification generalizes well; age classification degrades severely on live camera input regardless of condition; face detection itself fails on faces angled past ~45°.

```
PYTHONPATH=.:scripts ./venv/bin/python3 scripts/evaluate_real_world.py --condition <name> --true-age <bin> --true-gender <Male|Female>
PYTHONPATH=.:scripts ./venv/bin/python3 scripts/summarize_real_world_eval.py
```

- `scripts/real_world_eval/` — live-capture package (`constants.py`, `classify.py`, `capture.py`), reusing the existing `FaceDetector`.
- Each condition session opens a live preview (green box = correct, red = wrong) and appends per-frame predictions to `runs/real_world_eval/<condition>.csv`; press `q` to stop.
- Summarization writes `models/age_gender/real_world_eval_report.json`: face-detection rate and per-task accuracy per condition, compared against the fine-tuned classifier's test-set accuracy.

**Emotion** — expanded into a full emotion × condition matrix (7 emotions × 5 conditions). Fear and disgust are unreliable live at any distance (3-16% accuracy); happy/angry/neutral hold up well at close-to-medium range; non-frontal poses (side view, looking down) fail detection too often to even collect reliable per-emotion data. An early "severe close-range collapse" finding turned out to be a background object (a mannequin) being misdetected as a face, not a real model or lens issue — see the model_evaluation.md writeup for the full investigation.

```
PYTHONPATH=.:scripts ./venv/bin/python3 scripts/evaluate_emotion_real_world.py --condition <name> --true-emotion <label>
PYTHONPATH=.:scripts ./venv/bin/python3 scripts/summarize_emotion_real_world_eval.py
```

- `scripts/emotion_real_world_eval/` — live-capture package, mirroring `real_world_eval/`'s structure.
- Summarization writes `models/emotion/real_world_eval_report.json`, comparing each condition against the *specific* held-emotion's test-set recall (not the blended overall top1 — see the script's docstring for why that distinction matters here).

## Age regression (continuous age for live display)

The 4-bin classifier is coarse for live feedback ("18-30" isn't very informative). A separate model predicts a continuous age (e.g. "~25") for display, while the classifier continues to handle analytics/reporting. An earlier attempt to solve this by re-binning the classifier into narrower classes (7, then 10) was abandoned — adult age brackets plateaued at 50-65% F1 regardless of tuning; see `docs/models/age_rebinning_investigation.md`. Regression sidesteps that ceiling entirely since there are no bin boundaries to be confused across.

Ultralytics/YOLOv8 has no native regression task, so this model is a plain PyTorch/torchvision ResNet18 with a single-output regression head, trained separately from the YOLOv8 classifiers.

```
PYTHONPATH=scripts ./venv/bin/python3 scripts/prepare_age_regression.py    # builds train/val/test CSV manifests (path, age, gender)
PYTHONPATH=scripts ./venv/bin/python3 scripts/train_age_regression.py      # trains with early stopping on validation MAE
PYTHONPATH=scripts ./venv/bin/python3 scripts/evaluate_age_regression.py   # overall + per-age-group MAE on the held-out test split
```

- `scripts/age_regression_prep/` — builds CSV manifests instead of a folder-per-class layout (regression has no discrete classes); reuses `utkface_prep`'s filename parsing and stratified split.
- `scripts/age_regression/` — training package (`dataset.py`, `model.py`, `train.py`, `evaluate.py`, `plotting.py`).
- Weights saved to `models/age_gender/regression_age.pt`; evaluation writes `models/age_gender/regression_report.json` with overall MAE and MAE bucketed into the original 4-bin classifier ranges for a per-age-group breakdown (bucketing is for reporting only — the model itself is never trained against bins).
- Runs alongside the 4-bin classifier in the live pipeline: classifier output for analytics, regression output for display.
