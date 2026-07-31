# Unified real-time inference pipeline

## Overview

`src/retailvision/inference.py` combines face detection with age/gender
and emotion classification into a single per-frame call. One shared
`FaceDetector` (Haar cascade) locates faces; every detected crop is then
run through all three fine-tuned YOLOv8 classifiers
(`models/age_gender/final_age.pt`, `final_gender.pt`,
`models/emotion/final.pt`).

```python
from src.retailvision.inference import InferencePipeline

pipeline = InferencePipeline()
detections = pipeline.process_frame(frame)  # list[dict]
```

Each entry in the returned list has the shape:

```python
{
    "bbox": (x, y, w, h),
    "age_group": "18-30",
    "gender": "Male",
    "emotion": "happy",
    "confidence": {"age": 0.83, "gender": 0.97, "emotion": 0.91},
}
```

`confidence` is kept as a per-attribute dict rather than a single blended
scalar, since collapsing three independent classifier confidences into
one number would throw away information downstream consumers (logging,
zone-emotion correlation) are likely to need.

## Design decisions

**No separate bounding-box (IoU) matching step.** Age/gender and emotion
predictions come from the exact same detected crop, because one detector
feeds all three classification heads. There are never two independent
boxes for the same face to reconcile, so IoU-based matching (needed when
different models run their own detection passes) doesn't apply to this
architecture. If a future detector swap moves to per-model detection, a
matching step would need to be added at that point.

**Face detection stays on Haar cascade, not YOLOv8.** Everything
downstream — the age/gender and emotion classifiers, all real-world
evaluation work in `docs/model_evaluation.md` — was built and validated
against the existing Haar cascade `FaceDetector`. A deep-learning face
detector (YOLOv8 in detection mode, or a dedicated face model) would
likely generalize better across angle, distance, and lighting than Haar
cascade — particularly for the non-frontal-angle and false-positive
failure modes documented in `docs/model_evaluation.md` — but that is a
separate, untrained, unvalidated component. Swapping it in is deferred to
a future detection-pipeline ticket rather than folded into this one.

## Usage

```
PYTHONPATH=. ./venv/bin/python3 -m src.retailvision.pipeline_demo                      # live camera, default index 0
PYTHONPATH=. ./venv/bin/python3 -m src.retailvision.pipeline_demo --source path/to.mp4 # pre-recorded video file
PYTHONPATH=. ./venv/bin/python3 -m src.retailvision.pipeline_demo --benchmark          # headless, prints average FPS on exit
PYTHONPATH=. ./venv/bin/python3 -m src.retailvision.pipeline_demo --detector yolo-final # try a YOLOv8 face detector instead of Haar cascade
```

Per the project's testing convention, point `--source` at a pre-recorded
video file first — a deterministic input is much easier to debug against
than a live camera feed — before testing on the live camera.

`--detector` (`haar`, the default; or `yolo-baseline`/`yolo-final`,
loading `models/face_detection/baseline.pt`/`final.pt`) is a demo-only
override for visually trying an alternative face detector under
real-world evaluation. It does not change production behavior — `haar`
stays the default, and `FaceDetector` itself is untouched.

## Performance

Headless benchmark, single detected face per frame, Apple Silicon (MPS):
**~61 FPS**, well above the 10 FPS minimum. This measures the
classification cost (three YOLOv8 classifier calls per detected face)
with detection itself mocked to a fixed box, since the real bottleneck at
scale is per-face classification cost multiplying with face count, not
Haar cascade detection. Real end-to-end FPS on live camera/video input —
including actual detection cost and, in non-benchmark mode, display
overhead — should be measured per-deployment with `--benchmark` against
real footage; multiple simultaneous faces will scale classification cost
roughly linearly.