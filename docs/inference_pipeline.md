# Unified real-time inference pipeline

## Overview

`src/retailvision/inference.py` combines face detection with age/gender
and emotion classification into a single per-frame call. One shared
`FaceDetector` (a fine-tuned YOLOv8 detection-mode model) locates faces;
every detected crop is then run through all three fine-tuned YOLOv8
classifiers (`models/age_gender/final_age.pt`, `final_gender.pt`,
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

**Face detection runs on a fine-tuned YOLOv8 model, not Haar cascade.**
`FaceDetector` originally wrapped OpenCV's bundled Haar cascade
classifier. It was replaced after real-world evaluation (see
`docs/model_evaluation.md`) showed the YOLOv8 detector dramatically
outperforming Haar cascade on exactly the failure modes that motivated
looking for a replacement: detection rate on non-frontal angles jumped
from Haar's documented 27% (side view) / 46% (looking down) to ~97-100%,
and a background object that reliably fooled Haar cascade fooled the
YOLOv8 detector far less often. See `docs/models/widerface_baseline.md`
and `docs/models/widerface_finetune.md` for how the detector was trained.

## Usage

```
PYTHONPATH=. ./venv/bin/python3 -m src.retailvision.pipeline_demo                      # live camera, default index 0
PYTHONPATH=. ./venv/bin/python3 -m src.retailvision.pipeline_demo --source path/to.mp4 # pre-recorded video file
PYTHONPATH=. ./venv/bin/python3 -m src.retailvision.pipeline_demo --benchmark          # headless, prints average FPS on exit
```

Per the project's testing convention, point `--source` at a pre-recorded
video file first — a deterministic input is much easier to debug against
than a live camera feed — before testing on the live camera.

## Performance

*TBD — re-benchmark with `--benchmark` now that `FaceDetector` runs a
YOLOv8 model instead of Haar cascade. Detection itself now costs a real
model inference per frame (previously a lightweight classical
algorithm), on top of the three classifier calls per detected face — the
old ~61 FPS figure (measured with detection mocked to a fixed box, so it
never actually reflected Haar cascade's own cost either) should not be
assumed to still hold and needs a fresh number.*
