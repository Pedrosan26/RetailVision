# Real-world model evaluation (RV-006)

## Setup

The RV-005 fine-tuned classifiers (`models/age_gender/final_age.pt`,
`final_gender.pt`) were tested against live laptop-webcam video rather than
static dataset images, using `scripts/evaluate_real_world.py` (per-condition
capture session, one subject) and `scripts/summarize_real_world_eval.py`
(aggregation). Each session runs the existing `FaceDetector` (Haar cascade)
on every frame, classifies any detected face, and logs the result against a
manually-supplied ground truth. Full per-frame logs: `runs/real_world_eval/`;
aggregated numbers: `models/age_gender/real_world_eval_report.json`.

Four conditions were captured, one subject, one take each:

| Condition | Description |
|---|---|
| `normal_light` | Normal room lighting, frontal |
| `low_light` | Lights off/dimmed, frontal |
| `turned_away` | Body turned away / face partially out of frame |
| `angled_45` | Face turned to roughly a 45° profile |

**Methodology limitation**: each condition is a single uncontrolled take by
one subject, not repeated controlled trials — treat the numbers below as
indicative, not statistically rigorous. `normal_light` has far more frames
than the others (1,021 vs. ~150-450) simply because that session ran longer.

## Results

| Condition | Face detection rate | Age accuracy | Age vs. test set | Gender accuracy | Gender vs. test set |
|---|---|---|---|---|---|
| Test set (RV-005, static images) | — | 84.87% | — | 96.40% | — |
| `normal_light` | 79.5% | 25.49% | **-59.4 pts** | 91.13% | -5.3 pts |
| `low_light` | 95.9% | 34.04% | -50.8 pts | 100% | +3.6 pts |
| `turned_away` | 81.3% | 40.17% | -44.7 pts | 97.86% | +1.5 pts |
| `angled_45` | 52.4% | 29.96% | -54.9 pts | 95.78% | +0.6 pts |

## Finding 1: gender classification generalizes well to live camera

Gender accuracy stays close to (and in three of four conditions, above) the
static test-set number of 96.4%, across lighting, angle, and partial-frame
conditions. This is the one part of the pipeline validated as
production-viable as currently trained.

## Finding 2: age classification degrades severely — in every condition, including the best one

Age accuracy collapses to 25-40% in *all four* conditions, including
`normal_light` — the condition that should be easiest (frontal, well-lit)
and was expected to be the closest to the test-set number. Instead it's the
*worst* of the four (25.49%). Since the degradation is roughly uniform
across lighting and angle rather than getting worse as conditions get
harder, lighting/angle are not the primary cause.

Predictions are also **confidently wrong**, not uncertain: across every
condition, predictions skew heavily toward `0-17` regardless of the
subject's true age bracket (`18-30` or `31-50`), often at >0.8 confidence.
This is a stable bias, not noise.

**Investigated and ruled out as the cause**: a live-capture-specific code
bug in how frames are fed to the model. `classify_face()` passes raw BGR
numpy arrays (from `cv2.VideoCapture`) directly to `model.predict()`,
whereas RV-004/RV-005 evaluation always used file paths. Verified on a
known-label UTKFace test image that file-path input and BGR-array input
produce **identical** predictions and confidence (`18-30`, conf 0.8992,
both paths) — so the array-input path itself is not the problem.

**Most likely cause (not yet fixed, flagged for follow-up)**: a crop/domain
mismatch between training and live inference. UTKFace's images are
tightly-aligned, studio-quality face chips; the Haar cascade's live
bounding boxes are loose, unaligned, and pull in a webcam frame's different
lighting, compression, and lens characteristics. The age classifier —
already the weaker of the two models per RV-004/RV-005's per-class
breakdown — appears far more sensitive to this shift than the gender
classifier.

## Finding 3: face detection itself fails on non-frontal faces

`angled_45` drops to a 52.4% face-detection rate, well below the other
three conditions (79-96%). This happens *before* classification runs — for
roughly half the frames at a ~45° angle, no face box is produced at all, so
age/gender predictions are never attempted.

This was initially suspected to be caused by background clutter in the test
environment, but investigation ruled that out: `haarcascade_frontalface_default.xml`
is trained specifically on frontal faces and is a known, expected limitation
of Haar-cascade detection — it degrades past ~15-20° of yaw and commonly
fails outright beyond ~30-45°, independent of background content.

`low_light`'s detection rate (95.9%) being higher than `normal_light`'s
(79.5%) is counterintuitive and likely reflects session-specific framing/
distance differences rather than a real lighting effect — another instance
of the single-take methodology limitation above.

## Known failure cases (explicit)

- **Faces beyond ~45° yaw are frequently not detected at all** by the
  current Haar cascade stage (detection rate 52% at `angled_45` vs. 80%
  frontal) — a detector-stage limitation, not a classifier weakness.
- **Age classification is not reliable on live camera input in its current
  form**, regardless of lighting or angle: 25-40% accuracy live vs. 84.9%
  on the held-out test set, with a consistent bias toward predicting the
  youngest age bracket at high confidence. Not production-ready as
  deployed today.
- **Gender classification is reliable on live camera input**: 91-100%
  across all four tested conditions, consistent with its 96.4% test-set
  accuracy.

## Recommendations (follow-up work, out of scope for this ticket)

1. Replace the frontal-only Haar cascade with a pose-tolerant detector
   (`cv2.FaceDetectorYN`/YuNet is already available in the pinned OpenCV
   build, no new dependency) to fix the detection-stage failure on angled
   faces.
2. Add a frontality gate before classification (skip demographic
   classification on frames too angled to trust, while still counting the
   person for foot-traffic purposes) rather than assuming every detected
   face is classification-ready.
3. Investigate the age classifier's crop/domain sensitivity specifically —
   e.g. normalizing live face crops to match UTKFace's alignment/margin
   conventions, or augmenting training data with looser, detector-style
   crops — before relying on it in a live pipeline.

## Artifacts

- Per-frame session logs: `runs/real_world_eval/{condition}.csv`
- Aggregated report: `models/age_gender/real_world_eval_report.json`
- Evaluation tooling: `scripts/real_world_eval/`, `scripts/evaluate_real_world.py`,
  `scripts/summarize_real_world_eval.py`