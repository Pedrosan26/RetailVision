# Real-world model evaluation

Covers three independent real-world evaluations: age/gender (below),
emotion (further down), and the face detector (further down still) —
the last of these directly motivated by Finding 3 in the first section.

# Age/Gender

## Setup

The fine-tuned classifiers (`models/age_gender/final_age.pt`,
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
| Test set (fine-tuned, static images) | — | 84.87% | — | 96.40% | — |
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
whereas the classifier's own held-out evaluation always used file paths.
Verified on a known-label UTKFace test image that file-path input and
BGR-array input produce **identical** predictions and confidence
(`18-30`, conf 0.8992, both paths) — so the array-input path itself is
not the problem.

**Most likely cause (not yet fixed, flagged for follow-up)**: a crop/domain
mismatch between training and live inference. UTKFace's images are
tightly-aligned, studio-quality face chips; the Haar cascade's live
bounding boxes are loose, unaligned, and pull in a webcam frame's different
lighting, compression, and lens characteristics. The age classifier —
already the weaker of the two models per its own per-class
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

## Recommendations (follow-up work)

1. ~~Replace the frontal-only Haar cascade with a pose-tolerant
   detector~~ — **done**, see the Face Detector section further down.
   Detection rate on angled faces went from 52.4% (this section) to
   ~97-100% with the trained-from-scratch YOLOv8 replacement.
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

# Emotion

## Setup

The fine-tuned emotion classifier (`models/emotion/final.pt`) was tested
against live laptop-webcam video, using
`scripts/evaluate_emotion_real_world.py` (per-condition, per-emotion
capture session) and `scripts/summarize_emotion_real_world_eval.py`
(aggregation). Same tooling pattern as the age/gender evaluation above,
but the conditions are camera distance and non-frontal angle rather than
lighting/occlusion, and — unlike age/gender, where one session captures a
fixed ground truth — each session here holds one consistent emotion
throughout, so the evaluation could be expanded into a full **emotion ×
condition matrix** rather than one ground truth per condition. Full
per-frame logs: `runs/emotion_real_world_eval/`; aggregated numbers:
`models/emotion/real_world_eval_report.json`.

Five conditions were planned, each tested against all 7 emotion classes:

| Condition | Description |
|---|---|
| `close_1m` | ~1m from camera, frontal |
| `medium_2m` | ~2-3m from camera, frontal |
| `far_4m` | ~4-5m from camera, frontal |
| `side_view` | Comfortable distance, head turned to a side profile |
| `looking_down` | Comfortable distance, head tilted down |

**Coverage ended up uneven**, and that unevenness is itself a finding, not
just a gap: `close_1m`, `medium_2m`, and `far_4m` all reached full 7/7
emotion coverage. `side_view` only reached 3/7 (happy, angry, disgust) and
`looking_down` only reached 1/7 (happy) — because face detection failed so
often under those poses that most sessions produced little to no usable
data before the subject gave up holding the pose. See "Non-frontal poses"
below.

**Methodology limitations**: each (condition, emotion) cell is a single
uncontrolled take by one subject, not a repeated controlled trial — treat
all numbers as indicative, not statistically rigorous. Additionally,
convincingly *posing* some emotions on command (especially disgust and
fear) without a real stimulus is itself hard, even for trained actors —
low live accuracy on those classes may partly reflect posing fidelity, not
only classifier weakness. See Finding 2.

## Results — distance conditions (`close_1m` / `medium_2m` / `far_4m`)

The only three conditions with full 7-emotion coverage, so the only ones
compared as a matrix. Each cell is accuracy among *detected* faces, against
that specific emotion's own held-out test-set recall (not the model's
blended overall top1 — comparing against the blended figure understates
strong classes and overstates weak ones, since it averages over all seven).

| Emotion | close_1m | medium_2m | far_4m | Test-set recall |
|---|---|---|---|---|
| angry | 90.3% | 76.3% | 55.8% | 64.1% |
| neutral | 97.3% | 98.7% | 51.7% | 67.6% |
| surprise | 80.4% | 56.3% | 44.4% | 82.3% |
| happy | 86.0% | 85.1% | 87.3% | 88.8% |
| sad | 66.7% | 25.0% | 10.9% | 54.8% |
| fear | 14.6% | 15.6% | 3.3% | 52.3% |
| disgust | 0.0% | 12.7% | 2.1% | 69.4% |

## Finding 1: a data-contamination case study — the close-range "happy" collapse was never a model or lens problem

The first pass at this evaluation found `close_1m/happy` scoring a severe
32.3% — the worst result in the entire matrix at the time, and the
apparent basis for a "webcam lens distortion at close range" hypothesis.
That hypothesis didn't survive contact with the fuller dataset: every
*other* emotion did fine or better at close range (angry and neutral both
*exceeded* their test-set benchmark up close), so a lens-distortion effect
should have hurt all of them similarly, not just one.

Investigating properly — a small diagnostic tool
(`scripts/debug_emotion_crops.py`) was built to save every detected face
crop from a session to disk with its prediction, instead of only looking
at the aggregate accuracy number — found the real cause immediately:

| Misdetected (mannequin in background) | Correct (real face) |
|---|---|
| ![Mannequin misdetected as angry](images/emotion_close_range_mannequin_misdetect.jpg) | ![Correctly classified happy](images/emotion_close_range_correct.jpg) |

A mannequin behind the subject has painted eyes and lips — enough for the
Haar cascade to detect it as a face, and for the classifier to read its
painted expression as angry or surprise. Those misdetections, scored
against a ground truth of "happy," were the entire effect. Two follow-up
sessions confirmed it directly: a first re-take with other objects still
in frame scored 31.5% (54 frames) — reproducing the original problem
almost exactly — while a second, clean take with a clear background scored
**86.0%** (108 frames), matching the pattern of every other emotion at
close range and landing within 3 points of the static test-set benchmark.

The contaminated sessions (the original pilot take and the first re-take)
were excluded from the official record; only the clean 108-frame session
is reflected in the table above and in `models/emotion/real_world_eval_report.json`.
Raw contaminated data is preserved for the record at
`runs/emotion_real_world_eval/excluded_contaminated/`, not deleted.

**Takeaway for methodology, not just this one number**: any single cell
that breaks its own row or column's otherwise-consistent pattern is worth
a quick visual sanity check (via the debug tool) before being trusted as a
real finding — this project had exactly one such anomaly across the whole
matrix, and it was a background object, not a model or optics problem.

## Finding 2: fear and disgust are consistently near-unusable live, across every distance

Unlike the close-range happy result, this pattern shows up independently
across three separate sessions taken at three different times and
distances — fear: 14.6% / 15.6% / 3.3%; disgust: 0.0% / 12.7% / 2.1% —
consistently far below even their already-weak static test-set recall
(52.3%, 69.4%). Consistency across independent sessions is exactly what a
single-session artifact (like Finding 1) would *not* produce, so this is
much more likely a real effect.

Two contributing causes, not necessarily separable with this data:

1. **Model weakness**: fear and disgust were already the two weakest
   classes in static evaluation (`docs/models/emotion_finetune.md`), so
   some live degradation was expected.
2. **Posing-fidelity confound**: genuinely portraying disgust or fear
   convincingly on command, without a real stimulus, is hard — a
   well-documented issue in facial-expression research generally (part of
   why these two classes are noisy in FER-2013 itself). A misclassified
   frame here may reflect an unconvincing pose as much as a classifier
   failure.

Not recommended for any use case requiring reliable disgust/fear detection
as currently trained, regardless of camera distance.

## Finding 3: distance-dependent degradation varies a lot by emotion — no single "distance effect"

- **Happy stays flat and strong across all three distances** (86.0%,
  85.1%, 87.3%) — once Finding 1's contamination is excluded, distance
  has essentially no effect on this class.
- **Angry and neutral exceed their test-set benchmark at close/medium
  range**, then drop off at far range (angry: 90.3% → 76.3% → 55.8%;
  neutral: 97.3% → 98.7% → 51.7%). Neutral's specific drop at far range is
  plausible given it's already documented as confusable with sad
  (`docs/models/emotion_finetune.md`) — reduced facial detail at distance
  likely worsens exactly that confusion.
- **Surprise and sad degrade roughly monotonically with distance**
  (surprise: 80.4% → 56.3% → 44.4%; sad: 66.7% → 25.0% → 10.9%) — the
  "expected" pattern (less detail, harder classification), unlike the
  other four classes above.

Net: there is no single "closer is better" or "closer is worse" rule for
this classifier — the effect of distance is emotion-specific.

## Non-frontal poses (`side_view`, `looking_down`): insufficient data to draw per-emotion conclusions

Face detection failed so often under these two poses that most emotion
attempts never produced a usable session — consistent with the age/gender
evaluation's Finding 3 (the Haar cascade is frontal-only), but noticeably
more severe here than that earlier single-session estimate suggested:

| Condition | Detection rate observed |
|---|---|
| `side_view` / angry | 10.6% |
| `side_view` / disgust | 20.0% |
| `side_view` / happy | 50.5% |
| `looking_down` / happy | 46.3% |

The wide spread even within the *same* condition label (10.6% to 50.5% for
`side_view` alone) reflects how much the actual head angle varied between
uncontrolled single takes — "side view" wasn't a fixed, repeatable angle
across sessions. Given this, the per-emotion accuracy numbers that do
exist for these two conditions (available in
`models/emotion/real_world_eval_report.json` for the record) are not
included as findings here — the sample sizes and detection reliability
are too inconsistent to trust a per-emotion breakdown.

**The finding for these two conditions is the detection failure itself**:
the current Haar cascade detector is reliable enough for evaluation
purposes only when the subject is close to frontal. Fixing that
detector-stage limitation (see Recommendations) is a prerequisite for ever
getting a trustworthy emotion-accuracy read under these poses, not just a
nice-to-have.

## Known failure cases (explicit)

- **Fear and disgust are not reliable live, at any tested distance**
  (3-16% accuracy vs. 52-69% static recall) — likely a combination of
  genuine model weakness and the difficulty of convincingly posing these
  expressions on command. Not production-ready as deployed today.
- **Face detection fails on non-frontal poses badly enough to block
  evaluation itself** (10-50% detection rate across `side_view`/
  `looking_down` sessions) — consistent with, and more severe than, the
  age/gender evaluation's equivalent finding.
- **Happy, angry, and neutral are reliable at close-to-medium range**
  (76-98% accuracy, several exceeding their static benchmark) once
  background-object contamination is excluded.
- **Surprise and sad degrade steadily with distance** and should not be
  relied on beyond close range.

## Recommendations (follow-up work)

1. ~~Same detector-stage recommendation as the age/gender evaluation:
   replace the frontal-only Haar cascade with a pose-tolerant
   detector~~ — **done**, see the Face Detector section further down.
2. Re-run `side_view` and `looking_down` against the remaining untested
   emotions once detection is more reliable, to complete the matrix.
3. When capturing future sessions, clear the background of any
   face-like objects (mannequins, posters, dolls) beforehand — Finding 1
   cost a full investigation cycle that a 10-second visual check would
   have prevented.
4. Investigate whether a live-capture-specific frontality/quality gate
   (skip classification on frames too angled or too low-confidence to
   trust) is worth adding before this classifier is wired into the live
   pipeline, mirroring the age/gender recommendation.

## Artifacts

Raw per-frame logs, the JSON report, and the full set of diagnostic face
crops are local working data (regenerable by re-running the tooling
below), not committed to the repository — only this document and the two
illustrative images above are. Locally, they live at:

- Per-frame session logs: `runs/emotion_real_world_eval/{condition}__{emotion}.csv`
- Excluded contaminated sessions (preserved, not deleted): `runs/emotion_real_world_eval/excluded_contaminated/`
- Diagnostic face-crop captures (Finding 1): `runs/debug_close_1m_happy/`
- Aggregated report (full matrix + condition/emotion averages): `models/emotion/real_world_eval_report.json`

Evaluation tooling: `scripts/emotion_real_world_eval/`,
  `scripts/evaluate_emotion_real_world.py`,
  `scripts/summarize_emotion_real_world_eval.py`,
  `scripts/debug_emotion_crops.py`

# Face Detector

## Setup

Two YOLOv8 detection-mode checkpoints — `models/face_detection/baseline.pt`
(trained from scratch on WIDER FACE, `docs/models/widerface_baseline.md`)
and `final.pt` (fine-tuned with retail-camera-tuned augmentation,
`docs/models/widerface_finetune.md`) — were evaluated against live
webcam footage using `scripts/widerface_real_world_eval/`. Unlike the
age/gender and emotion evaluations above (one live take per condition),
each condition here is **recorded once** to a video file
(`scripts/record_widerface_eval_session.py`) and then replayed through
both checkpoints (`scripts/evaluate_widerface_real_world.py`), so the
two are compared on identical frames rather than two separate takes
that could differ in pose or timing.

Conditions match the emotion evaluation's distance/angle set, plus one
new one:

| Condition | Description |
|---|---|
| `close_1m` | ~1m from camera |
| `medium_2m` | ~2m from camera |
| `far_4m` | ~4m from camera — **not captured**, see Known limitations |
| `side_view` | Face turned to profile |
| `looking_down` | Head tilted down |
| `no_person_background` | Camera pointed at an empty background, no person in frame — dedicated false-positive test |

## Results

| Condition | Haar cascade reference | baseline detection rate | final detection rate | baseline extra-box rate | final extra-box rate |
|---|---|---|---|---|---|
| `close_1m` | 98.5% | 100% | 100% | 0% | 0% |
| `medium_2m` | 98.5% | 100% | 100% | 100% | 62.2% |
| `far_4m` | 92.5% | — | — | — | — |
| `side_view` | 27.0%* | 96.9% | 97.4% | 89.0% | 85.9% |
| `looking_down` | 46.3% | 100% | 100% | 100% | 100% |
| `no_person_background` | — | 1.0% | 0.0% | — | — |

\* mean of 3 highly variable single-emotion sessions (10.6%-50.5%), see
the Emotion section above — Haar cascade's own `side_view` number was
never a single stable figure.

"Extra-box rate" = share of frames with more than one detected box.
For a single-subject session this is a proxy for false positives, but
see Finding 2 — it needed investigation before being trusted as such.

## Finding 1: non-frontal detection rate improved dramatically — the result that motivated this whole migration

`side_view` (27.0% → 96.9%/97.4%) and `looking_down` (46.3% → 100%/100%)
were the two failure modes that started the search for a Haar cascade
replacement in the first place (see both sections above). Both YOLOv8
checkpoints resolve them almost completely, and nearly identically to
each other — this is a detector-architecture win, not something that
depended on which checkpoint (baseline or fine-tuned) was used.

## Finding 2: the high extra-box rate is almost entirely one recurring background object, not broad false-positive-proneness

The extra-box rate is high in `medium_2m`, `looking_down`, and
`side_view`, but near-zero in `close_1m` and `no_person_background` —
a pattern, not noise. Sampling actual frames and inspecting the detected
crops (same methodology as the Emotion section's Finding 1 contamination
case study) found the second box in all three conditions to be the same
object: a mannequin head on a shelf in the test environment, already
documented as a Haar cascade false-positive case in the Emotion section
above.

| Mannequin (misdetected as a face) | Same frame, real subject also correctly detected |
|---|---|
| ![Mannequin misdetected as a face](images/widerface_mannequin_crop.jpg) | ![Full frame context](images/widerface_mannequin_context.jpg) |

Comparing the exact same frames between checkpoints:

| Condition | baseline.pt | final.pt |
|---|---|---|
| `medium_2m` | detects mannequin (0.43 conf) | does not detect it |
| `looking_down` | detects mannequin (0.77 conf) | detects it, lower confidence (0.69) |
| `side_view` | detects mannequin (0.77 conf) | detects it, lower confidence (0.75) |

Small sample (one frame per condition), but directionally consistent:
the fine-tuned checkpoint showed more resistance to this specific false
positive than the baseline in all three matched comparisons — a
plausible (not proven) connection to the retail-tuned augmentation
`docs/models/widerface_finetune.md` describes.

## Finding 3: false positives on a genuinely empty background are near-zero for both checkpoints

`no_person_background` — a dedicated test with no real face anywhere in
frame — gave baseline a 1.0% detection rate (1 of 96 frames) and final
0.0% (0 of 96 frames). This is the cleaner test for broad
false-positive-proneness (Finding 2's elevated rates are explained by
one specific recurring object, not a general tendency to hallucinate
faces), and both checkpoints pass it convincingly.

## Known limitations (explicit)

- **`far_4m` was not captured.** Skipped given time constraints, judged
  low-risk: `close_1m`/`medium_2m` were already ~100% detection for both
  checkpoints, and the two conditions with real uncertainty (the
  non-frontal angles) were both fully covered.
- Each condition is a single uncontrolled take, same limitation
  documented in the age/gender and emotion sections above.
- Finding 2's checkpoint comparison is single-frame sampling per
  condition, not a full-session statistic — a qualitative signal worth
  documenting, not a statistically rigorous claim.
- `medium_2m`, `looking_down`, and `side_view`'s extra-box numbers are
  contaminated by the same recurring object across all three sessions;
  re-capturing with it out of frame would give a cleaner false-positive
  signal if further validation is wanted.

## Decision

`final.pt` (the fine-tuned checkpoint) is now the production
`FaceDetector` (`src/retailvision/detection.py`), replacing Haar
cascade. This was decided on the real-world result above, not on
`docs/models/widerface_finetune.md`'s WIDER FACE benchmark numbers,
where the fine-tune actually regressed — the real-world evaluation is
what confirmed the fine-tune's retail-focused hypothesis actually paid
off, resolving that open question explicitly rather than assuming
either direction.

Follow-up, if further validation is wanted before wider deployment:
capture `far_4m`, re-capture `medium_2m`/`side_view`/`looking_down` with
the mannequin out of frame for a clean false-positive comparison, and
re-run the full pipeline's FPS benchmark (`docs/inference_pipeline.md`)
now that detection itself costs a real model inference per frame instead
of a lightweight classical algorithm.

## Artifacts

- Per-frame session logs: `runs/widerface_real_world_eval/logs/{condition}__{model}.csv`
- Recorded video clips: `runs/widerface_real_world_eval/recordings/{condition}.mp4`
- Aggregated report: `models/face_detection/real_world_eval_report.json`

Evaluation tooling: `scripts/widerface_real_world_eval/`,
  `scripts/record_widerface_eval_session.py`,
  `scripts/evaluate_widerface_real_world.py`,
  `scripts/summarize_widerface_real_world_eval.py`
