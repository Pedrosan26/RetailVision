# Emotion output consolidation: angry/disgust/fear/sad → "negative"

## Context

`docs/models/emotion_finetune.md` already documents Fear, Disgust, and
Neutral as accepted known limitations of the 7-class FER-2013 classifier.
Retail engineering review of that same weak cluster concluded these
finer-grained negative emotions aren't reliably distinguishable from each
other in practice -- unlike Happy (a big smile) or Surprise (an open
mouth), Angry/Disgust/Fear/Sad don't have one highly-salient visual
feature each; they sit on a shared, subtle spectrum. The proposal: stop
asking the pipeline to distinguish between them and report a single
`negative` label instead. This doc is the data behind that decision,
gathered before making the change, not after.

## FER-2013 held-out test set (lab benchmark)

From `models/emotion/final_report.json`:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| happy | 0.886 | 0.888 | 0.887 | 1774 |
| surprise | 0.804 | 0.823 | 0.813 | 831 |
| disgust | 0.755 | 0.694 | 0.723 | 111 (small sample) |
| neutral | 0.609 | 0.676 | 0.640 | 1233 |
| angry | 0.605 | 0.641 | 0.622 | 958 |
| sad | 0.583 | 0.548 | 0.565 | 1247 |
| fear | 0.602 | 0.523 | 0.560 | 1024 |

Happy and Surprise are clearly separated from the rest; the other five
cluster in a materially weaker 0.56-0.72 F1 band.

## Real-world evaluation (the more relevant signal)

From `models/emotion/real_world_eval_report.json` -- actual recorded
webcam sessions across several conditions (`close_1m`, `medium_2m`,
`far_4m`, `side_view`, `looking_down`), not the FER-2013 lab benchmark:

| Class | Mean real-world accuracy | FER-2013 recall (for comparison) |
|---|---|---|
| neutral | 0.825 | 0.676 |
| happy | 0.807 | 0.888 |
| surprise | 0.604 | 0.823 |
| **angry** | **0.565** | 0.641 |
| sad | 0.342 | 0.548 |
| fear | 0.112 | 0.523 |
| disgust | 0.037 | 0.694 |

Disgust and Fear collapse almost entirely under real-world conditions
(3.7%, 11.2%) despite looking passable on the lab benchmark -- the
largest lab-to-real-world gap of any class by a wide margin. Sad also
degrades sharply (54.8% → 34.2%). **Angry is the outlier of the four**:
it holds up meaningfully better (56.5%), closer to Surprise (60.4%) than
to the other three negative classes. It's included in the consolidation
anyway, as a product decision made after seeing this data, not because
the data alone demanded it -- see "Decision" below.

## Confusion matrix: the five weak classes confuse each other, not Happy/Surprise

![Fine-tuned emotion confusion matrix (normalized)](../../runs/emotion_finetune/emotion/confusion_matrix_normalized.png)

Reading the matrix (columns = true class, rows = predicted class):
Happy and Surprise sit almost entirely on their own diagonal cell with
minimal leakage anywhere else. Angry/Disgust/Fear/Sad/Neutral, by
contrast, leak heavily into *each other* -- e.g. true Angry → predicted
Disgust 20%, true Neutral → predicted Sad 19%, true Sad → predicted Fear
14%. This is a genuine tangled cluster, not five independently-noisy
classes.

## What consolidation actually buys: reconstructed from the same matrix

Merging Angry/Disgust/Fear/Sad into one `negative` bucket and
re-deriving precision/recall against the FER-2013 test set (weighted by
each true class's support):

| Metric | Value |
|---|---|
| Recall of `negative` | **~81.3%** |
| Precision of `negative` | **~86.7%** |
| F1 | **~0.84** |

That's a large jump over any of the four individual classes (best was
Disgust at F1 0.723, on a support of only 111) -- because most of what
was dragging each class down was confusion *with the other three*, which
stops being an error once they're one label.

**Caveat found in the same reconstruction:** ~20% of true `neutral`
frames get misclassified into the negative cluster (mostly as Sad), vs.
5% leakage from true `happy` and 10% from true `surprise`. Since neutral
is presumably the single most common expression in real retail footage,
this means `negative` counts will be somewhat inflated by genuinely
neutral people, not purely by genuinely negative ones. This leakage rate
is derived from the FER-2013 test set specifically; it has not been
separately re-confirmed against the real-world recordings.

## Decision

**Collapse Angry, Disgust, Fear, and Sad into a single `negative` label**
in the pipeline's output. Implemented as a **post-hoc remap of the
existing classifier's predictions**, not a retrain: the underlying
7-class model in `models/emotion/final.pt` is unchanged; `InferencePipeline._classify_emotion()`
(`src/retailvision/inference.py`) takes its top1 prediction and, if it
falls in `{angry, disgust, fear, sad}`, reports `negative` with a
confidence equal to the *summed* probability mass of those four classes
(not the original single-class confidence) -- the correct number for "how
confident is the model that this is one of the negative emotions,"
regardless of which one.

A full retrain on relabeled data (four coarse classes instead of seven)
was considered and rejected for now: the reconstructed post-hoc numbers
above (~81% recall / ~87% precision) already represent a substantial
improvement with zero retraining cost, so retraining is left as a
possible future improvement rather than a blocker, revisit only if the
post-hoc remap's real-world performance turns out to fall short of these
FER-2013-derived estimates.

No downstream schema or server change was needed: `emotion` has always
been a free-form string field (`docs/schema.md`), and the dashboard's
aggregate emotion distribution already buckets by whatever string
appears -- `negative` just shows up as a new bucket automatically, the
same way real `zone_id` values will once zone configuration lands.

## Not affected

- `models/emotion/final.pt` itself -- still the 7-class classifier
  described in `docs/models/emotion_finetune.md`.
- `age_group`/`gender` classification -- unrelated classifiers, untouched.
- The frozen 8-field log schema (`docs/schema.md`) -- `emotion` was
  already a plain string, no shape change.
