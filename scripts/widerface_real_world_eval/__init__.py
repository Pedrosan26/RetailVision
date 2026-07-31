"""
widerface_real_world_eval

Live-camera real-world evaluation of the two YOLOv8 face detectors
(baseline.pt, final.pt) against the Haar cascade FaceDetector's already-
documented real-world detection rates (docs/model_evaluation.md). Each
condition is recorded once to a video file and then replayed through
both checkpoints, so they're compared on identical frames rather than
two separate live takes that could differ in pose/timing.
"""
