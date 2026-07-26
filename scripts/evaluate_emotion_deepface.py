"""
evaluate_emotion_deepface.py

CLI entry point for RET-9's DeepFace fallback: evaluates DeepFace's
pre-trained emotion model (itself trained on FER-2013) on the exact same
held-out FER-2013 test split our custom yolov8n-cls classifier was
evaluated on, for a direct, apples-to-apples comparison. Per RET-9's
acceptance criteria, this fallback was triggered because two fine-tuning
iterations of our own classifier both failed the 80% Neutral recall
threshold, with confusion-matrix evidence pointing to a structural
neutral/sad overlap rather than a fixable tuning problem — see
docs/models/emotion_finetune.md for the full iteration history.

`detector_backend="skip"` and `enforce_detection=False` are used since
FER-2013 images are already tightly-cropped single-face chips; DeepFace's
own face detection stage would be redundant and could only hurt accuracy
by re-cropping an already-correct crop.

Usage: PYTHONPATH=scripts ./venv/bin/python3 scripts/evaluate_emotion_deepface.py
"""

import json
from pathlib import Path

from deepface import DeepFace
from emotion_baseline.evaluate import per_class_metrics
from emotion_finetune.constants import DATA_DIR, MIN_CLASS_RECALL, MODEL_OUT_DIR

REPORT_PATH = MODEL_OUT_DIR / "deepface_report.json"


def run_test_predictions(data_dir: Path) -> tuple[list[str], list[str], list[str]]:
    """Run DeepFace's emotion model over every test image and return (y_true, y_pred, class_names)."""
    class_names = sorted(p.name for p in (data_dir / "test").iterdir() if p.is_dir())
    y_true: list[str] = []
    y_pred: list[str] = []
    for class_name in class_names:
        image_paths = sorted((data_dir / "test" / class_name).iterdir())
        for image_path in image_paths:
            result = DeepFace.analyze(
                img_path=str(image_path),
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="skip",
                silent=True,
            )[0]
            y_true.append(class_name)
            y_pred.append(result["dominant_emotion"])
    return y_true, y_pred, class_names


def main() -> None:
    """Evaluate DeepFace's pre-trained emotion model and write the RET-9 fallback comparison report."""
    print("Running DeepFace emotion model on the FER-2013 test split...")
    y_true, y_pred, class_names = run_test_predictions(DATA_DIR)
    class_metrics = per_class_metrics(y_true, y_pred, class_names)
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

    all_passed = True
    for class_name, threshold in MIN_CLASS_RECALL.items():
        recall = class_metrics[class_name]["recall"]
        passed = recall >= threshold
        all_passed = all_passed and passed
        class_metrics[class_name]["min_recall_threshold"] = threshold
        class_metrics[class_name]["passed_threshold"] = passed

    report = {
        "model": "DeepFace facial_expression_model (pre-trained)",
        "accuracy": round(accuracy, 4),
        "min_class_recall_thresholds": MIN_CLASS_RECALL,
        "passed_thresholds": all_passed,
        "per_class_metrics": class_metrics,
    }
    print(f"accuracy={report['accuracy']}")
    for class_name, metrics in class_metrics.items():
        print(f"  {class_name}: precision={metrics['precision']} recall={metrics['recall']} f1={metrics['f1']}")
    for class_name, threshold in MIN_CLASS_RECALL.items():
        status = "PASS" if class_metrics[class_name]["passed_threshold"] else "FAIL"
        print(f"  threshold check — {class_name}: recall={class_metrics[class_name]['recall']} "
              f"vs. {threshold} required [{status}]")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nDeepFace comparison report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()