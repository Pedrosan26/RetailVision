"""
evaluate_emotion_finetune.py

CLI entry point for fine-tune evaluation: computes top1/top5 accuracy and
per-class precision/recall/F1 on the held-out FER-2013 test split, saves a
loss/accuracy curve plot, checks recall against the required per-class
thresholds (80% for Happy and Neutral specifically — the two classes
considered commercially relevant for retail, not an aggregate top1 bar),
and writes final_report.json. Run this after scripts/finetune_emotion.py
has finished.

Fear and Disgust are expected weak classes (FER-2013 is noisy for both,
and Disgust is badly underrepresented) and are not held to a threshold —
they're documented as known limitations instead.

If Happy/Neutral fail to clear 80% recall after this fine-tuning
iteration, do not attempt a third automatically — fall back to
pre-trained DeepFace emotion weights and document the decision.

Usage: PYTHONPATH=scripts ./venv/bin/python3 scripts/evaluate_emotion_finetune.py
"""

import json

from emotion_baseline.evaluate import load_loss_curves, per_class_metrics, run_test_predictions, top_k_accuracy
from emotion_baseline.plotting import save_loss_curves
from emotion_finetune.constants import DATA_DIR, MIN_CLASS_RECALL, MODEL_OUT_DIR, REPORT_PATH, RUNS_DIR, WEIGHTS_PATH, resolve_device
from ultralytics import YOLO


def main() -> None:
    """Evaluate the fine-tuned emotion classifier, check per-class thresholds, and write the final report."""
    device = resolve_device()
    model = YOLO(str(WEIGHTS_PATH))

    class_names = sorted(p.name for p in (DATA_DIR / "test").iterdir() if p.is_dir())
    y_true, y_pred = run_test_predictions(model, DATA_DIR)
    class_metrics = per_class_metrics(y_true, y_pred, class_names)
    accuracy = top_k_accuracy(model, DATA_DIR, device)

    results_csv = RUNS_DIR / "emotion" / "results.csv"
    curves = load_loss_curves(results_csv)
    plot_path = MODEL_OUT_DIR / "final_loss_curves.png"
    save_loss_curves(curves, plot_path, title="emotion fine-tune: loss & accuracy curves")

    all_passed = True
    for class_name, threshold in MIN_CLASS_RECALL.items():
        recall = class_metrics[class_name]["recall"]
        passed = recall >= threshold
        all_passed = all_passed and passed
        class_metrics[class_name]["min_recall_threshold"] = threshold
        class_metrics[class_name]["passed_threshold"] = passed

    report = {
        "weights": str(WEIGHTS_PATH),
        "top_k_accuracy": accuracy,
        "min_class_recall_thresholds": MIN_CLASS_RECALL,
        "passed_thresholds": all_passed,
        "per_class_metrics": class_metrics,
        "loss_curve_plot": str(plot_path),
    }
    print(f"top1={accuracy['top1_accuracy']}  top5={accuracy['top5_accuracy']}")
    for class_name, metrics in class_metrics.items():
        print(f"  {class_name}: precision={metrics['precision']} recall={metrics['recall']} f1={metrics['f1']}")
    for class_name, threshold in MIN_CLASS_RECALL.items():
        status = "PASS" if class_metrics[class_name]["passed_threshold"] else "FAIL"
        print(f"  threshold check — {class_name}: recall={class_metrics[class_name]['recall']} "
              f"vs. {threshold} required [{status}]")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nFinal report written to {REPORT_PATH}")
    if not all_passed:
        print(
            "\nHappy and/or Neutral did not clear their required recall threshold. "
            "If this is the second fine-tuning iteration, fall back to pre-trained "
            "DeepFace emotion weights and document the decision instead of running "
            "a third iteration."
        )


if __name__ == "__main__":
    main()