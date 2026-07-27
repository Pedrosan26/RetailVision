"""
evaluate_emotion_baseline.py

CLI entry point for baseline evaluation: computes top1/top5 accuracy and
per-class precision/recall/F1 on the held-out FER-2013 test split, saves a
loss/accuracy curve plot, and writes baseline_report.json. Run this after
scripts/train_emotion_baseline.py has finished.

FER-2013 is known to be noisy, particularly for Fear and Disgust (the
latter is also badly underrepresented — 392 of ~23,000 train images);
underperformance on these two classes is an expected, documented
limitation, not a training failure.

Usage: PYTHONPATH=scripts ./venv/bin/python3 scripts/evaluate_emotion_baseline.py
"""

import json

from emotion_baseline.constants import DATA_DIR, MODEL_OUT_DIR, REPORT_PATH, RUNS_DIR, WEIGHTS_PATH, resolve_device
from emotion_baseline.evaluate import load_loss_curves, per_class_metrics, run_test_predictions, top_k_accuracy
from emotion_baseline.plotting import save_loss_curves
from ultralytics import YOLO


def main() -> None:
    """Evaluate the emotion baseline and write the metrics report."""
    device = resolve_device()
    model = YOLO(str(WEIGHTS_PATH))

    class_names = sorted(p.name for p in (DATA_DIR / "test").iterdir() if p.is_dir())
    y_true, y_pred = run_test_predictions(model, DATA_DIR)
    class_metrics = per_class_metrics(y_true, y_pred, class_names)
    accuracy = top_k_accuracy(model, DATA_DIR, device)

    results_csv = RUNS_DIR / "emotion" / "results.csv"
    curves = load_loss_curves(results_csv)
    plot_path = MODEL_OUT_DIR / "baseline_loss_curves.png"
    save_loss_curves(curves, plot_path)

    report = {
        "weights": str(WEIGHTS_PATH),
        "top_k_accuracy": accuracy,
        "per_class_metrics": class_metrics,
        "loss_curve_plot": str(plot_path),
    }
    print(f"top1={accuracy['top1_accuracy']}  top5={accuracy['top5_accuracy']}")
    for class_name, metrics in class_metrics.items():
        print(f"  {class_name}: precision={metrics['precision']} recall={metrics['recall']} f1={metrics['f1']}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nBaseline report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()