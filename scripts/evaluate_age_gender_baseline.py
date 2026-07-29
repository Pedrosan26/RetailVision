"""
evaluate_age_gender_baseline.py

CLI entry point for baseline evaluation: for each trained baseline (age,
gender), computes top1/top5 accuracy and per-class precision/recall/F1 on
the held-out UTKFace test split, saves a loss/accuracy curve plot, and
writes a combined baseline_report.json. Run this after
scripts/train_age_gender_baseline.py has finished.

Usage: ./venv/bin/python3 scripts/evaluate_age_gender_baseline.py
"""

import json

from age_gender_baseline.constants import MODEL_OUT_DIR, REPORT_PATH, RUNS_DIR, TASKS, resolve_device
from age_gender_baseline.evaluate import load_loss_curves, per_class_metrics, run_test_predictions, top_k_accuracy
from age_gender_baseline.plotting import save_loss_curves
from ultralytics import YOLO


def main() -> None:
    """Evaluate both baselines and write the combined metrics report."""
    device = resolve_device()
    report: dict = {}

    for task_name, data_dir in TASKS.items():
        print(f"\n=== Evaluating baseline: {task_name} ===")
        weights_path = MODEL_OUT_DIR / f"baseline_{task_name}.pt"
        model = YOLO(str(weights_path))

        class_names = sorted(p.name for p in (data_dir / "test").iterdir() if p.is_dir())
        y_true, y_pred = run_test_predictions(model, data_dir)
        class_metrics = per_class_metrics(y_true, y_pred, class_names)
        accuracy = top_k_accuracy(model, data_dir, device)

        results_csv = RUNS_DIR / task_name / "results.csv"
        curves = load_loss_curves(results_csv)
        plot_path = MODEL_OUT_DIR / f"baseline_{task_name}_loss_curves.png"
        save_loss_curves(curves, task_name, plot_path)

        report[task_name] = {
            "weights": str(weights_path),
            "top_k_accuracy": accuracy,
            "per_class_metrics": class_metrics,
            "loss_curve_plot": str(plot_path),
        }
        print(f"  top1={accuracy['top1_accuracy']}  top5={accuracy['top5_accuracy']}")
        for class_name, metrics in class_metrics.items():
            print(f"  {class_name}: precision={metrics['precision']} recall={metrics['recall']} f1={metrics['f1']}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nBaseline report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
