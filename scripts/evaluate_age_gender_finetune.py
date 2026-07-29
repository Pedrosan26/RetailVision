"""
evaluate_age_gender_finetune.py

CLI entry point for fine-tune evaluation: for each fine-tuned model (age,
gender), computes top1/top5 accuracy and per-class precision/recall/F1 on
the held-out UTKFace test split, saves a loss/accuracy curve plot, checks
top1 accuracy against the minimum required thresholds (75% age, 85%
gender), and writes a combined final_report.json. Run this after
scripts/finetune_age_gender.py has finished.

If either task fails its threshold after two fine-tuning iterations, do not
attempt a third automatically — fall back to pre-trained DeepFace weights
for that task and document the decision (why fine-tuning didn't clear the
bar, what DeepFace provides instead) rather than looping on hyperparameters
indefinitely.

Usage: PYTHONPATH=scripts ./venv/bin/python3 scripts/evaluate_age_gender_finetune.py
"""

import json

from age_gender_baseline.evaluate import load_loss_curves, per_class_metrics, run_test_predictions, top_k_accuracy
from age_gender_baseline.plotting import save_loss_curves
from age_gender_finetune.constants import MIN_ACCURACY, MODEL_OUT_DIR, REPORT_PATH, RUNS_DIR, TASKS, resolve_device
from ultralytics import YOLO


def main() -> None:
    """Evaluate both fine-tuned models, check thresholds, and write the final report."""
    device = resolve_device()
    report: dict = {}
    all_passed = True

    for task_name, data_dir in TASKS.items():
        print(f"\n=== Evaluating fine-tuned model: {task_name} ===")
        weights_path = MODEL_OUT_DIR / f"final_{task_name}.pt"
        model = YOLO(str(weights_path))

        class_names = sorted(p.name for p in (data_dir / "test").iterdir() if p.is_dir())
        y_true, y_pred = run_test_predictions(model, data_dir)
        class_metrics = per_class_metrics(y_true, y_pred, class_names)
        accuracy = top_k_accuracy(model, data_dir, device)

        results_csv = RUNS_DIR / task_name / "results.csv"
        curves = load_loss_curves(results_csv)
        plot_path = MODEL_OUT_DIR / f"final_{task_name}_loss_curves.png"
        save_loss_curves(curves, task_name, plot_path)

        threshold = MIN_ACCURACY[task_name]
        passed = accuracy["top1_accuracy"] >= threshold
        all_passed = all_passed and passed

        report[task_name] = {
            "weights": str(weights_path),
            "top_k_accuracy": accuracy,
            "min_accuracy_threshold": threshold,
            "passed_threshold": passed,
            "per_class_metrics": class_metrics,
            "loss_curve_plot": str(plot_path),
        }
        status = "PASS" if passed else "FAIL"
        print(f"  top1={accuracy['top1_accuracy']}  top5={accuracy['top5_accuracy']}  "
              f"threshold={threshold}  [{status}]")
        for class_name, metrics in class_metrics.items():
            print(f"  {class_name}: precision={metrics['precision']} recall={metrics['recall']} f1={metrics['f1']}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nFinal report written to {REPORT_PATH}")
    if not all_passed:
        print(
            "\nOne or more tasks did not clear their required accuracy threshold. "
            "If this is the second fine-tuning iteration for that task, fall back "
            "to pre-trained DeepFace weights and document the decision instead of "
            "running a third iteration."
        )


if __name__ == "__main__":
    main()