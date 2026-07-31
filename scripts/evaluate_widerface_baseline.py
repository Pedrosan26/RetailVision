"""
evaluate_widerface_baseline.py

CLI entry point for baseline evaluation: computes overall mAP@0.5 /
mAP@0.5:0.95 / precision / recall on our held-out test split, recall
against WIDER FACE's official Easy/Medium/Hard ground-truth partition
(restricted to our test split), saves a loss/mAP curve plot, and writes
baseline_report.json. Run this after
scripts/train_widerface_baseline.py has finished.

Usage: PYTHONPATH=scripts ./venv/bin/python3 scripts/evaluate_widerface_baseline.py
"""

import json

from ultralytics import YOLO
from widerface_baseline.constants import MODEL_OUT_DIR, REPORT_PATH, RUNS_DIR, WEIGHTS_PATH, resolve_device
from widerface_baseline.evaluate import load_loss_curves, overall_metrics
from widerface_baseline.official_eval import official_difficulty_recall
from widerface_baseline.plotting import save_loss_curves


def main() -> None:
    """Evaluate the WIDER FACE baseline detector and write the metrics report."""
    device = resolve_device()
    model = YOLO(str(WEIGHTS_PATH))

    print("Computing overall test-split metrics...")
    overall = overall_metrics(model, device)
    print(f"  mAP50={overall['mAP50']} mAP50-95={overall['mAP50_95']} "
          f"precision={overall['precision']} recall={overall['recall']}")

    print("Computing official Easy/Medium/Hard recall (test split only)...")
    difficulty_recall = official_difficulty_recall(model, device)
    for difficulty, stats in difficulty_recall.items():
        print(f"  {difficulty}: recall={stats['recall']} "
              f"({stats['gt_faces']} gt faces across {stats['images_evaluated']} images)")

    results_csv = RUNS_DIR / "widerface" / "results.csv"
    curves = load_loss_curves(results_csv)
    plot_path = MODEL_OUT_DIR / "baseline_loss_curves.png"
    save_loss_curves(curves, plot_path)

    report = {
        "weights": str(WEIGHTS_PATH),
        "overall_test_split_metrics": overall,
        "official_difficulty_recall": difficulty_recall,
        "loss_curve_plot": str(plot_path),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nBaseline report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()