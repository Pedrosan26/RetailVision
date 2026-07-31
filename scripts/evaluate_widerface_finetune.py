"""
evaluate_widerface_finetune.py

CLI entry point for fine-tune evaluation: computes overall mAP@0.5 /
mAP@0.5:0.95 / precision / recall and official Easy/Medium/Hard recall
on the held-out test split (reusing widerface_baseline's evaluation
helpers, which are generic across any trained checkpoint), compares
mAP@0.5 against the baseline's own report, checks the minimum acceptable
Hard-difficulty recall threshold, saves a loss/mAP curve plot, and
writes final_report.json. Run this after scripts/finetune_widerface.py
has finished.

Usage: PYTHONPATH=scripts ./venv/bin/python3 scripts/evaluate_widerface_finetune.py
"""

import json

from ultralytics import YOLO
from widerface_baseline.constants import REPORT_PATH as BASELINE_REPORT_PATH
from widerface_baseline.evaluate import load_loss_curves, overall_metrics
from widerface_baseline.official_eval import official_difficulty_recall
from widerface_baseline.plotting import save_loss_curves
from widerface_finetune.constants import MIN_HARD_RECALL, MODEL_OUT_DIR, REPORT_PATH, RUNS_DIR, WEIGHTS_PATH, resolve_device


def main() -> None:
    """Evaluate the fine-tuned WIDER FACE detector and write the metrics report."""
    device = resolve_device()
    model = YOLO(str(WEIGHTS_PATH))

    print("Computing overall test-split metrics...")
    overall = overall_metrics(model, device)
    print(f"  mAP50={overall['mAP50']} mAP50-95={overall['mAP50_95']} "
          f"precision={overall['precision']} recall={overall['recall']}")

    baseline_map50 = None
    if BASELINE_REPORT_PATH.exists():
        baseline_report = json.loads(BASELINE_REPORT_PATH.read_text())
        baseline_map50 = baseline_report["overall_test_split_metrics"]["mAP50"]
        delta = overall["mAP50"] - baseline_map50
        print(f"  vs. baseline mAP50 ({baseline_map50}): {delta:+.4f}")

    print("Computing official Easy/Medium/Hard recall (test split only)...")
    difficulty_recall = official_difficulty_recall(model, device)
    for difficulty, stats in difficulty_recall.items():
        print(f"  {difficulty}: recall={stats['recall']} "
              f"({stats['gt_faces']} gt faces across {stats['images_evaluated']} images)")

    hard_recall = difficulty_recall["hard"]["recall"]
    passed_threshold = hard_recall is not None and hard_recall >= MIN_HARD_RECALL
    print(f"  Hard recall threshold ({MIN_HARD_RECALL}): {'PASS' if passed_threshold else 'FAIL'} ({hard_recall})")

    results_csv = RUNS_DIR / "widerface" / "results.csv"
    curves = load_loss_curves(results_csv)
    plot_path = MODEL_OUT_DIR / "final_loss_curves.png"
    save_loss_curves(curves, plot_path, title="widerface fine-tune: loss & mAP curves")

    report = {
        "weights": str(WEIGHTS_PATH),
        "overall_test_split_metrics": overall,
        "baseline_mAP50": baseline_map50,
        "mAP50_delta_vs_baseline": (overall["mAP50"] - baseline_map50) if baseline_map50 is not None else None,
        "official_difficulty_recall": difficulty_recall,
        "min_hard_recall_threshold": MIN_HARD_RECALL,
        "hard_recall_threshold_passed": passed_threshold,
        "loss_curve_plot": str(plot_path),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nFine-tune report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
