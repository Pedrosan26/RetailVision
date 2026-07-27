"""
summarize_emotion_real_world_eval.py

CLI entry point for the emotion classifier's real-world evaluation
aggregation: reads every (condition, emotion) CSV logged by
scripts/evaluate_emotion_real_world.py, computes face-detection rate and
emotion accuracy for each combination, and writes a full condition x
emotion matrix to models/emotion/real_world_eval_report.json. Use this
report as the data source when writing the per-condition analysis and
known-failure-case list into docs/model_evaluation.md.

Each session holds one true emotion throughout, so degradation is compared
against that specific class's held-out test-set *recall*
(models/emotion/final_report.json's per_class_metrics), not the model's
overall top1 accuracy — the overall figure is blended across all classes
(several much weaker than others) and would understate how well a
single-emotion session should perform, making some conditions look like
false improvements over the test set.

Usage: PYTHONPATH=.:scripts ./venv/bin/python3 scripts/summarize_emotion_real_world_eval.py
"""

import csv
import json
from collections import defaultdict

from emotion_finetune.constants import REPORT_PATH as FINETUNE_REPORT_PATH
from emotion_real_world_eval.constants import LOG_DIR, REPO_ROOT


def summarize_cell(rows: list[dict], class_recall: float | None) -> dict:
    """Compute detection rate and emotion accuracy for one (condition, emotion) cell's logged frames."""
    total = len(rows)
    detected = [r for r in rows if r["face_detected"] == "True"]
    emotion_correct = sum(1 for r in detected if r["emotion_correct"] == "True")
    accuracy = round(emotion_correct / len(detected), 4) if detected else None

    return {
        "total_frames": total,
        "faces_detected": len(detected),
        "face_detection_rate": round(len(detected) / total, 4) if total else 0.0,
        "emotion_accuracy": accuracy,
        "test_set_recall_for_this_emotion": class_recall,
        "emotion_degradation_vs_test_set": (
            round(class_recall - accuracy, 4) if accuracy is not None and class_recall is not None else None
        ),
    }


def mean_of(values: list[float]) -> float | None:
    """Return the rounded mean of a list of numbers, or None if empty."""
    return round(sum(values) / len(values), 4) if values else None


def main() -> None:
    """Aggregate every (condition, emotion) CSV under LOG_DIR and write the full matrix report."""
    finetune_report = json.loads(FINETUNE_REPORT_PATH.read_text())
    overall_top1 = finetune_report["top_k_accuracy"]["top1_accuracy"]
    per_class_recall = {name: metrics["recall"] for name, metrics in finetune_report["per_class_metrics"].items()}

    csv_paths = sorted(LOG_DIR.glob("*.csv"))
    if not csv_paths:
        print(f"No logs found in {LOG_DIR} — run scripts/evaluate_emotion_real_world.py first.")
        return

    # Group rows by (condition, true_emotion) as recorded in the data itself,
    # not the filename, so results are correct even if sessions were re-run
    # into a shared file.
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for csv_path in csv_paths:
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                cells[(row["condition"], row["true_emotion"])].append(row)

    matrix: dict[str, dict[str, dict]] = defaultdict(dict)
    for (condition, emotion), rows in sorted(cells.items()):
        matrix[condition][emotion] = summarize_cell(rows, per_class_recall.get(emotion))
        print(f"{condition} / {emotion}: {matrix[condition][emotion]}")

    condition_averages = {
        condition: {
            "mean_accuracy": mean_of([c["emotion_accuracy"] for c in emotions.values() if c["emotion_accuracy"] is not None]),
            "mean_degradation": mean_of(
                [c["emotion_degradation_vs_test_set"] for c in emotions.values() if c["emotion_degradation_vs_test_set"] is not None]
            ),
            "emotions_tested": sorted(emotions.keys()),
        }
        for condition, emotions in matrix.items()
    }

    emotion_totals: dict[str, list[dict]] = defaultdict(list)
    for emotions in matrix.values():
        for emotion, cell in emotions.items():
            emotion_totals[emotion].append(cell)
    emotion_averages = {
        emotion: {
            "mean_accuracy": mean_of([c["emotion_accuracy"] for c in cells_ if c["emotion_accuracy"] is not None]),
            "mean_degradation": mean_of(
                [c["emotion_degradation_vs_test_set"] for c in cells_ if c["emotion_degradation_vs_test_set"] is not None]
            ),
            "conditions_tested": len(cells_),
        }
        for emotion, cells_ in emotion_totals.items()
    }

    report = {
        "test_set_overall_top1": overall_top1,
        "test_set_per_class_recall": per_class_recall,
        "matrix": matrix,
        "condition_averages": condition_averages,
        "emotion_averages": emotion_averages,
    }

    out_path = REPO_ROOT / "models" / "emotion" / "real_world_eval_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nReal-world evaluation report written to {out_path}")


if __name__ == "__main__":
    main()