"""
summarize_real_world_eval.py

CLI entry point for the age/gender classifiers' real-world evaluation
aggregation: reads every per-condition CSV logged by
scripts/evaluate_real_world.py, computes face-detection rate and
age/gender accuracy per condition, compares each against the fine-tuned
held-out test-set accuracy (models/age_gender/final_report.json) to
quantify degradation, and writes the combined numbers to
models/age_gender/real_world_eval_report.json. Use this report as the
data source when writing the per-condition analysis and known-failure-case
list into docs/model_evaluation.md.

Usage: PYTHONPATH=.:scripts ./venv/bin/python3 scripts/summarize_real_world_eval.py
"""

import csv
import json

from age_gender_finetune.constants import REPORT_PATH as FINETUNE_REPORT_PATH
from real_world_eval.constants import LOG_DIR, REPO_ROOT


def summarize_condition(csv_path) -> dict:
    """Compute detection rate and per-task accuracy for one condition's logged frames."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    detected = [r for r in rows if r["face_detected"] == "True"]
    age_correct = sum(1 for r in detected if r["age_correct"] == "True")
    gender_correct = sum(1 for r in detected if r["gender_correct"] == "True")

    return {
        "total_frames": total,
        "faces_detected": len(detected),
        "face_detection_rate": round(len(detected) / total, 4) if total else 0.0,
        "age_accuracy": round(age_correct / len(detected), 4) if detected else None,
        "gender_accuracy": round(gender_correct / len(detected), 4) if detected else None,
    }


def main() -> None:
    """Aggregate every condition CSV under LOG_DIR and write the real-world evaluation report."""
    baseline = json.loads(FINETUNE_REPORT_PATH.read_text())
    test_set_accuracy = {
        "age": baseline["age"]["top_k_accuracy"]["top1_accuracy"],
        "gender": baseline["gender"]["top_k_accuracy"]["top1_accuracy"],
    }

    report: dict = {"test_set_accuracy": test_set_accuracy, "conditions": {}}
    csv_paths = sorted(LOG_DIR.glob("*.csv"))
    if not csv_paths:
        print(f"No condition logs found in {LOG_DIR} — run scripts/evaluate_real_world.py first.")
        return

    for csv_path in csv_paths:
        condition = csv_path.stem
        summary = summarize_condition(csv_path)
        for task in ("age", "gender"):
            accuracy = summary[f"{task}_accuracy"]
            summary[f"{task}_degradation_vs_test_set"] = (
                round(test_set_accuracy[task] - accuracy, 4) if accuracy is not None else None
            )
        report["conditions"][condition] = summary
        print(f"{condition}: {summary}")

    out_path = REPO_ROOT / "models" / "age_gender" / "real_world_eval_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nReal-world evaluation report written to {out_path}")


if __name__ == "__main__":
    main()