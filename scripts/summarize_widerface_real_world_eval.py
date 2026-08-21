"""
summarize_widerface_real_world_eval.py

CLI entry point that aggregates every logged (condition, model) session
under runs/widerface_real_world_eval/logs/ into a report: detection rate
per condition per checkpoint, compared directly against the Haar
cascade's own documented real-world detection rates
(RESULTS.md), plus a false-positive rate for the dedicated
no_person_background condition (any detection there is unambiguously a
false positive, not a matter of degree).

Usage: PYTHONPATH=.:scripts ./venv/bin/python3 scripts/summarize_widerface_real_world_eval.py
"""

import csv
import json
from collections import defaultdict

from widerface_real_world_eval.constants import HAAR_REFERENCE_DETECTION_RATE, LOG_DIR, REPORT_PATH


def load_sessions() -> dict:
    """Group every logged CSV's rows by (condition, model)."""
    sessions: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for log_path in sorted(LOG_DIR.glob("*.csv")):
        with open(log_path, newline="") as f:
            for row in csv.DictReader(f):
                sessions[(row["condition"], row["model"])].append(row)
    return sessions


def summarize_session(rows: list[dict]) -> dict:
    """Compute detection rate and extra-box (likely false positive) rate for one session's frames."""
    total = len(rows)
    detected = sum(1 for r in rows if r["face_detected"] == "True")
    extra_boxes = sum(1 for r in rows if int(r["num_boxes"]) > 1)
    return {
        "total_frames": total,
        "detection_rate": round(detected / total, 4) if total else 0.0,
        "frames_with_extra_boxes": extra_boxes,
        "extra_box_rate": round(extra_boxes / total, 4) if total else 0.0,
    }


def main() -> None:
    """Aggregate every logged session into a report and print a summary table."""
    sessions = load_sessions()
    if not sessions:
        print(f"No logged sessions found under {LOG_DIR} -- run evaluate_widerface_real_world.py first")
        return

    matrix: dict[str, dict] = defaultdict(dict)
    for (condition, model), rows in sessions.items():
        matrix[condition][model] = summarize_session(rows)

    for condition, by_model in matrix.items():
        haar_rate = HAAR_REFERENCE_DETECTION_RATE.get(condition)
        print(f"\n{condition}" + (f"  (Haar reference: {haar_rate:.1%})" if haar_rate is not None else ""))
        for model, stats in by_model.items():
            print(
                f"  {model}: detection_rate={stats['detection_rate']:.1%}  "
                f"extra_box_rate={stats['extra_box_rate']:.1%}  ({stats['total_frames']} frames)"
            )

    report = {
        "haar_reference_detection_rate": HAAR_REFERENCE_DETECTION_RATE,
        "matrix": matrix,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
