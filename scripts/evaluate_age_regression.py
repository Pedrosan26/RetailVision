"""
evaluate_age_regression.py

CLI entry point for RET-31 evaluation: computes overall and per-age-group
MAE for the trained regression model on the held-out UTKFace test split,
and writes models/age_gender/regression_report.json. Run this after
scripts/train_age_regression.py has finished.

Usage: PYTHONPATH=scripts ./venv/bin/python3 scripts/evaluate_age_regression.py
"""

import json

from age_gender_baseline.constants import resolve_device
from age_regression.constants import REPORT_PATH
from age_regression.evaluate import evaluate_regression


def main() -> None:
    """Evaluate the age-regression model and write the RET-31 metrics report."""
    device = resolve_device()
    report = evaluate_regression(device)

    print(f"Overall MAE: {report['overall_mae']} years")
    for age_group, stats in report["per_age_group_mae"].items():
        print(f"  {age_group}: MAE={stats['mae']} (n={stats['support']})")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nRegression report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()