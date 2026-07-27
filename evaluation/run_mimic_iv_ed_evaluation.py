from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.mimic_iv_ed import evaluate_mimic_patient_risk  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Folder containing MIMIC-IV-ED edstays and triage CSV files.")
    parser.add_argument("--label", default="admitted", choices=["admitted", "high_acuity", "prolonged_los", "critical_proxy"])
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()

    metrics = evaluate_mimic_patient_risk(args.data_dir, label_name=args.label, limit=args.limit)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
