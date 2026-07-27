from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.nhamcs_ed import evaluate_nhamcs_ed, summarize_nhamcs_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Path to NHAMCS ED CSV or ZIP file.")
    parser.add_argument("--summary", action="store_true", help="Print dataset summary instead of task metrics.")
    parser.add_argument(
        "--task",
        default="high_acuity",
        choices=["high_acuity", "critical_vitals", "prolonged_wait", "very_prolonged_wait", "revisit_72h"],
    )
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--threshold", type=float, default=0.55)
    args = parser.parse_args()

    if args.summary:
        print(json.dumps(summarize_nhamcs_dataset(args.data_path), indent=2))
        return

    metrics = evaluate_nhamcs_ed(args.data_path, task=args.task, limit=args.limit, threshold=args.threshold)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
