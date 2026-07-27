from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation import compact_summary, run_synthetic_benchmark, write_evaluation_outputs  # noqa: E402


def main() -> None:
    outputs = write_evaluation_outputs(ROOT / "evaluation_outputs")
    print("Saved evaluation outputs:")
    print(json.dumps(outputs, indent=2))
    print("\nCompact summary:")
    comparison = run_synthetic_benchmark()
    print(json.dumps(compact_summary(comparison), indent=2))


if __name__ == "__main__":
    main()
