from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.additional_evaluations import evaluate_component_contribution  # noqa: E402


SUMMARY_COLUMNS = [
    "configuration",
    "precision",
    "recall",
    "action_quality",
    "recommendation_delay",
    "escalation_recall",
    "escalation_target_accuracy",
    "alert_burden",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real-agentic",
        action="store_true",
        help="Use the real LLM orchestration agent. Requires OPENAI_API_KEY and may incur API cost.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "evaluation_outputs"),
        help="Directory for JSON, CSV, and Markdown outputs.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = evaluate_component_contribution(use_real_agentic=args.real_agentic)
    suffix = "llm" if args.real_agentic else "reference"
    json_path = output_dir / f"component_contribution_results_{suffix}.json"
    csv_path = output_dir / f"component_contribution_table_{suffix}.csv"
    md_path = output_dir / f"component_contribution_results_{suffix}.md"

    written: dict[str, str] = {}
    write_warnings: list[str] = []
    try:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        written["json"] = str(json_path)
    except OSError as exc:
        _record_write_failure("json", json_path, exc, written, write_warnings)
    try:
        _write_csv(csv_path, results["summary"])
        written["csv"] = str(csv_path)
    except OSError as exc:
        _record_write_failure("csv", csv_path, exc, written, write_warnings)
    try:
        _write_markdown(md_path, results, csv_path)
        written["markdown"] = str(md_path)
    except OSError as exc:
        _record_write_failure("markdown", md_path, exc, written, write_warnings)

    print(json.dumps({
        "mode": results["mode"],
        "scenario_count": results["scenario_count"],
        "outputs": written,
        "write_warnings": write_warnings,
        "summary": results["summary"],
    }, indent=2))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in SUMMARY_COLUMNS})


def _record_write_failure(
    output_name: str,
    path: Path,
    exc: OSError,
    written: dict[str, str],
    write_warnings: list[str],
) -> None:
    if path.exists():
        written[output_name] = str(path)
        write_warnings.append(f"Could not overwrite {path}; existing file was retained: {exc}")
    else:
        write_warnings.append(f"Could not write {output_name} output to {path}: {exc}")


def _write_markdown(path: Path, results: dict[str, Any], csv_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Component Contribution Analysis",
        "",
        f"Mode: `{results['mode']}`",
        f"Scenario count: `{results['scenario_count']}`",
        "",
        "Goal: quantify how safety validation, state management, and follow-up tracking contribute to workflow recommendation quality, safety coverage, stateful continuity, and alert burden.",
        "",
        f"CSV table: `{csv_path}`",
        "",
        "| Configuration | Precision | Recall | Action Quality | Recommendation Delay | Escalation Recall | Escalation Target Accuracy | Alert Burden |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results["summary"]:
        lines.append(
            "| {configuration} | {precision:.3f} | {recall:.3f} | {action_quality:.3f} | {recommendation_delay:.3f} | {escalation_recall:.3f} | {escalation_target_accuracy:.3f} | {alert_burden:.3f} |".format(
                **row
            )
        )
    lines.extend([
        "",
        "Interpretation notes:",
        "",
        "- Lower recommendation delay is better.",
        "- Alert burden is computed from generated follow-up tasks, so disabling follow-up tracking should reduce this value.",
        "- State-management variants use the same scenario data but receive patient arrivals as sequential updates; without state management, only the latest update is available at decision time.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
