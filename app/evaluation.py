from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any

from app.agentic_system import (
    EDOrchestrationAgent,
    build_evaluation_scenarios,
    evaluate_all_systems,
)


SYSTEM_LABELS = {
    "esi_triage_baseline": "ESI triage",
    "news2_qsofa_baseline": "NEWS2/qSOFA",
    "nedocs_edwin_crowding_baseline": "Crowding score",
    "prediction_only_baseline": "Prediction only",
    "rule_based_baseline": "Rule based",
    "non_agentic_integrated_baseline": "Non-agentic",
    "agentic_orchestration": "Agentic",
}

KEY_METRICS = (
    "precision",
    "recall",
    "avg_action_quality",
    "avg_alert_burden",
    "avg_recommendation_delay",
    "avg_explanation_quality",
    "avg_response_time_ms",
    "escalation_recall",
    "escalation_target_accuracy",
)


def run_synthetic_benchmark() -> dict[str, Any]:
    """Runs the default 180-scenario benchmark used for agentic comparison."""

    return evaluate_all_systems()


def compact_summary(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for system_name, result in comparison.items():
        metrics = result["summary_metrics"]
        row = {"system": system_name, "label": SYSTEM_LABELS.get(system_name, system_name)}
        for metric in KEY_METRICS:
            row[metric] = metrics.get(metric)
        rows.append(row)
    return rows


def write_evaluation_outputs(output_dir: str | Path = "evaluation_outputs") -> dict[str, str]:
    """Writes JSON, CSV, and one example agentic output.

    Figure/chart files are intentionally not generated in the public release.
    """

    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    write_warnings: list[str] = []
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        write_warnings.append(f"Could not create output directory {output_path}: {exc}")

    comparison = run_synthetic_benchmark()
    summary_rows = compact_summary(comparison)

    full_json = output_path / "synthetic_comparison_full.json"
    summary_csv = output_path / "synthetic_summary_metrics.csv"
    example_json = output_path / "example_agentic_output.json"

    _safe_write_text(full_json, json.dumps(comparison, indent=2), write_warnings)
    _safe_write_text(summary_csv, _summary_to_csv(summary_rows), write_warnings)

    first_scenario = build_evaluation_scenarios(count=1)[0]
    example = EDOrchestrationAgent(use_llm_summary=False).decide(first_scenario.payload)
    _safe_write_text(example_json, json.dumps(example.model_dump(), indent=2), write_warnings)

    outputs = {
        "full_json": str(full_json),
        "summary_csv": str(summary_csv),
        "example_json": str(example_json),
    }
    if write_warnings:
        outputs["write_warnings"] = " | ".join(write_warnings)
    return outputs


def _safe_write_text(path: Path, content: str, warnings: list[str]) -> None:
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        warnings.append(f"Could not write {path}: {exc}")


def _summary_to_csv(rows: list[dict[str, Any]]) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["system", "label", *KEY_METRICS])
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()
