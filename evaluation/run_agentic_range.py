from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agentic_system import EDOrchestrationAgent, build_evaluation_scenarios, score_response  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="Zero-based start index.")
    parser.add_argument("--count", type=int, default=10, help="Number of scenarios to run.")
    parser.add_argument("--totals-only", action="store_true", help="Print only aggregate totals for this range.")
    args = parser.parse_args()

    scenarios = build_evaluation_scenarios()
    selected = scenarios[args.start : args.start + args.count]
    runner = EDOrchestrationAgent(use_llm_summary=False)
    scores = []
    scenario_rows = []

    for offset, scenario in enumerate(selected, start=args.start):
        start_time = time.perf_counter()
        response = runner.decide(scenario.payload)
        response_time_ms = (time.perf_counter() - start_time) * 1000.0
        score = score_response(scenario, response, response_time_ms=response_time_ms)
        scores.append(score)
        scenario_rows.append(
            {
                "index": offset,
                "scenario": scenario.name,
                "expected": sorted(score.expected_actions),
                "predicted": sorted(score.predicted_actions),
                "tp": score.true_positives,
                "fp": score.false_positives,
                "fn": score.false_negatives,
                "action_quality": score.action_quality,
                "alert_count": score.alert_count,
                "delay": score.simulated_delay,
                "explanation": score.explanation_quality,
                "response_time_ms": round(score.response_time_ms, 3),
                "escalation_expected": "escalate_patient" in score.expected_actions,
                "escalation_hit": score.escalation_hit,
                "escalation_target_hit": score.escalation_target_hit,
            }
        )

    escalation_cases = [score for score in scores if "escalate_patient" in score.expected_actions]
    totals = {
        "start": args.start,
        "count": len(scores),
        "tp": sum(score.true_positives for score in scores),
        "fp": sum(score.false_positives for score in scores),
        "fn": sum(score.false_negatives for score in scores),
        "action_quality_sum": round(sum(score.action_quality for score in scores), 6),
        "alert_count_sum": sum(score.alert_count for score in scores),
        "delay_sum": round(sum(score.simulated_delay for score in scores), 6),
        "explanation_sum": round(sum(score.explanation_quality for score in scores), 6),
        "response_time_ms_sum": round(sum(score.response_time_ms for score in scores), 3),
        "escalation_cases": len(escalation_cases),
        "escalation_hits": sum(1 for score in escalation_cases if score.escalation_hit),
        "escalation_target_hits": sum(1 for score in escalation_cases if score.escalation_target_hit),
    }

    payload = {"totals": totals}
    if not args.totals_only:
        payload["scenarios"] = scenario_rows
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
