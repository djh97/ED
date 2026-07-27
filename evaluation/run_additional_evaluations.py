from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.additional_evaluations import (  # noqa: E402
    evaluate_component_contribution,
    evaluate_ablation_baselines,
    evaluate_safety_validation,
    evaluate_stateful_replanning,
)
from app.agentic_system import EDOrchestrationAgent, RuleBasedEDBaseline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["all", "ablation", "safety", "stateful", "component"],
        default="all",
    )
    parser.add_argument(
        "--real-agentic",
        action="store_true",
        help="Use the real LLM agent for the stateful re-planning evaluation. Requires OPENAI_API_KEY.",
    )
    args = parser.parse_args()

    results = {}
    if args.mode in {"all", "ablation"}:
        results["ablation"] = evaluate_ablation_baselines()
    if args.mode in {"all", "component"}:
        results["component_contribution"] = evaluate_component_contribution(use_real_agentic=args.real_agentic)
    if args.mode in {"all", "safety"}:
        results["safety_validation"] = evaluate_safety_validation()
    if args.mode in {"all", "stateful"}:
        if args.real_agentic:
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("--real-agentic requires OPENAI_API_KEY.")
            runner = EDOrchestrationAgent(use_llm_summary=False)
            runner_name = "agentic_llm"
        else:
            runner = RuleBasedEDBaseline()
            runner_name = "rule_based_reference"
        results["stateful_replanning"] = {
            "runner": runner_name,
            **evaluate_stateful_replanning(runner),
        }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
