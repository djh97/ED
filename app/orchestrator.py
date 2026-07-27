from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from app.config import settings
from app.models import EDDecisionResponse, EDRequest, RecommendationItem, ToolOutputs
from app.tools.bed_management import run_bed_management
from app.tools.flow_prediction import run_flow_prediction
from app.tools.patient_risk import run_patient_risk
from app.tools.staffing import run_staffing_availability

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None


class EDOrchestrationAgent:
    def __init__(self) -> None:
        self._client = OpenAI(api_key=settings.openai_api_key) if (OpenAI and settings.openai_api_key) else None

    def decide(self, ed_input: EDRequest) -> EDDecisionResponse:
        flow = run_flow_prediction(ed_input)
        risk = run_patient_risk(ed_input)
        staffing = run_staffing_availability(ed_input)
        bed = run_bed_management(ed_input)

        recommendations: list[RecommendationItem] = []

        top_patient = risk.flagged_patients[0] if risk.flagged_patients else None
        if top_patient and top_patient.risk_level in {"critical", "high"}:
            recommendations.append(
                RecommendationItem(
                    action="escalate_patient",
                    priority="urgent" if top_patient.risk_level == "critical" else "high",
                    target_id=top_patient.patient_id,
                    reason=f"Highest-risk patient is {top_patient.patient_id} with risk level {top_patient.risk_level}.",
                )
            )

        if flow.bottleneck_level in {"high", "critical"}:
            recommendations.append(
                RecommendationItem(
                    action="reprioritize_queue",
                    priority="high" if flow.bottleneck_level == "high" else "urgent",
                    reason=f"Flow pressure is {flow.bottleneck_level} with predicted wait of {flow.predicted_wait_minutes} minutes.",
                )
            )

        if staffing.staffing_level in {"strained", "critical"}:
            recommendations.append(
                RecommendationItem(
                    action="staffing_alert",
                    priority="high" if staffing.staffing_level == "strained" else "urgent",
                    reason=f"Staffing is {staffing.staffing_level} with pressure score {staffing.staffing_pressure_score}.",
                )
            )

        if bed.action_window in {"tight", "critical"}:
            recommendations.append(
                RecommendationItem(
                    action="reassign_bed",
                    priority="high" if bed.action_window == "tight" else "urgent",
                    reason=f"Bed action window is {bed.action_window} with {bed.available_beds_now} beds immediately available.",
                )
            )

        if top_patient and top_patient.risk_level in {"high", "critical"} and bed.available_beds_now > 0:
            recommendations.append(
                RecommendationItem(
                    action="admit_support",
                    priority="high",
                    target_id=top_patient.patient_id,
                    reason="High-risk patient and bed capacity support early admission or higher-acuity placement review.",
                )
            )

        if not recommendations:
            recommendations.append(
                RecommendationItem(
                    action="monitor",
                    priority="medium",
                    reason="No critical operational or patient-level trigger detected. Continue monitoring.",
                )
            )

        system_state = self._derive_system_state(flow.bottleneck_level, staffing.staffing_level, bed.action_window, top_patient)
        tool_outputs = ToolOutputs(
            flow_prediction=flow,
            patient_risk=risk,
            staffing=staffing,
            bed_management=bed,
        )
        summary = self._build_summary(ed_input, tool_outputs, recommendations, system_state)

        return EDDecisionResponse(
            summary=summary,
            system_state=system_state,
            recommendations=recommendations,
            tool_outputs=tool_outputs,
        )

    def _derive_system_state(self, flow_level: str, staffing_level: str, bed_window: str, top_patient: Any) -> str:
        if (
            flow_level == "critical"
            or staffing_level == "critical"
            or bed_window == "critical"
            or (top_patient and top_patient.risk_level == "critical")
        ):
            return "critical"
        if flow_level == "high" or staffing_level == "strained" or bed_window == "tight" or (top_patient and top_patient.risk_level == "high"):
            return "strained"
        if flow_level == "moderate" or (top_patient and top_patient.risk_level == "moderate"):
            return "watch"
        return "stable"

    def _build_summary(
        self,
        ed_input: EDRequest,
        tool_outputs: ToolOutputs,
        recommendations: list[RecommendationItem],
        system_state: str,
    ) -> str:
        if self._client:
            llm_summary = self._maybe_generate_llm_summary(ed_input, tool_outputs, recommendations, system_state)
            if llm_summary:
                return llm_summary

        top_patient = tool_outputs.patient_risk.flagged_patients[0] if tool_outputs.patient_risk.flagged_patients else None
        parts = [
            f"ED state is {system_state}.",
            f"Flow is {tool_outputs.flow_prediction.bottleneck_level} with predicted wait {tool_outputs.flow_prediction.predicted_wait_minutes} minutes.",
            f"Staffing is {tool_outputs.staffing.staffing_level}.",
            f"Beds are {tool_outputs.bed_management.action_window}.",
        ]
        if top_patient:
            parts.append(
                f"Highest-risk patient is {top_patient.patient_id} at {top_patient.risk_level} risk."
            )
        parts.append(f"{len(recommendations)} recommendation(s) were generated.")
        return " ".join(parts)

    def _maybe_generate_llm_summary(
        self,
        ed_input: EDRequest,
        tool_outputs: ToolOutputs,
        recommendations: list[RecommendationItem],
        system_state: str,
    ) -> str | None:
        try:
            prompt = {
                "system_state": system_state,
                "input_snapshot": ed_input.model_dump(),
                "tool_outputs": tool_outputs.model_dump(),
                "recommendations": [item.model_dump() for item in recommendations],
                "task": (
                    "Write a concise clinical operations summary for an emergency department decision-support system. "
                    "Do not invent facts. Use only the supplied tool outputs. Mention that this is decision support."
                ),
            }
            response = self._client.responses.create(
                model=settings.openai_model,
                input=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": json.dumps(prompt)}],
                    }
                ],
                reasoning={"effort": "medium"},
            )
            return response.output_text.strip()
        except Exception:
            return None


class RuleBasedEDBaseline:
    """Fixed-threshold baseline using the same four tools without agentic orchestration."""

    def decide(self, ed_input: EDRequest) -> EDDecisionResponse:
        flow = run_flow_prediction(ed_input)
        risk = run_patient_risk(ed_input)
        staffing = run_staffing_availability(ed_input)
        bed = run_bed_management(ed_input)

        recommendations: list[RecommendationItem] = []
        top_patient = risk.flagged_patients[0] if risk.flagged_patients else None

        if top_patient and top_patient.risk_level in {"high", "critical"}:
            recommendations.append(
                RecommendationItem(
                    action="escalate_patient",
                    priority="urgent" if top_patient.risk_level == "critical" else "high",
                    target_id=top_patient.patient_id,
                    reason="Triggered by fixed patient-risk threshold.",
                )
            )

        if flow.predicted_wait_minutes > 120 or flow.bottleneck_level in {"high", "critical"}:
            recommendations.append(
                RecommendationItem(
                    action="reprioritize_queue",
                    priority="urgent" if flow.bottleneck_level == "critical" else "high",
                    reason="Triggered by wait-time or congestion threshold.",
                )
            )

        if staffing.staffing_pressure_score > 0.8:
            recommendations.append(
                RecommendationItem(
                    action="staffing_alert",
                    priority="urgent" if staffing.staffing_pressure_score > 0.95 else "high",
                    reason="Triggered by staffing pressure threshold.",
                )
            )

        if bed.occupancy_rate > 0.9 or bed.available_beds_now <= 1:
            recommendations.append(
                RecommendationItem(
                    action="reassign_bed",
                    priority="urgent" if bed.action_window == "critical" else "high",
                    reason="Triggered by occupancy or available-bed threshold.",
                )
            )

        if top_patient and top_patient.risk_level in {"high", "critical"} and bed.available_beds_now > 0:
            recommendations.append(
                RecommendationItem(
                    action="admit_support",
                    priority="high",
                    target_id=top_patient.patient_id,
                    reason="Triggered by high patient risk with any immediate bed available.",
                )
            )

        if not recommendations:
            recommendations.append(
                RecommendationItem(
                    action="monitor",
                    priority="medium",
                    reason="No fixed threshold crossed.",
                )
            )

        system_state = "stable"
        if any(item.priority == "urgent" for item in recommendations):
            system_state = "critical"
        elif any(item.priority == "high" for item in recommendations):
            system_state = "strained"

        return EDDecisionResponse(
            summary="Rule-based baseline generated recommendations from static thresholds applied to the four tools.",
            system_state=system_state,
            recommendations=recommendations,
            tool_outputs=ToolOutputs(
                flow_prediction=flow,
                patient_risk=risk,
                staffing=staffing,
                bed_management=bed,
            ),
        )


@dataclass
class EvaluationScenario:
    name: str
    description: str
    payload: EDRequest
    expected_actions: set[str]
    urgent_targets: set[str]


@dataclass
class ScenarioScore:
    scenario_name: str
    predicted_actions: set[str]
    expected_actions: set[str]
    true_positives: int
    false_positives: int
    false_negatives: int
    action_quality: float
    escalation_hit: bool
    escalation_target_hit: bool
    alert_count: int
    simulated_delay: float


def _scenario_copy(base: EDRequest, updates: dict[str, Any]) -> EDRequest:
    payload = base.model_dump()
    payload.update(updates)
    return EDRequest(**payload)


def build_evaluation_scenarios() -> list[EvaluationScenario]:
    base = EDRequest(
        timestamp="2026-05-19T10:00:00Z",
        current_queue_length=18,
        arrivals_last_hour=21,
        average_wait_minutes=74,
        boarding_patients=6,
        patients=[
            {
                "patient_id": "ED-001",
                "age": 78,
                "triage_level": 2,
                "heart_rate": 126,
                "systolic_bp": 86,
                "respiratory_rate": 28,
                "oxygen_saturation": 90,
                "temperature_c": 38.8,
                "has_abnormal_labs": True,
                "suspected_sepsis": True,
                "pain_score": 7,
                "waiting_minutes": 45,
            },
            {
                "patient_id": "ED-002",
                "age": 44,
                "triage_level": 3,
                "heart_rate": 98,
                "systolic_bp": 118,
                "respiratory_rate": 18,
                "oxygen_saturation": 96,
                "temperature_c": 37.2,
                "has_abnormal_labs": False,
                "suspected_sepsis": False,
                "pain_score": 5,
                "waiting_minutes": 132,
            },
            {
                "patient_id": "ED-003",
                "age": 67,
                "triage_level": 3,
                "heart_rate": 108,
                "systolic_bp": 102,
                "respiratory_rate": 22,
                "oxygen_saturation": 93,
                "temperature_c": 37.7,
                "has_abnormal_labs": True,
                "suspected_sepsis": False,
                "pain_score": 4,
                "waiting_minutes": 96,
            },
        ],
        staffing={
            "available_nurses": 4,
            "available_physicians": 2,
            "nurse_capacity_per_hour": 4,
            "physician_capacity_per_hour": 6,
            "staff_absence_flag": True,
        },
        beds={
            "total_beds": 24,
            "occupied_beds": 23,
            "discharge_ready_beds": 2,
            "high_acuity_beds_available": 1,
        },
    )

    base_patients = [patient.model_dump() for patient in base.patients]

    return [
        EvaluationScenario(
            name="critical_congestion_with_sepsis",
            description="Crowded ED with a critically unstable patient and major staffing pressure.",
            payload=base,
            expected_actions={"escalate_patient", "reprioritize_queue", "staffing_alert", "reassign_bed", "admit_support"},
            urgent_targets={"ED-001"},
        ),
        EvaluationScenario(
            name="stable_day_shift",
            description="Lower-pressure scenario where monitoring should dominate.",
            payload=_scenario_copy(base, {
                "current_queue_length": 6,
                "arrivals_last_hour": 7,
                "average_wait_minutes": 22,
                "boarding_patients": 1,
                "patients": [
                    {
                        "patient_id": "ED-001",
                        "age": 78,
                        "triage_level": 4,
                        "heart_rate": 88,
                        "systolic_bp": 124,
                        "respiratory_rate": 18,
                        "oxygen_saturation": 97,
                        "temperature_c": 37.1,
                        "has_abnormal_labs": False,
                        "suspected_sepsis": False,
                        "pain_score": 2,
                        "waiting_minutes": 30,
                    },
                    base_patients[1],
                    base_patients[2],
                ],
                "staffing": {
                    "available_nurses": 7,
                    "available_physicians": 4,
                    "nurse_capacity_per_hour": 4,
                    "physician_capacity_per_hour": 6,
                    "staff_absence_flag": False,
                },
                "beds": {
                    "total_beds": 24,
                    "occupied_beds": 16,
                    "discharge_ready_beds": 3,
                    "high_acuity_beds_available": 2,
                },
            }),
            expected_actions={"monitor"},
            urgent_targets=set(),
        ),
        EvaluationScenario(
            name="isolated_high_risk_patient",
            description="Overall ED is manageable, but one patient clearly requires urgent escalation.",
            payload=_scenario_copy(base, {
                "current_queue_length": 6,
                "arrivals_last_hour": 7,
                "average_wait_minutes": 22,
                "boarding_patients": 1,
                "patients": [
                    base_patients[0],
                    {
                        "patient_id": "ED-002",
                        "age": 44,
                        "triage_level": 2,
                        "heart_rate": 132,
                        "systolic_bp": 82,
                        "respiratory_rate": 30,
                        "oxygen_saturation": 89,
                        "temperature_c": 39.0,
                        "has_abnormal_labs": True,
                        "suspected_sepsis": True,
                        "pain_score": 6,
                        "waiting_minutes": 18,
                    },
                    base_patients[2],
                ],
                "staffing": {
                    "available_nurses": 7,
                    "available_physicians": 4,
                    "nurse_capacity_per_hour": 4,
                    "physician_capacity_per_hour": 6,
                    "staff_absence_flag": False,
                },
                "beds": {
                    "total_beds": 24,
                    "occupied_beds": 16,
                    "discharge_ready_beds": 3,
                    "high_acuity_beds_available": 2,
                },
            }),
            expected_actions={"escalate_patient", "admit_support"},
            urgent_targets={"ED-002"},
        ),
        EvaluationScenario(
            name="staffing_crunch",
            description="Moderate flow with severe staffing shortage.",
            payload=_scenario_copy(base, {
                "current_queue_length": 14,
                "arrivals_last_hour": 15,
                "average_wait_minutes": 68,
                "boarding_patients": 2,
                "staffing": {
                    "available_nurses": 2,
                    "available_physicians": 1,
                    "nurse_capacity_per_hour": 4,
                    "physician_capacity_per_hour": 6,
                    "staff_absence_flag": True,
                },
                "beds": {
                    "total_beds": 24,
                    "occupied_beds": 18,
                    "discharge_ready_beds": 2,
                    "high_acuity_beds_available": 1,
                },
            }),
            expected_actions={"staffing_alert", "reprioritize_queue"},
            urgent_targets=set(),
        ),
        EvaluationScenario(
            name="bed_block",
            description="Bed saturation with otherwise moderate demand.",
            payload=_scenario_copy(base, {
                "current_queue_length": 10,
                "arrivals_last_hour": 9,
                "average_wait_minutes": 48,
                "boarding_patients": 2,
                "staffing": {
                    "available_nurses": 6,
                    "available_physicians": 3,
                    "nurse_capacity_per_hour": 4,
                    "physician_capacity_per_hour": 6,
                    "staff_absence_flag": False,
                },
                "beds": {
                    "total_beds": 24,
                    "occupied_beds": 24,
                    "discharge_ready_beds": 0,
                    "high_acuity_beds_available": 0,
                },
            }),
            expected_actions={"reassign_bed"},
            urgent_targets=set(),
        ),
        EvaluationScenario(
            name="queue_overload",
            description="Heavy inflow and queue pressure without a single obviously unstable patient.",
            payload=_scenario_copy(base, {
                "current_queue_length": 22,
                "arrivals_last_hour": 24,
                "average_wait_minutes": 116,
                "boarding_patients": 5,
                "patients": [
                    {
                        "patient_id": "ED-001",
                        "age": 42,
                        "triage_level": 3,
                        "heart_rate": 92,
                        "systolic_bp": 122,
                        "respiratory_rate": 18,
                        "oxygen_saturation": 97,
                        "temperature_c": 36.9,
                        "has_abnormal_labs": False,
                        "suspected_sepsis": False,
                        "pain_score": 2,
                        "waiting_minutes": 70,
                    },
                    base_patients[1],
                    base_patients[2],
                ],
                "staffing": {
                    "available_nurses": 5,
                    "available_physicians": 3,
                    "nurse_capacity_per_hour": 4,
                    "physician_capacity_per_hour": 6,
                    "staff_absence_flag": False,
                },
                "beds": {
                    "total_beds": 24,
                    "occupied_beds": 22,
                    "discharge_ready_beds": 1,
                    "high_acuity_beds_available": 1,
                },
            }),
            expected_actions={"reprioritize_queue"},
            urgent_targets=set(),
        ),
    ]


def score_response(scenario: EvaluationScenario, response: EDDecisionResponse) -> ScenarioScore:
    predicted_actions = {item.action for item in response.recommendations}
    expected_actions = set(scenario.expected_actions)
    tp = len(predicted_actions & expected_actions)
    fp = len(predicted_actions - expected_actions)
    fn = len(expected_actions - predicted_actions)
    precision = tp / max(len(predicted_actions), 1)
    recall = tp / max(len(expected_actions), 1)
    escalation_expected = "escalate_patient" in expected_actions
    escalation_hit = (not escalation_expected) or ("escalate_patient" in predicted_actions)
    target_hit = True

    if escalation_expected and scenario.urgent_targets:
        predicted_targets = {
            item.target_id
            for item in response.recommendations
            if item.action == "escalate_patient" and item.target_id is not None
        }
        target_hit = bool(predicted_targets & scenario.urgent_targets)

    return ScenarioScore(
        scenario_name=scenario.name,
        predicted_actions=predicted_actions,
        expected_actions=expected_actions,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        action_quality=round((precision + recall) / 2.0, 3),
        escalation_hit=escalation_hit,
        escalation_target_hit=target_hit,
        alert_count=len(response.recommendations),
        simulated_delay=round(max(0.0, fn * 8.0 + fp * 2.0), 2),
    )


def summarize_scores(scores: list[ScenarioScore]) -> dict[str, float]:
    tp = sum(score.true_positives for score in scores)
    fp = sum(score.false_positives for score in scores)
    fn = sum(score.false_negatives for score in scores)

    return {
        "precision": round(tp / max(tp + fp, 1), 3),
        "recall": round(tp / max(tp + fn, 1), 3),
        "avg_action_quality": round(sum(score.action_quality for score in scores) / max(len(scores), 1), 3),
        "avg_alert_burden": round(sum(score.alert_count for score in scores) / max(len(scores), 1), 3),
        "avg_recommendation_delay": round(sum(score.simulated_delay for score in scores) / max(len(scores), 1), 3),
        "escalation_recall": round(sum(1 for score in scores if score.escalation_hit) / max(len(scores), 1), 3),
        "escalation_target_accuracy": round(sum(1 for score in scores if score.escalation_target_hit) / max(len(scores), 1), 3),
    }


def evaluate_runner(name: str, runner: EDOrchestrationAgent | RuleBasedEDBaseline) -> dict[str, Any]:
    scenario_results = []
    scores = []

    for scenario in build_evaluation_scenarios():
        response = runner.decide(scenario.payload)
        score = score_response(scenario, response)
        scores.append(score)
        scenario_results.append(
            {
                "scenario": scenario.name,
                "description": scenario.description,
                "expected_actions": sorted(scenario.expected_actions),
                "predicted_actions": sorted(score.predicted_actions),
                "metrics": asdict(score),
            }
        )

    return {
        "system": name,
        "summary_metrics": summarize_scores(scores),
        "scenario_results": scenario_results,
    }


# Compatibility exports: the upgraded implementation lives in app.agentic_system,
# while existing scripts and the FastAPI app continue importing app.orchestrator.
from app.agentic_system import (  # noqa: E402,F401
    DecisionRunner,
    EDOrchestrationAgent,
    CrowdingScoreBaseline,
    ESITriageBaseline,
    EarlyWarningScoreBaseline,
    EvaluationScenario,
    NonAgenticIntegratedBaseline,
    PredictionOnlyBaseline,
    RuleBasedEDBaseline,
    ScenarioScore,
    build_core_evaluation_scenarios,
    build_evaluation_scenarios,
    build_mixed_evaluation_scenarios,
    evaluate_all_systems,
    evaluate_runner,
    score_response,
    summarize_scores,
)
