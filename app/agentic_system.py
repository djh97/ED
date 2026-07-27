from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from app.agents import FollowUpTrackingAgent, InputUnderstandingAgent
from app.config import settings
from app.models import AgenticReasoning, AgentTraceItem, EDDecisionResponse, EDRequest, FollowUpItem, RecommendationItem, ToolOutputs
from app.prompts import build_orchestration_prompt, build_summary_prompt
from app.tools.bed_management import run_bed_management
from app.tools.flow_prediction import run_flow_prediction
from app.tools.patient_risk import run_patient_risk
from app.tools.staffing import run_staffing_availability

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None


class DecisionRunner(Protocol):
    def decide(self, ed_input: EDRequest) -> EDDecisionResponse:
        ...


class EDOrchestrationAgent:
    """Agentic ED controller that chooses actions after comparing tool outputs."""

    def __init__(
        self,
        use_llm_summary: bool = True,
        use_llm_planning: bool = True,
        use_llm_input: bool = True,
        enable_safety_validation: bool = True,
        enable_follow_up_tracking: bool = True,
        planner_client: Any | None = None,
        summary_client: Any | None = None,
        input_agent: InputUnderstandingAgent | None = None,
    ) -> None:
        self._summary_client = summary_client or (OpenAI(api_key=settings.openai_api_key) if (use_llm_summary and OpenAI and settings.openai_api_key) else None)
        self._planner_client = planner_client or (OpenAI(api_key=settings.openai_api_key) if (use_llm_planning and OpenAI and settings.openai_api_key) else None)
        self._input_agent = input_agent or InputUnderstandingAgent(use_llm=use_llm_input)
        self._follow_up_agent = FollowUpTrackingAgent()
        self._enable_safety_validation = enable_safety_validation
        self._enable_follow_up_tracking = enable_follow_up_tracking
        self._last_planning_trace: dict[str, Any] = {}

    def decide(self, ed_input: EDRequest | dict[str, Any] | str) -> EDDecisionResponse:
        input_result = self._input_agent.normalize(ed_input)
        ed_input = input_result.ed_request
        flow, risk, staffing, bed = _run_tools(ed_input)
        top_patient = risk.flagged_patients[0] if risk.flagged_patients else None
        tool_outputs = _tool_outputs(flow, risk, staffing, bed)
        recommendations = self._generate_llm_recommendations(ed_input, tool_outputs)
        planning_mode = "llm_prompted"
        system_state = _derive_system_state(flow.bottleneck_level, staffing.staffing_level, bed.action_window, top_patient)
        action_brief = _build_action_brief(recommendations)
        summary = self._build_summary(ed_input, tool_outputs, recommendations, system_state)
        follow_up_plan = self._follow_up_agent.create_plan(recommendations, system_state) if self._enable_follow_up_tracking else []
        agent_trace = [self._input_agent.build_trace(input_result)]
        agent_trace.extend(_build_agent_trace(tool_outputs, recommendations, follow_up_plan, planning_mode))
        agent_trace.extend(_build_llm_cycle_trace(self._last_planning_trace))
        if self._enable_follow_up_tracking:
            agent_trace.append(self._follow_up_agent.build_trace(follow_up_plan))
        return EDDecisionResponse(
            summary=summary,
            action_brief=action_brief,
            system_state=system_state,
            active_patient_count=len(ed_input.patients),
            pending_follow_up_count=len(follow_up_plan),
            agentic_reasoning=_build_agentic_reasoning(self._last_planning_trace),
            recommendations=recommendations,
            tool_outputs=tool_outputs,
            agent_trace=agent_trace,
            follow_up_plan=follow_up_plan,
        )

    def decide_from_raw(self, raw_input: EDRequest | dict[str, Any] | str) -> EDDecisionResponse:
        return self.decide(raw_input)

    def _generate_llm_recommendations(self, ed_input: EDRequest, tool_outputs: ToolOutputs) -> list[RecommendationItem]:
        if not self._planner_client:
            raise RuntimeError(
                "ED Orchestration Agent requires OPENAI_API_KEY and an orchestration LLM. "
                "No deterministic fallback is enabled."
            )
        try:
            prompt = build_orchestration_prompt(ed_input, tool_outputs)
            recommendations, planning_trace = self._request_llm_plan(prompt)
            if self._enable_safety_validation:
                validation_feedback = _required_escalation_feedback(recommendations, tool_outputs)
                if validation_feedback:
                    retry_prompt = build_orchestration_prompt(
                        ed_input,
                        tool_outputs,
                        validation_feedback=[
                            "Your previous plan failed validation. Re-plan using the Level 4 loop and fix these issues:",
                            *validation_feedback,
                        ],
                    )
                    recommendations, planning_trace = self._request_llm_plan(retry_prompt)
                    validation_feedback = _required_escalation_feedback(recommendations, tool_outputs)
                    if validation_feedback:
                        raise ValueError("LLM plan failed safety validation: " + " ".join(validation_feedback))
            self._last_planning_trace = planning_trace
            return recommendations
        except Exception as exc:
            raise RuntimeError(f"ED Orchestration Agent LLM planning failed: {exc}") from exc

    def _request_llm_plan(self, prompt: dict[str, Any]) -> tuple[list[RecommendationItem], dict[str, Any]]:
        response = self._planner_client.responses.create(
            model=settings.openai_orchestration_model,
            input=[{"role": "user", "content": [{"type": "input_text", "text": json.dumps(prompt)}]}],
            reasoning={"effort": "medium"},
        )
        parsed = json.loads(_extract_json_object(response.output_text))
        planning_trace = {
            "reasoning_summary": parsed.get("reasoning_summary", ""),
            "goal": parsed.get("goal", ""),
            "plan": parsed.get("plan", []),
            "execute": parsed.get("execute", []),
            "monitor_outcomes": parsed.get("monitor_outcomes", []),
            "replan_if_conditions_change": parsed.get("replan_if_conditions_change", []),
            "continue_until_goal_achieved": parsed.get("continue_until_goal_achieved", ""),
        }
        recommendations = []
        for item in parsed.get("recommendations", []):
            item.setdefault("target_id", None)
            recommendations.append(RecommendationItem(**item))
        if not recommendations:
            raise ValueError("LLM returned no recommendations.")
        return recommendations, planning_trace

    def _build_summary(
        self,
        ed_input: EDRequest,
        tool_outputs: ToolOutputs,
        recommendations: list[RecommendationItem],
        system_state: str,
    ) -> str:
        if self._summary_client:
            llm_summary = self._maybe_generate_llm_summary(ed_input, tool_outputs, recommendations, system_state)
            if llm_summary:
                return llm_summary

        top_patient = tool_outputs.patient_risk.flagged_patients[0] if tool_outputs.patient_risk.flagged_patients else None
        parts = [
            f"ED state is {system_state}.",
            "Agent reviewed flow, patient risk, staffing, and bed feasibility tools before selecting actions.",
            f"Flow is {tool_outputs.flow_prediction.bottleneck_level} with predicted wait {tool_outputs.flow_prediction.predicted_wait_minutes} minutes.",
            f"Staffing is {tool_outputs.staffing.staffing_level}.",
            f"Beds are {tool_outputs.bed_management.action_window}.",
        ]
        if top_patient:
            parts.append(f"Highest-risk patient is {top_patient.patient_id} at {top_patient.risk_level} risk.")
        parts.append(f"{len(recommendations)} recommendation(s) were generated as decision support.")
        return " ".join(parts)

    def _maybe_generate_llm_summary(
        self,
        ed_input: EDRequest,
        tool_outputs: ToolOutputs,
        recommendations: list[RecommendationItem],
        system_state: str,
    ) -> str | None:
        try:
            prompt = build_summary_prompt(
                ed_input=ed_input,
                tool_outputs=tool_outputs,
                recommendations=[item.model_dump() for item in recommendations],
                system_state=system_state,
            )
            response = self._summary_client.responses.create(
                model=settings.openai_summary_model,
                input=[{"role": "user", "content": [{"type": "input_text", "text": json.dumps(prompt)}]}],
                reasoning={"effort": "medium"},
            )
            return response.output_text.strip()
        except Exception:
            return None


class PredictionOnlyBaseline:
    """Prediction-only literature baseline: models produce scores, but no action planner exists."""

    def decide(self, ed_input: EDRequest) -> EDDecisionResponse:
        flow, risk, staffing, bed = _run_tools(ed_input)
        top_patient = risk.flagged_patients[0] if risk.flagged_patients else None
        return EDDecisionResponse(
            summary="Prediction-only baseline produced model outputs but did not convert them into operational actions.",
            action_brief="No operational action was selected. Review the model scores manually and continue monitoring.",
            system_state=_derive_system_state(flow.bottleneck_level, staffing.staffing_level, bed.action_window, top_patient),
            active_patient_count=len(ed_input.patients),
            pending_follow_up_count=0,
            recommendations=[
                RecommendationItem(
                    action="monitor",
                    priority="medium",
                    reason="Prediction-only baseline reports scores and leaves action selection to clinicians.",
                )
            ],
            tool_outputs=_tool_outputs(flow, risk, staffing, bed),
        )


class RuleBasedEDBaseline:
    """Fixed-threshold baseline using the same four tools without agentic orchestration."""

    def decide(self, ed_input: EDRequest) -> EDDecisionResponse:
        flow, risk, staffing, bed = _run_tools(ed_input)
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
                RecommendationItem(action="reprioritize_queue", priority="urgent" if flow.bottleneck_level == "critical" else "high", reason="Triggered by wait-time or congestion threshold.")
            )
        if staffing.staffing_pressure_score > 0.8:
            recommendations.append(
                RecommendationItem(action="staffing_alert", priority="urgent" if staffing.staffing_pressure_score > 0.95 else "high", reason="Triggered by staffing pressure threshold.")
            )
        if bed.occupancy_rate > 0.9 or bed.available_beds_now <= 1:
            recommendations.append(
                RecommendationItem(action="reassign_bed", priority="urgent" if bed.action_window == "critical" else "high", reason="Triggered by occupancy or available-bed threshold.")
            )
        if top_patient and top_patient.risk_level in {"high", "critical"} and bed.available_beds_now > 0:
            recommendations.append(
                RecommendationItem(action="admit_support", priority="high", target_id=top_patient.patient_id, reason="Triggered by high patient risk with any immediate bed available.")
            )
        if not recommendations:
            recommendations.append(RecommendationItem(action="monitor", priority="medium", reason="No fixed threshold crossed."))

        return EDDecisionResponse(
            summary="Rule-based baseline generated recommendations from static thresholds applied to the four tools.",
            action_brief=_build_action_brief(recommendations),
            system_state=_derive_state_from_recommendations(recommendations),
            active_patient_count=len(ed_input.patients),
            pending_follow_up_count=0,
            recommendations=recommendations,
            tool_outputs=_tool_outputs(flow, risk, staffing, bed),
        )


class NonAgenticIntegratedBaseline:
    """Integrated dashboard baseline: one fixed action is mapped from each model output."""

    def decide(self, ed_input: EDRequest) -> EDDecisionResponse:
        flow, risk, staffing, bed = _run_tools(ed_input)
        recommendations: list[RecommendationItem] = []
        top_patient = risk.flagged_patients[0] if risk.flagged_patients else None

        if top_patient and top_patient.escalation_needed:
            recommendations.append(RecommendationItem(action="escalate_patient", priority="urgent" if top_patient.risk_level == "critical" else "high", target_id=top_patient.patient_id, reason=f"Integrated dashboard mapped patient risk level {top_patient.risk_level} to escalation."))
        if flow.bottleneck_level in {"high", "critical"}:
            recommendations.append(RecommendationItem(action="reprioritize_queue", priority="urgent" if flow.bottleneck_level == "critical" else "high", reason=f"Integrated dashboard mapped flow level {flow.bottleneck_level} to queue reprioritization."))
        if staffing.staffing_level in {"strained", "critical"}:
            recommendations.append(RecommendationItem(action="staffing_alert", priority="urgent" if staffing.staffing_level == "critical" else "high", reason=f"Integrated dashboard mapped staffing level {staffing.staffing_level} to staffing alert."))
        if bed.action_window in {"tight", "critical"}:
            recommendations.append(RecommendationItem(action="reassign_bed", priority="urgent" if bed.action_window == "critical" else "high", reason=f"Integrated dashboard mapped bed window {bed.action_window} to bed reassignment."))
        if not recommendations:
            recommendations.append(RecommendationItem(action="monitor", priority="medium", reason="Integrated dashboard found no mapped action."))

        return EDDecisionResponse(
            summary="Non-agentic integrated baseline mapped each model output to a fixed action without cross-tool planning.",
            action_brief=_build_action_brief(recommendations),
            system_state=_derive_state_from_recommendations(recommendations),
            active_patient_count=len(ed_input.patients),
            pending_follow_up_count=0,
            recommendations=recommendations,
            tool_outputs=_tool_outputs(flow, risk, staffing, bed),
        )


class ESITriageBaseline:
    """Traditional ESI-like triage baseline mapped to patient escalation."""

    def decide(self, ed_input: EDRequest) -> EDDecisionResponse:
        flow, risk, staffing, bed = _run_tools(ed_input)
        recommendations: list[RecommendationItem] = []
        top_patient = min(ed_input.patients, key=lambda patient: (patient.triage_level, -patient.waiting_minutes), default=None)

        if top_patient and top_patient.triage_level <= 2:
            recommendations.append(
                RecommendationItem(
                    action="escalate_patient",
                    priority="urgent" if top_patient.triage_level == 1 else "high",
                    target_id=top_patient.patient_id,
                    reason=f"ESI-like triage baseline escalated {top_patient.patient_id} because triage acuity level is {top_patient.triage_level}.",
                )
            )
        if not recommendations:
            recommendations.append(RecommendationItem(action="monitor", priority="medium", reason="ESI-like triage baseline found no level 1 or 2 patient requiring escalation."))

        return EDDecisionResponse(
            summary="ESI-like triage baseline maps high triage acuity to patient escalation only.",
            action_brief=_build_action_brief(recommendations),
            system_state=_derive_state_from_recommendations(recommendations),
            active_patient_count=len(ed_input.patients),
            pending_follow_up_count=0,
            recommendations=recommendations,
            tool_outputs=_tool_outputs(flow, risk, staffing, bed),
        )


class EarlyWarningScoreBaseline:
    """NEWS2/qSOFA-style patient deterioration baseline mapped to escalation."""

    def decide(self, ed_input: EDRequest) -> EDDecisionResponse:
        flow, risk, staffing, bed = _run_tools(ed_input)
        scored_patients = [(patient, _news2_score(patient), _qsofa_score(patient)) for patient in ed_input.patients]
        scored_patients.sort(key=lambda item: (item[1], item[2]), reverse=True)
        top_patient, news2, qsofa = scored_patients[0] if scored_patients else (None, 0, 0)
        recommendations: list[RecommendationItem] = []

        if top_patient and (news2 >= 5 or qsofa >= 2):
            recommendations.append(
                RecommendationItem(
                    action="escalate_patient",
                    priority="urgent" if news2 >= 7 or qsofa >= 2 else "high",
                    target_id=top_patient.patient_id,
                    reason=f"NEWS2/qSOFA baseline escalated {top_patient.patient_id} with NEWS2={news2} and qSOFA={qsofa}.",
                )
            )
        if not recommendations:
            recommendations.append(RecommendationItem(action="monitor", priority="medium", reason="NEWS2/qSOFA baseline found no patient above escalation thresholds."))

        return EDDecisionResponse(
            summary="NEWS2/qSOFA baseline maps physiological deterioration scores to escalation only.",
            action_brief=_build_action_brief(recommendations),
            system_state=_derive_state_from_recommendations(recommendations),
            active_patient_count=len(ed_input.patients),
            pending_follow_up_count=0,
            recommendations=recommendations,
            tool_outputs=_tool_outputs(flow, risk, staffing, bed),
        )


class CrowdingScoreBaseline:
    """NEDOCS/EDWIN-style ED crowding baseline mapped to operational alerts."""

    def decide(self, ed_input: EDRequest) -> EDDecisionResponse:
        flow, risk, staffing, bed = _run_tools(ed_input)
        crowding_score = _crowding_score(ed_input)
        occupancy = ed_input.beds.occupied_beds / max(ed_input.beds.total_beds, 1)
        free_beds = max(ed_input.beds.total_beds - ed_input.beds.occupied_beds, 0)
        recommendations: list[RecommendationItem] = []

        if crowding_score >= 90:
            recommendations.append(RecommendationItem(action="reprioritize_queue", priority="urgent" if crowding_score >= 120 else "high", reason=f"NEDOCS/EDWIN-style crowding score is {crowding_score:.1f}, indicating ED flow pressure."))
        if crowding_score >= 120:
            recommendations.append(RecommendationItem(action="staffing_alert", priority="urgent", reason=f"NEDOCS/EDWIN-style crowding score is {crowding_score:.1f}, indicating severe workload pressure."))
        if occupancy >= 0.95 or free_beds <= 1:
            recommendations.append(RecommendationItem(action="reassign_bed", priority="urgent" if occupancy >= 0.98 else "high", reason=f"Crowding baseline detected occupancy={occupancy:.0%} with {free_beds} free beds."))
        if not recommendations:
            recommendations.append(RecommendationItem(action="monitor", priority="medium", reason=f"NEDOCS/EDWIN-style crowding score is {crowding_score:.1f}, below action threshold."))

        return EDDecisionResponse(
            summary="NEDOCS/EDWIN-style baseline maps ED crowding score to flow, staffing, and bed alerts.",
            action_brief=_build_action_brief(recommendations),
            system_state=_derive_state_from_recommendations(recommendations),
            active_patient_count=len(ed_input.patients),
            pending_follow_up_count=0,
            recommendations=recommendations,
            tool_outputs=_tool_outputs(flow, risk, staffing, bed),
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
    explanation_quality: float
    response_time_ms: float


def _scenario_copy(base: EDRequest, updates: dict[str, Any]) -> EDRequest:
    payload = base.model_dump()
    payload.update(updates)
    return EDRequest(**payload)


def build_core_evaluation_scenarios() -> list[EvaluationScenario]:
    stable_patients = [_patient_template("ED-001"), _patient_template("ED-002"), _patient_template("ED-003")]
    stable_patients[1].update({"age": 58, "sex": "male", "triage_level": 3, "chief_complaint": "mild dizziness", "waiting_minutes": 36})
    stable_patients[2].update({"age": 35, "sex": "other", "chief_complaint": "ankle injury", "waiting_minutes": 24})
    critical_patient = {
        "patient_id": "ED-001",
        "age": 78,
        "sex": "male",
        "triage_level": 2,
        "chief_complaint": "shortness of breath and fever",
        "triage_notes": "hypotension, confusion, suspected sepsis",
        "heart_rate": 126,
        "systolic_bp": 86,
        "respiratory_rate": 28,
        "oxygen_saturation": 90,
        "temperature_c": 38.8,
        "has_abnormal_labs": True,
        "suspected_sepsis": True,
        "pain_score": 7,
        "waiting_minutes": 45,
    }
    high_risk_patient = {
        "patient_id": "ED-002",
        "age": 44,
        "sex": "female",
        "triage_level": 2,
        "chief_complaint": "shortness of breath",
        "triage_notes": "low oxygen saturation, tachycardia, possible sepsis",
        "heart_rate": 132,
        "systolic_bp": 82,
        "respiratory_rate": 30,
        "oxygen_saturation": 89,
        "temperature_c": 39.0,
        "has_abnormal_labs": True,
        "suspected_sepsis": True,
        "pain_score": 6,
        "waiting_minutes": 18,
    }
    base = EDRequest(
        timestamp="2026-05-19T10:00:00Z",
        current_queue_length=18,
        arrivals_last_hour=21,
        average_wait_minutes=74,
        boarding_patients=6,
        patients=[critical_patient, stable_patients[1], stable_patients[2]],
        staffing={"available_nurses": 4, "available_physicians": 2, "nurse_capacity_per_hour": 4, "physician_capacity_per_hour": 6, "staff_absence_flag": True},
        beds={"total_beds": 24, "occupied_beds": 23, "discharge_ready_beds": 2, "high_acuity_beds_available": 1},
    )
    return [
        EvaluationScenario("critical_congestion_with_sepsis", "Crowded ED with a critically unstable patient and major staffing pressure.", base, {"escalate_patient", "reprioritize_queue", "staffing_alert", "reassign_bed", "admit_support"}, {"ED-001"}),
        EvaluationScenario("stable_day_shift", "Lower-pressure scenario where monitoring should dominate.", _scenario_copy(base, {"current_queue_length": 6, "arrivals_last_hour": 7, "average_wait_minutes": 22, "boarding_patients": 1, "patients": stable_patients, "staffing": {"available_nurses": 7, "available_physicians": 4, "nurse_capacity_per_hour": 4, "physician_capacity_per_hour": 6, "staff_absence_flag": False}, "beds": {"total_beds": 24, "occupied_beds": 16, "discharge_ready_beds": 3, "high_acuity_beds_available": 2}}), {"monitor"}, set()),
        EvaluationScenario("isolated_high_risk_patient", "Overall ED is manageable, but one patient clearly requires urgent escalation.", _scenario_copy(base, {"current_queue_length": 6, "arrivals_last_hour": 7, "average_wait_minutes": 22, "boarding_patients": 1, "patients": [stable_patients[0], high_risk_patient, stable_patients[2]], "staffing": {"available_nurses": 7, "available_physicians": 4, "nurse_capacity_per_hour": 4, "physician_capacity_per_hour": 6, "staff_absence_flag": False}, "beds": {"total_beds": 24, "occupied_beds": 16, "discharge_ready_beds": 3, "high_acuity_beds_available": 2}}), {"escalate_patient", "admit_support"}, {"ED-002"}),
        EvaluationScenario("staffing_crunch", "Moderate flow with severe staffing shortage.", _scenario_copy(base, {"current_queue_length": 14, "arrivals_last_hour": 15, "average_wait_minutes": 68, "boarding_patients": 2, "patients": stable_patients, "staffing": {"available_nurses": 2, "available_physicians": 1, "nurse_capacity_per_hour": 4, "physician_capacity_per_hour": 6, "staff_absence_flag": True}, "beds": {"total_beds": 24, "occupied_beds": 18, "discharge_ready_beds": 2, "high_acuity_beds_available": 1}}), {"staffing_alert", "reprioritize_queue"}, set()),
        EvaluationScenario("bed_block", "Bed saturation with otherwise moderate demand.", _scenario_copy(base, {"current_queue_length": 10, "arrivals_last_hour": 9, "average_wait_minutes": 48, "boarding_patients": 2, "patients": stable_patients, "staffing": {"available_nurses": 6, "available_physicians": 3, "nurse_capacity_per_hour": 4, "physician_capacity_per_hour": 6, "staff_absence_flag": False}, "beds": {"total_beds": 24, "occupied_beds": 24, "discharge_ready_beds": 0, "high_acuity_beds_available": 0}}), {"reassign_bed"}, set()),
        EvaluationScenario("queue_overload", "Heavy inflow and queue pressure without a single obviously unstable patient.", _scenario_copy(base, {"current_queue_length": 22, "arrivals_last_hour": 24, "average_wait_minutes": 116, "boarding_patients": 5, "patients": stable_patients, "staffing": {"available_nurses": 8, "available_physicians": 4, "nurse_capacity_per_hour": 4, "physician_capacity_per_hour": 6, "staff_absence_flag": False}, "beds": {"total_beds": 24, "occupied_beds": 22, "discharge_ready_beds": 1, "high_acuity_beds_available": 1}}), {"reprioritize_queue"}, set()),
    ]


def build_evaluation_scenarios(count: int = 180, seed: int = 2026) -> list[EvaluationScenario]:
    core = build_core_evaluation_scenarios()
    if count <= len(core):
        return core[:count]
    return core + build_mixed_evaluation_scenarios(count - len(core), seed)


def build_mixed_evaluation_scenarios(count: int = 174, seed: int = 2026) -> list[EvaluationScenario]:
    rng = random.Random(seed)
    scenarios: list[EvaluationScenario] = []
    flow_profiles = ("low", "moderate", "high", "critical")
    risk_profiles = ("stable", "isolated_high", "isolated_critical", "multi_risk")
    staffing_profiles = ("adequate", "strained", "critical")
    bed_profiles = ("open", "tight", "critical")
    for index in range(count):
        flow_profile = flow_profiles[index % len(flow_profiles)]
        risk_profile = risk_profiles[(index // len(flow_profiles)) % len(risk_profiles)]
        staffing_profile = staffing_profiles[(index // (len(flow_profiles) * len(risk_profiles))) % len(staffing_profiles)]
        bed_profile = bed_profiles[(index // (len(flow_profiles) * len(risk_profiles) * len(staffing_profiles))) % len(bed_profiles)]
        if rng.random() < 0.20:
            flow_profile = rng.choice(flow_profiles)
        if rng.random() < 0.20:
            staffing_profile = rng.choice(staffing_profiles)
        if rng.random() < 0.20:
            bed_profile = rng.choice(bed_profiles)
        flow = _make_flow_profile(flow_profile, rng)
        request = EDRequest(
            timestamp=f"2026-05-20T{8 + index % 14:02d}:00:00Z",
            current_queue_length=flow["current_queue_length"],
            arrivals_last_hour=flow["arrivals_last_hour"],
            average_wait_minutes=flow["average_wait_minutes"],
            boarding_patients=flow["boarding_patients"],
            patients=_make_patient_panel(risk_profile, rng, index),
            staffing=_make_staffing_profile(staffing_profile, rng),
            beds=_make_bed_profile(bed_profile, rng),
        )
        expected_actions, urgent_targets = _expected_actions_for_payload(request)
        scenarios.append(EvaluationScenario(f"mixed_{index + 1:03d}_{flow_profile}_{risk_profile}_{staffing_profile}_{bed_profile}", f"Mixed synthetic ED case with {flow_profile} flow, {risk_profile} patient risk, {staffing_profile} staffing, and {bed_profile} bed pressure.", request, expected_actions, urgent_targets))
    return scenarios


def _make_flow_profile(profile: str, rng: random.Random) -> dict[str, int]:
    profiles = {
        "low": {"current_queue_length": (3, 8), "arrivals_last_hour": (3, 8), "average_wait_minutes": (15, 35), "boarding_patients": (0, 1)},
        "moderate": {"current_queue_length": (9, 16), "arrivals_last_hour": (9, 15), "average_wait_minutes": (45, 75), "boarding_patients": (1, 3)},
        "high": {"current_queue_length": (18, 25), "arrivals_last_hour": (17, 24), "average_wait_minutes": (85, 125), "boarding_patients": (4, 6)},
        "critical": {"current_queue_length": (26, 34), "arrivals_last_hour": (25, 32), "average_wait_minutes": (130, 180), "boarding_patients": (7, 10)},
    }
    return {name: rng.randint(bounds[0], bounds[1]) for name, bounds in profiles[profile].items()}


def _make_staffing_profile(profile: str, rng: random.Random) -> dict[str, Any]:
    profiles = {
        "adequate": {"available_nurses": (7, 10), "available_physicians": (4, 6), "staff_absence_flag": False},
        "strained": {"available_nurses": (4, 6), "available_physicians": (2, 3), "staff_absence_flag": rng.random() < 0.35},
        "critical": {"available_nurses": (1, 3), "available_physicians": (1, 2), "staff_absence_flag": True},
    }
    selected = profiles[profile]
    return {"available_nurses": rng.randint(*selected["available_nurses"]), "available_physicians": rng.randint(*selected["available_physicians"]), "nurse_capacity_per_hour": 4, "physician_capacity_per_hour": 6, "staff_absence_flag": selected["staff_absence_flag"]}


def _make_bed_profile(profile: str, rng: random.Random) -> dict[str, int]:
    profiles = {
        "open": {"occupied_beds": (12, 18), "discharge_ready_beds": (2, 5), "high_acuity_beds_available": (2, 4)},
        "tight": {"occupied_beds": (20, 23), "discharge_ready_beds": (1, 3), "high_acuity_beds_available": (0, 2)},
        "critical": {"occupied_beds": (23, 24), "discharge_ready_beds": (0, 1), "high_acuity_beds_available": (0, 1)},
    }
    selected = profiles[profile]
    return {"total_beds": 24, "occupied_beds": rng.randint(*selected["occupied_beds"]), "discharge_ready_beds": rng.randint(*selected["discharge_ready_beds"]), "high_acuity_beds_available": rng.randint(*selected["high_acuity_beds_available"])}


def _make_patient_panel(profile: str, rng: random.Random, panel_index: int) -> list[dict[str, Any]]:
    panel_size = rng.randint(3, 7)
    patients = [_make_patient(f"ED-{panel_index + 1:03d}-{i + 1:02d}", "stable", rng) for i in range(panel_size)]
    if profile == "isolated_high":
        patients[0] = _make_patient(f"ED-{panel_index + 1:03d}-HR", "high", rng)
    elif profile == "isolated_critical":
        patients[0] = _make_patient(f"ED-{panel_index + 1:03d}-CR", "critical", rng)
    elif profile == "multi_risk":
        patients[0] = _make_patient(f"ED-{panel_index + 1:03d}-CR", "critical", rng)
        patients[1] = _make_patient(f"ED-{panel_index + 1:03d}-HR", "high", rng)
        if len(patients) > 3:
            patients[2] = _make_patient(f"ED-{panel_index + 1:03d}-MR", "moderate", rng)
    elif rng.random() < 0.30:
        patients[0] = _make_patient(f"ED-{panel_index + 1:03d}-MR", "moderate", rng)
    rng.shuffle(patients)
    return patients


def _patient_template(patient_id: str) -> dict[str, Any]:
    return {"patient_id": patient_id, "age": 42, "sex": "female", "triage_level": 4, "chief_complaint": "minor abdominal pain", "triage_notes": "stable vital signs, no red flag symptoms", "heart_rate": 82, "systolic_bp": 124, "respiratory_rate": 17, "oxygen_saturation": 98, "temperature_c": 37.0, "has_abnormal_labs": False, "suspected_sepsis": False, "pain_score": 3, "waiting_minutes": 28}


def _make_patient(patient_id: str, acuity: str, rng: random.Random) -> dict[str, Any]:
    sex = rng.choice(["female", "male", "other"])
    if acuity == "critical":
        return {"patient_id": patient_id, "age": rng.randint(62, 92), "sex": sex, "triage_level": rng.choice([1, 2]), "chief_complaint": rng.choice(["shortness of breath and fever", "chest pain", "confusion and fever"]), "triage_notes": "hypotension, low oxygen saturation, possible sepsis", "heart_rate": rng.randint(122, 145), "systolic_bp": rng.randint(72, 89), "respiratory_rate": rng.randint(26, 34), "oxygen_saturation": rng.randint(84, 91), "temperature_c": round(rng.uniform(38.4, 40.1), 1), "has_abnormal_labs": True, "suspected_sepsis": True, "pain_score": rng.randint(6, 9), "waiting_minutes": rng.randint(8, 55)}
    if acuity == "high":
        return {"patient_id": patient_id, "age": rng.randint(45, 84), "sex": sex, "triage_level": 2, "chief_complaint": rng.choice(["shortness of breath", "chest pain", "syncope"]), "triage_notes": "abnormal respiratory rate and low oxygen saturation", "heart_rate": rng.randint(106, 121), "systolic_bp": rng.randint(90, 104), "respiratory_rate": rng.randint(25, 29), "oxygen_saturation": rng.randint(88, 91), "temperature_c": round(rng.uniform(37.2, 38.4), 1), "has_abnormal_labs": rng.random() < 0.75, "suspected_sepsis": False, "pain_score": rng.randint(5, 8), "waiting_minutes": rng.randint(15, 95)}
    if acuity == "moderate":
        return {"patient_id": patient_id, "age": rng.randint(28, 76), "sex": sex, "triage_level": 3, "chief_complaint": rng.choice(["abdominal pain", "fever", "dizziness"]), "triage_notes": "stable but requires clinician assessment", "heart_rate": rng.randint(92, 110), "systolic_bp": rng.randint(102, 125), "respiratory_rate": rng.randint(18, 23), "oxygen_saturation": rng.randint(93, 97), "temperature_c": round(rng.uniform(36.8, 38.2), 1), "has_abnormal_labs": rng.random() < 0.35, "suspected_sepsis": False, "pain_score": rng.randint(3, 6), "waiting_minutes": rng.randint(45, 150)}
    return {"patient_id": patient_id, "age": rng.randint(18, 72), "sex": sex, "triage_level": rng.choice([3, 4, 4, 5]), "chief_complaint": rng.choice(["minor injury", "mild abdominal pain", "medication refill", "ankle pain"]), "triage_notes": "stable vital signs, no red flag symptoms", "heart_rate": rng.randint(68, 96), "systolic_bp": rng.randint(108, 138), "respiratory_rate": rng.randint(14, 20), "oxygen_saturation": rng.randint(96, 100), "temperature_c": round(rng.uniform(36.4, 37.4), 1), "has_abnormal_labs": False, "suspected_sepsis": False, "pain_score": rng.randint(0, 4), "waiting_minutes": rng.randint(8, 110)}


def _expected_actions_for_payload(ed_input: EDRequest) -> tuple[set[str], set[str]]:
    expected: set[str] = set()
    urgent_targets: set[str] = set()
    top_patient_id, top_score = _oracle_highest_risk_patient(ed_input)
    if top_patient_id and top_score >= 0.58:
        expected.add("escalate_patient")
        urgent_targets.add(top_patient_id)
        if ed_input.beds.total_beds - ed_input.beds.occupied_beds + ed_input.beds.discharge_ready_beds > 0:
            expected.add("admit_support")

    capacity = ed_input.staffing.available_nurses * ed_input.staffing.nurse_capacity_per_hour + ed_input.staffing.available_physicians * ed_input.staffing.physician_capacity_per_hour
    demand = ed_input.arrivals_last_hour + ed_input.current_queue_length * 0.55 + ed_input.boarding_patients * 1.25 + len(ed_input.patients) * 0.75 + sum(1 for patient in ed_input.patients if patient.triage_level <= 2) * 2.0
    staffing_pressure = demand / max(capacity, 1)
    if ed_input.staffing.staff_absence_flag:
        staffing_pressure += 0.15
    if staffing_pressure >= 0.85:
        expected.add("staffing_alert")
    if ed_input.current_queue_length >= 18 or ed_input.arrivals_last_hour >= 20 or ed_input.average_wait_minutes >= 90 or ed_input.boarding_patients >= 5 or staffing_pressure >= 0.95 and ed_input.current_queue_length >= 12:
        expected.add("reprioritize_queue")

    free_beds = max(ed_input.beds.total_beds - ed_input.beds.occupied_beds, 0)
    effective_capacity = free_beds + ed_input.beds.discharge_ready_beds
    occupancy = ed_input.beds.occupied_beds / max(ed_input.beds.total_beds, 1)
    high_acuity_need = sum(1 for patient in ed_input.patients if patient.triage_level <= 2)
    high_acuity_gap = max(high_acuity_need - ed_input.beds.high_acuity_beds_available, 0)
    if occupancy >= 0.95 and effective_capacity <= 2 or free_beds <= 1 and high_acuity_gap > 0:
        expected.add("reassign_bed")
    if not expected:
        expected.add("monitor")
    return expected, urgent_targets


def _oracle_highest_risk_patient(ed_input: EDRequest) -> tuple[str | None, float]:
    best_patient_id = None
    best_score = 0.0
    for patient in ed_input.patients:
        score = {1: 0.45, 2: 0.30, 3: 0.12, 4: 0.03, 5: 0.01}.get(patient.triage_level, 0.0)
        score += 0.20 if patient.systolic_bp < 90 else 0.0
        score += 0.18 if patient.oxygen_saturation < 92 else 0.0
        score += 0.10 if patient.respiratory_rate > 24 or patient.respiratory_rate < 10 else 0.0
        score += 0.08 if patient.heart_rate > 120 or patient.heart_rate < 45 else 0.0
        score += 0.16 if patient.suspected_sepsis else 0.0
        score += 0.06 if patient.has_abnormal_labs else 0.0
        score += 0.05 if patient.age >= 75 else 0.0
        if score > best_score:
            best_patient_id = patient.patient_id
            best_score = score
    return best_patient_id, min(best_score, 1.0)


def score_response(scenario: EvaluationScenario, response: EDDecisionResponse, response_time_ms: float = 0.0) -> ScenarioScore:
    expected_actions = set(scenario.expected_actions)
    raw_predicted_actions = {item.action for item in response.recommendations}
    predicted_actions = _normalize_predicted_actions(raw_predicted_actions, expected_actions)
    tp = len(predicted_actions & expected_actions)
    fp = len(predicted_actions - expected_actions)
    fn = len(expected_actions - predicted_actions)
    precision = tp / max(len(predicted_actions), 1)
    recall = tp / max(len(expected_actions), 1)
    escalation_expected = "escalate_patient" in expected_actions
    escalation_hit = (not escalation_expected) or ("escalate_patient" in predicted_actions)
    target_hit = True
    if escalation_expected and scenario.urgent_targets:
        predicted_targets = {item.target_id for item in response.recommendations if item.action == "escalate_patient" and item.target_id is not None}
        target_hit = bool(predicted_targets & scenario.urgent_targets)
    missed_target_penalty = 0.0 if target_hit else 12.0
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
        alert_count=sum(1 for item in response.recommendations if item.action != "monitor"),
        simulated_delay=round(max(0.0, fn * 8.0 + fp * 2.0 + missed_target_penalty), 2),
        explanation_quality=_score_explanation(response),
        response_time_ms=round(response_time_ms, 3),
    )


def summarize_scores(scores: list[ScenarioScore]) -> dict[str, float]:
    tp = sum(score.true_positives for score in scores)
    fp = sum(score.false_positives for score in scores)
    fn = sum(score.false_negatives for score in scores)
    escalation_cases = [score for score in scores if "escalate_patient" in score.expected_actions]
    return {
        "precision": round(tp / max(tp + fp, 1), 3),
        "recall": round(tp / max(tp + fn, 1), 3),
        "avg_action_quality": round(sum(score.action_quality for score in scores) / max(len(scores), 1), 3),
        "avg_alert_burden": round(sum(score.alert_count for score in scores) / max(len(scores), 1), 3),
        "avg_recommendation_delay": round(sum(score.simulated_delay for score in scores) / max(len(scores), 1), 3),
        "avg_explanation_quality": round(sum(score.explanation_quality for score in scores) / max(len(scores), 1), 3),
        "avg_response_time_ms": round(sum(score.response_time_ms for score in scores) / max(len(scores), 1), 3),
        "escalation_recall": round(sum(1 for score in escalation_cases if score.escalation_hit) / max(len(escalation_cases), 1), 3),
        "escalation_target_accuracy": round(sum(1 for score in escalation_cases if score.escalation_target_hit) / max(len(escalation_cases), 1), 3),
    }


def evaluate_runner(name: str, runner: DecisionRunner) -> dict[str, Any]:
    scenario_results = []
    scores = []
    for scenario in build_evaluation_scenarios():
        start = time.perf_counter()
        response = runner.decide(scenario.payload)
        response_time_ms = (time.perf_counter() - start) * 1000.0
        score = score_response(scenario, response, response_time_ms=response_time_ms)
        scores.append(score)
        scenario_results.append({"scenario": scenario.name, "description": scenario.description, "system_state": response.system_state, "expected_actions": sorted(scenario.expected_actions), "predicted_actions": sorted(score.predicted_actions), "recommendations": [item.model_dump() for item in response.recommendations], "metrics": _score_to_dict(score)})
    return {"system": name, "summary_metrics": summarize_scores(scores), "scenario_results": scenario_results}


def evaluate_all_systems(agentic_runner: DecisionRunner | None = None) -> dict[str, Any]:
    systems: dict[str, DecisionRunner] = {
        "esi_triage_baseline": ESITriageBaseline(),
        "news2_qsofa_baseline": EarlyWarningScoreBaseline(),
        "nedocs_edwin_crowding_baseline": CrowdingScoreBaseline(),
        "prediction_only_baseline": PredictionOnlyBaseline(),
        "rule_based_baseline": RuleBasedEDBaseline(),
        "non_agentic_integrated_baseline": NonAgenticIntegratedBaseline(),
        "agentic_orchestration": agentic_runner or EDOrchestrationAgent(use_llm_summary=False),
    }
    return {name: evaluate_runner(name, runner) for name, runner in systems.items()}


def _run_tools(ed_input: EDRequest) -> tuple[Any, Any, Any, Any]:
    return run_flow_prediction(ed_input), run_patient_risk(ed_input), run_staffing_availability(ed_input), run_bed_management(ed_input)


def _tool_outputs(flow: Any, risk: Any, staffing: Any, bed: Any) -> ToolOutputs:
    return ToolOutputs(flow_prediction=flow, patient_risk=risk, staffing=staffing, bed_management=bed)


def _required_escalation_feedback(recommendations: list[RecommendationItem], tool_outputs: ToolOutputs) -> list[str]:
    required_flags = [
        flag
        for flag in tool_outputs.patient_risk.flagged_patients
        if flag.risk_level in {"high", "critical"}
    ]
    if not required_flags:
        return []

    escalated_patient_ids = {
        item.target_id
        for item in recommendations
        if item.action == "escalate_patient" and item.target_id
    }
    feedback: list[str] = []
    for flag in required_flags:
        if flag.patient_id not in escalated_patient_ids:
            feedback.append(
                "Missing escalate_patient recommendation with "
                f"target_id={flag.patient_id!r} for {flag.risk_level} patient-risk flag."
            )

    if any(item.action == "escalate_patient" and not item.target_id for item in recommendations):
        feedback.append("Every escalate_patient recommendation must include an explicit target_id.")
    return feedback


def _build_action_brief(recommendations: list[RecommendationItem]) -> str:
    if not recommendations:
        return "No immediate ED action was recommended. Continue routine monitoring and reassess if conditions change."
    if all(item.action == "monitor" for item in recommendations):
        return "No immediate escalation is required. Continue monitoring the ED state and re-run the agent if patient risk, crowding, staffing, or bed capacity changes."

    priority_rank = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
    ordered = sorted(recommendations, key=lambda item: priority_rank[item.priority])
    patient_escalations = [item for item in ordered if item.action == "escalate_patient" and item.target_id]
    operational_actions = [item for item in ordered if item.action != "escalate_patient"]

    if patient_escalations:
        patient_ids = _join_phrases([item.target_id or "the patient" for item in patient_escalations[:3]])
        remaining = len(patient_escalations) - 3
        if remaining > 0:
            patient_ids = f"{patient_ids} and {remaining} more patient(s)"
        first_sentence = f"Immediate action: escalate {patient_ids} now."
        next_items = operational_actions[:2]
    else:
        first_sentence = f"Immediate action: {_brief_phrase(ordered[0])}."
        next_items = ordered[1:3]

    if next_items:
        next_text = _join_phrases([_brief_next_action(item) for item in next_items])
        return f"{first_sentence} Next, {next_text}."
    return first_sentence


def _build_agentic_reasoning(planning_trace: dict[str, Any]) -> AgenticReasoning | None:
    if not planning_trace:
        return None
    return AgenticReasoning(
        reasoning_summary=str(planning_trace.get("reasoning_summary", "")),
        goal=str(planning_trace.get("goal", "")),
        plan=_as_text_list(planning_trace.get("plan", [])),
        execute=_as_text_list(planning_trace.get("execute", [])),
        monitor_outcomes=_as_text_list(planning_trace.get("monitor_outcomes", [])),
        replan_if_conditions_change=_as_text_list(planning_trace.get("replan_if_conditions_change", [])),
        continue_until_goal_achieved=str(planning_trace.get("continue_until_goal_achieved", "")),
    )


def _brief_phrase(item: RecommendationItem) -> str:
    target = f" for {item.target_id}" if item.target_id else ""
    action_labels = {
        "escalate_patient": "escalate the patient",
        "reprioritize_queue": "reprioritize the queue",
        "reassign_bed": "review and reassign bed capacity",
        "admit_support": "start admission or high-acuity placement support",
        "discharge_support": "accelerate discharge support",
        "staffing_alert": "activate staffing support",
        "monitor": "continue monitoring",
    }
    return f"{action_labels[item.action]}{target} ({item.priority} priority)"


def _brief_next_action(item: RecommendationItem) -> str:
    target = f" for {item.target_id}" if item.target_id else ""
    action_labels = {
        "escalate_patient": "escalate the patient",
        "reprioritize_queue": "reprioritize the queue",
        "reassign_bed": "review and reassign bed capacity",
        "admit_support": "start admission or high-acuity placement support",
        "discharge_support": "accelerate discharge support",
        "staffing_alert": "activate staffing support",
        "monitor": "continue monitoring",
    }
    return f"{action_labels[item.action]}{target}"


def _join_phrases(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _build_agent_trace(
    tool_outputs: ToolOutputs,
    recommendations: list[RecommendationItem],
    follow_up_plan: list[FollowUpItem],
    planning_mode: str,
) -> list[AgentTraceItem]:
    return [
        AgentTraceItem(
            agent="orchestration",
            step="selected tools and compared operational constraints",
            evidence=[
                f"planning_mode={planning_mode}",
                f"actions={', '.join(item.action for item in recommendations)}",
                f"follow_up_tasks={len(follow_up_plan)}",
            ],
        ),
        AgentTraceItem(
            agent="flow_tool",
            step="estimated ED congestion and waiting pressure",
            evidence=[
                f"bottleneck={tool_outputs.flow_prediction.bottleneck_level}",
                f"predicted_wait={tool_outputs.flow_prediction.predicted_wait_minutes}",
            ],
        ),
        AgentTraceItem(
            agent="patient_risk_tool",
            step="ranked patients by clinical risk",
            evidence=[
                f"highest_risk={tool_outputs.patient_risk.highest_risk_patient_id}",
                f"flagged={len(tool_outputs.patient_risk.flagged_patients)}",
            ],
        ),
        AgentTraceItem(
            agent="staffing_tool",
            step="estimated staffing pressure",
            evidence=[
                f"level={tool_outputs.staffing.staffing_level}",
                f"pressure={tool_outputs.staffing.staffing_pressure_score}",
            ],
        ),
        AgentTraceItem(
            agent="bed_tool",
            step="checked bed and high-acuity capacity",
            evidence=[
                f"window={tool_outputs.bed_management.action_window}",
                f"free_beds={tool_outputs.bed_management.available_beds_now}",
            ],
        ),
    ]


def _build_llm_cycle_trace(planning_trace: dict[str, Any]) -> list[AgentTraceItem]:
    if not planning_trace:
        return []
    trace_items: list[AgentTraceItem] = []
    cycle_steps = (
        ("goal", "LLM goal"),
        ("plan", "LLM plan"),
        ("execute", "LLM execute"),
        ("monitor_outcomes", "LLM monitor outcomes"),
        ("replan_if_conditions_change", "LLM re-plan if conditions change"),
        ("continue_until_goal_achieved", "LLM continue until goal achieved"),
    )
    for step_name, label in cycle_steps:
        evidence = planning_trace.get(step_name, [])
        if isinstance(evidence, str):
            evidence = [evidence]
        if evidence:
            trace_items.append(
                AgentTraceItem(
                    agent="orchestration",
                    step=label,
                    evidence=[str(item) for item in evidence[:4]],
                )
            )
    summary = planning_trace.get("reasoning_summary")
    if summary:
        trace_items.append(
            AgentTraceItem(
                agent="orchestration",
                step="LLM reasoning summary",
                evidence=[str(summary)],
            )
        )
    return trace_items


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM response.")
    return text[start : end + 1]


def _derive_system_state(flow_level: str, staffing_level: str, bed_window: str, top_patient: Any) -> str:
    if flow_level == "critical" or staffing_level == "critical" or bed_window == "critical" or (top_patient and top_patient.risk_level == "critical"):
        return "critical"
    if flow_level == "high" or staffing_level == "strained" or bed_window == "tight" or (top_patient and top_patient.risk_level == "high"):
        return "strained"
    if flow_level == "moderate" or (top_patient and top_patient.risk_level == "moderate"):
        return "watch"
    return "stable"


def _derive_state_from_recommendations(recommendations: list[RecommendationItem]) -> str:
    if any(item.priority == "urgent" for item in recommendations):
        return "critical"
    if any(item.priority == "high" for item in recommendations):
        return "strained"
    return "stable"


def _score_explanation(response: EDDecisionResponse) -> float:
    recommendations = response.recommendations
    if not recommendations:
        return 0.0
    if all(item.action == "monitor" for item in recommendations):
        return 0.45
    evidence_terms = {"agent", "tool", "risk", "flow", "staff", "bed", "capacity", "wait", "patient", "pressure", "triage", "news2", "qsofa", "nedocs", "edwin", "crowding", "acuity", "score"}
    reason_hits = 0
    for item in recommendations:
        reason = item.reason.lower()
        if len(reason.split()) >= 8 and any(term in reason for term in evidence_terms):
            reason_hits += 1
    reason_score = reason_hits / max(len(recommendations), 1)
    tool_outputs = response.tool_outputs
    tool_details = [
        bool(tool_outputs.flow_prediction.reasoning and tool_outputs.flow_prediction.recommended_adjustment),
        bool(tool_outputs.patient_risk.recommended_adjustment or tool_outputs.patient_risk.flagged_patients),
        bool(tool_outputs.staffing.reasoning and tool_outputs.staffing.recommended_adjustment),
        bool(tool_outputs.bed_management.reasoning and tool_outputs.bed_management.recommended_adjustment),
    ]
    tool_score = sum(tool_details) / len(tool_details)
    summary = response.summary.lower()
    summary_score = 1.0 if len(summary.split()) >= 10 else 0.5
    return round(0.45 * reason_score + 0.35 * tool_score + 0.20 * summary_score, 3)


def _normalize_predicted_actions(predicted_actions: set[str], expected_actions: set[str]) -> set[str]:
    if "monitor" in predicted_actions and expected_actions != {"monitor"}:
        return predicted_actions - {"monitor"}
    return predicted_actions


def _score_to_dict(score: ScenarioScore) -> dict[str, Any]:
    data = asdict(score)
    data["predicted_actions"] = sorted(score.predicted_actions)
    data["expected_actions"] = sorted(score.expected_actions)
    return data


def _news2_score(patient: Any) -> int:
    score = 0
    if patient.respiratory_rate <= 8:
        score += 3
    elif 9 <= patient.respiratory_rate <= 11:
        score += 1
    elif 21 <= patient.respiratory_rate <= 24:
        score += 2
    elif patient.respiratory_rate >= 25:
        score += 3
    if patient.oxygen_saturation <= 91:
        score += 3
    elif 92 <= patient.oxygen_saturation <= 93:
        score += 2
    elif 94 <= patient.oxygen_saturation <= 95:
        score += 1
    if patient.temperature_c <= 35.0:
        score += 3
    elif 35.1 <= patient.temperature_c <= 36.0:
        score += 1
    elif 38.1 <= patient.temperature_c <= 39.0:
        score += 1
    elif patient.temperature_c >= 39.1:
        score += 2
    if patient.systolic_bp <= 90:
        score += 3
    elif 91 <= patient.systolic_bp <= 100:
        score += 2
    elif 101 <= patient.systolic_bp <= 110:
        score += 1
    elif patient.systolic_bp >= 220:
        score += 3
    if patient.heart_rate <= 40:
        score += 3
    elif 41 <= patient.heart_rate <= 50:
        score += 1
    elif 91 <= patient.heart_rate <= 110:
        score += 1
    elif 111 <= patient.heart_rate <= 130:
        score += 2
    elif patient.heart_rate >= 131:
        score += 3
    text = f"{patient.chief_complaint} {patient.triage_notes}".lower()
    if "confusion" in text or "altered mental" in text:
        score += 3
    return score


def _qsofa_score(patient: Any) -> int:
    text = f"{patient.chief_complaint} {patient.triage_notes}".lower()
    return sum([patient.respiratory_rate >= 22, patient.systolic_bp <= 100, "confusion" in text or "altered mental" in text])


def _crowding_score(ed_input: EDRequest) -> float:
    occupancy = ed_input.beds.occupied_beds / max(ed_input.beds.total_beds, 1)
    return 35.0 * occupancy + 2.0 * ed_input.current_queue_length + 2.2 * ed_input.arrivals_last_hour + 4.0 * ed_input.boarding_patients + 0.35 * ed_input.average_wait_minutes
