from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from app.additional_evaluations import evaluate_safety_validation, evaluate_stateful_replanning
from app.agentic_system import _build_agentic_reasoning, _required_escalation_feedback
from app.models import EDRequest, EDStateUpdate, FollowUpItem, RecommendationItem, ToolOutputs
from app.nhamcs_ed import _build_request, _labels, _temperature_c
from app.orchestrator import EDOrchestrationAgent, RuleBasedEDBaseline, build_evaluation_scenarios, evaluate_all_systems
from app.state_manager import EDStateManager
from app.tools.bed_management import run_bed_management
from app.tools.flow_prediction import run_flow_prediction
from app.tools.patient_risk import run_patient_risk
from app.tools.staffing import run_staffing_availability


class OrchestratorTests(unittest.TestCase):
    @unittest.skipUnless(os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY is required for real LLM orchestration tests.")
    def test_high_risk_patient_triggers_escalation(self) -> None:
        payload = json.loads(Path("demo_data/sample_case.json").read_text(encoding="utf-8"))
        request = EDRequest(**payload)

        result = EDOrchestrationAgent(use_llm_summary=False).decide(request)

        actions = {item.action for item in result.recommendations}
        self.assertIn("escalate_patient", actions)
        self.assertIn("staffing_alert", actions)
        self.assertIn("reassign_bed", actions)
        self.assertEqual(result.system_state, "critical")
        self.assertGreaterEqual(len(result.follow_up_plan), len(result.recommendations))
        self.assertIn("input_understanding", {item.agent for item in result.agent_trace})
        self.assertIn("follow_up_tracking", {item.agent for item in result.agent_trace})
        orchestration_trace = [item for item in result.agent_trace if item.agent == "orchestration"][0]
        self.assertIn("planning_mode=llm_prompted", orchestration_trace.evidence)
        self.assertIn("LLM plan", {item.step for item in result.agent_trace})
        self.assertIn("LLM monitor outcomes", {item.step for item in result.agent_trace})

    def test_evaluation_includes_all_comparison_systems(self) -> None:
        comparison = evaluate_all_systems(agentic_runner=RuleBasedEDBaseline())

        self.assertEqual(
            set(comparison),
            {
                "esi_triage_baseline",
                "news2_qsofa_baseline",
                "nedocs_edwin_crowding_baseline",
                "prediction_only_baseline",
                "rule_based_baseline",
                "non_agentic_integrated_baseline",
                "agentic_orchestration",
            },
        )
        agentic_metrics = comparison["agentic_orchestration"]["summary_metrics"]
        rule_metrics = comparison["rule_based_baseline"]["summary_metrics"]
        prediction_metrics = comparison["prediction_only_baseline"]["summary_metrics"]

        self.assertGreaterEqual(agentic_metrics["avg_action_quality"], rule_metrics["avg_action_quality"])
        self.assertGreater(agentic_metrics["avg_action_quality"], prediction_metrics["avg_action_quality"])
        self.assertIn("avg_response_time_ms", agentic_metrics)
        self.assertIn("avg_explanation_quality", agentic_metrics)

    def test_default_evaluation_uses_180_scenarios(self) -> None:
        self.assertEqual(len(build_evaluation_scenarios()), 180)

    def test_state_manager_keeps_previous_patient_when_new_patient_arrives(self) -> None:
        payload = json.loads(Path("demo_data/sample_case.json").read_text(encoding="utf-8"))
        initial = EDRequest(**payload)
        manager = EDStateManager()
        manager.update_from_full_snapshot(initial)

        new_patient = {
            "patient_id": "ED-NEW",
            "age": 63,
            "triage_level": 2,
            "heart_rate": 118,
            "systolic_bp": 94,
            "respiratory_rate": 26,
            "oxygen_saturation": 91,
            "temperature_c": 38.1,
            "has_abnormal_labs": True,
            "suspected_sepsis": False,
            "pain_score": 8,
            "waiting_minutes": 5,
        }
        updated = manager.apply_update(
            EDStateUpdate(
                timestamp="2026-05-19T10:12:00Z",
                current_queue_length=19,
                patients=[new_patient],
            )
        )

        patient_ids = {patient.patient_id for patient in updated.patients}
        self.assertIn("ED-001", patient_ids)
        self.assertIn("ED-NEW", patient_ids)
        self.assertEqual(len(updated.patients), 4)

    def test_state_manager_removes_discharged_patient(self) -> None:
        payload = json.loads(Path("demo_data/sample_case.json").read_text(encoding="utf-8"))
        manager = EDStateManager()
        manager.update_from_full_snapshot(EDRequest(**payload))

        updated = manager.apply_update(EDStateUpdate(discharged_patient_ids=["ED-002"]))
        patient_ids = {patient.patient_id for patient in updated.patients}

        self.assertNotIn("ED-002", patient_ids)
        self.assertEqual(len(updated.patients), 2)

    def test_state_manager_deduplicates_operational_followups(self) -> None:
        manager = EDStateManager()
        first = FollowUpItem(
            task_id="FU-001",
            linked_action="staffing_alert",
            owner="charge_nurse",
            due_minutes=10,
            escalation_rule="Escalate if not accepted.",
            reason="Staffing tool shows critical pressure.",
        )
        second = FollowUpItem(
            task_id="FU-001",
            linked_action="staffing_alert",
            owner="charge_nurse",
            due_minutes=10,
            escalation_rule="Escalate if not accepted.",
            reason="Staffing tool still shows critical pressure after another patient arrived.",
        )

        manager.merge_follow_up_plan([first])
        pending = manager.merge_follow_up_plan([second])

        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].task_id, "SFU-001")
        self.assertIn("still shows critical pressure", pending[0].reason)

    def test_required_escalation_validator_requires_targeted_high_risk_patient(self) -> None:
        payload = json.loads(Path("demo_data/sample_case.json").read_text(encoding="utf-8"))
        request = EDRequest(**payload)
        tool_outputs = ToolOutputs(
            flow_prediction=run_flow_prediction(request),
            patient_risk=run_patient_risk(request),
            staffing=run_staffing_availability(request),
            bed_management=run_bed_management(request),
        )

        missing = _required_escalation_feedback(
            [RecommendationItem(action="monitor", priority="medium", reason="Continue monitoring the ED state.")],
            tool_outputs,
        )
        satisfied = _required_escalation_feedback(
            [
                RecommendationItem(
                    action="escalate_patient",
                    priority="urgent",
                    target_id="ED-001",
                    reason="Patient risk tool flagged ED-001 as critical.",
                )
            ],
            tool_outputs,
        )

        self.assertTrue(any("ED-001" in item for item in missing))
        self.assertEqual(satisfied, [])

    def test_agentic_reasoning_formats_planning_trace(self) -> None:
        reasoning = _build_agentic_reasoning(
            {
                "reasoning_summary": "Goal -> Plan -> Execute -> Monitor -> Re-plan -> Continue.",
                "goal": "Reduce ED risk while maintaining throughput.",
                "plan": ["Use all active patients and tools."],
                "execute": ["Run risk, flow, staffing, and bed tools."],
                "monitor_outcomes": ["Watch unresolved escalations."],
                "replan_if_conditions_change": ["Escalate priority if staffing worsens."],
                "continue_until_goal_achieved": "Continue until escalated or safely monitored.",
            }
        )

        self.assertIsNotNone(reasoning)
        self.assertEqual(reasoning.goal, "Reduce ED risk while maintaining throughput.")
        self.assertIn("Run risk", reasoning.execute[0])

    def test_nhamcs_row_maps_to_ed_request_and_labels(self) -> None:
        row = {
            "year": "2022",
            "visit_month": "5",
            "day_of_week": "2",
            "arrival_time": "1430",
            "age": "77",
            "sex": "Female",
            "temp": "101.3",
            "heart_rate": "132",
            "resp_rate": "28",
            "sys_bp": "84",
            "spo2": "89",
            "pain_score": "7",
            "target_triage_acuity": "2",
            "wait_time_minutes": "95",
            "ems_arrival": "Yes",
            "seen_last_72h": "1",
            "chief_complaint_text": "fever. shortness of breath",
        }

        request = _build_request(row, "NHAMCS-TEST", {"2022-5-2-14": 6})
        labels = _labels(row)

        self.assertEqual(request.patients[0].patient_id, "NHAMCS-TEST")
        self.assertEqual(request.patients[0].triage_level, 2)
        self.assertEqual(request.patients[0].sex, "female")
        self.assertTrue(request.patients[0].suspected_sepsis)
        self.assertTrue(labels["high_acuity"])
        self.assertTrue(labels["critical_vitals"])
        self.assertTrue(labels["prolonged_wait"])
        self.assertTrue(labels["revisit_72h"])
        self.assertAlmostEqual(_temperature_c("101.3"), 38.5)

    def test_safety_validation_evaluation_catches_missing_escalations(self) -> None:
        metrics = evaluate_safety_validation()

        self.assertEqual(metrics["scenarios"], 180)
        self.assertGreater(metrics["scenarios_requiring_escalation"], 0)
        self.assertEqual(metrics["missing_plan_detection_rate"], 1.0)
        self.assertEqual(metrics["valid_plan_pass_rate"], 1.0)

    def test_stateful_replanning_evaluation_tracks_active_patients(self) -> None:
        metrics = evaluate_stateful_replanning(RuleBasedEDBaseline())

        self.assertTrue(metrics["memory_retention_pass"])
        self.assertTrue(metrics["replanning_pass"])
        self.assertEqual([step["active_patient_count"] for step in metrics["steps"]], [1, 2, 3])
        self.assertIn("ED-003", metrics["steps"][-1]["critical_targets"])

    def test_agentic_planner_requires_llm(self) -> None:
        payload = json.loads(Path("demo_data/sample_case.json").read_text(encoding="utf-8"))
        request = EDRequest(**payload)

        with self.assertRaises(RuntimeError):
            EDOrchestrationAgent(use_llm_summary=False, use_llm_planning=False, use_llm_input=False).decide(request)

    @unittest.skipUnless(os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY is required for real LLM free-text extraction tests.")
    def test_free_text_input_uses_llm_extractor(self) -> None:
        text = (
            "Queue 18, arrivals 22, average wait 95, boarding 6. "
            "Nurses 3, physicians 1, total beds 24, occupied beds 23. "
            "Patient age 77, triage 2, HR 128, SBP 86, RR 30, SpO2 89, "
            "temp 38.9, shortness of breath, fever, sepsis."
        )
        result = EDOrchestrationAgent(use_llm_summary=False).decide(text)
        actions = {item.action for item in result.recommendations}
        self.assertIn("escalate_patient", actions)
        self.assertIn("input_understanding", {item.agent for item in result.agent_trace})


if __name__ == "__main__":
    unittest.main()
