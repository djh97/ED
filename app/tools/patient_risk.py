from __future__ import annotations

from app.enhanced_models import XGBoostRiskPredictor
from app.models import EDRequest, PatientRiskToolResult


_MODEL = XGBoostRiskPredictor()


def run_patient_risk(ed_input: EDRequest) -> PatientRiskToolResult:
    return _MODEL.predict(ed_input)
