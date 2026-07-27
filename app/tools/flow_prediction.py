from __future__ import annotations

from app.enhanced_models import TCNFlowForecaster
from app.models import EDRequest, FlowToolResult


_MODEL = TCNFlowForecaster()


def run_flow_prediction(ed_input: EDRequest) -> FlowToolResult:
    return _MODEL.predict(ed_input)
