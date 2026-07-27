from __future__ import annotations

from app.enhanced_models import TwoStageStaffingModel
from app.models import EDRequest, StaffingToolResult


_MODEL = TwoStageStaffingModel()


def run_staffing_availability(ed_input: EDRequest) -> StaffingToolResult:
    return _MODEL.predict(ed_input)
