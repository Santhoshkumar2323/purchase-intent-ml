from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SessionIn(BaseModel):
    model_config = ConfigDict(extra="forbid") 

    Administrative: int = Field(0, ge=0, le=100)
    Administrative_Duration: float = Field(0, ge=0)
    Informational: int = Field(0, ge=0, le=100)
    Informational_Duration: float = Field(0, ge=0)
    ProductRelated: int = Field(0, ge=0, le=2000)
    ProductRelated_Duration: float = Field(0, ge=0)
    BounceRates: float = Field(0, ge=0, le=1)
    ExitRates: float = Field(0, ge=0, le=1)
    PageValues: float = Field(0, ge=0)
    SpecialDay: float = Field(0, ge=0, le=1)
    Weekend: bool = False
    OperatingSystems: int = Field(2, ge=1, le=20)
    Browser: int = Field(2, ge=1, le=20)
    Region: int = Field(1, ge=1, le=20)
    TrafficType: int = Field(2, ge=1, le=30)
    VisitorType: Literal["Returning_Visitor", "New_Visitor", "Other"] = "Returning_Visitor"

    def to_row(self) -> dict:
        """The shape the training pipeline expects: categories as text, Weekend as 0/1."""
        row = self.model_dump()
        row["Weekend"] = int(row["Weekend"])
        for col in ("OperatingSystems", "Browser", "Region", "TrafficType"):
            row[col] = str(row[col])
        return row


class PredictOut(BaseModel):
    probability: float
    action: str
    model_version: str


class HealthOut(BaseModel):
    status: str
    model_version: str
    trained_at: str | None = None
    detail: str | None = None


class Reason(BaseModel):
    feature: str
    value: float | int | str | None = None
    effect: float
    direction: Literal["up", "down"]


class ExplainOut(PredictOut):
    reasons: list[Reason]
    note: str = ("Effects are SHAP values in log-odds of the raw XGBoost score, before "
                 "calibration. They show direction and relative size, not exact probability points.")
