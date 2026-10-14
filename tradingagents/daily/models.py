from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PROMPT_VERSION = "daily-v1"
Role = Literal["market", "news", "social", "fundamentals", "decision"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("A timezone-aware timestamp is required")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


class DailyForecast(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    close: float = Field(gt=0, strict=True, description="Predicted regular-session close in USD")
    low: float = Field(gt=0, strict=True, description="Lower end of an 80% close-price interval")
    high: float = Field(gt=0, strict=True, description="Upper end of an 80% close-price interval")
    up_probability: float = Field(
        ge=0, le=1, strict=True,
        description="Probability that close is strictly above the frozen previous-session close",
    )
    rationale: str = Field(min_length=1, max_length=6000)
    risks: str = Field(min_length=1, max_length=4000)

    @model_validator(mode="after")
    def ordered_interval(self):
        if not self.low <= self.close <= self.high:
            raise ValueError("Require low <= close <= high")
        return self


class LessonCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Role
    condition: str = Field(min_length=1, max_length=800)
    advice: str = Field(min_length=1, max_length=1200)


class LessonEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lesson_id: int = Field(gt=0)
    verdict: Literal["supports", "contradicts", "untested"]
    evidence: str = Field(min_length=1, max_length=1200)


class DailyReflection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis: str = Field(min_length=1, max_length=6000)
    candidate: LessonCandidate | None
    evaluations: list[LessonEvaluation] = Field(max_length=8)


def score(forecast: DailyForecast, reference: float, actual: float) -> dict:
    """Price, binary direction, fixed 80% interval, and probability scores."""
    if not (0 < reference < float("inf") and 0 < actual < float("inf")):
        raise ValueError("Reference and actual prices must be finite and positive")
    error = forecast.close - actual
    # No retrospective neutral band: a flat close is explicitly not an up day.
    up = actual > reference
    width = forecast.high - forecast.low
    interval_score = width
    if actual < forecast.low:
        interval_score += 10 * (forecast.low - actual)
    elif actual > forecast.high:
        interval_score += 10 * (actual - forecast.high)
    return {
        "error": error,
        "absolute_error": abs(error),
        "absolute_percentage_error": abs(error) / actual,
        "direction_hit": (forecast.close > reference) == up,
        "interval_hit": forecast.low <= actual <= forecast.high,
        "interval_width": width,
        "interval_score": interval_score,
        "brier": (forecast.up_probability - int(up)) ** 2,
        "actual_up": up,
        "actual_return": actual / reference - 1,
        "naive_absolute_error": abs(reference - actual),
        "naive_absolute_percentage_error": abs(reference - actual) / actual,
    }
