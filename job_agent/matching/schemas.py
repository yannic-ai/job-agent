from typing import Literal

from pydantic import BaseModel, Field, field_validator

DIMENSIONS = (
    "skills",
    "years",
    "education",
    "location",
    "responsibilities",
)

Recommendation = Literal["推荐", "待定", "不推荐"]
DimensionName = Literal[
    "skills",
    "years",
    "education",
    "location",
    "responsibilities",
]


def _coerce_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


class JobRequirement(BaseModel):
    title: str | None = None
    must_have_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    years_required: str | None = None
    education_required: str | None = None
    location: str | None = None
    responsibilities: list[str] = Field(default_factory=list)

    @field_validator(
        "must_have_skills",
        "nice_to_have_skills",
        "responsibilities",
        mode="before",
    )
    @classmethod
    def _lists(cls, value: object) -> list[str]:
        return _coerce_str_list(value)


class DimensionScore(BaseModel):
    dimension: DimensionName
    score: int
    evidence: str

    @field_validator("score")
    @classmethod
    def _score_range(cls, value: int) -> int:
        if value not in {1, 2, 3, 4, 5}:
            raise ValueError("score must be 1-5")
        return value

    @field_validator("evidence")
    @classmethod
    def _evidence_non_empty(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("evidence must be non-empty")
        return text


def label_from_average(average: float) -> Recommendation:
    if average >= 4.0:
        return "推荐"
    if average >= 3.0:
        return "待定"
    return "不推荐"


class Decision(BaseModel):
    average: float
    recommendation: Recommendation
