from __future__ import annotations

from typing import Annotated, NotRequired, TypedDict

from job_agent.matching.schemas import Decision, DimensionScore, JobRequirement
from job_agent.resume.schema import Resume


def merge_scores(
    left: dict[str, DimensionScore] | None,
    right: dict[str, DimensionScore] | None,
) -> dict[str, DimensionScore]:
    """Merge parallel dimension score updates by key."""

    return {**(left or {}), **(right or {})}


class MatchingState(TypedDict):
    jd_path: str
    resume_path: str
    job_requirement: NotRequired[JobRequirement | None]
    resume: NotRequired[Resume | None]
    dimension_scores: Annotated[dict[str, DimensionScore], merge_scores]
    decision: NotRequired[Decision | None]
    report: NotRequired[str | None]
