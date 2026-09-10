from __future__ import annotations

from job_agent.matching.schemas import Decision, DimensionScore, JobRequirement
from job_agent.matching.state import MatchingState
from job_agent.resume.schema import Resume


async def jd_parse_node(state: MatchingState) -> dict[str, JobRequirement]:
    """Stub JD parsing node for graph compilation."""

    return {"job_requirement": JobRequirement()}


async def resume_extract_node(state: MatchingState) -> dict[str, Resume]:
    """Stub resume extraction node for graph compilation."""

    return {"resume": Resume()}


async def parse_join_node(state: MatchingState) -> dict[str, object]:
    """Synchronization node that waits for JD and resume parsing."""

    return {}


def _dimension_score(dimension: str) -> dict[str, dict[str, DimensionScore]]:
    score = DimensionScore(
        dimension=dimension,
        score=3,
        evidence="stub",
    )
    return {"dimension_scores": {dimension: score}}


async def eval_skills_node(state: MatchingState) -> dict[str, dict[str, DimensionScore]]:
    """Stub skills evaluation node."""

    return _dimension_score("skills")


async def eval_years_node(state: MatchingState) -> dict[str, dict[str, DimensionScore]]:
    """Stub years evaluation node."""

    return _dimension_score("years")


async def eval_education_node(
    state: MatchingState,
) -> dict[str, dict[str, DimensionScore]]:
    """Stub education evaluation node."""

    return _dimension_score("education")


async def eval_location_node(
    state: MatchingState,
) -> dict[str, dict[str, DimensionScore]]:
    """Stub location evaluation node."""

    return _dimension_score("location")


async def eval_responsibilities_node(
    state: MatchingState,
) -> dict[str, dict[str, DimensionScore]]:
    """Stub responsibilities evaluation node."""

    return _dimension_score("responsibilities")


async def decide_node(state: MatchingState) -> dict[str, Decision]:
    """Stub decision node."""

    return {
        "decision": Decision(
            average=3.0,
            recommendation="待定",
        )
    }


async def report_node(state: MatchingState) -> dict[str, str]:
    """Stub report node."""

    return {"report": "stub report"}
