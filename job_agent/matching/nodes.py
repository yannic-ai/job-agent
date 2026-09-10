from __future__ import annotations

import logging

from job_agent.config import load_llm_config
from job_agent.matching.jd_prompts import build_jd_prompt
from job_agent.matching.jd_tools import read_and_split_jd
from job_agent.matching.runtime import run_expert
from job_agent.matching.schemas import Decision, DimensionScore, JobRequirement
from job_agent.matching.state import MatchingState
from job_agent.resume.schema import Resume

logger = logging.getLogger(__name__)


async def jd_parse_node(state: MatchingState) -> dict[str, JobRequirement]:
    """Parse the JD file into a structured job requirement."""
    logger.info("parsing jd requirement", extra={"jd_path": state["jd_path"]})
    prompt = build_jd_prompt()
    messages = await prompt.aformat_messages(jd_path=state["jd_path"])
    result = await run_expert(
        messages=messages,
        tools=[read_and_split_jd],
        output_schema=JobRequirement,
        config=load_llm_config(),
    )
    return {"job_requirement": result}


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
