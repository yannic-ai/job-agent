from __future__ import annotations

import asyncio
import logging
from typing import Callable

from langchain.tools import tool

from job_agent.matching.scoring import (
    score_education,
    score_location,
    score_responsibilities,
    score_skills,
    score_years,
)
from job_agent.matching.schemas import JobRequirement
from job_agent.resume.schema import Resume

logger = logging.getLogger(__name__)


async def _score_to_json(
    scorer: Callable[[JobRequirement, Resume], object],
    job_json: str,
    resume_json: str,
) -> str:
    job = JobRequirement.model_validate_json(job_json)
    resume = Resume.model_validate_json(resume_json)
    score = await asyncio.to_thread(scorer, job, resume)
    return score.model_dump_json()


@tool
async def score_skills_tool(job_json: str, resume_json: str) -> str:
    """Score the skills dimension and return DimensionScore JSON."""
    logger.info("scoring skills dimension")
    return await _score_to_json(score_skills, job_json, resume_json)


@tool
async def score_years_tool(job_json: str, resume_json: str) -> str:
    """Score the years dimension and return DimensionScore JSON."""
    logger.info("scoring years dimension")
    return await _score_to_json(score_years, job_json, resume_json)


@tool
async def score_education_tool(job_json: str, resume_json: str) -> str:
    """Score the education dimension and return DimensionScore JSON."""
    logger.info("scoring education dimension")
    return await _score_to_json(score_education, job_json, resume_json)


@tool
async def score_location_tool(job_json: str, resume_json: str) -> str:
    """Score the location dimension and return DimensionScore JSON."""
    logger.info("scoring location dimension")
    return await _score_to_json(score_location, job_json, resume_json)


@tool
async def score_responsibilities_tool(job_json: str, resume_json: str) -> str:
    """Score the responsibilities dimension and return DimensionScore JSON."""
    logger.info("scoring responsibilities dimension")
    return await _score_to_json(score_responsibilities, job_json, resume_json)
