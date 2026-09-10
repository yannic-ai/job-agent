from __future__ import annotations
import json
import logging
from typing import Callable

from pydantic import BaseModel

from job_agent.config import LLMConfig, load_llm_config
from job_agent.kb.models import ResumeProfile
from job_agent.kb.profile import build_resume_profile
from job_agent.matching.decide_prompts import build_decide_prompt
from job_agent.matching.decide_tools import average_and_label_tool
from job_agent.matching.decision import average_and_label
from job_agent.matching.errors import MatchingExtractError
from job_agent.matching.evaluate_prompts import build_evaluate_prompt
from job_agent.matching.evaluate_tools import (
    score_education_tool,
    score_location_tool,
    score_responsibilities_tool,
    score_skills_tool,
    score_years_tool,
)
from job_agent.matching.jd_prompts import build_jd_prompt
from job_agent.matching.jd_tools import read_and_split_jd
from job_agent.matching.report import render_report
from job_agent.matching.report_prompts import build_report_prompt
from job_agent.matching.report_tools import render_report_tool
from job_agent.matching.runtime import run_expert
from job_agent.matching.scoring import (
    score_education,
    score_location,
    score_responsibilities,
    score_skills,
    score_years,
)
from job_agent.matching.schemas import (
    DIMENSIONS,
    Decision,
    DimensionScore,
    JobRequirement,
)
from job_agent.matching.resume_prompts import build_resume_prompt
from job_agent.matching.resume_tools import parse_resume_file
from job_agent.matching.state import MatchingState
from job_agent.resume.schema import Resume

logger = logging.getLogger(__name__)

DimensionScorer = Callable[[JobRequirement, Resume], DimensionScore]
ProfileScorer = Callable[[JobRequirement, ResumeProfile], DimensionScore]


class ReportOutput(BaseModel):
    """Structured output schema for the report expert."""

    report: str


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
    """Parse the resume file into a structured resume."""
    logger.info("parsing resume", extra={"resume_path": state["resume_path"]})
    prompt = build_resume_prompt()
    messages = await prompt.aformat_messages(resume_path=state["resume_path"])
    result = await run_expert(
        messages=messages,
        tools=[parse_resume_file],
        output_schema=Resume,
        config=load_llm_config(),
    )
    return {"resume": result}


async def parse_join_node(state: MatchingState) -> dict[str, object]:
    """Synchronization node that waits for JD and resume parsing."""

    return {}


async def eval_skills_node(state: MatchingState) -> dict[str, dict[str, DimensionScore]]:
    """Evaluate the skills dimension."""
    job, resume = _require_job_and_resume(state)
    job_slice = JobRequirement(
        must_have_skills=job.must_have_skills,
        nice_to_have_skills=job.nice_to_have_skills,
    )
    resume_slice = Resume(skills=resume.skills)
    return await _evaluate_dimension_node(
        config=load_llm_config(),
        dimension_label="技能",
        tool=score_skills_tool,
        scorer=score_skills,
        job_slice=job_slice,
        resume_slice=resume_slice,
    )


async def eval_years_node(state: MatchingState) -> dict[str, dict[str, DimensionScore]]:
    """Evaluate the years dimension."""
    job, resume = _require_job_and_resume(state)
    job_slice = JobRequirement(years_required=job.years_required)
    profile_slice = build_resume_profile(resume)
    return await _evaluate_profile_dimension_node(
        config=load_llm_config(),
        dimension_label="年限",
        tool=score_years_tool,
        scorer=score_years,
        job_slice=job_slice,
        profile_slice=profile_slice,
    )


async def eval_education_node(
    state: MatchingState,
) -> dict[str, dict[str, DimensionScore]]:
    """Evaluate the education dimension."""
    job, resume = _require_job_and_resume(state)
    job_slice = JobRequirement(education_required=job.education_required)
    profile_slice = build_resume_profile(resume)
    return await _evaluate_profile_dimension_node(
        config=load_llm_config(),
        dimension_label="学历",
        tool=score_education_tool,
        scorer=score_education,
        job_slice=job_slice,
        profile_slice=profile_slice,
    )


async def eval_location_node(
    state: MatchingState,
) -> dict[str, dict[str, DimensionScore]]:
    """Evaluate the location dimension."""
    job, resume = _require_job_and_resume(state)
    job_slice = JobRequirement(location=job.location)
    profile_slice = build_resume_profile(resume)
    return await _evaluate_profile_dimension_node(
        config=load_llm_config(),
        dimension_label="地点",
        tool=score_location_tool,
        scorer=score_location,
        job_slice=job_slice,
        profile_slice=profile_slice,
    )


async def eval_responsibilities_node(
    state: MatchingState,
) -> dict[str, dict[str, DimensionScore]]:
    """Evaluate the responsibilities dimension."""
    job, resume = _require_job_and_resume(state)
    job_slice = JobRequirement(responsibilities=job.responsibilities)
    resume_slice = Resume(
        work_experience=resume.work_experience,
        projects=resume.projects,
    )
    return await _evaluate_dimension_node(
        config=load_llm_config(),
        dimension_label="职责",
        tool=score_responsibilities_tool,
        scorer=score_responsibilities,
        job_slice=job_slice,
        resume_slice=resume_slice,
    )


async def decide_node(state: MatchingState) -> dict[str, Decision]:
    """Make the final decision from the five dimension scores."""
    score_items = _ordered_dimension_scores(state)
    score_values = [item.score for item in score_items]
    prompt = build_decide_prompt()
    messages = await prompt.aformat_messages(
        scores_json=_to_json([item.model_dump() for item in score_items])
    )
    await run_expert(
        messages=messages,
        tools=[average_and_label_tool],
        output_schema=Decision,
        config=load_llm_config(),
    )
    return {"decision": average_and_label(score_values)}


async def report_node(state: MatchingState) -> dict[str, str]:
    """Render the final matching report."""
    job_requirement = state.get("job_requirement")
    resume = state.get("resume")
    decision = state.get("decision")
    if job_requirement is None or resume is None or decision is None:
        raise MatchingExtractError("报告节点缺少岗位、简历或决策结果")

    score_items = _ordered_dimension_scores(state)
    title = job_requirement.title
    candidate_name = resume.personal_info.name
    scores_json = _to_json([item.model_dump() for item in score_items])
    decision_json = decision.model_dump_json()
    prompt = build_report_prompt()
    messages = await prompt.aformat_messages(
        title=title or "未提供",
        candidate_name=candidate_name or "未提供",
        scores_json=scores_json,
        decision_json=decision_json,
    )
    await run_expert(
        messages=messages,
        tools=[render_report_tool],
        output_schema=ReportOutput,
        config=load_llm_config(),
    )
    report = render_report(
        title=title,
        candidate_name=candidate_name,
        scores=score_items,
        decision=decision,
    )
    return {"report": report}


def _require_job_and_resume(state: MatchingState) -> tuple[JobRequirement, Resume]:
    """Validate that the current state has both job and resume objects."""
    job_requirement = state.get("job_requirement")
    resume = state.get("resume")
    if job_requirement is None or resume is None:
        raise MatchingExtractError("评估节点缺少岗位需求或简历信息")
    return job_requirement, resume


async def _evaluate_dimension_node(
    *,
    config: LLMConfig,
    dimension_label: str,
    tool: object,
    scorer: DimensionScorer,
    job_slice: JobRequirement,
    resume_slice: Resume,
) -> dict[str, dict[str, DimensionScore]]:
    """Run one evaluation expert and persist the deterministic Python score."""
    prompt = build_evaluate_prompt()
    job_json = job_slice.model_dump_json()
    resume_json = resume_slice.model_dump_json()
    messages = await prompt.aformat_messages(
        dimension_label=dimension_label,
        job_requirement_json=job_json,
        resume_json=resume_json,
    )
    await run_expert(
        messages=messages,
        tools=[tool],
        output_schema=DimensionScore,
        config=config,
    )
    score = scorer(job_slice, resume_slice)
    return {"dimension_scores": {score.dimension: score}}


async def _evaluate_profile_dimension_node(
    *,
    config: LLMConfig,
    dimension_label: str,
    tool: object,
    scorer: ProfileScorer,
    job_slice: JobRequirement,
    profile_slice: ResumeProfile,
) -> dict[str, dict[str, DimensionScore]]:
    """Run one profile-based evaluation expert and persist the Python score."""
    prompt = build_evaluate_prompt()
    job_json = job_slice.model_dump_json()
    profile_json = profile_slice.model_dump_json()
    messages = await prompt.aformat_messages(
        dimension_label=dimension_label,
        job_requirement_json=job_json,
        resume_json=profile_json,
    )
    await run_expert(
        messages=messages,
        tools=[tool],
        output_schema=DimensionScore,
        config=config,
    )
    score = scorer(job_slice, profile_slice)
    return {"dimension_scores": {score.dimension: score}}


def _ordered_dimension_scores(state: MatchingState) -> list[DimensionScore]:
    """Collect and validate the five dimension scores in fixed order."""
    score_map = state.get("dimension_scores") or {}
    missing = [dimension for dimension in DIMENSIONS if dimension not in score_map]
    if missing:
        raise MatchingExtractError(f"流程缺少维度分数：{', '.join(missing)}")
    return [score_map[dimension] for dimension in DIMENSIONS]


def _to_json(value: object) -> str:
    """Serialize prompt payloads into JSON for human messages."""
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    return json.dumps(value, ensure_ascii=False)
