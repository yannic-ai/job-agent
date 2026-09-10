from __future__ import annotations
import json
import logging
from typing import Callable

from pydantic import BaseModel

from job_agent.config import LLMConfig, load_llm_config
from job_agent.kb.models import ResumeChunkHit, ResumeProfile
from job_agent.kb.pipeline import open_kb
from job_agent.kb.retrieve import search_resume_chunks
from job_agent.matching.decide_prompts import build_decide_prompt
from job_agent.matching.decide_tools import average_and_label_tool
from job_agent.matching.decision import average_and_label
from job_agent.matching.errors import MatchingExtractError
from job_agent.matching.evaluate_prompts import build_evaluate_prompt
from job_agent.matching.evaluate_tools import (
    missing_retrieval_score_tool,
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
    missing_retrieval_score,
    score_education,
    score_location,
    score_responsibilities,
    score_skills,
    score_years,
)
from job_agent.matching.schemas import (
    DIMENSIONS,
    Decision,
    DimensionName,
    DimensionScore,
    JobRequirement,
)
from job_agent.matching.resume_prompts import build_resume_prompt
from job_agent.matching.resume_tools import (
    ingest_resume_file,
    is_resume_id_ref,
    load_resume_by_id,
)
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


async def resume_extract_node(state: MatchingState) -> dict[str, object]:
    """Load or ingest a resume reference into a structured profile."""
    resume_ref = state["resume_ref"]
    id_mode = is_resume_id_ref(resume_ref)
    logger.info("loading resume", extra={"resume_ref": resume_ref})
    prompt = build_resume_prompt(is_id=id_mode)
    messages = await prompt.aformat_messages(resume_ref=resume_ref)
    result = await run_expert(
        messages=messages,
        tools=[load_resume_by_id if id_mode else ingest_resume_file],
        output_schema=ResumeProfile,
        config=load_llm_config(),
    )
    if result.id is None:
        raise MatchingExtractError("简历档案缺少 ID")
    return {"resume_id": result.id, "resume_profile": result}


async def parse_join_node(state: MatchingState) -> dict[str, object]:
    """Synchronization node that waits for JD and resume parsing."""
    _require_job_and_profile(state)
    return {}


async def eval_skills_node(state: MatchingState) -> dict[str, dict[str, DimensionScore]]:
    """Evaluate the skills dimension."""
    job, _ = _require_job_and_profile(state)
    resume_id = _require_resume_id(state)
    job_slice = JobRequirement(
        must_have_skills=job.must_have_skills,
        nice_to_have_skills=job.nice_to_have_skills,
    )
    query = ", ".join(job.must_have_skills + job.nice_to_have_skills)
    async with await open_kb() as handles:
        hits = await search_resume_chunks(
            resume_id,
            ["skills"],
            query,
            1,
            mysql=handles.mysql,
            milvus=handles.milvus,
            embedder=handles.embedder,
        )
    resume_slice = build_skills_resume_slice(hits)
    if resume_slice is None:
        return await _evaluate_missing_retrieval_node(
            config=load_llm_config(),
            dimension_label="技能",
            dimension="skills",
            job_slice=job_slice,
        )
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
    job, profile = _require_job_and_profile(state)
    job_slice = JobRequirement(years_required=job.years_required)
    return await _evaluate_profile_dimension_node(
        config=load_llm_config(),
        dimension_label="年限",
        tool=score_years_tool,
        scorer=score_years,
        job_slice=job_slice,
        profile_slice=profile,
    )


async def eval_education_node(
    state: MatchingState,
) -> dict[str, dict[str, DimensionScore]]:
    """Evaluate the education dimension."""
    job, profile = _require_job_and_profile(state)
    job_slice = JobRequirement(education_required=job.education_required)
    return await _evaluate_profile_dimension_node(
        config=load_llm_config(),
        dimension_label="学历",
        tool=score_education_tool,
        scorer=score_education,
        job_slice=job_slice,
        profile_slice=profile,
    )


async def eval_location_node(
    state: MatchingState,
) -> dict[str, dict[str, DimensionScore]]:
    """Evaluate the location dimension."""
    job, profile = _require_job_and_profile(state)
    job_slice = JobRequirement(location=job.location)
    return await _evaluate_profile_dimension_node(
        config=load_llm_config(),
        dimension_label="地点",
        tool=score_location_tool,
        scorer=score_location,
        job_slice=job_slice,
        profile_slice=profile,
    )


async def eval_responsibilities_node(
    state: MatchingState,
) -> dict[str, dict[str, DimensionScore]]:
    """Evaluate the responsibilities dimension."""
    job, _ = _require_job_and_profile(state)
    resume_id = _require_resume_id(state)
    job_slice = JobRequirement(responsibilities=job.responsibilities)
    query = "\n".join(job.responsibilities)
    async with await open_kb() as handles:
        hits = await search_resume_chunks(
            resume_id,
            ["work_experience", "project"],
            query,
            5,
            mysql=handles.mysql,
            milvus=handles.milvus,
            embedder=handles.embedder,
        )
    resume_slice = build_responsibilities_resume_slice(hits)
    if resume_slice is None:
        return await _evaluate_missing_retrieval_node(
            config=load_llm_config(),
            dimension_label="职责",
            dimension="responsibilities",
            job_slice=job_slice,
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
    profile = state.get("resume_profile")
    decision = state.get("decision")
    if job_requirement is None or profile is None or decision is None:
        raise MatchingExtractError("报告节点缺少岗位、简历或决策结果")

    score_items = _ordered_dimension_scores(state)
    title = job_requirement.title
    candidate_name = profile.name
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


def build_skills_resume_slice(hits: list[ResumeChunkHit]) -> Resume | None:
    """Build the skills-only resume slice from retrieval hits."""
    for hit in hits:
        if hit.chunk_type == "skills":
            return Resume(skills=hit.payload.get("skills"))
    return None


def build_responsibilities_resume_slice(
    hits: list[ResumeChunkHit],
) -> Resume | None:
    """Build work and project resume slices from retrieval hits."""
    work_experience = [
        hit.payload for hit in hits if hit.chunk_type == "work_experience"
    ]
    projects = [hit.payload for hit in hits if hit.chunk_type == "project"]
    if not work_experience and not projects:
        return None
    return Resume(
        work_experience=work_experience,
        projects=projects,
    )


def _require_job_and_profile(
    state: MatchingState,
) -> tuple[JobRequirement, ResumeProfile]:
    """Validate that the current state has both job and resume profile."""
    job_requirement = state.get("job_requirement")
    profile = state.get("resume_profile")
    if job_requirement is None or profile is None:
        raise MatchingExtractError("评估节点缺少岗位需求或简历信息")
    return job_requirement, profile


def _require_resume_id(state: MatchingState) -> int:
    """Return the resume ID required by retrieval nodes."""
    resume_id = state.get("resume_id")
    if resume_id is None:
        raise MatchingExtractError("评估节点缺少简历 ID")
    return resume_id


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


async def _evaluate_missing_retrieval_node(
    *,
    config: LLMConfig,
    dimension_label: str,
    dimension: DimensionName,
    job_slice: JobRequirement,
) -> dict[str, dict[str, DimensionScore]]:
    """Run the expert with the neutral no-retrieval scoring tool."""
    prompt = build_evaluate_prompt()
    messages = await prompt.aformat_messages(
        dimension_label=dimension_label,
        job_requirement_json=job_slice.model_dump_json(),
        resume_json=Resume().model_dump_json(),
    )
    await run_expert(
        messages=messages,
        tools=[missing_retrieval_score_tool],
        output_schema=DimensionScore,
        config=config,
    )
    score = missing_retrieval_score(dimension)
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
