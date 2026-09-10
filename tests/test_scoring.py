from datetime import date

from job_agent.matching.schemas import JobRequirement
from job_agent.matching.scoring import (
    hit_ratio_to_score,
    score_education,
    score_location,
    score_responsibilities,
    score_skills,
    score_years,
)
from job_agent.resume.schema import Education, PersonalInfo, Resume, WorkExperience


def test_hit_ratio_to_score_buckets():
    assert hit_ratio_to_score(0.0) == 1
    assert hit_ratio_to_score(0.25) == 2
    assert hit_ratio_to_score(0.5) == 3
    assert hit_ratio_to_score(0.89) == 4
    assert hit_ratio_to_score(0.9) == 5


def test_empty_must_have_skills_is_three():
    job = JobRequirement(must_have_skills=[])
    resume = Resume(skills=["python"])
    result = score_skills(job, resume)
    assert result.dimension == "skills"
    assert result.score == 3


def test_all_skills_hit_is_five():
    job = JobRequirement(must_have_skills=["python", "api"])
    resume = Resume(skills=["python", "fastapi"])
    assert score_skills(job, resume).score == 5


def test_both_locations_empty_is_three():
    job = JobRequirement(location=None)
    resume = Resume(personal_info=PersonalInfo(location=None))
    assert score_location(job, resume).score == 3


def test_years_meets_requirement():
    job = JobRequirement(years_required="3年以上")
    resume = Resume(
        work_experience=[
            WorkExperience(start_date="2018-01", end_date="present"),
        ]
    )
    result = score_years(job, resume, today=date(2026, 9, 10))
    assert result.score >= 4


def test_education_bachelor_required():
    job = JobRequirement(education_required="本科及以上")
    resume = Resume(education=[Education(degree="本科")])
    assert score_education(job, resume).score == 4


def test_responsibilities_hit_is_five():
    job = JobRequirement(responsibilities=["设计 Agent Runtime", "优化 记忆 系统"])
    resume = Resume(
        work_experience=[
            WorkExperience(
                responsibilities=["负责设计 agent runtime 架构"],
                achievements=["主导记忆系统优化"],
            )
        ]
    )
    assert score_responsibilities(job, resume).score == 5
