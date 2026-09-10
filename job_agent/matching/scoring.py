from __future__ import annotations

import re
from datetime import date

from job_agent.kb.profile import (
    _calculate_experience_years,
    _degree_label,
    _highest_resume_degree,
    _months_between,
    _parse_year_month,
)
from job_agent.matching.schemas import DimensionScore, JobRequirement
from job_agent.resume.schema import Resume

_RESPONSIBILITY_TOKEN_PATTERN = re.compile(r"[A-Za-z]{2,}|[\u4e00-\u9fff]{2,}")


def hit_ratio_to_score(ratio: float) -> int:
    """Map a hit ratio to the configured 1-5 score buckets."""
    if ratio <= 0.0:
        return 1
    if ratio <= 0.25:
        return 2
    if ratio <= 0.5:
        return 3
    if ratio < 0.9:
        return 4
    return 5


def score_skills(job: JobRequirement, resume: Resume) -> DimensionScore:
    """Score required-skill coverage using exact or substring matching."""
    required_skills = [_normalize_text(skill) for skill in job.must_have_skills if skill.strip()]
    resume_skills = [_normalize_text(skill) for skill in resume.skills if skill.strip()]
    if not required_skills:
        return DimensionScore(dimension="skills", score=3, evidence="岗位未提供必须技能")

    hit_count = sum(
        1
        for required_skill in required_skills
        if any(_contains_either_way(required_skill, resume_skill) for resume_skill in resume_skills)
    )
    total = len(required_skills)
    score = hit_ratio_to_score(hit_count / total)
    return DimensionScore(
        dimension="skills",
        score=score,
        evidence=f"必须技能 {hit_count}/{total} 命中",
    )


def score_years(
    job: JobRequirement,
    resume: Resume,
    *,
    today: date | None = None,
) -> DimensionScore:
    """Score years of experience from the earliest start to latest end month."""
    today_value = today or date.today()
    required_years = _extract_required_years(job.years_required)
    if required_years is None:
        return DimensionScore(dimension="years", score=3, evidence="岗位年限要求缺失")

    actual_years = _calculate_experience_years(resume, today_value)
    if actual_years is None:
        return DimensionScore(dimension="years", score=3, evidence="简历缺少可解析工作年限")

    if actual_years >= required_years + 2:
        score = 5
    elif actual_years >= required_years:
        score = 4
    elif actual_years >= required_years - 1:
        score = 3
    elif actual_years >= required_years - 3:
        score = 2
    else:
        score = 1
    return DimensionScore(
        dimension="years",
        score=score,
        evidence=f"工作年限约 {actual_years:.1f} 年，要求 {required_years} 年",
    )


def score_education(job: JobRequirement, resume: Resume) -> DimensionScore:
    """Score education against the highest degree mentioned in the JD."""
    required_level = _highest_required_degree(job.education_required)
    if required_level is None:
        return DimensionScore(dimension="education", score=3, evidence="岗位未提供学历要求")
    if not resume.education:
        return DimensionScore(dimension="education", score=3, evidence="简历缺少学历信息")

    candidate_level = _highest_resume_degree(resume)
    candidate_label = _degree_label(candidate_level) or "未识别"

    if required_level == 3:
        score = 5 if candidate_level == 3 else 2
    elif required_level == 2:
        if candidate_level == 3:
            score = 5
        elif candidate_level == 2:
            score = 4
        elif candidate_level == 1:
            score = 3
        else:
            score = 2
    else:
        score = 4 if candidate_level >= 1 else 2

    return DimensionScore(
        dimension="education",
        score=score,
        evidence=f"最高学历 {candidate_label}，对照岗位学历要求评分",
    )


def score_location(job: JobRequirement, resume: Resume) -> DimensionScore:
    """Score location by normalized substring containment."""
    job_location = _compact_text(job.location)
    resume_location = _compact_text(resume.personal_info.location)

    if not job_location and not resume_location:
        return DimensionScore(dimension="location", score=3, evidence="岗位与候选人地点均缺失")
    if job_location and resume_location and (
        job_location in resume_location or resume_location in job_location
    ):
        return DimensionScore(
            dimension="location",
            score=5,
            evidence=f"地点匹配：岗位 {job_location}，候选人 {resume_location}",
        )
    return DimensionScore(
        dimension="location",
        score=2,
        evidence=f"地点不匹配：岗位 {job_location or '缺失'}，候选人 {resume_location or '缺失'}",
    )


def score_responsibilities(job: JobRequirement, resume: Resume) -> DimensionScore:
    """Score JD responsibilities by keyword hits in work and project history."""
    responsibilities = [text.strip() for text in job.responsibilities if text.strip()]
    if not responsibilities:
        return DimensionScore(
            dimension="responsibilities",
            score=3,
            evidence="岗位未提供职责要求",
        )

    corpus = _build_responsibility_corpus(resume)
    if not corpus:
        return DimensionScore(
            dimension="responsibilities",
            score=1,
            evidence=f"职责 0/{len(responsibilities)} 命中",
        )

    hit_count = 0
    for responsibility in responsibilities:
        keywords = _RESPONSIBILITY_TOKEN_PATTERN.findall(responsibility)
        if any(keyword.lower() in corpus for keyword in keywords):
            hit_count += 1

    total = len(responsibilities)
    score = hit_ratio_to_score(hit_count / total)
    return DimensionScore(
        dimension="responsibilities",
        score=score,
        evidence=f"职责 {hit_count}/{total} 命中",
    )


def _normalize_text(value: str) -> str:
    return value.strip().lower()


def _compact_text(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", "", value)


def _contains_either_way(left: str, right: str) -> bool:
    return bool(left and right and (left in right or right in left))


def _extract_required_years(text: str | None) -> int | None:
    if not text:
        return None
    match = re.search(r"\d+", text)
    return int(match.group(0)) if match else None


def _highest_required_degree(text: str | None) -> int | None:
    if not text:
        return None
    if "博士" in text:
        return 3
    if "硕士" in text:
        return 2
    if "本科" in text:
        return 1
    return None


def _build_responsibility_corpus(resume: Resume) -> str:
    texts: list[str] = []
    for experience in resume.work_experience:
        texts.extend(experience.responsibilities)
        texts.extend(experience.achievements)
    for project in resume.projects:
        texts.extend(project.responsibilities)
        texts.extend(project.achievements)
    return "\n".join(texts).lower()
