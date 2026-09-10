from __future__ import annotations

import re

from job_agent.resume.schema import (
    Education,
    PersonalInfo,
    Project,
    Resume,
    WorkExperience,
)

_PLACEHOLDERS = {"xxx", "保密", "na", "n/a"}
_DATE_RE = re.compile(
    r"^\s*(\d{4})\s*[.\-/年]\s*(\d{1,2})\s*月?\s*$"
)
_RANGE_SPLIT = re.compile(r"\s+[-–—~]\s+")
_PRESENT = {"至今", "现在", "present", "current", "ongoing"}


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if not text or text.casefold() in _PLACEHOLDERS:
        return None
    return text


def normalize_phone(value: str | None) -> str | None:
    text = _blank_to_none(value)
    if text is None:
        return None
    digits = re.sub(r"\D", "", text)
    return digits or None


def normalize_date(value: str | None) -> str | None:
    text = _blank_to_none(value)
    if text is None:
        return None
    if text.casefold() in _PRESENT:
        return "present"
    match = _DATE_RE.match(text)
    if not match:
        return None
    return f"{match.group(1)}-{int(match.group(2)):02d}"


def normalize_date_pair(
    start: str | None, end: str | None
) -> tuple[str | None, str | None]:
    start_text = (start or "").strip()
    end_text = (end or "").strip()
    if start_text and not end_text:
        parts = [part for part in _RANGE_SPLIT.split(start_text) if part]
        if len(parts) == 2:
            return normalize_date(parts[0]), normalize_date(parts[1])
    return normalize_date(start), normalize_date(end)


def normalize_skills(skills: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in skills:
        token = item.strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _clean_list(items: list[str]) -> list[str]:
    return [item.strip() for item in items if item.strip()]


def normalize_resume(resume: Resume) -> Resume:
    info = resume.personal_info
    return Resume(
        personal_info=PersonalInfo(
            name=_blank_to_none(info.name),
            phone=normalize_phone(info.phone),
            email=_blank_to_none(info.email),
            location=_blank_to_none(info.location),
        ),
        education=[
            Education(
                school=_blank_to_none(item.school),
                degree=_blank_to_none(item.degree),
                major=_blank_to_none(item.major),
                start_date=start,
                end_date=end,
            )
            for item in resume.education
            for start, end in [normalize_date_pair(item.start_date, item.end_date)]
        ],
        work_experience=[
            WorkExperience(
                company=_blank_to_none(item.company),
                title=_blank_to_none(item.title),
                start_date=start,
                end_date=end,
                responsibilities=_clean_list(item.responsibilities),
                achievements=_clean_list(item.achievements),
            )
            for item in resume.work_experience
            for start, end in [normalize_date_pair(item.start_date, item.end_date)]
        ],
        projects=[
            Project(
                name=_blank_to_none(item.name),
                role=_blank_to_none(item.role),
                start_date=start,
                end_date=end,
                responsibilities=_clean_list(item.responsibilities),
                achievements=_clean_list(item.achievements),
            )
            for item in resume.projects
            for start, end in [normalize_date_pair(item.start_date, item.end_date)]
        ],
        skills=normalize_skills(resume.skills),
        summary=_blank_to_none(resume.summary),
        target_role=_blank_to_none(resume.target_role),
    )
