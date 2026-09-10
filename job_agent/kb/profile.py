from __future__ import annotations

import re
from datetime import date

from job_agent.kb.models import ResumeProfile
from job_agent.resume.schema import Resume

_DATE_PATTERN = re.compile(r"^(?P<year>\d{4})-(?P<month>\d{2})$")


def build_resume_profile(
    resume: Resume,
    *,
    today: date | None = None,
) -> ResumeProfile:
    """Derive searchable scalar fields from a parsed resume."""
    today_value = today or date.today()
    highest_degree = _degree_label(_highest_resume_degree(resume))
    experience_months = _calculate_experience_months(resume, today_value)
    personal_info = resume.personal_info
    return ResumeProfile(
        source_path="",
        name=personal_info.name,
        location=personal_info.location,
        phone=personal_info.phone,
        email=personal_info.email,
        highest_degree=highest_degree,
        experience_months=experience_months,
    )


def _calculate_experience_years(resume: Resume, today: date) -> float | None:
    """Calculate experience across the earliest start and latest end."""
    months = _calculate_experience_months(resume, today)
    return months / 12 if months is not None else None


def _calculate_experience_months(resume: Resume, today: date) -> int | None:
    """Calculate months across the earliest start and latest end."""
    parsed_ranges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for experience in resume.work_experience:
        start_value = _parse_year_month(experience.start_date, today)
        end_value = _parse_year_month(experience.end_date, today)
        if start_value is None or end_value is None:
            continue
        parsed_ranges.append((start_value, end_value))

    if not parsed_ranges:
        return None

    earliest_start = min(start for start, _ in parsed_ranges)
    latest_end = max(end for _, end in parsed_ranges)
    return max(0, _months_between(earliest_start, latest_end))


def _parse_year_month(value: str | None, today: date) -> tuple[int, int] | None:
    """Parse YYYY-MM and resolve ``present`` against the supplied date."""
    if not value:
        return None
    if value.strip().lower() == "present":
        return today.year, today.month

    match = _DATE_PATTERN.fullmatch(value.strip())
    if not match:
        return None

    year = int(match.group("year"))
    month = int(match.group("month"))
    if month < 1 or month > 12:
        return None
    return year, month


def _months_between(start: tuple[int, int], end: tuple[int, int]) -> int:
    """Return the signed calendar-month difference between two dates."""
    return (end[0] - start[0]) * 12 + (end[1] - start[1])


def _highest_resume_degree(resume: Resume) -> int:
    """Return the highest recognized resume degree as a numeric level."""
    highest = 0
    for education in resume.education:
        degree_text = education.degree or ""
        if "博士" in degree_text:
            highest = max(highest, 3)
        elif "硕士" in degree_text:
            highest = max(highest, 2)
        elif "本科" in degree_text:
            highest = max(highest, 1)
    return highest


def _degree_label(level: int) -> str | None:
    """Map a recognized numeric degree level to its canonical label."""
    if level == 3:
        return "博士"
    if level == 2:
        return "硕士"
    if level == 1:
        return "本科"
    return None
