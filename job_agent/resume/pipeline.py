from pathlib import Path
import re

from job_agent.config import load_llm_config
from job_agent.resume.extractor import extract_resume
from job_agent.resume.loader import load_markdown
from job_agent.resume.normalizer import normalize_resume
from job_agent.resume.schema import Resume
from job_agent.resume.splitter import heading_name, split_sections

_LOCATION_RE = re.compile(r"期望城市[:：]\s*(\S+)")
_PHONE_RE = re.compile(r"电话[:：]\s*([0-9\-\s　]+)")
_TARGET_ROLE_RE = re.compile(
    r"(?:求职岗位|求职方向|期望职位|意向岗位|目标岗位)[:：]\s*(.+)"
)


def _fill_personal_info(resume: Resume, sections: dict[str, str]) -> Resume:
    info = resume.personal_info
    if not info.name:
        info.name = heading_name(sections)
    if not info.location:
        match = _LOCATION_RE.search(sections.get("personal_info") or "")
        if match:
            info.location = match.group(1)
    if not info.phone:
        match = _PHONE_RE.search(sections.get("personal_info") or "")
        if match:
            info.phone = match.group(1)
    return resume


def _fill_target_role(resume: Resume, sections: dict[str, str]) -> Resume:
    if resume.target_role:
        return resume
    blob = "\n".join(
        sections.get(key) or ""
        for key in ("target_role", "personal_info", "title", "other")
    )
    match = _TARGET_ROLE_RE.search(blob)
    if match:
        resume.target_role = match.group(1).strip()
    return resume


def parse_resume(path: str | Path) -> Resume:
    markdown = load_markdown(path)
    sections = split_sections(markdown)
    config = load_llm_config()
    extracted = extract_resume(sections, config)
    filled = _fill_personal_info(extracted, sections)
    filled = _fill_target_role(filled, sections)
    return normalize_resume(filled)
