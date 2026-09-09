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


def parse_resume(path: str | Path) -> Resume:
    markdown = load_markdown(path)
    sections = split_sections(markdown)
    config = load_llm_config()
    extracted = extract_resume(sections, config)
    filled = _fill_personal_info(extracted, sections)
    return normalize_resume(filled)
