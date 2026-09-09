from pathlib import Path

from job_agent.config import load_llm_config
from job_agent.resume.extractor import extract_resume
from job_agent.resume.loader import load_markdown
from job_agent.resume.normalizer import normalize_resume
from job_agent.resume.schema import Resume
from job_agent.resume.splitter import split_sections


def parse_resume(path: str | Path) -> Resume:
    markdown = load_markdown(path)
    sections = split_sections(markdown)
    config = load_llm_config()
    extracted = extract_resume(sections, config)
    return normalize_resume(extracted)
