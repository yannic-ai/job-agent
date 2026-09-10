from __future__ import annotations

import asyncio
import logging

from langchain.tools import tool

from job_agent.matching.jd_loader import load_jd
from job_agent.matching.jd_splitter import format_jd_sections, split_jd_sections

logger = logging.getLogger(__name__)


def _load_and_format_jd(path: str) -> str:
    raw_jd = load_jd(path)
    sections = split_jd_sections(raw_jd)
    return format_jd_sections(sections)


@tool
async def read_and_split_jd(path: str) -> str:
    """Read a JD file, split sections, and return normalized markdown."""
    logger.info("reading and splitting jd", extra={"jd_path": path})
    return await asyncio.to_thread(_load_and_format_jd, path)
