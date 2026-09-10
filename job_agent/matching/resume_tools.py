from __future__ import annotations

import asyncio
import logging

from langchain.tools import tool

from job_agent.matching.errors import MatchingExtractError
from job_agent.resume.errors import ResumeExtractError
from job_agent.resume.pipeline import parse_resume

logger = logging.getLogger(__name__)


def _parse_resume_to_json(path: str) -> str:
    resume = parse_resume(path)
    return resume.model_dump_json()


@tool
async def parse_resume_file(path: str) -> str:
    """Parse a resume file into normalized JSON. 日后可改为从向量库按 ID 读取，本章不实现"""
    logger.info("parsing resume file", extra={"resume_path": path})
    try:
        return await asyncio.to_thread(_parse_resume_to_json, path)
    except ResumeExtractError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise MatchingExtractError(f"简历解析失败：{exc}") from exc
