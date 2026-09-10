from __future__ import annotations

import logging

from langchain.tools import tool

from job_agent.kb.errors import KbConfigError, KbNotFoundError, KbStoreError
from job_agent.kb.pipeline import ingest_resume, load_profile, open_kb, open_mysql
from job_agent.matching.errors import MatchingExtractError
from job_agent.resume.errors import (
    ResumeConfigError,
    ResumeExtractError,
    ResumeFileError,
)

logger = logging.getLogger(__name__)


def is_resume_id_ref(value: str) -> bool:
    """Return whether a resume reference is a decimal knowledge-base ID."""
    return value.isdecimal()


@tool
async def ingest_resume_file(path: str) -> str:
    """Ingest a resume file and return its vectorized profile JSON."""
    logger.info("ingesting resume file", extra={"resume_path": path})
    try:
        async with await open_kb() as handles:
            resume_id = await ingest_resume(
                path,
                mysql=handles.mysql,
                milvus=handles.milvus,
                embedder=handles.embedder,
            )
            profile = await load_profile(resume_id, mysql=handles.mysql)
            return profile.model_dump_json()
    except (
        KbConfigError,
        KbNotFoundError,
        KbStoreError,
        ResumeConfigError,
        ResumeExtractError,
        ResumeFileError,
    ):
        raise
    except Exception as exc:  # noqa: BLE001
        raise MatchingExtractError(f"简历入库失败：{exc}") from exc


@tool
async def load_resume_by_id(resume_id: int) -> str:
    """Load a vectorized resume profile by knowledge-base ID."""
    logger.info("loading resume profile", extra={"resume_id": resume_id})
    async with open_mysql() as mysql:
        profile = await load_profile(resume_id, mysql=mysql)
        return profile.model_dump_json()
