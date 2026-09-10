from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from sqlalchemy import func, select

from job_agent.kb.models import ChunkType
from job_agent.kb.mysql_store import ResumeChunkRow
from job_agent.kb.pipeline import ingest_resume, load_profile, open_kb
from job_agent.kb.stores import ResumeMysqlStore

DEFAULT_SAMPLE = "/Users/yannic/面试/宗艳云简历v9.md"

ALL_CHUNK_TYPES: list[ChunkType] = [
    "personal_info",
    "skills",
    "education",
    "work_experience",
    "project",
    "summary",
]


def _sample_path() -> Path:
    return Path(os.environ.get("RESUME_SAMPLE_PATH", DEFAULT_SAMPLE))


async def _count_chunks(mysql: ResumeMysqlStore, resume_id: int) -> int:
    async with mysql._sessions() as session:
        count = await session.scalar(
            select(func.count())
            .select_from(ResumeChunkRow)
            .where(ResumeChunkRow.resume_id == resume_id)
        )
    return int(count or 0)


async def _count_vectorized(mysql: ResumeMysqlStore, resume_id: int) -> int:
    async with mysql._sessions() as session:
        count = await session.scalar(
            select(func.count())
            .select_from(ResumeChunkRow)
            .where(
                ResumeChunkRow.resume_id == resume_id,
                ResumeChunkRow.vector_status == "vectorized",
            )
        )
    return int(count or 0)


@pytest.mark.live
def test_ingest_sample_resume_live() -> None:
    path = _sample_path()
    if not path.is_file():
        pytest.skip(f"sample resume not found: {path}")

    async def _run() -> None:
        async with await open_kb() as handles:
            resume_id = await ingest_resume(
                str(path),
                mysql=handles.mysql,
                milvus=handles.milvus,
                embedder=handles.embedder,
            )
            profile = await load_profile(resume_id, mysql=handles.mysql)
            assert profile.experience_months or profile.highest_degree

            chunk_count = await _count_chunks(handles.mysql, resume_id)
            assert chunk_count >= 4

            vectorized_count = await _count_vectorized(handles.mysql, resume_id)
            query_vector = handles.embedder.embed(["resume chunks"])[0]
            hits = await handles.milvus.search(
                resume_id=resume_id,
                chunk_types=list(ALL_CHUNK_TYPES),
                query_vector=query_vector,
                top_k=100,
            )
            assert len(hits) == vectorized_count

    asyncio.run(_run())
