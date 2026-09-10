from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from job_agent.kb.pipeline import ingest_resume, open_kb
from job_agent.kb.retrieve import search_resume_chunks

DEFAULT_SAMPLE = "/Users/yannic/面试/宗艳云简历v9.md"


def _sample_path() -> Path:
    return Path(os.environ.get("RESUME_SAMPLE_PATH", DEFAULT_SAMPLE))


@pytest.mark.live
def test_retrieve_skills_and_responsibilities_live() -> None:
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

            skills_hits = await search_resume_chunks(
                resume_id,
                ["skills"],
                "精通 python 与大模型应用",
                top_k=1,
                mysql=handles.mysql,
                milvus=handles.milvus,
                embedder=handles.embedder,
            )
            assert skills_hits, "skills retrieval returned no hits"
            assert skills_hits[0].chunk_type == "skills"

            resp_hits = await search_resume_chunks(
                resume_id,
                ["work_experience", "project"],
                "Agent 平台 任务编排 记忆系统",
                top_k=5,
                mysql=handles.mysql,
                milvus=handles.milvus,
                embedder=handles.embedder,
            )
            assert resp_hits, "responsibilities retrieval returned no hits"
            assert resp_hits[0].chunk_type in ("work_experience", "project")

    asyncio.run(_run())
