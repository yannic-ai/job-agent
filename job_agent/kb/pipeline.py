from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from job_agent.config import KbConfig, load_kb_config
from job_agent.kb.chunker import chunk_resume
from job_agent.kb.embedder import BgeM3Embedder
from job_agent.kb.errors import KbNotFoundError, KbStoreError
from job_agent.kb.milvus_store import PyMilvusResumeStore
from job_agent.kb.models import ResumeProfile
from job_agent.kb.mysql_store import SqlAlchemyResumeStore
from job_agent.kb.profile import build_resume_profile
from job_agent.kb.stores import Embedder, ResumeMilvusStore, ResumeMysqlStore
from job_agent.resume.loader import load_markdown
from job_agent.resume.pipeline import parse_resume
from job_agent.resume.schema import Resume

logger = logging.getLogger(__name__)


@dataclass
class KbHandles:
    """Concrete knowledge-base services with async resource cleanup."""

    mysql: SqlAlchemyResumeStore
    milvus: PyMilvusResumeStore
    embedder: BgeM3Embedder

    async def __aenter__(self) -> KbHandles:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        """Close storage clients owned by these handles."""
        await self.milvus.close()
        await self.mysql.close()


async def open_kb(config: KbConfig | None = None) -> KbHandles:
    """Open concrete knowledge-base services from configuration."""
    resolved_config = config or load_kb_config()
    mysql = SqlAlchemyResumeStore(resolved_config)
    milvus_result, embedder_result = await asyncio.gather(
        asyncio.to_thread(
            PyMilvusResumeStore,
            uri=resolved_config.milvus_uri,
            token=resolved_config.milvus_token,
        ),
        asyncio.to_thread(
            BgeM3Embedder,
            model_name=resolved_config.bge_m3_model,
        ),
        return_exceptions=True,
    )
    created_milvus = (
        milvus_result
        if isinstance(milvus_result, PyMilvusResumeStore)
        else None
    )
    init_errors = [
        result
        for result in (milvus_result, embedder_result)
        if isinstance(result, BaseException)
    ]
    if init_errors:
        if created_milvus is not None:
            await created_milvus.close()
        await mysql.close()
        raise init_errors[0]
    return KbHandles(
        mysql=mysql,
        milvus=milvus_result,
        embedder=embedder_result,
    )


async def ingest_resume(
    path: str,
    *,
    mysql: ResumeMysqlStore | None = None,
    milvus: ResumeMilvusStore | None = None,
    embedder: Embedder | None = None,
    parse: Callable[[str], Resume] = parse_resume,
) -> int:
    """Parse a resume file, persist chunks, and vectorize them."""
    stores = (mysql, milvus, embedder)
    if all(store is None for store in stores):
        async with await open_kb() as handles:
            return await _ingest_resume(
                path,
                mysql=handles.mysql,
                milvus=handles.milvus,
                embedder=handles.embedder,
                parse=parse,
            )
    if any(store is None for store in stores):
        raise KbStoreError(
            "mysql, milvus, and embedder must be provided together"
        )
    return await _ingest_resume(
        path,
        mysql=mysql,
        milvus=milvus,
        embedder=embedder,
        parse=parse,
    )


async def _ingest_resume(
    path: str,
    *,
    mysql: ResumeMysqlStore,
    milvus: ResumeMilvusStore,
    embedder: Embedder,
    parse: Callable[[str], Resume],
) -> int:
    """Run ingest with fully resolved knowledge-base services."""
    load_markdown(path)
    initialize = getattr(mysql, "initialize", None)
    if initialize is not None:
        await initialize()
    source_hash = _source_hash(path)
    resume = parse(path)
    profile = build_resume_profile(resume).model_copy(
        update={"source_path": path, "source_hash": source_hash}
    )

    resume_id = await mysql.upsert_resume(
        source_path=path,
        source_hash=source_hash,
        profile=profile,
    )
    await milvus.delete_by_resume_id(resume_id)

    chunks = chunk_resume(resume)
    stored_chunks = await mysql.replace_chunks(resume_id, chunks)

    try:
        for chunk in stored_chunks:
            if chunk.id is None:
                raise KbStoreError("chunk id missing after replace_chunks")
            vector = embedder.embed([chunk.embed_text])[0]
            await milvus.upsert_vectors(
                [
                    (
                        chunk.id,
                        resume_id,
                        chunk.chunk_type,
                        chunk.chunk_index,
                        vector,
                    )
                ]
            )
            await mysql.mark_chunk_vectorized(chunk.id, chunk.id)
    except Exception as exc:
        logger.exception("vectorization failed", extra={"resume_id": resume_id})
        await mysql.mark_resume_status(resume_id, "failed")
        if isinstance(exc, KbStoreError):
            raise
        raise KbStoreError("vectorization failed") from exc

    await mysql.mark_resume_status(resume_id, "vectorized")
    return resume_id


async def load_profile(
    resume_id: int,
    *,
    mysql: ResumeMysqlStore,
) -> ResumeProfile:
    """Load a vectorized resume profile by id."""
    profile = await mysql.get_profile(resume_id)
    if profile is None:
        raise KbNotFoundError(f"resume not found: {resume_id}")
    if profile.status != "vectorized":
        raise KbStoreError(
            f"resume {resume_id} is not vectorized (status={profile.status})"
        )
    return profile


def _source_hash(path: str) -> str:
    """Return the SHA-256 hex digest of the resume file bytes."""
    file_bytes = Path(path).read_bytes()
    return hashlib.sha256(file_bytes).hexdigest()
