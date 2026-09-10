from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from pathlib import Path

from job_agent.kb.chunker import chunk_resume
from job_agent.kb.errors import KbNotFoundError, KbStoreError
from job_agent.kb.models import ResumeProfile
from job_agent.kb.profile import build_resume_profile
from job_agent.kb.stores import Embedder, ResumeMilvusStore, ResumeMysqlStore
from job_agent.resume.loader import load_markdown
from job_agent.resume.pipeline import parse_resume
from job_agent.resume.schema import Resume

logger = logging.getLogger(__name__)


async def ingest_resume(
    path: str,
    *,
    mysql: ResumeMysqlStore,
    milvus: ResumeMilvusStore,
    embedder: Embedder,
    parse: Callable[[str], Resume] = parse_resume,
) -> int:
    """Parse a resume file, persist chunks, and vectorize them."""
    load_markdown(path)
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
