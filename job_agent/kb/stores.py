from __future__ import annotations

from typing import Protocol

from job_agent.kb.models import ResumeChunk, ResumeProfile


class Embedder(Protocol):
    """Text embedder used during resume ingest and retrieval."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class ResumeMysqlStore(Protocol):
    """Persistent resume profile and chunk storage."""

    async def upsert_resume(
        self,
        *,
        source_path: str,
        source_hash: str,
        profile: ResumeProfile,
    ) -> int: ...

    async def replace_chunks(
        self,
        resume_id: int,
        chunks: list[ResumeChunk],
    ) -> list[ResumeChunk]: ...

    async def mark_chunk_vectorized(self, chunk_id: int, vector_id: int) -> None: ...

    async def mark_resume_status(self, resume_id: int, status: str) -> None: ...

    async def get_profile(self, resume_id: int) -> ResumeProfile | None: ...

    async def get_chunks_by_ids(self, ids: list[int]) -> list[ResumeChunk]: ...


class ResumeMilvusStore(Protocol):
    """Vector storage for resume chunks."""

    async def delete_by_resume_id(self, resume_id: int) -> None: ...

    async def upsert_vectors(
        self,
        rows: list[tuple[int, int, str, int, list[float]]],
    ) -> None:
        """Upsert rows as (id, resume_id, chunk_type, chunk_index, embedding)."""

    async def search(
        self,
        *,
        resume_id: int,
        chunk_types: list[str],
        query_vector: list[float],
        top_k: int,
    ) -> list[tuple[int, float]]: ...
