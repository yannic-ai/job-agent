from __future__ import annotations

import math

from job_agent.kb.models import BGE_M3_DIM, ResumeChunk, ResumeProfile


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


class FakeEmbedder:
    """In-memory embedder for tests."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * BGE_M3_DIM for _ in texts]


class FakeMysql:
    """In-memory MySQL store for tests."""

    def __init__(self) -> None:
        self._next_resume_id = 1
        self._next_chunk_id = 1
        self._resumes: dict[int, ResumeProfile] = {}
        self._source_path_index: dict[str, int] = {}
        self._chunks: dict[int, ResumeChunk] = {}

    async def upsert_resume(
        self,
        *,
        source_path: str,
        source_hash: str,
        profile: ResumeProfile,
    ) -> int:
        existing_id = self._source_path_index.get(source_path)
        stored = profile.model_copy(
            update={
                "source_path": source_path,
                "source_hash": source_hash,
                "status": "pending",
            }
        )
        if existing_id is not None:
            stored.id = existing_id
            self._resumes[existing_id] = stored
            return existing_id

        resume_id = self._next_resume_id
        self._next_resume_id += 1
        stored.id = resume_id
        self._resumes[resume_id] = stored
        self._source_path_index[source_path] = resume_id
        return resume_id

    async def replace_chunks(
        self,
        resume_id: int,
        chunks: list[ResumeChunk],
    ) -> list[ResumeChunk]:
        self._chunks = {
            chunk_id: chunk
            for chunk_id, chunk in self._chunks.items()
            if chunk.resume_id != resume_id
        }
        stored_chunks: list[ResumeChunk] = []
        for chunk in chunks:
            chunk_id = self._next_chunk_id
            self._next_chunk_id += 1
            stored = chunk.model_copy(
                update={
                    "id": chunk_id,
                    "resume_id": resume_id,
                    "vector_status": "pending",
                    "vector_id": None,
                }
            )
            self._chunks[chunk_id] = stored
            stored_chunks.append(stored)
        return stored_chunks

    async def mark_chunk_vectorized(self, chunk_id: int, vector_id: int) -> None:
        chunk = self._chunks[chunk_id]
        self._chunks[chunk_id] = chunk.model_copy(
            update={"vector_status": "vectorized", "vector_id": vector_id}
        )

    async def mark_resume_status(self, resume_id: int, status: str) -> None:
        profile = self._resumes[resume_id]
        self._resumes[resume_id] = profile.model_copy(
            update={"status": status}  # type: ignore[arg-type]
        )

    async def get_profile(self, resume_id: int) -> ResumeProfile | None:
        return self._resumes.get(resume_id)

    async def get_chunks_by_ids(self, ids: list[int]) -> list[ResumeChunk]:
        return [self._chunks[chunk_id] for chunk_id in ids if chunk_id in self._chunks]

    def chunks_for(self, resume_id: int) -> list[ResumeChunk]:
        """Return all chunks for a resume (test helper)."""
        return [
            chunk
            for chunk in self._chunks.values()
            if chunk.resume_id == resume_id
        ]


class FakeMilvus:
    """In-memory Milvus store for tests."""

    def __init__(self) -> None:
        self._vectors: dict[int, tuple[int, str, int, list[float]]] = {}

    async def delete_by_resume_id(self, resume_id: int) -> None:
        self._vectors = {
            vector_id: row
            for vector_id, row in self._vectors.items()
            if row[0] != resume_id
        }

    async def upsert_vectors(
        self,
        rows: list[tuple[int, int, str, int, list[float]]],
    ) -> None:
        for vector_id, resume_id, chunk_type, chunk_index, embedding in rows:
            self._vectors[vector_id] = (
                resume_id,
                chunk_type,
                chunk_index,
                embedding,
            )

    async def search(
        self,
        *,
        resume_id: int,
        chunk_types: list[str],
        query_vector: list[float],
        top_k: int,
    ) -> list[tuple[int, float]]:
        hits: list[tuple[int, float]] = []
        for vector_id, row in self._vectors.items():
            row_resume_id, chunk_type, _chunk_index, embedding = row
            if row_resume_id != resume_id or chunk_type not in chunk_types:
                continue
            score = _cosine_similarity(query_vector, embedding)
            hits.append((vector_id, score))
        hits.sort(key=lambda item: item[1], reverse=True)
        return hits[:top_k]

    def ids_for(self, resume_id: int) -> set[int]:
        """Return vector ids for a resume (test helper)."""
        return {
            vector_id
            for vector_id, row in self._vectors.items()
            if row[0] == resume_id
        }
