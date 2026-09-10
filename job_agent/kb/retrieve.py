from __future__ import annotations

from job_agent.kb.models import ResumeChunkHit
from job_agent.kb.stores import Embedder, ResumeMilvusStore, ResumeMysqlStore


async def search_resume_chunks(
    resume_id: int,
    chunk_types: list[str],
    query: str,
    top_k: int,
    *,
    mysql: ResumeMysqlStore,
    milvus: ResumeMilvusStore,
    embedder: Embedder,
) -> list[ResumeChunkHit]:
    """Search resume chunks by vector similarity with type and resume filters."""
    if not query.strip() or top_k <= 0:
        return []

    vector = embedder.embed([query])[0]
    pairs = await milvus.search(
        resume_id=resume_id,
        chunk_types=chunk_types,
        query_vector=vector,
        top_k=top_k,
    )
    if not pairs:
        return []

    chunks = await mysql.get_chunks_by_ids([chunk_id for chunk_id, _ in pairs])
    by_id = {chunk.id: chunk for chunk in chunks if chunk.id is not None}

    hits: list[ResumeChunkHit] = []
    for chunk_id, score in pairs:
        chunk = by_id.get(chunk_id)
        if chunk is None:
            continue
        hits.append(
            ResumeChunkHit(
                id=chunk_id,
                chunk_type=chunk.chunk_type,
                score=score,
                embed_text=chunk.embed_text,
                payload=chunk.payload,
            )
        )
    return hits
