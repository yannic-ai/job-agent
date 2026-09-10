from __future__ import annotations

import asyncio
import json
from typing import Any

from job_agent.kb.models import BGE_M3_DIM

try:
    from pymilvus import DataType, MilvusClient
except ModuleNotFoundError:  # pragma: no cover - depends on optional runtime package
    DataType = None  # type: ignore[assignment,misc]
    MilvusClient = None  # type: ignore[assignment,misc]

COLLECTION_NAME = "resume_chunks"


class PyMilvusResumeStore:
    """Persist and search resume chunk vectors through PyMilvus."""

    def __init__(self, uri: str, token: str = "") -> None:
        if MilvusClient is None or DataType is None:
            raise RuntimeError("pymilvus is required to use Milvus storage")
        self._client: Any = MilvusClient(uri=uri, token=token)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the locked resume chunk collection and index when absent."""
        if self._client.has_collection(collection_name=COLLECTION_NAME):
            return

        schema = self._client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )
        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=False,
        )
        schema.add_field(
            field_name="embedding",
            datatype=DataType.FLOAT_VECTOR,
            dim=BGE_M3_DIM,
        )
        schema.add_field(field_name="resume_id", datatype=DataType.INT64)
        schema.add_field(
            field_name="chunk_type",
            datatype=DataType.VARCHAR,
            max_length=32,
        )
        schema.add_field(field_name="chunk_index", datatype=DataType.INT64)

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="FLAT",
            metric_type="COSINE",
        )
        self._client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params,
        )

    async def close(self) -> None:
        """Close the PyMilvus client when supported by its version."""
        close = getattr(self._client, "close", None)
        if close is not None:
            await asyncio.to_thread(close)

    async def delete_by_resume_id(self, resume_id: int) -> None:
        """Delete all vectors belonging to one resume."""
        await asyncio.to_thread(
            self._client.delete,
            collection_name=COLLECTION_NAME,
            filter=f"resume_id == {resume_id}",
        )

    async def upsert_vectors(
        self,
        rows: list[tuple[int, int, str, int, list[float]]],
    ) -> None:
        """Upsert rows as (id, resume_id, chunk_type, index, embedding)."""
        if not rows:
            return
        data = [
            {
                "id": vector_id,
                "embedding": embedding,
                "resume_id": resume_id,
                "chunk_type": chunk_type,
                "chunk_index": chunk_index,
            }
            for vector_id, resume_id, chunk_type, chunk_index, embedding in rows
        ]
        await asyncio.to_thread(
            self._client.upsert,
            collection_name=COLLECTION_NAME,
            data=data,
        )

    async def search(
        self,
        *,
        resume_id: int,
        chunk_types: list[str],
        query_vector: list[float],
        top_k: int,
    ) -> list[tuple[int, float]]:
        """Search one resume using COSINE similarity and chunk-type filters."""
        if not chunk_types or top_k <= 0:
            return []
        chunk_type_values = json.dumps(chunk_types, ensure_ascii=False)
        result = await asyncio.to_thread(
            self._client.search,
            collection_name=COLLECTION_NAME,
            data=[query_vector],
            filter=(
                f"resume_id == {resume_id} "
                f"and chunk_type in {chunk_type_values}"
            ),
            limit=top_k,
        )
        if not result:
            return []
        return [
            (int(hit["id"]), float(hit.get("distance", hit.get("score"))))
            for hit in result[0]
        ]
