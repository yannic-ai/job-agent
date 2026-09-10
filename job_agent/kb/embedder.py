from __future__ import annotations

from typing import Any

from job_agent.kb.errors import KbStoreError
from job_agent.kb.models import BGE_M3_DIM

try:
    from FlagEmbedding import BGEM3FlagModel
except ModuleNotFoundError:  # pragma: no cover - depends on optional runtime package
    BGEM3FlagModel = None  # type: ignore[assignment,misc]


class BgeM3Embedder:
    """Generate 1024-dimensional dense vectors with BGE-M3."""

    def __init__(self, model_name: str) -> None:
        if BGEM3FlagModel is None:
            raise KbStoreError("FlagEmbedding is required to use BGE-M3")
        self._model: Any = BGEM3FlagModel(model_name, use_fp16=False)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts and validate the locked BGE-M3 vector dimension."""
        encoded = self._model.encode(texts)
        dense_vectors = encoded["dense_vecs"]
        if hasattr(dense_vectors, "tolist"):
            dense_vectors = dense_vectors.tolist()
        vectors = [
            [float(value) for value in vector]
            for vector in dense_vectors
        ]
        if any(len(vector) != BGE_M3_DIM for vector in vectors):
            raise KbStoreError(
                f"BGE-M3 embedding dimension must be {BGE_M3_DIM}"
            )
        return vectors
