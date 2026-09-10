import pytest

from job_agent.kb.embedder import BgeM3Embedder
from job_agent.kb.errors import KbStoreError


def test_bge_embedder_reads_dense_vecs(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def encode(self, texts: list[str], **kwargs: object) -> dict[str, object]:
            return {
                "dense_vecs": [[1.0] + [0.0] * 1023 for _ in texts],
            }

    monkeypatch.setattr(
        "job_agent.kb.embedder.BGEM3FlagModel",
        lambda *args, **kwargs: FakeModel(),
    )

    embedder = BgeM3Embedder(model_name="BAAI/bge-m3")
    vectors = embedder.embed(["hello"])

    assert len(vectors[0]) == 1024
    assert vectors[0][0] == 1.0


def test_bge_embedder_rejects_invalid_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel:
        def encode(self, texts: list[str], **kwargs: object) -> dict[str, object]:
            return {"dense_vecs": [[1.0, 0.0] for _ in texts]}

    monkeypatch.setattr(
        "job_agent.kb.embedder.BGEM3FlagModel",
        lambda *args, **kwargs: FakeModel(),
    )

    embedder = BgeM3Embedder(model_name="BAAI/bge-m3")

    with pytest.raises(KbStoreError, match="1024"):
        embedder.embed(["hello"])
