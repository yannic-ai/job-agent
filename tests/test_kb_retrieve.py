import asyncio

from job_agent.kb.fake_stores import FakeMilvus, FakeMysql
from job_agent.kb.models import BGE_M3_DIM, ResumeChunk, ResumeProfile
from job_agent.kb.retrieve import search_resume_chunks


def _unit_vector(dim: int, index: int) -> list[float]:
    vec = [0.0] * dim
    vec[index] = 1.0
    return vec


class MappingEmbedder:
    """Test embedder that maps exact text to fixed vectors."""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._mapping.get(t, [0.0] * BGE_M3_DIM) for t in texts]


def test_search_resume_chunks_ranking_and_payload() -> None:
    async def _run() -> None:
        mysql = FakeMysql()
        milvus = FakeMilvus()
        vec_work_a = _unit_vector(BGE_M3_DIM, 0)
        vec_work_b = _unit_vector(BGE_M3_DIM, 1)
        vec_skills = _unit_vector(BGE_M3_DIM, 2)
        embedder = MappingEmbedder({"match work a": vec_work_a})

        profile = ResumeProfile(source_path="/a.md", source_hash="abc")
        resume_id = await mysql.upsert_resume(
            source_path="/a.md",
            source_hash="abc",
            profile=profile,
        )

        chunks = await mysql.replace_chunks(
            resume_id,
            [
                ResumeChunk(
                    chunk_type="work_experience",
                    chunk_index=0,
                    title="公司A",
                    embed_text="work a",
                    payload={"company": "A"},
                ),
                ResumeChunk(
                    chunk_type="work_experience",
                    chunk_index=1,
                    title="公司B",
                    embed_text="work b",
                    payload={"company": "B"},
                ),
                ResumeChunk(
                    chunk_type="skills",
                    chunk_index=0,
                    title="技能",
                    embed_text="skills",
                    payload={"items": ["Python"]},
                ),
            ],
        )
        work_a, work_b, skills = chunks
        assert work_a.id is not None
        assert work_b.id is not None
        assert skills.id is not None

        await milvus.upsert_vectors(
            [
                (work_a.id, resume_id, "work_experience", 0, vec_work_a),
                (work_b.id, resume_id, "work_experience", 1, vec_work_b),
                (skills.id, resume_id, "skills", 0, vec_skills),
            ]
        )

        hits = await search_resume_chunks(
            resume_id,
            ["work_experience", "project"],
            "match work a",
            5,
            mysql=mysql,
            milvus=milvus,
            embedder=embedder,
        )

        assert len(hits) == 2
        assert [h.id for h in hits] == [work_a.id, work_b.id]
        assert hits[0].score > hits[1].score
        assert hits[0].payload == {"company": "A"}
        assert hits[0].embed_text == "work a"
        assert hits[0].chunk_type == "work_experience"

        skills_only = await search_resume_chunks(
            resume_id,
            ["skills"],
            "match work a",
            5,
            mysql=mysql,
            milvus=milvus,
            embedder=embedder,
        )
        assert len(skills_only) == 1
        assert skills_only[0].id == skills.id
        assert skills_only[0].chunk_type == "skills"
        assert all(h.chunk_type != "work_experience" for h in skills_only)

    asyncio.run(_run())


def test_search_resume_chunks_empty_store() -> None:
    async def _run() -> None:
        mysql = FakeMysql()
        milvus = FakeMilvus()
        embedder = MappingEmbedder({})

        hits = await search_resume_chunks(
            1,
            ["work_experience"],
            "query",
            5,
            mysql=mysql,
            milvus=milvus,
            embedder=embedder,
        )
        assert hits == []

    asyncio.run(_run())


def test_search_resume_chunks_empty_query_returns_empty() -> None:
    async def _run() -> None:
        mysql = FakeMysql()
        milvus = FakeMilvus()
        embedder = MappingEmbedder({})

        assert (
            await search_resume_chunks(
                1,
                ["work_experience"],
                "",
                5,
                mysql=mysql,
                milvus=milvus,
                embedder=embedder,
            )
            == []
        )
        assert (
            await search_resume_chunks(
                1,
                ["work_experience"],
                "   ",
                5,
                mysql=mysql,
                milvus=milvus,
                embedder=embedder,
            )
            == []
        )

    asyncio.run(_run())


def test_search_resume_chunks_non_positive_top_k_returns_empty() -> None:
    async def _run() -> None:
        mysql = FakeMysql()
        milvus = FakeMilvus()
        embedder = MappingEmbedder({})

        assert (
            await search_resume_chunks(
                1,
                ["work_experience"],
                "query",
                0,
                mysql=mysql,
                milvus=milvus,
                embedder=embedder,
            )
            == []
        )

    asyncio.run(_run())
