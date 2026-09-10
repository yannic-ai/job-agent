import asyncio
from pathlib import Path

import pytest

from job_agent.kb.errors import KbNotFoundError, KbStoreError
from job_agent.kb.fake_stores import FakeEmbedder, FakeMilvus, FakeMysql
from job_agent.kb.models import ResumeChunk, ResumeProfile
from job_agent.kb.pipeline import ingest_resume, load_profile
from job_agent.resume.schema import Resume, WorkExperience


def test_reingest_same_path_keeps_id_and_replaces_chunks(tmp_path) -> None:
    async def _run() -> None:
        path = tmp_path / "r.md"
        path.write_text("# x\n", encoding="utf-8")
        mysql, milvus, embedder = FakeMysql(), FakeMilvus(), FakeEmbedder()
        resume = Resume(skills=["Python"])
        rid1 = await ingest_resume(
            str(path),
            mysql=mysql,
            milvus=milvus,
            embedder=embedder,
            parse=lambda _: resume,
        )
        resume2 = Resume(
            skills=["Python", "Go"],
            work_experience=[
                WorkExperience(company="A", start_date="2020-01", end_date="2021-01")
            ],
        )
        rid2 = await ingest_resume(
            str(path),
            mysql=mysql,
            milvus=milvus,
            embedder=embedder,
            parse=lambda _: resume2,
        )
        assert rid1 == rid2
        chunks = mysql.chunks_for(rid1)
        assert len([c for c in chunks if c.chunk_type == "skills"]) == 1
        assert len([c for c in chunks if c.chunk_type == "work_experience"]) == 1
        assert milvus.ids_for(rid1) == {c.id for c in chunks}

    asyncio.run(_run())


@pytest.mark.parametrize("failure_point", ["delete", "replace"])
def test_ingest_marks_resume_failed_when_store_preparation_fails(
    tmp_path: Path,
    failure_point: str,
) -> None:
    class FailingMysql(FakeMysql):
        async def replace_chunks(
            self,
            resume_id: int,
            chunks: list[ResumeChunk],
        ) -> list[ResumeChunk]:
            if failure_point == "replace":
                raise RuntimeError("replace failed")
            return await super().replace_chunks(resume_id, chunks)

    class FailingMilvus(FakeMilvus):
        async def delete_by_resume_id(self, resume_id: int) -> None:
            if failure_point == "delete":
                raise RuntimeError("delete failed")
            await super().delete_by_resume_id(resume_id)

    async def _run() -> None:
        path = tmp_path / "r.md"
        path.write_text("# x\n", encoding="utf-8")
        mysql = FailingMysql()

        with pytest.raises(KbStoreError):
            await ingest_resume(
                str(path),
                mysql=mysql,
                milvus=FailingMilvus(),
                embedder=FakeEmbedder(),
                parse=lambda _: Resume(skills=["Python"]),
            )

        profile = await mysql.get_profile(1)
        assert profile is not None
        assert profile.status == "failed"

    asyncio.run(_run())


def test_load_profile_missing_id_raises_not_found() -> None:
    async def _run() -> None:
        mysql = FakeMysql()
        with pytest.raises(KbNotFoundError):
            await load_profile(999, mysql=mysql)

    asyncio.run(_run())


def test_load_profile_non_vectorized_raises_store_error() -> None:
    async def _run() -> None:
        mysql = FakeMysql()
        profile = ResumeProfile(source_path="/a.md", source_hash="abc")
        resume_id = await mysql.upsert_resume(
            source_path="/a.md",
            source_hash="abc",
            profile=profile,
        )
        await mysql.mark_resume_status(resume_id, "pending")
        with pytest.raises(KbStoreError):
            await load_profile(resume_id, mysql=mysql)

    asyncio.run(_run())


def test_load_profile_vectorized_returns_profile() -> None:
    async def _run() -> None:
        mysql = FakeMysql()
        profile = ResumeProfile(
            source_path="/a.md",
            source_hash="abc",
            name="张三",
        )
        resume_id = await mysql.upsert_resume(
            source_path="/a.md",
            source_hash="abc",
            profile=profile,
        )
        await mysql.mark_resume_status(resume_id, "vectorized")
        loaded = await load_profile(resume_id, mysql=mysql)
        assert loaded.id == resume_id
        assert loaded.name == "张三"
        assert loaded.status == "vectorized"

    asyncio.run(_run())
