from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
    delete,
    func,
    select,
)
from sqlalchemy.ext.asyncio import (
    AsyncAttrs,
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from job_agent.config import KbConfig
from job_agent.kb.models import ResumeChunk, ResumeProfile


class Base(AsyncAttrs, DeclarativeBase):
    """Declarative base for resume knowledge-base tables."""


class ResumeRow(Base):
    """SQLAlchemy row for one source resume."""

    __tablename__ = "resumes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    source_path: Mapped[str] = mapped_column(String(1024), unique=True)
    source_hash: Mapped[str] = mapped_column(String(64))
    name: Mapped[str | None] = mapped_column(String(255))
    location: Mapped[str | None] = mapped_column(String(255))
    phone: Mapped[str | None] = mapped_column(String(255))
    email: Mapped[str | None] = mapped_column(String(255))
    highest_degree: Mapped[str | None] = mapped_column(String(16))
    experience_months: Mapped[int | None]
    status: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
    )


class ResumeChunkRow(Base):
    """SQLAlchemy row for one retrievable resume chunk."""

    __tablename__ = "resume_chunks"
    __table_args__ = (
        UniqueConstraint(
            "resume_id",
            "chunk_type",
            "chunk_index",
            name="uq_resume_chunk_position",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    resume_id: Mapped[int] = mapped_column(ForeignKey("resumes.id"))
    chunk_type: Mapped[str] = mapped_column(String(32))
    chunk_index: Mapped[int]
    title: Mapped[str] = mapped_column(String(255))
    embed_text: Mapped[str] = mapped_column(Text)
    payload_json: Mapped[dict[str, object]] = mapped_column(JSON)
    vector_status: Mapped[str] = mapped_column(String(32))
    vector_id: Mapped[int | None]


class SqlAlchemyResumeStore:
    """Persist resume profiles and chunks through SQLAlchemy async sessions."""

    def __init__(self, config: KbConfig) -> None:
        connection_url = (
            "mysql+aiomysql://"
            f"{config.mysql_user}:{config.mysql_password}"
            f"@{config.mysql_host}:{config.mysql_port}/{config.mysql_db}"
            "?charset=utf8mb4"
        )
        self.engine = create_async_engine(connection_url)
        self._sessions = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
        )

    async def initialize(self) -> None:
        """Create development tables when they do not already exist."""
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        """Dispose of the underlying SQLAlchemy engine."""
        await self.engine.dispose()

    async def upsert_resume(
        self,
        *,
        source_path: str,
        source_hash: str,
        profile: ResumeProfile,
    ) -> int:
        """Insert a source resume or refresh its scalar profile fields."""
        values = {
            "source_hash": source_hash,
            "name": profile.name,
            "location": profile.location,
            "phone": profile.phone,
            "email": profile.email,
            "highest_degree": profile.highest_degree,
            "experience_months": profile.experience_months,
            "status": "pending",
        }
        async with self._sessions.begin() as session:
            row = await session.scalar(
                select(ResumeRow).where(ResumeRow.source_path == source_path)
            )
            if row is None:
                row = ResumeRow(source_path=source_path, **values)
                session.add(row)
            else:
                for field_name, value in values.items():
                    setattr(row, field_name, value)
            await session.flush()
            return row.id

    async def replace_chunks(
        self,
        resume_id: int,
        chunks: list[ResumeChunk],
    ) -> list[ResumeChunk]:
        """Replace all chunks for a resume and return rows with generated ids."""
        async with self._sessions.begin() as session:
            await session.execute(
                delete(ResumeChunkRow).where(
                    ResumeChunkRow.resume_id == resume_id
                )
            )
            rows = [
                ResumeChunkRow(
                    resume_id=resume_id,
                    chunk_type=chunk.chunk_type,
                    chunk_index=chunk.chunk_index,
                    title=chunk.title,
                    embed_text=chunk.embed_text,
                    payload_json=chunk.payload,
                    vector_status="pending",
                    vector_id=None,
                )
                for chunk in chunks
            ]
            session.add_all(rows)
            await session.flush()
            return [self._chunk_from_row(row) for row in rows]

    async def mark_chunk_vectorized(
        self,
        chunk_id: int,
        vector_id: int,
    ) -> None:
        """Mark a chunk as successfully written to vector storage."""
        async with self._sessions.begin() as session:
            row = await session.get(ResumeChunkRow, chunk_id)
            if row is not None:
                row.vector_status = "vectorized"
                row.vector_id = vector_id

    async def mark_resume_status(self, resume_id: int, status: str) -> None:
        """Update a resume's vectorization status."""
        async with self._sessions.begin() as session:
            row = await session.get(ResumeRow, resume_id)
            if row is not None:
                row.status = status

    async def get_profile(self, resume_id: int) -> ResumeProfile | None:
        """Return the scalar profile for a resume id."""
        async with self._sessions() as session:
            row = await session.get(ResumeRow, resume_id)
            if row is None:
                return None
            return ResumeProfile(
                id=row.id,
                source_path=row.source_path,
                source_hash=row.source_hash,
                name=row.name,
                location=row.location,
                phone=row.phone,
                email=row.email,
                highest_degree=row.highest_degree,
                experience_months=row.experience_months,
                status=row.status,
            )

    async def get_chunks_by_ids(self, ids: list[int]) -> list[ResumeChunk]:
        """Return all chunks matching the supplied primary keys."""
        if not ids:
            return []
        async with self._sessions() as session:
            rows = (
                await session.scalars(
                    select(ResumeChunkRow).where(ResumeChunkRow.id.in_(ids))
                )
            ).all()
            return [self._chunk_from_row(row) for row in rows]

    @staticmethod
    def _chunk_from_row(row: ResumeChunkRow) -> ResumeChunk:
        """Convert an ORM row to the public chunk model."""
        return ResumeChunk(
            id=row.id,
            resume_id=row.resume_id,
            chunk_type=row.chunk_type,
            chunk_index=row.chunk_index,
            title=row.title,
            embed_text=row.embed_text,
            payload=row.payload_json,
            vector_status=row.vector_status,
            vector_id=row.vector_id,
        )
