from __future__ import annotations

from job_agent.kb.embed_text import render_embed_text
from job_agent.kb.models import ChunkType, ResumeChunk
from job_agent.resume.schema import Resume


def chunk_resume(resume: Resume) -> list[ResumeChunk]:
    """Split a parsed resume into independently searchable chunks."""
    chunks: list[ResumeChunk] = []
    personal_info = resume.personal_info
    if _has_text(personal_info.name) or _has_text(personal_info.location):
        _append_chunk(
            chunks,
            chunk_type="personal_info",
            chunk_index=0,
            title="location",
            payload={
                "name": personal_info.name,
                "location": personal_info.location,
            },
            resume=resume,
        )

    if resume.skills:
        _append_chunk(
            chunks,
            chunk_type="skills",
            chunk_index=0,
            title="skills",
            payload={"skills": resume.skills},
            resume=resume,
        )

    for index, education in enumerate(resume.education):
        payload = education.model_dump()
        _append_chunk(
            chunks,
            chunk_type="education",
            chunk_index=index,
            title=education.school or "education",
            payload=payload,
            resume=resume,
        )

    for index, experience in enumerate(resume.work_experience):
        payload = experience.model_dump()
        _append_chunk(
            chunks,
            chunk_type="work_experience",
            chunk_index=index,
            title=experience.company or "work",
            payload=payload,
            resume=resume,
        )

    for index, project in enumerate(resume.projects):
        payload = project.model_dump()
        _append_chunk(
            chunks,
            chunk_type="project",
            chunk_index=index,
            title=project.name or "project",
            payload=payload,
            resume=resume,
        )

    if _has_text(resume.summary):
        _append_chunk(
            chunks,
            chunk_type="summary",
            chunk_index=0,
            title="summary",
            payload={"summary": resume.summary},
            resume=resume,
        )
    return chunks


def _append_chunk(
    chunks: list[ResumeChunk],
    *,
    chunk_type: ChunkType,
    chunk_index: int,
    title: str,
    payload: dict[str, object],
    resume: Resume,
) -> None:
    """Render and append one chunk while excluding personal contacts."""
    embed_text = render_embed_text(chunk_type, payload)
    for contact in (resume.personal_info.phone, resume.personal_info.email):
        if contact:
            embed_text = embed_text.replace(contact, "")
    chunks.append(
        ResumeChunk(
            chunk_type=chunk_type,
            chunk_index=chunk_index,
            title=title,
            embed_text=embed_text,
            payload=payload,
        )
    )


def _has_text(value: str | None) -> bool:
    return bool(value and value.strip())
