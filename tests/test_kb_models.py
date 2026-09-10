import pytest
from pydantic import ValidationError

from job_agent.kb.models import (
    BGE_M3_DIM,
    ResumeChunk,
    ResumeChunkHit,
    ResumeProfile,
)


def test_resume_profile_optional_scalars():
    profile = ResumeProfile(id=1, source_path="/a.md")

    assert profile.highest_degree is None
    assert profile.experience_months is None


def test_resume_profile_rejects_unknown_degree():
    with pytest.raises(ValidationError):
        ResumeProfile(source_path="/a.md", highest_degree="大专")


def test_bge_m3_dim_is_1024():
    assert BGE_M3_DIM == 1024


def test_resume_chunk_defaults_vector_fields():
    chunk = ResumeChunk(
        chunk_type="skills",
        chunk_index=0,
        title="技能",
        embed_text="Python",
        payload={"items": ["Python"]},
    )

    assert chunk.id is None
    assert chunk.resume_id is None
    assert chunk.vector_status == "pending"
    assert chunk.vector_id is None


def test_resume_chunk_hit_keeps_payload():
    hit = ResumeChunkHit(
        id=1,
        chunk_type="project",
        score=0.9,
        embed_text="Agent 项目",
        payload={"name": "Agent"},
    )

    assert hit.payload == {"name": "Agent"}
