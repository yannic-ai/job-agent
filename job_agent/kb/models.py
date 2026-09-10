from typing import Literal

from pydantic import BaseModel, Field

BGE_M3_DIM = 1024

ChunkType = Literal[
    "personal_info",
    "skills",
    "education",
    "work_experience",
    "project",
    "summary",
]
VectorStatus = Literal["pending", "vectorized", "failed"]


class ResumeProfile(BaseModel):
    """简历的可筛选标量信息。"""

    id: int | None = None
    source_path: str
    source_hash: str = ""
    name: str | None = None
    location: str | None = None
    phone: str | None = None
    email: str | None = None
    highest_degree: Literal["博士", "硕士", "本科"] | None = None
    experience_months: int | None = None
    status: VectorStatus = "pending"


class ResumeChunk(BaseModel):
    """待向量化或已向量化的简历切片。"""

    id: int | None = None
    resume_id: int | None = None
    chunk_type: ChunkType
    chunk_index: int
    title: str
    embed_text: str
    payload: dict[str, object] = Field(default_factory=dict)
    vector_status: VectorStatus = "pending"
    vector_id: int | None = None


class ResumeChunkHit(BaseModel):
    """简历切片向量检索命中。"""

    id: int
    chunk_type: ChunkType
    score: float
    embed_text: str
    payload: dict[str, object] = Field(default_factory=dict)
