from __future__ import annotations

import asyncio
from types import SimpleNamespace

from job_agent.kb.models import ResumeChunkHit, ResumeProfile
from job_agent.matching.schemas import JobRequirement


def _hit(chunk_type: str, payload: dict[str, object]) -> ResumeChunkHit:
    return ResumeChunkHit(
        id=1,
        chunk_type=chunk_type,
        score=0.9,
        embed_text="matched",
        payload=payload,
    )


def test_build_skills_resume_slice_returns_none_for_empty_hits() -> None:
    from job_agent.matching.nodes import build_skills_resume_slice

    assert build_skills_resume_slice([]) is None


def test_build_skills_resume_slice_uses_hit_payload() -> None:
    from job_agent.matching.nodes import build_skills_resume_slice

    resume = build_skills_resume_slice(
        [_hit("skills", {"skills": ["Python", "LangGraph"]})]
    )

    assert resume is not None
    assert resume.skills == ["Python", "LangGraph"]


def test_build_responsibilities_resume_slice_routes_chunk_types() -> None:
    from job_agent.matching.nodes import build_responsibilities_resume_slice

    resume = build_responsibilities_resume_slice(
        [
            _hit(
                "work_experience",
                {"company": "A", "responsibilities": ["建设检索服务"]},
            ),
            _hit(
                "project",
                {"name": "B", "responsibilities": ["设计智能体"]},
            ),
        ]
    )

    assert resume is not None
    assert resume.work_experience[0].company == "A"
    assert resume.projects[0].name == "B"


class _FakeHandles:
    mysql = object()
    milvus = object()
    embedder = object()

    async def __aenter__(self) -> _FakeHandles:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def close(self) -> None:
        return None


def test_eval_skills_empty_retrieval_uses_missing_score_tool(monkeypatch) -> None:
    from job_agent.matching import nodes

    searches: list[dict[str, object]] = []
    expert_tools: list[object] = []
    expert_messages: list[object] = []

    async def fake_open_kb() -> _FakeHandles:
        return _FakeHandles()

    async def fake_search_resume_chunks(
        resume_id: int,
        chunk_types: list[str],
        query: str,
        top_k: int,
        **stores: object,
    ) -> list[ResumeChunkHit]:
        searches.append(
            {
                "resume_id": resume_id,
                "chunk_types": chunk_types,
                "query": query,
                "top_k": top_k,
                **stores,
            }
        )
        return []

    async def fake_run_expert(**kwargs: object) -> object:
        expert_tools.extend(kwargs["tools"])
        expert_messages.extend(kwargs["messages"])
        return SimpleNamespace()

    monkeypatch.setattr(nodes, "open_kb", fake_open_kb)
    monkeypatch.setattr(nodes, "search_resume_chunks", fake_search_resume_chunks)
    monkeypatch.setattr(nodes, "run_expert", fake_run_expert)
    monkeypatch.setattr(nodes, "load_llm_config", lambda: object())

    result = asyncio.run(
        nodes.eval_skills_node(
            {
                "job_requirement": JobRequirement(
                    must_have_skills=["Python"],
                    nice_to_have_skills=["LangGraph"],
                ),
                "resume_profile": ResumeProfile(
                    id=12,
                    source_path="resume.md",
                    status="vectorized",
                ),
                "resume_id": 12,
            }
        )
    )

    assert searches[0]["query"] == "Python, LangGraph"
    assert searches[0]["chunk_types"] == ["skills"]
    assert searches[0]["top_k"] == 1
    assert expert_tools == [nodes.missing_retrieval_score_tool]
    prompt_text = "\n".join(str(message.content) for message in expert_messages)
    assert "dimension" in prompt_text
    assert "resume_json" not in prompt_text
    assert "profile_json" not in prompt_text
    assert result["dimension_scores"]["skills"].score == 3
    assert result["dimension_scores"]["skills"].evidence == "未召回到相关条目"


def test_eval_responsibilities_searches_expected_chunks(monkeypatch) -> None:
    from job_agent.matching import nodes

    searches: list[dict[str, object]] = []
    expert_messages: list[object] = []

    async def fake_open_kb() -> _FakeHandles:
        return _FakeHandles()

    async def fake_search_resume_chunks(
        resume_id: int,
        chunk_types: list[str],
        query: str,
        top_k: int,
        **stores: object,
    ) -> list[ResumeChunkHit]:
        searches.append(
            {
                "resume_id": resume_id,
                "chunk_types": chunk_types,
                "query": query,
                "top_k": top_k,
                **stores,
            }
        )
        return [
            _hit(
                "work_experience",
                {"company": "A", "responsibilities": ["建设检索平台"]},
            )
        ]

    async def fake_run_expert(**kwargs: object) -> object:
        expert_messages.extend(kwargs["messages"])
        return SimpleNamespace()

    monkeypatch.setattr(nodes, "open_kb", fake_open_kb)
    monkeypatch.setattr(nodes, "search_resume_chunks", fake_search_resume_chunks)
    monkeypatch.setattr(nodes, "run_expert", fake_run_expert)
    monkeypatch.setattr(nodes, "load_llm_config", lambda: object())

    result = asyncio.run(
        nodes.eval_responsibilities_node(
            {
                "job_requirement": JobRequirement(
                    responsibilities=["建设检索平台", "维护智能体"],
                ),
                "resume_profile": ResumeProfile(
                    id=12,
                    source_path="resume.md",
                    status="vectorized",
                ),
                "resume_id": 12,
            }
        )
    )

    assert searches[0]["query"] == "建设检索平台\n维护智能体"
    assert searches[0]["chunk_types"] == ["work_experience", "project"]
    assert searches[0]["top_k"] == 5
    assert result["dimension_scores"]["responsibilities"].dimension == (
        "responsibilities"
    )
    prompt_text = "\n".join(str(message.content) for message in expert_messages)
    assert "resume_json" in prompt_text
    assert "profile_json" not in prompt_text


def test_eval_profile_dimension_prompt_uses_profile_json(monkeypatch) -> None:
    from job_agent.matching import nodes

    expert_messages: list[object] = []

    async def fake_run_expert(**kwargs: object) -> object:
        expert_messages.extend(kwargs["messages"])
        return SimpleNamespace()

    monkeypatch.setattr(nodes, "run_expert", fake_run_expert)
    monkeypatch.setattr(nodes, "load_llm_config", lambda: object())

    asyncio.run(
        nodes.eval_years_node(
            {
                "job_requirement": JobRequirement(years_required="3年以上"),
                "resume_profile": ResumeProfile(
                    id=12,
                    source_path="resume.md",
                    experience_months=48,
                    status="vectorized",
                ),
                "resume_id": 12,
            }
        )
    )

    prompt_text = "\n".join(str(message.content) for message in expert_messages)
    assert "profile_json" in prompt_text
    assert "resume_json" not in prompt_text


def test_parallel_retrieval_nodes_share_one_kb_open(monkeypatch) -> None:
    from job_agent.matching import nodes

    open_count = 0

    async def fake_open_kb() -> _FakeHandles:
        nonlocal open_count
        open_count += 1
        return _FakeHandles()

    async def fake_search_resume_chunks(
        resume_id: int,
        chunk_types: list[str],
        query: str,
        top_k: int,
        **stores: object,
    ) -> list[ResumeChunkHit]:
        return []

    async def fake_run_expert(**_: object) -> object:
        return SimpleNamespace()

    monkeypatch.setattr(nodes, "open_kb", fake_open_kb)
    monkeypatch.setattr(nodes, "search_resume_chunks", fake_search_resume_chunks)
    monkeypatch.setattr(nodes, "run_expert", fake_run_expert)
    monkeypatch.setattr(nodes, "load_llm_config", lambda: object())
    state = {
        "job_requirement": JobRequirement(
            must_have_skills=["Python"],
            responsibilities=["建设检索平台"],
        ),
        "resume_profile": ResumeProfile(
            id=12,
            source_path="resume.md",
            status="vectorized",
        ),
        "resume_id": 12,
    }

    async def _run() -> None:
        async with nodes.matching_kb_context():
            await asyncio.gather(
                nodes.eval_skills_node(state),
                nodes.eval_responsibilities_node(state),
            )

    asyncio.run(_run())

    assert open_count == 1
