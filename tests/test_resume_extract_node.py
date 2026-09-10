import asyncio

from job_agent.kb.models import ResumeProfile


class FakeResumeTool:
    name = "ingest_resume_file"

    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, args: dict[str, str]) -> str:
        self.calls += 1
        return ResumeProfile(
            id=12,
            source_path=args["path"],
            name="Tool Candidate",
            status="vectorized",
        ).model_dump_json()


async def _fake_run_expert(**kwargs: object) -> ResumeProfile:
    return ResumeProfile(
        id=12,
        source_path="/tmp/resume.md",
        name="LLM Candidate",
        status="vectorized",
    )


def test_resume_extract_node_does_not_preinvoke_resume_tool(monkeypatch) -> None:
    from job_agent.matching import nodes

    fake_tool = FakeResumeTool()
    expected_profile = asyncio.run(_fake_run_expert())
    captured_tools: list[object] = []
    parse_resume_calls: list[str] = []

    async def fake_run_expert(**kwargs: object) -> ResumeProfile:
        captured_tools.extend(kwargs["tools"])
        return await _fake_run_expert(**kwargs)

    monkeypatch.setattr(nodes, "ingest_resume_file", fake_tool)
    monkeypatch.setattr(
        nodes,
        "parse_resume",
        lambda path: parse_resume_calls.append(path),
        raising=False,
    )
    monkeypatch.setattr(nodes, "run_expert", fake_run_expert)
    monkeypatch.setattr(nodes, "load_llm_config", lambda: object())

    result = asyncio.run(
        nodes.resume_extract_node({"resume_ref": "/tmp/resume.md"})
    )

    assert fake_tool.calls == 0
    assert captured_tools == [fake_tool]
    assert parse_resume_calls == []
    assert result == {"resume_id": 12, "resume_profile": expected_profile}


def test_resume_extract_node_binds_only_id_tool(monkeypatch) -> None:
    from job_agent.matching import nodes

    expected_profile = ResumeProfile(
        id=12,
        source_path="/tmp/resume.md",
        name="Candidate",
        status="vectorized",
    )
    captured_tools: list[object] = []

    async def fake_run_expert(**kwargs: object) -> ResumeProfile:
        captured_tools.extend(kwargs["tools"])
        return expected_profile

    monkeypatch.setattr(nodes, "run_expert", fake_run_expert)
    monkeypatch.setattr(nodes, "load_llm_config", lambda: object())

    result = asyncio.run(nodes.resume_extract_node({"resume_ref": "12"}))

    assert captured_tools == [nodes.load_resume_by_id]
    assert result == {"resume_id": 12, "resume_profile": expected_profile}
