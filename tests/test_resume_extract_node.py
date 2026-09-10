import asyncio

from job_agent.resume.schema import Resume


class FakeResumeTool:
    name = "parse_resume_file"

    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, args: dict[str, str]) -> str:
        self.calls += 1
        return Resume(
            personal_info={"name": "Tool Candidate"},
            skills=["python"],
        ).model_dump_json()


async def _fake_run_expert(**kwargs: object) -> Resume:
    return Resume(personal_info={"name": "LLM Candidate"})


def test_resume_extract_node_does_not_preinvoke_resume_tool(monkeypatch) -> None:
    from job_agent.matching import nodes

    fake_tool = FakeResumeTool()
    expected_resume = Resume(
        personal_info={"name": "Python Candidate"},
        skills=["langgraph"],
    )
    captured_tools: list[object] = []

    async def fake_run_expert(**kwargs: object) -> Resume:
        captured_tools.extend(kwargs["tools"])
        return await _fake_run_expert(**kwargs)

    monkeypatch.setattr(nodes, "parse_resume_file", fake_tool)
    monkeypatch.setattr(nodes, "parse_resume", lambda path: expected_resume, raising=False)
    monkeypatch.setattr(nodes, "run_expert", fake_run_expert)
    monkeypatch.setattr(nodes, "load_llm_config", lambda: object())

    result = asyncio.run(
        nodes.resume_extract_node({"resume_path": "/tmp/resume.md"})
    )

    assert fake_tool.calls == 0
    assert captured_tools == [fake_tool]
    assert result == {"resume": expected_resume}
