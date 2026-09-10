import asyncio
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel

from job_agent.config import LLMConfig
from job_agent.kb.errors import KbNotFoundError, KbStoreError
from job_agent.matching.runtime import run_expert


class Out(BaseModel):
    ok: bool = True


class FakeLLM:
    def __init__(self) -> None:
        self.tool_round_calls = 0

    def bind_tools(self, tools: list[object]) -> "FakeLLM":
        self._tools = tools
        return self

    async def ainvoke(self, messages: list[object]) -> Any:
        self.tool_round_calls += 1
        if self.tool_round_calls > 2:
            raise AssertionError("third tool-bound call is not allowed")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "noop",
                    "args": {},
                    "id": f"call-{self.tool_round_calls}",
                    "type": "tool_call",
                }
            ],
        )

    def with_structured_output(
        self,
        schema: type[BaseModel],
        method: str = "json_mode",
    ) -> object:
        assert schema is Out
        assert method == "json_mode"

        class Structured:
            async def ainvoke(inner_self, messages: list[object]) -> Out:
                assert any(isinstance(message, ToolMessage) for message in messages)
                return Out(ok=True)

        return Structured()


class FakeTool:
    name = "noop"

    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, args: dict[str, Any]) -> str:
        self.calls += 1
        if self.calls > 2:
            raise AssertionError("third tool execution is not allowed")
        return "ok"


def test_runtime_stops_after_two_tool_calls() -> None:
    llm = FakeLLM()

    result = asyncio.run(
        run_expert(
            messages=[{"role": "user", "content": "go"}],
            tools=[FakeTool()],
            output_schema=Out,
            config=LLMConfig(api_key="x", base_url="http://x", model="x"),
            llm=llm,
        )
    )

    assert result.ok is True
    assert llm.tool_round_calls == 2


class FakeMultiToolCallLLM:
    def __init__(self) -> None:
        self.tool_round_calls = 0

    def bind_tools(self, tools: list[object]) -> "FakeMultiToolCallLLM":
        self._tools = tools
        return self

    async def ainvoke(self, messages: list[object]) -> Any:
        self.tool_round_calls += 1
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "noop",
                    "args": {"index": 1},
                    "id": "call-1",
                    "type": "tool_call",
                },
                {
                    "name": "noop",
                    "args": {"index": 2},
                    "id": "call-2",
                    "type": "tool_call",
                },
                {
                    "name": "noop",
                    "args": {"index": 3},
                    "id": "call-3",
                    "type": "tool_call",
                },
            ],
        )

    def with_structured_output(
        self,
        schema: type[BaseModel],
        method: str = "json_mode",
    ) -> object:
        assert schema is Out
        assert method == "json_mode"

        class Structured:
            async def ainvoke(inner_self, messages: list[object]) -> Out:
                tool_messages = [
                    message
                    for message in messages
                    if isinstance(message, ToolMessage)
                ]
                assert len(tool_messages) == 2
                return Out(ok=True)

        return Structured()


def test_runtime_limits_executed_tool_calls_per_response() -> None:
    llm = FakeMultiToolCallLLM()
    tool = FakeTool()

    result = asyncio.run(
        run_expert(
            messages=[{"role": "user", "content": "go"}],
            tools=[tool],
            output_schema=Out,
            config=LLMConfig(api_key="x", base_url="http://x", model="x"),
            llm=llm,
        )
    )

    assert result.ok is True
    assert tool.calls == 2
    assert llm.tool_round_calls == 1


@pytest.mark.parametrize(
    "error",
    [KbNotFoundError("missing"), KbStoreError("not vectorized")],
)
def test_runtime_preserves_kb_tool_errors(error: Exception) -> None:
    class FailingTool:
        name = "noop"

        async def ainvoke(self, args: dict[str, Any]) -> str:
            raise error

    with pytest.raises(type(error), match=str(error)):
        asyncio.run(
            run_expert(
                messages=[{"role": "user", "content": "go"}],
                tools=[FailingTool()],
                output_schema=Out,
                config=LLMConfig(
                    api_key="x",
                    base_url="http://x",
                    model="x",
                ),
                llm=FakeLLM(),
            )
        )
