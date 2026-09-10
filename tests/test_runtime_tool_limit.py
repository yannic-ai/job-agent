import asyncio
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel

from job_agent.config import LLMConfig
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

    async def ainvoke(self, args: dict[str, Any]) -> str:
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
