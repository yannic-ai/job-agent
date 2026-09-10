from __future__ import annotations

import json
from typing import Any, TypeVar

from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from job_agent.config import LLMConfig
from job_agent.matching.errors import MatchingExtractError

SchemaT = TypeVar("SchemaT", bound=BaseModel)


def _build_llm(config: LLMConfig) -> ChatOpenAI:
    return ChatOpenAI(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=0,
        max_tokens=8192,
        timeout=180,
    )


def _stringify_tool_result(result: object) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, BaseModel):
        return result.model_dump_json()
    if isinstance(result, (dict, list)):
        return json.dumps(result, ensure_ascii=False)
    return str(result)


async def _run_tool(tool: object, args: dict[str, Any]) -> object:
    if hasattr(tool, "ainvoke"):
        return await tool.ainvoke(args)
    if hasattr(tool, "invoke"):
        return tool.invoke(args)
    raise MatchingExtractError("工具缺少 ainvoke 或 invoke")


async def run_expert(
    *,
    messages: list[object],
    tools: list[object],
    output_schema: type[SchemaT],
    config: LLMConfig,
    llm: Any | None = None,
    max_tool_calls: int = 2,
) -> SchemaT:
    """Run one expert node with a bounded tool loop then structured output."""

    runtime_llm = llm or _build_llm(config)
    history: list[object] = list(messages)
    tool_map = {tool.name: tool for tool in tools}
    calls = 0

    try:
        bound_llm = runtime_llm.bind_tools(tools)
        while calls < max_tool_calls:
            response = await bound_llm.ainvoke(history)
            calls += 1
            history.append(response)

            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                break

            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool = tool_map.get(tool_name)
                if tool is None:
                    raise MatchingExtractError(f"未知工具：{tool_name}")

                tool_result = await _run_tool(tool, tool_call.get("args", {}))
                history.append(
                    ToolMessage(
                        content=_stringify_tool_result(tool_result),
                        tool_call_id=tool_call["id"],
                    )
                )

            # 达到上限后直接离开 tools 循环，转入结构化输出阶段。
            if calls >= max_tool_calls:
                break

        structured_llm = runtime_llm.with_structured_output(
            output_schema,
            method="json_mode",
        )
        result = await structured_llm.ainvoke(history)
    except MatchingExtractError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise MatchingExtractError(f"专家节点运行失败：{exc}") from exc

    if not isinstance(result, output_schema):
        raise MatchingExtractError("模型未返回目标结构")
    return result
