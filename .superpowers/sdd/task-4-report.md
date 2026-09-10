# Task 4 Report

## RED

- 新增 `tests/test_runtime_tool_limit.py`
- 运行 `python -m pytest tests/test_runtime_tool_limit.py -v`
- 结果：`ModuleNotFoundError: No module named 'job_agent.matching.runtime'`
- 结论：失败原因符合预期，证明测试先于实现存在且有效

## GREEN

- 新增 `job_agent/matching/runtime.py`
- 实现 `run_expert(...)`：
  - 绑定 tools 后最多进行 `max_tool_calls=2` 轮 `ainvoke`
  - 每轮执行返回的 tool calls，并将 `ToolMessage` 追加到 history
  - 达到上限后不再进行第三次带 tools 的模型调用
  - 转入 `with_structured_output(..., method="json_mode")`
  - 异常或非目标 schema 返回时抛出 `MatchingExtractError`
- 运行 `python -m pytest tests/test_runtime_tool_limit.py -v`
- 结果：PASS

## Notes

- 生产路径使用 `ChatOpenAI(model, api_key, base_url, temperature=0, max_tokens=8192, timeout=180)`
- 工具优先走 `ainvoke(args)`，否则退回 `invoke(args)`

## After Review Fix

### RED

- 保留原有“两轮各 1 次 tool call”测试，并新增“首轮返回 3 个 `tool_calls`”回归测试。
- 运行 `python -m pytest tests/test_runtime_tool_limit.py -v`
- 结果：`test_runtime_limits_executed_tool_calls_per_response` 失败，`FakeTool.ainvoke` 在第 3 次执行时触发 `AssertionError: third tool execution is not allowed`。
- 结论：当前实现按模型轮次计数，未按实际执行的工具数截断，同一响应中的第 3 个工具被错误执行，失败原因符合预期。

### GREEN

- 修改 `job_agent/matching/runtime.py`：
  - `calls` 仅在实际执行工具后递增，不再按带 tools 的模型轮次递增。
  - 对每次模型响应的 `tool_calls` 按剩余额度 `tool_calls[: max_tool_calls - calls]` 截断。
  - 命中上限后立即退出 bind_tools 循环，即使该响应里还有未执行的 `tool_calls`。
- 重新运行 `python -m pytest tests/test_runtime_tool_limit.py -v`
- 结果：2 个测试全部通过。

### Full Verification

- 运行 `python -m pytest`
- 结果：`42 passed in 4.59s`
