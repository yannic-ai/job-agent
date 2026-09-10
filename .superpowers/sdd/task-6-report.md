# Task 6 Report

## RED

- 新增 `tests/test_jd_prompts.py`
- 运行 `python -m pytest tests/test_jd_prompts.py -v`
- 结果：`ModuleNotFoundError: No module named 'job_agent.matching.jd_prompts'`
- 结论：失败原因符合预期，证明 prompt smoke test 先于实现存在且有效

## GREEN

- 新增 `job_agent/matching/jd_prompts.py`
  - 定义 `SYSTEM_PROMPT`
  - 定义 `HUMAN_PROMPT`
  - 实现 `build_jd_prompt() -> ChatPromptTemplate`
- 新增 `job_agent/matching/jd_tools.py`
  - 实现 `@tool async def read_and_split_jd(path: str) -> str`
  - 串联 `load_jd`、`split_jd_sections`、`format_jd_sections`
- 更新 `job_agent/matching/nodes.py`
  - 替换 stub `jd_parse_node`
  - 接入 `build_jd_prompt()`、`read_and_split_jd`、`run_expert(...)`
  - 使用 `output_schema=JobRequirement` 与 `load_llm_config()`
- 保持其他 expert 节点为 stub，未额外实现
- 运行 `python -m pytest tests/test_jd_prompts.py -v`
- 结果：PASS

## Full Verification

- 运行 `python -m pytest`
- 结果：`44 passed in 4.78s`
