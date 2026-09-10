# Task 8 实施报告：接入匹配图与双入口 CLI

## 状态

已完成。

## 实现

- `MatchingState` 改为保存 `resume_ref`、`resume_id` 和 `resume_profile`，不再保存完整 `Resume`。
- `run_matching` 与 CLI 支持简历文件路径或十进制知识库 ID；ID 模式跳过文件读取。
- 简历专家按入口只绑定 `ingest_resume_file` 或 `load_resume_by_id`，输出 `ResumeProfile`。
- 年限、学历、地点直接基于 profile 评分；报告使用 `profile.name`。
- 技能与职责节点先检索知识库切片，分别使用指定查询、类型和 Top-K。
- 空召回绑定 `missing_retrieval_score_tool` 并写入中性 3 分，不执行命中评分规则。
- 知识库工具错误可穿透专家运行时，CLI 按约定映射退出码。

## 测试

- TDD RED：新增及更新测试初次运行得到 12 个预期失败。
- 目标回归：`36 passed in 5.42s`。
- 全量非 live：`python3 -m pytest -q`，`129 passed in 4.47s`。
- IDE lint：任务相关文件无诊断。

## 注意事项

- 未运行需要真实 LLM、MySQL、Milvus 和嵌入模型的 live 匹配评估。
- 未暂存或修改工作区中与 Task 8 无关的既有脏文件。
# Task 8 报告

## 完成情况

- 新增 `job_agent/matching/pipeline.py`，提供 `async def run_matching(...)`，负责构建匹配图并以初始状态调用 `ainvoke`。
- 新增 `job_agent/match.py`，提供 `python -m job_agent.match <jd.md> <resume.md>` CLI。
- CLI 在进入图前先调用 `load_jd` 与 `load_markdown` 做本地文件校验。
- CLI 使用 `asyncio.run(run_matching(...))` 执行匹配，并按要求映射退出码与标准输出/错误输出。

## 测试

- 新增 `tests/test_match_cli.py`，覆盖参数个数、文件校验、`run_matching` 入参、异常退出码映射、空报告与成功输出。
- 已执行 `python -m pytest tests/test_match_cli.py -v`。
- 已执行 `python -m pytest`，57 个测试全部通过。

## 备注

- 报告文件按要求保留在 `.superpowers/sdd/`，未纳入提交。
