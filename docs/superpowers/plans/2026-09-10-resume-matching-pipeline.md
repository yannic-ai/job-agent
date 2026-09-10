# 简历匹配主流程（第 2 章）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 LangGraph 跑通「岗位解析 ∥ 简历提取 → 五维并行评估 → 决策 → 报告」，CLI 打印 Markdown 报告；需求解析按样例 JD 做细，其余专家做薄。

**Architecture:** `StateGraph` 扇出/扇入。每个专家节点：`@tool` 列表 + 最多 2 次 tool call + `with_structured_output(..., method="json_mode")`。确定性规则（JD 切分、五维打分、平均阈值、报告模板）是 Python 函数，tools 只做薄封装。简历提取 tool 内部调用第 1 章 `parse_resume`。

**Tech Stack:** Python 3.11+、pydantic v2、langchain / langchain-core / langchain-openai（本机已是 langchain 1.3.x）、**新增 langgraph≥1.0**、pytest。API 以 Context7 2026-09-10 查询为准：
- `from langgraph.graph import START, END, StateGraph`
- 并行扇入：多节点 `add_edge(START, a/b)`，再 `add_edge(a, join)` + `add_edge(b, join)`，join 等待全部完成
- `from langchain.tools import tool`
- `model.bind_tools([tool])`；`response.tool_calls`
- `ChatPromptTemplate.from_messages`（`langchain_core.prompts`，与第 1 章一致）
- `ChatOpenAI(...).with_structured_output(Model, method="json_mode")`（DeepSeek 不支持 json_schema）
- 图执行：`await graph.ainvoke(...)`；CLI `asyncio.run`

## Global Constraints

- 入口：`python -m job_agent.match <jd.md> <resume.md>`；JD 后缀 `.md`/`.txt`，简历 `.md`。
- 每节点最多 2 次 tool call；禁止 `create_react_agent` / 无限 recursion。
- 不做 RAG、向量库、Supervisor、开放 Agent Loop、聊天页。
- 不修改第 1 章 Resume schema / 提取 Prompt。
- JSON 英文 key；标量缺省 `null`，数组缺省 `[]`。
- 决策：五维算术平均；≥4.0 推荐；≥3.0 且 <4.0 待定；<3.0 不推荐。
- 每维分数只能是 1–5；以 score tool 返回值为准。
- 环境变量仍是 `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`。
- 简历原文不入库；样例 JD 可入库。
- Prompt / JD 提取 / 全图：不写假 LLM 单测，改评估集（`@pytest.mark.live`，缺配置失败不 skip）。其余确定性代码 TDD。
- LangGraph / LangChain API 与本文不一致时停下来对照 Context7，不自行换 SDK。
- 节点与 LLM 调用用 async；`parse_resume` 保持同步，在 tool 里直接调。

## File Map

| Path | Responsibility |
| --- | --- |
| `pyproject.toml` | 增加 `langgraph>=1.0` |
| `job_agent/matching/errors.py` | `MatchingFileError`、`MatchingExtractError` |
| `job_agent/matching/schemas.py` | `JobRequirement`、`DimensionScore`、`Decision` |
| `job_agent/matching/state.py` | `MatchingState` + `dimension_scores` reducer |
| `job_agent/matching/jd_splitter.py` | JD ATX 切分 |
| `job_agent/matching/jd_loader.py` | 读 JD 文件 |
| `job_agent/matching/scoring.py` | 五维规则 + `hit_ratio_to_score` |
| `job_agent/matching/decision.py` | `average_and_label` |
| `job_agent/matching/report.py` | `render_report` |
| `job_agent/matching/runtime.py` | 2 次 tool 上限循环 |
| `job_agent/matching/jd_prompts.py` | 需求解析 Prompt |
| `job_agent/matching/jd_tools.py` | `read_and_split_jd` |
| `job_agent/matching/resume_prompts.py` | 简历提取 Prompt |
| `job_agent/matching/resume_tools.py` | `parse_resume_file` |
| `job_agent/matching/evaluate_prompts.py` | 五维 Prompt |
| `job_agent/matching/evaluate_tools.py` | 五个 `score_*` |
| `job_agent/matching/decide_prompts.py` | 决策 Prompt |
| `job_agent/matching/decide_tools.py` | `average_and_label_tool` |
| `job_agent/matching/report_prompts.py` | 报告 Prompt |
| `job_agent/matching/report_tools.py` | `render_report_tool` |
| `job_agent/matching/nodes.py` | 各 async 节点 |
| `job_agent/matching/graph.py` | 编译 StateGraph |
| `job_agent/matching/pipeline.py` | `run_matching(jd, resume)` |
| `job_agent/match.py` | CLI |
| `tests/fixtures/xagent_jd.md` | 样例 JD |
| `tests/fixtures/xagent_jd.gold.json` | 需求解析 gold |
| `tests/eval_jd_parse.py` | 需求解析 live |
| `tests/eval_matching_pipeline.py` | 全图 live |

---

### Task 1: 匹配 schema 与错误类型

**Files:**
- Create: `job_agent/matching/__init__.py`
- Create: `job_agent/matching/errors.py`
- Create: `job_agent/matching/schemas.py`
- Test: `tests/test_matching_schema.py`

**Interfaces:**
- Consumes: 无
- Produces: `JobRequirement`、`DimensionScore`、`Decision`、`MatchingFileError`、`MatchingExtractError`；`DIMENSIONS = ("skills", "years", "education", "location", "responsibilities")`；`label_from_average(average: float) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_matching_schema.py
from job_agent.matching.schemas import DIMENSIONS, Decision, JobRequirement, label_from_average


def test_job_requirement_defaults():
    dumped = JobRequirement().model_dump()
    assert dumped["title"] is None
    assert dumped["must_have_skills"] == []
    assert dumped["nice_to_have_skills"] == []
    assert dumped["years_required"] is None
    assert dumped["education_required"] is None
    assert dumped["location"] is None
    assert dumped["responsibilities"] == []


def test_label_from_average_thresholds():
    assert label_from_average(4.0) == "推荐"
    assert label_from_average(3.0) == "待定"
    assert label_from_average(2.9) == "不推荐"


def test_decision_round_trip():
    decision = Decision(average=3.6, recommendation="待定")
    assert decision.recommendation == "待定"
    assert DIMENSIONS[0] == "skills"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_matching_schema.py -v`

Expected: FAIL（`job_agent.matching` 不存在）

- [ ] **Step 3: Write minimal implementation**

`job_agent/matching/__init__.py` 空文件。

`job_agent/matching/errors.py`:

```python
class MatchingFileError(Exception):
    """本地岗位文件无法读取。"""


class MatchingExtractError(Exception):
    """匹配流程中模型提取、工具循环或结构校验失败。"""
```

`job_agent/matching/schemas.py`:

```python
from typing import Literal

from pydantic import BaseModel, Field, field_validator

DIMENSIONS = (
    "skills",
    "years",
    "education",
    "location",
    "responsibilities",
)

Recommendation = Literal["推荐", "待定", "不推荐"]
DimensionName = Literal[
    "skills",
    "years",
    "education",
    "location",
    "responsibilities",
]


def _coerce_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


class JobRequirement(BaseModel):
    title: str | None = None
    must_have_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    years_required: str | None = None
    education_required: str | None = None
    location: str | None = None
    responsibilities: list[str] = Field(default_factory=list)

    @field_validator(
        "must_have_skills",
        "nice_to_have_skills",
        "responsibilities",
        mode="before",
    )
    @classmethod
    def _lists(cls, value: object) -> list[str]:
        return _coerce_str_list(value)


class DimensionScore(BaseModel):
    dimension: DimensionName
    score: int
    evidence: str

    @field_validator("score")
    @classmethod
    def _score_range(cls, value: int) -> int:
        if value not in {1, 2, 3, 4, 5}:
            raise ValueError("score must be 1-5")
        return value


def label_from_average(average: float) -> Recommendation:
    if average >= 4.0:
        return "推荐"
    if average >= 3.0:
        return "待定"
    return "不推荐"


class Decision(BaseModel):
    average: float
    recommendation: Recommendation
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_matching_schema.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add job_agent/matching/__init__.py job_agent/matching/errors.py job_agent/matching/schemas.py tests/test_matching_schema.py
git commit -m "feat: add matching schemas for JD, scores, and decision"
```

---

### Task 2: JD 读取与切分

**Files:**
- Create: `job_agent/matching/jd_loader.py`
- Create: `job_agent/matching/jd_splitter.py`
- Test: `tests/test_jd_splitter.py`
- Test: `tests/test_jd_loader.py`

**Interfaces:**
- Consumes: `MatchingFileError`
- Produces: `load_jd(path: str | Path) -> str`；`split_jd_sections(markdown: str) -> dict[str, str]`；`format_jd_sections(sections: dict[str, str]) -> str`

切分规则：`#` 与 `##` 都是区块边界（与简历不同：JD 样例是 `#职位介绍` 无空格）。标题正则：`^(#{1,6})\s*(.+?)\s*$`。只把 1–2 级当边界，更深标题留在当前块。映射先匹配更具体：

1. 标题含「加分」→ `nice_to_have`
2. 含「岗位要求」或「任职要求」或「职位要求」→ `requirements`
3. 含「岗位职责」或「工作职责」→ `responsibilities`
4. 含「职位介绍」或「岗位介绍」或「职位描述」→ `intro`
5. 其余 → `other`

无任何 1–2 级标题：全文 `{"other": text}`。

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_jd_splitter.py
from job_agent.matching.jd_splitter import format_jd_sections, split_jd_sections

SAMPLE = """#职位介绍
我们正在构建 Agent 平台。

#岗位职责
参与编排系统

#岗位要求
精通 Python

##加分项
有 LangChain 经验
"""


def test_split_maps_hash_headings_without_space():
    sections = split_jd_sections(SAMPLE)
    assert "Agent 平台" in sections["intro"]
    assert "编排" in sections["responsibilities"]
    assert "Python" in sections["requirements"]
    assert "LangChain" in sections["nice_to_have"]


def test_split_without_headings_goes_to_other():
    assert split_jd_sections("纯文本岗位") == {"other": "纯文本岗位"}


def test_format_includes_section_titles():
    text = format_jd_sections({"intro": "hello", "other": "x"})
    assert "职位介绍" in text or "intro" in text
    assert "hello" in text
```

```python
# tests/test_jd_loader.py
from pathlib import Path

import pytest

from job_agent.matching.errors import MatchingFileError
from job_agent.matching.jd_loader import load_jd


def test_load_jd_rejects_missing(tmp_path: Path):
    with pytest.raises(MatchingFileError):
        load_jd(tmp_path / "nope.md")


def test_load_jd_rejects_pdf(tmp_path: Path):
    path = tmp_path / "a.pdf"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(MatchingFileError):
        load_jd(path)


def test_load_jd_reads_md(tmp_path: Path):
    path = tmp_path / "jd.md"
    path.write_text("#职位介绍\nhello", encoding="utf-8")
    assert "hello" in load_jd(path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_jd_splitter.py tests/test_jd_loader.py -v`

Expected: FAIL（模块不存在）

- [ ] **Step 3: Write minimal implementation**

`jd_loader.py`：存在性、`is_file`、后缀 `.md`/`.txt`（小写比较），UTF-8 读取，否则 `MatchingFileError` 中文原因。

`jd_splitter.py`：按上面规则切分。`format_jd_sections` 用固定中文标题输出：

```text
# 职位介绍
{intro}

# 岗位职责
{responsibilities}

# 岗位要求
{requirements}

# 加分项
{nice_to_have}

# 其他
{other}
```

缺块写「（无）」。

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_jd_splitter.py tests/test_jd_loader.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add job_agent/matching/jd_loader.py job_agent/matching/jd_splitter.py tests/test_jd_splitter.py tests/test_jd_loader.py
git commit -m "feat: split JD markdown into intro, requirements, and perk sections"
```

---

### Task 3: 五维打分、决策与报告渲染（纯 Python）

**Files:**
- Create: `job_agent/matching/scoring.py`
- Create: `job_agent/matching/decision.py`
- Create: `job_agent/matching/report.py`
- Test: `tests/test_scoring.py`
- Test: `tests/test_decision.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `JobRequirement`、`DimensionScore`、`Decision`、`label_from_average`；第 1 章 `Resume`
- Produces:
  - `hit_ratio_to_score(ratio: float) -> int`
  - `score_skills(job: JobRequirement, resume: Resume) -> DimensionScore`
  - `score_years(job: JobRequirement, resume: Resume, *, today: date | None = None) -> DimensionScore`
  - `score_education(job: JobRequirement, resume: Resume) -> DimensionScore`
  - `score_location(job: JobRequirement, resume: Resume) -> DimensionScore`
  - `score_responsibilities(job: JobRequirement, resume: Resume) -> DimensionScore`
  - `average_and_label(scores: list[int]) -> Decision`（必须恰好 5 个 1–5，否则 `ValueError`）
  - `render_report(*, title: str | None, candidate_name: str | None, scores: list[DimensionScore], decision: Decision) -> str`

打分规则按 spec 原文实现。`today` 默认 `date.today()`，测试传入固定日。年限：最早 `start_date` 到最晚 `end_date` 的月份差/12；`present` 用 `today` 的 YYYY-MM。日期只认 `YYYY-MM`。

职责关键词：每条职责用 `re.findall(r"[A-Za-z]{2,}|[\u4e00-\u9fff]{2,}", text)`。

报告必须含：岗位 title、候选人姓名、表或列表中的「技能」「年限」「学历」「地点」「职责」、平均分 1 位小数、结论词。

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scoring.py
from datetime import date

from job_agent.matching.schemas import JobRequirement
from job_agent.matching.scoring import (
    hit_ratio_to_score,
    score_education,
    score_location,
    score_skills,
    score_years,
)
from job_agent.resume.schema import Education, PersonalInfo, Resume, WorkExperience


def test_hit_ratio_to_score_buckets():
    assert hit_ratio_to_score(0.0) == 1
    assert hit_ratio_to_score(0.25) == 2
    assert hit_ratio_to_score(0.5) == 3
    assert hit_ratio_to_score(0.89) == 4
    assert hit_ratio_to_score(0.9) == 5


def test_empty_must_have_skills_is_three():
    job = JobRequirement(must_have_skills=[])
    resume = Resume(skills=["python"])
    result = score_skills(job, resume)
    assert result.dimension == "skills"
    assert result.score == 3


def test_all_skills_hit_is_five():
    job = JobRequirement(must_have_skills=["python", "api"])
    resume = Resume(skills=["python", "fastapi"])
    assert score_skills(job, resume).score == 5


def test_both_locations_empty_is_three():
    job = JobRequirement(location=None)
    resume = Resume(personal_info=PersonalInfo(location=None))
    assert score_location(job, resume).score == 3


def test_years_meets_requirement():
    job = JobRequirement(years_required="3年以上")
    resume = Resume(
        work_experience=[
            WorkExperience(start_date="2018-01", end_date="present"),
        ]
    )
    result = score_years(job, resume, today=date(2026, 9, 10))
    assert result.score >= 4


def test_education_bachelor_required():
    job = JobRequirement(education_required="本科及以上")
    resume = Resume(education=[Education(degree="本科")])
    assert score_education(job, resume).score == 4
```

```python
# tests/test_decision.py
import pytest

from job_agent.matching.decision import average_and_label


def test_average_thresholds():
    assert average_and_label([5, 5, 5, 5, 5]).recommendation == "推荐"
    assert average_and_label([4, 4, 4, 4, 4]).recommendation == "推荐"
    assert average_and_label([3, 3, 3, 3, 3]).recommendation == "待定"
    assert average_and_label([2, 3, 3, 3, 3]).recommendation == "待定"
    assert average_and_label([1, 1, 1, 1, 1]).recommendation == "不推荐"


def test_requires_five_scores():
    with pytest.raises(ValueError):
        average_and_label([1, 2, 3, 4])
```

```python
# tests/test_report.py
from job_agent.matching.report import render_report
from job_agent.matching.schemas import Decision, DimensionScore


def test_report_contains_dimensions_and_verdict():
    scores = [
        DimensionScore(dimension="skills", score=4, evidence="技能 2/2 命中"),
        DimensionScore(dimension="years", score=5, evidence="年限充足"),
        DimensionScore(dimension="education", score=4, evidence="本科学历"),
        DimensionScore(dimension="location", score=3, evidence="地点缺失"),
        DimensionScore(dimension="responsibilities", score=4, evidence="职责 3/5"),
    ]
    text = render_report(
        title="Agent 平台核心开发",
        candidate_name="宗艳云",
        scores=scores,
        decision=Decision(average=4.0, recommendation="推荐"),
    )
    for word in ("技能", "年限", "学历", "地点", "职责", "推荐", "宗艳云", "Agent"):
        assert word in text
    assert "4.0" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_scoring.py tests/test_decision.py tests/test_report.py -v`

Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

按 spec 规则实现三个模块。`average_and_label`：`average = sum(scores)/5`，`recommendation = label_from_average(average)`。

`score_years`：从 `years_required` 用 `re.search(r"\d+", text)` 取 N；解析工作日期；缺数据 score=3，evidence 说明原因。

中文维度名映射：`skills→技能` 等，写在 `report.py` 的 `DIMENSION_LABELS`。

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_scoring.py tests/test_decision.py tests/test_report.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add job_agent/matching/scoring.py job_agent/matching/decision.py job_agent/matching/report.py tests/test_scoring.py tests/test_decision.py tests/test_report.py
git commit -m "feat: add deterministic matching scores, decision, and report"
```

---

### Task 4: 单节点 runtime（最多 2 次 tool call）

**Files:**
- Create: `job_agent/matching/runtime.py`
- Test: `tests/test_runtime_tool_limit.py`

**Interfaces:**
- Consumes: `LLMConfig`、`MatchingExtractError`
- Produces: `async def run_expert(*, messages: list, tools: list, output_schema: type[BaseModel], config: LLMConfig, llm: Any | None = None, max_tool_calls: int = 2) -> BaseModel`

行为（禁止 create_react_agent）：

1. `bound = llm.bind_tools(tools)`（测试注入假 llm；生产用 `ChatOpenAI` 与第 1 章相同参数 + `temperature=0`）。
2. `calls = 0`；循环 `await bound.ainvoke(history)`。
3. 若 `response.tool_calls` 非空：只执行直到 `calls == max_tool_calls` 的调用，把 `ToolMessage` 追加进 history；达到上限则离开循环，**不再**向带 tools 的模型发请求。
4. 然后 `structured = (llm 或新 ChatOpenAI).with_structured_output(output_schema, method="json_mode")`；`await structured.ainvoke(history)`。
5. 失败或非 schema 实例 → `MatchingExtractError`。

工具执行：`tool_map = {t.name: t for t in tools}`；`await tool.ainvoke(tool_call["args"])` 若无 `ainvoke` 则 `tool.invoke`。`ToolMessage` 从 `langchain_core.messages` 导入。

假 llm 必须记录 `bind_tools` 后 `ainvoke` 次数：第三次带 tools 的调用不得发生。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runtime_tool_limit.py
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

    def bind_tools(self, tools: list) -> "FakeLLM":
        self._tools = tools
        return self

    async def ainvoke(self, messages: list) -> Any:
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

    def with_structured_output(self, schema: type[BaseModel], method: str = "json_mode"):
        class Structured:
            async def ainvoke(inner_self, messages: list) -> Out:
                assert any(isinstance(m, ToolMessage) for m in messages)
                return Out(ok=True)

        return Structured()


class FakeTool:
    name = "noop"

    async def ainvoke(self, args: dict) -> str:
        return "ok"


def test_runtime_stops_after_two_tool_calls():
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_runtime_tool_limit.py -v`

Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

按上面 5 步实现 `run_expert`。生产路径：`llm is None` 时用 `ChatOpenAI(model=config.model, api_key=config.api_key, base_url=config.base_url, temperature=0, max_tokens=8192, timeout=180)`。

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_runtime_tool_limit.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add job_agent/matching/runtime.py tests/test_runtime_tool_limit.py
git commit -m "feat: cap expert tool calls at two per LangGraph node"
```

---

### Task 5: 依赖、状态与图拓扑

**Files:**
- Modify: `pyproject.toml`（dependencies 增加 `"langgraph>=1.0"`）
- Create: `job_agent/matching/state.py`
- Create: `job_agent/matching/nodes.py`（先放 stub，后续任务替换函数体）
- Create: `job_agent/matching/graph.py`
- Test: `tests/test_matching_graph.py`

**Interfaces:**
- Consumes: `JobRequirement`、`Resume`、`DimensionScore`、`Decision`
- Produces: `MatchingState`；`merge_scores(left, right) -> dict`；`build_matching_graph() -> 编译后图`；节点名：`jd_parse`、`resume_extract`、`parse_join`、`eval_skills`、`eval_years`、`eval_education`、`eval_location`、`eval_responsibilities`、`decide`、`report`

`MatchingState`（TypedDict）：

```python
from typing import Annotated, NotRequired, TypedDict

class MatchingState(TypedDict):
    jd_path: str
    resume_path: str
    job_requirement: NotRequired[JobRequirement | None]
    resume: NotRequired[Resume | None]
    dimension_scores: Annotated[dict[str, DimensionScore], merge_scores]
    decision: NotRequired[Decision | None]
    report: NotRequired[str | None]
```

`merge_scores`：`{**(left or {}), **(right or {})}`。

图边（Context7：join 等待全部入边）：

```
START → jd_parse
START → resume_extract
jd_parse → parse_join
resume_extract → parse_join
parse_join → eval_skills / eval_years / eval_education / eval_location / eval_responsibilities
eval_* → decide
decide → report
report → END
```

本任务 stub 节点：async，返回空 dict 或占位分数，保证能 `compile()`。不要在本任务调 LLM。安装：`pip install -e ".[dev]"`。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_matching_graph.py
from job_agent.matching.graph import build_matching_graph

REQUIRED = {
    "jd_parse",
    "resume_extract",
    "eval_skills",
    "eval_years",
    "eval_education",
    "eval_location",
    "eval_responsibilities",
    "decide",
    "report",
}


def test_compiled_graph_contains_required_nodes():
    app = build_matching_graph()
    names = set(app.nodes)
    assert REQUIRED <= names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_matching_graph.py -v`

Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

更新 `pyproject.toml` 依赖。实现 `state.py`、`graph.py`、`nodes.py` stub（`parse_join` 返回 `{}`）。`from langgraph.graph import END, START, StateGraph`。

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_matching_graph.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml job_agent/matching/state.py job_agent/matching/nodes.py job_agent/matching/graph.py tests/test_matching_graph.py
git commit -m "feat: compile LangGraph matching topology with parallel fan-in"
```

---

### Task 6: 需求解析专家（Prompt + tool + 节点）

**Files:**
- Create: `job_agent/matching/jd_prompts.py`
- Create: `job_agent/matching/jd_tools.py`
- Modify: `job_agent/matching/nodes.py`（实现 `jd_parse_node`）
- Test: `tests/test_jd_prompts.py`（只测模板变量，不调 LLM）

**Interfaces:**
- Consumes: `load_jd`、`split_jd_sections`、`format_jd_sections`、`run_expert`、`JobRequirement`、`load_llm_config`
- Produces: `build_jd_prompt() -> ChatPromptTemplate`；`read_and_split_jd(path: str) -> str`（`@tool`）；`async def jd_parse_node(state: MatchingState) -> dict`

System Prompt 必须包含：中文岗位需求解析专家；不编造；不写简历信息；地点没有则 null；技能短标签小写；年限学历保留原文；最多 2 次 tool call；必须输出完整 JobRequirement。

Human：`JD 文件路径：{jd_path}`，要求先调 `read_and_split_jd`。

`read_and_split_jd`：`load_jd` + `split_jd_sections` + `format_jd_sections`。

节点：`messages = await prompt.ainvoke({"jd_path": state["jd_path"]})`（或 `prompt.format_messages`）；`run_expert(..., tools=[read_and_split_jd], output_schema=JobRequirement)`；返回 `{"job_requirement": result}`。

- [ ] **Step 1: Write the failing prompt smoke test**

```python
# tests/test_jd_prompts.py
from job_agent.matching.jd_prompts import SYSTEM_PROMPT, build_jd_prompt


def test_jd_prompt_has_path_variable_and_role():
    prompt = build_jd_prompt()
    assert "jd_path" in prompt.input_variables
    assert "岗位需求解析" in SYSTEM_PROMPT
    assert "不编造" in SYSTEM_PROMPT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_jd_prompts.py -v`

Expected: FAIL

- [ ] **Step 3: Write prompts, tool, and node**

`from langchain.tools import tool`。`from langchain_core.prompts import ChatPromptTemplate`。

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_jd_prompts.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add job_agent/matching/jd_prompts.py job_agent/matching/jd_tools.py job_agent/matching/nodes.py tests/test_jd_prompts.py
git commit -m "feat: add JD parsing expert prompt, tool, and graph node"
```

---

### Task 7: 简历 / 评估 / 决策 / 报告专家（做薄）

**Files:**
- Create: `job_agent/matching/resume_prompts.py`
- Create: `job_agent/matching/resume_tools.py`
- Create: `job_agent/matching/evaluate_prompts.py`
- Create: `job_agent/matching/evaluate_tools.py`
- Create: `job_agent/matching/decide_prompts.py`
- Create: `job_agent/matching/decide_tools.py`
- Create: `job_agent/matching/report_prompts.py`
- Create: `job_agent/matching/report_tools.py`
- Modify: `job_agent/matching/nodes.py`
- Test: `tests/test_thin_expert_prompts.py`

**Interfaces:**
- Consumes: `parse_resume`、`score_*`、`average_and_label`、`render_report`、`run_expert`
- Produces: 各 `@tool` 与 async 节点，写入对应状态字段

简历 tool 文档字符串必须含「日后可改为从向量库按 ID 读取，本章不实现」。内部同步调用 `parse_resume`，返回 `resume.model_dump_json()`。失败包装为 `MatchingExtractError` 或透传 `ResumeExtractError`。节点把 JSON 再 `Resume.model_validate_json` 后写入 `resume`。

五维每个节点：Human 注入该维 JD/简历切片 JSON；tool 调用对应 `score_*`，返回 `DimensionScore.model_dump_json()`。节点**以 tool 结果为准**写入 `dimension_scores`（即使模型 structured 输出不同）。实现方式：`run_expert` 之后若能从最后一次 tool 输出解析 `DimensionScore` 则覆盖。最简单：节点直接 `score = score_skills(job, resume)` 并仍走 `run_expert` 让模型调用同一个 tool；写状态用 Python `score_*` 结果（与 tool 同一函数，保证一致）。仍必须 `run_expert` + 该维 tool，满足「有 tool list」。

决策节点：收集五维 score 列表，`run_expert` 调 `average_and_label_tool`，写状态用 `average_and_label(scores)`。

报告节点：`run_expert` 调 `render_report_tool`，写状态用 `render_report(...)`。

缺 `job_requirement` 或 `resume` 时评估节点抛 `MatchingExtractError`（正常图不会发生）。

- [ ] **Step 1: Write the failing prompt smoke test**

```python
# tests/test_thin_expert_prompts.py
from job_agent.matching.decide_prompts import SYSTEM_PROMPT as DECIDE
from job_agent.matching.evaluate_prompts import SYSTEM_PROMPT as EVAL
from job_agent.matching.report_prompts import SYSTEM_PROMPT as REPORT
from job_agent.matching.resume_prompts import SYSTEM_PROMPT as RESUME


def test_role_prompts_exist():
    assert "简历" in RESUME
    assert "评估" in EVAL or "打分" in EVAL
    assert "决策" in DECIDE
    assert "报告" in REPORT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_thin_expert_prompts.py -v`

Expected: FAIL

- [ ] **Step 3: Implement tools, prompts, and nodes**

评估 System：只打本维；证据引用规则结果；不编造公司项目。决策 System：必须调用平均 tool，不自己算。报告 System：必须调用渲染 tool。

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_thin_expert_prompts.py tests/test_matching_graph.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add job_agent/matching/resume_prompts.py job_agent/matching/resume_tools.py job_agent/matching/evaluate_prompts.py job_agent/matching/evaluate_tools.py job_agent/matching/decide_prompts.py job_agent/matching/decide_tools.py job_agent/matching/report_prompts.py job_agent/matching/report_tools.py job_agent/matching/nodes.py tests/test_thin_expert_prompts.py
git commit -m "feat: add thin resume, evaluation, decision, and report experts"
```

---

### Task 8: 流水线与 CLI

**Files:**
- Create: `job_agent/matching/pipeline.py`
- Create: `job_agent/match.py`
- Test: `tests/test_match_cli.py`

**Interfaces:**
- Consumes: `build_matching_graph`、`load_jd`、第 1 章 loader 校验逻辑 / `ResumeFileError`、`ResumeConfigError`
- Produces: `async def run_matching(jd_path: str, resume_path: str) -> MatchingState`；`main(argv: list[str] | None = None) -> int`

CLI：

- 不是两个参数 → stderr `用法：python -m job_agent.match <jd.md> <resume.md>`，码 2
- 先校验 JD（`load_jd`）与简历文件（复用 `load_markdown` 做存在性/后缀，不在 CLI 里解析 LLM）→ 码 2
- `asyncio.run(run_matching(...))`；缺配置 `ResumeConfigError` 码 2
- `MatchingExtractError` / `ResumeExtractError` 码 1
- 成功：stdout 打印 `state["report"]`，码 0；报告为空则码 1

`run_matching`：`graph = build_matching_graph()`；`await graph.ainvoke({"jd_path": ..., "resume_path": ..., "dimension_scores": {}})`。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_match_cli.py
from job_agent.match import main


def test_cli_requires_two_paths():
    assert main([]) == 2
    assert main(["only.md"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_match_cli.py -v`

Expected: FAIL

- [ ] **Step 3: Implement pipeline and CLI**

`match.py` 的 `if __name__ == "__main__"` 与 `parse.py` 相同模式。

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_match_cli.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add job_agent/matching/pipeline.py job_agent/match.py tests/test_match_cli.py
git commit -m "feat: add matching pipeline CLI"
```

---

### Task 9: 样例 JD、gold 与 live 评估

**Files:**
- Create: `tests/fixtures/xagent_jd.md`（用户提供的 JD 原文，保留 `#职位介绍` 等标题）
- Create: `tests/fixtures/xagent_jd.gold.json`
- Create: `tests/eval_jd_parse.py`
- Create: `tests/eval_matching_pipeline.py`

**Interfaces:**
- Consumes: `jd_parse_node` / `run_matching` / `JobRequirement`
- Produces: live 评估（缺 `.env` 失败不 skip；简历缺失可 skip，与第 1 章 eval 对文件缺失的处理一致——**配置缺失不得 skip**）

`xagent_jd.md` 使用用户粘贴的全文。

gold 至少含：

```json
{
  "title": "Agent 平台核心框架开发",
  "must_have_skills": ["python"],
  "nice_to_have_skills": ["langchain"],
  "years_required": "3年以上后端开发、AI系统或平台开发经验",
  "education_required": "计算机相关专业本科及以上学历",
  "location": null,
  "responsibilities": [
    "参与 Agent 平台核心框架开发，包括任务编排、规划系统、记忆系统和工具执行机制",
    "构建支持多步骤推理和任务执行的 Agent Runtime",
    "设计和实现 Agent 的记忆系统，包括短期上下文、长期记忆"
  ]
}
```

`eval_jd_parse.py` 硬性断言按 spec：七字段、`location is None`、skills 含 python、nice 含 langchain、年限含 3、学历含本科、职责≥3 且「编排|Runtime|记忆」至少命中两项。

`eval_matching_pipeline.py`：`RESUME_SAMPLE_PATH` 默认 `/Users/yannic/面试/宗艳云简历v9.md`；跑 `run_matching`；报告含「技能」「年限」「学历」「地点」「职责」以及「推荐」或「待定」或「不推荐」。缺 API 配置让 `load_llm_config` 抛错使测试失败。

本任务不写假 LLM。可先加一个不调模型的 fixture 文件存在性断言（确定性）：`tests/test_xagent_fixture.py` 读 md 含「职位介绍」。

- [ ] **Step 1: Write fixture existence test + eval files**

确定性：

```python
# tests/test_xagent_fixture.py
from pathlib import Path

def test_xagent_jd_fixture_exists():
    text = (Path(__file__).parent / "fixtures" / "xagent_jd.md").read_text(encoding="utf-8")
    assert "职位介绍" in text
    assert "加分项" in text
```

live 评估文件按上面硬性规则编写，`@pytest.mark.live`。

- [ ] **Step 2: Run deterministic test (expect fail before fixture)**

Run: `python -m pytest tests/test_xagent_fixture.py -v`

Expected: FAIL then after Step 3 PASS

- [ ] **Step 3: Write fixture, gold, evals**

- [ ] **Step 4: Run default pytest (live excluded) then live if `.env` present**

Run: `python -m pytest`

Expected: 既有第 1 章测试 + 本章确定性测试全部 PASS；live 被 `addopts = -m 'not live'` 排除。

Run: `python -m pytest -m live tests/eval_jd_parse.py tests/eval_matching_pipeline.py`

Expected: `.env` 配好则 PASS；全图较慢可接受。

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/xagent_jd.md tests/fixtures/xagent_jd.gold.json tests/eval_jd_parse.py tests/eval_matching_pipeline.py tests/test_xagent_fixture.py
git commit -m "test: add JD parse and matching pipeline live evaluations"
```

---

## Self-review vs spec

| Spec 项 | Task |
| --- | --- |
| 并行解析 + 五维并行 + 决策 + 报告 | 5–8 |
| 每节点 2 次 tool call | 4 |
| `@tool` + PromptTemplate | 6–7 |
| JobRequirement 精简集 | 1、6、9 |
| 五维 1–5 + 平均阈值 | 1、3 |
| 简历复用 parse_resume，预留读库注释 | 7 |
| 样例 JD 评估集 | 9 |
| CLI 退出码 | 8 |
| 不做 RAG / Agent Loop / Supervisor | 全局约束 |
| join 等两侧完成 | 5 的 `parse_join` |
