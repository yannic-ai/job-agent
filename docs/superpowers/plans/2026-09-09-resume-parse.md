# 简历解析（第 1 章）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从本地中文 Markdown 简历跑通「读取 → `##` 切块 → 一次结构化提取 → 规则后处理」，库函数与 CLI 都输出符合契约的 JSON。

**Architecture:** 确定性 Python 负责 loader / splitter / normalizer；LangChain `ChatPromptTemplate` + `ChatOpenAI.with_structured_output(Resume)` 只调用一次模型。应用侧只走 OpenAI 协议，地址/模型/密钥全部来自 `.env`。无工具、无 Agent 循环。

**Tech Stack:** Python 3.11+、pydantic v2、langchain / langchain-core / langchain-openai、python-dotenv、pytest。API 以 Context7 2026-09-09 查询为准：`from langchain_core.prompts import ChatPromptTemplate`；`from langchain_openai import ChatOpenAI`；`ChatOpenAI(model=..., api_key=..., base_url=...)`；`llm.with_structured_output(Model, method="json_schema")`；`from dotenv import load_dotenv`。

## Global Constraints

- 仅中文 Markdown；仅本地文件；不支持 PDF/Word/OCR。
- 库 + CLI：`parse_resume(path) -> Resume`；`python -m job_agent.parse <resume.md>`。
- JSON 英文 key；标量缺省 `null`，数组缺省 `[]`。
- 切分只认 ATX：`#` → `title`，`##` → 区块，`###+` 留在所属 `##` 内。
- 一次模型调用完成全部字段；禁止手写 JSON 正则解析。
- 环境变量：`OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`。
- 简历原文不入库；评估读 `/Users/yannic/面试/宗艳云简历v9.md`（可用 `RESUME_SAMPLE_PATH` 覆盖）。
- 不引入 LangGraph、向量库、Langfuse。
- Prompt / 提取不写假 LLM 单测；改为评估集。其余确定性代码走 TDD。
- 实现 LangChain / dotenv 前若与本文 API 不一致，停下来用 Context7 核对，不自行换 SDK。

## File Map

| Path | Responsibility |
| --- | --- |
| `pyproject.toml` | 包元数据、依赖、pytest `pythonpath` |
| `.gitignore` | `.env`、venv、pycache |
| `.env.example` | 三个 OpenAI 协议变量示例 |
| `job_agent/__init__.py` | 包标记 |
| `job_agent/parse.py` | CLI |
| `job_agent/config.py` | 读 `.env`，返回 `LLMConfig` |
| `job_agent/resume/__init__.py` | 导出 `parse_resume` |
| `job_agent/resume/errors.py` | `ResumeFileError`、`ResumeConfigError`、`ResumeExtractError` |
| `job_agent/resume/schema.py` | Pydantic 输出契约 |
| `job_agent/resume/loader.py` | `load_markdown` |
| `job_agent/resume/splitter.py` | `split_sections` |
| `job_agent/resume/normalizer.py` | `normalize_resume` |
| `job_agent/resume/prompts.py` | System / Human 模板 |
| `job_agent/resume/extractor.py` | 一次结构化提取 |
| `job_agent/resume/pipeline.py` | `parse_resume` 串联 |
| `tests/test_schema.py` | schema 默认值 / dump |
| `tests/test_loader.py` | 文件读取失败与成功 |
| `tests/test_splitter.py` | 标题映射与边界 |
| `tests/test_normalizer.py` | 日期/电话/技能/占位 |
| `tests/test_config.py` | 缺环境变量 |
| `tests/eval_resume_parse.py` | 真简历硬性核对 |
| `tests/fixtures/zong_yanyun.gold.json` | 评估期望（无手机号原文文件） |

---

### Task 1: 脚手架与输出 schema

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `job_agent/__init__.py`
- Create: `job_agent/resume/__init__.py`
- Create: `job_agent/resume/errors.py`
- Create: `job_agent/resume/schema.py`
- Create: `tests/test_schema.py`

**Interfaces:**
- Consumes: 无
- Produces: `PersonalInfo`、`Education`、`WorkExperience`、`Project`、`Resume`；`ResumeFileError`、`ResumeConfigError`、`ResumeExtractError`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schema.py
from job_agent.resume.schema import Resume


def test_resume_defaults_and_nulls():
    resume = Resume()
    dumped = resume.model_dump()
    assert dumped["personal_info"]["name"] is None
    assert dumped["education"] == []
    assert dumped["work_experience"] == []
    assert dumped["projects"] == []
    assert dumped["skills"] == []
    assert dumped["summary"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_schema.py::test_resume_defaults_and_nulls -v`

Expected: FAIL（包或 `Resume` 不存在）

- [ ] **Step 3: Write minimal implementation**

`pyproject.toml`:

```toml
[project]
name = "job-agent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "langchain>=0.3",
    "langchain-core>=0.3",
    "langchain-openai>=0.3",
    "pydantic>=2",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = ["pytest>=8"]

[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["job_agent*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

`.gitignore`:

```
.env
.venv/
venv/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
dist/
build/
```

`.env.example`:

```
OPENAI_API_KEY=
OPENAI_BASE_URL=
OPENAI_MODEL=
```

`job_agent/resume/errors.py`:

```python
class ResumeFileError(Exception):
    """本地文件无法作为 Markdown 简历读取。"""


class ResumeConfigError(Exception):
    """缺少或无效的模型配置。"""


class ResumeExtractError(Exception):
    """模型提取或结构校验失败。"""
```

`job_agent/resume/schema.py`:

```python
from pydantic import BaseModel, Field


class PersonalInfo(BaseModel):
    name: str | None = None
    phone: str | None = None
    email: str | None = None
    location: str | None = None


class Education(BaseModel):
    school: str | None = None
    degree: str | None = None
    major: str | None = None
    start_date: str | None = None
    end_date: str | None = None


class WorkExperience(BaseModel):
    company: str | None = None
    title: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    responsibilities: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)


class Project(BaseModel):
    name: str | None = None
    role: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    responsibilities: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)


class Resume(BaseModel):
    personal_info: PersonalInfo = Field(default_factory=PersonalInfo)
    education: list[Education] = Field(default_factory=list)
    work_experience: list[WorkExperience] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    summary: str | None = None
```

空 `__init__.py` 两个。`job_agent/resume/__init__.py` 先空，Task 7 再导出 `parse_resume`。

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_schema.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore .env.example job_agent tests/test_schema.py
git commit -m "feat: add resume JSON schema and project scaffold"
```

---

### Task 2: Markdown loader

**Files:**
- Create: `job_agent/resume/loader.py`
- Create: `tests/test_loader.py`

**Interfaces:**
- Consumes: `ResumeFileError`
- Produces: `load_markdown(path: str | Path) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_loader.py
from pathlib import Path

import pytest

from job_agent.resume.errors import ResumeFileError
from job_agent.resume.loader import load_markdown


def test_load_markdown_reads_utf8(tmp_path: Path):
    path = tmp_path / "resume.md"
    path.write_text("# 张三\n", encoding="utf-8")
    assert load_markdown(path) == "# 张三\n"


def test_load_markdown_missing_file(tmp_path: Path):
    with pytest.raises(ResumeFileError, match="不存在"):
        load_markdown(tmp_path / "nope.md")


def test_load_markdown_rejects_non_md(tmp_path: Path):
    path = tmp_path / "resume.txt"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ResumeFileError, match="Markdown"):
        load_markdown(path)


def test_load_markdown_accepts_uppercase_md(tmp_path: Path):
    path = tmp_path / "resume.MD"
    path.write_text("# A\n", encoding="utf-8")
    assert "# A" in load_markdown(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_loader.py -v`

Expected: FAIL（`load_markdown` 未定义）

- [ ] **Step 3: Write minimal implementation**

```python
from pathlib import Path


from job_agent.resume.errors import ResumeFileError


def load_markdown(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise ResumeFileError(f"文件不存在：{file_path}")
    if not file_path.is_file():
        raise ResumeFileError(f"不是文件：{file_path}")
    if file_path.suffix.lower() != ".md":
        raise ResumeFileError(f"仅支持 Markdown（.md）文件：{file_path}")
    return file_path.read_text(encoding="utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_loader.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add job_agent/resume/loader.py tests/test_loader.py
git commit -m "feat: load local markdown resume files"
```

---

### Task 3: 按 ## 切块

**Files:**
- Create: `job_agent/resume/splitter.py`
- Create: `tests/test_splitter.py`

**Interfaces:**
- Consumes: Markdown 字符串
- Produces: `split_sections(markdown: str) -> dict[str, str]`；已知 id：`title`、`personal_info`、`skills`、`work_experience`、`projects`、`education`、`other`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_splitter.py
from job_agent.resume.splitter import split_sections

SAMPLE = """# 张三

## 基本信息
电话：123

## 专业能力
Python

## 工作履历
A 公司

### 不该单独成块
细节

## 项目经历
项目甲

## 教育经历
某大学

## 个人实践学习
自学

## 奇怪标题
额外
"""


def test_split_maps_known_headings_and_keeps_h3():
    sections = split_sections(SAMPLE)
    assert "张三" in sections["title"]
    assert "电话：123" in sections["personal_info"]
    assert "Python" in sections["skills"]
    assert "A 公司" in sections["work_experience"]
    assert "### 不该单独成块" in sections["work_experience"]
    assert "项目甲" in sections["projects"]
    assert "某大学" in sections["education"]
    assert "自学" in sections["other"]
    assert "额外" in sections["other"]


def test_split_without_headings_goes_to_other():
    sections = split_sections("纯文本简历")
    assert sections == {"other": "纯文本简历"}


def test_split_concatenates_same_id():
    text = "## 技能\nA\n\n## 专业能力\nB\n"
    sections = split_sections(text)
    assert "A" in sections["skills"]
    assert "B" in sections["skills"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_splitter.py -v`

Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# job_agent/resume/splitter.py
from __future__ import annotations

import re

_H_RE = re.compile(r"^(#{1,6})\s+(.*)$")

_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("项目经历", "项目经验"), "projects"),
    (("工作履历", "工作经历", "任职"), "work_experience"),
    (("教育经历", "教育背景"), "education"),
    (("基本信息", "个人信息", "联系"), "personal_info"),
    (("专业能力", "技能"), "skills"),
)


def _map_heading(title: str) -> str:
    for keywords, section_id in _RULES:
        if any(keyword in title for keyword in keywords):
            return section_id
    return "other"


def _append(sections: dict[str, str], key: str, text: str) -> None:
    body = text.strip()
    if not body:
        return
    if key in sections:
        sections[key] = sections[key] + "\n\n" + body
    else:
        sections[key] = body


def split_sections(markdown: str) -> dict[str, str]:
    lines = markdown.splitlines()
    sections: dict[str, str] = {}
    current_id: str | None = None
    buf: list[str] = []
    saw_heading = False

    def flush() -> None:
        nonlocal buf, current_id
        if current_id is None:
            buf = []
            return
        _append(sections, current_id, "\n".join(buf))
        buf = []

    for line in lines:
        match = _H_RE.match(line)
        if match:
            hashes, title = match.group(1), match.group(2).strip()
            if len(hashes) == 1:
                flush()
                saw_heading = True
                current_id = "title"
                buf = [line]
                continue
            if len(hashes) == 2:
                flush()
                saw_heading = True
                current_id = _map_heading(title)
                buf = [line]
                continue
        buf.append(line)

    flush()
    if not saw_heading:
        body = markdown.strip()
        return {"other": body} if body else {}
    return sections
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_splitter.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add job_agent/resume/splitter.py tests/test_splitter.py
git commit -m "feat: split markdown resumes by heading sections"
```

---

### Task 4: 规则后处理

**Files:**
- Create: `job_agent/resume/normalizer.py`
- Create: `tests/test_normalizer.py`

**Interfaces:**
- Consumes: `Resume` 及子模型
- Produces: `normalize_resume(resume: Resume) -> Resume`（返回新模型，不原地改）

- [ ] **Step 1: Write the failing test**

```python
# tests/test_normalizer.py
from job_agent.resume.normalizer import normalize_resume
from job_agent.resume.schema import PersonalInfo, Resume, WorkExperience


def test_normalize_dates_phone_skills_and_placeholders():
    resume = Resume(
        personal_info=PersonalInfo(
            name="XXX",
            phone="138-0000-1234",
            email="保密",
            location="杭州",
        ),
        work_experience=[
            WorkExperience(
                company="丁香园",
                start_date="2021.08",
                end_date="至今",
                responsibilities=["  ", "做题"],
                achievements=["TP99 下降"],
            )
        ],
        skills=["Python", "python", " Redis ", ""],
        summary="后端",
    )
    out = normalize_resume(resume)
    assert out.personal_info.name is None
    assert out.personal_info.phone == "13800001234"
    assert out.personal_info.email is None
    assert out.personal_info.location == "杭州"
    assert out.work_experience[0].start_date == "2021-08"
    assert out.work_experience[0].end_date == "present"
    assert out.work_experience[0].responsibilities == ["做题"]
    assert out.skills == ["python", "redis"]


def test_normalize_chinese_month_and_garbage_date():
    resume = Resume(
        work_experience=[
            WorkExperience(start_date="2015年07月", end_date="sometime"),
        ]
    )
    out = normalize_resume(resume)
    assert out.work_experience[0].start_date == "2015-07"
    assert out.work_experience[0].end_date is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_normalizer.py -v`

Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# job_agent/resume/normalizer.py
from __future__ import annotations

import re

from job_agent.resume.schema import (
    Education,
    PersonalInfo,
    Project,
    Resume,
    WorkExperience,
)

_PLACEHOLDERS = {"xxx", "保密", "na", "n/a"}
_DATE_RE = re.compile(
    r"^\s*(\d{4})\s*[.\-/年]\s*(\d{1,2})\s*月?\s*$"
)
_PRESENT = {"至今", "现在", "present", "current", "ongoing"}


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if not text or text.casefold() in _PLACEHOLDERS:
        return None
    return text


def normalize_phone(value: str | None) -> str | None:
    text = _blank_to_none(value)
    if text is None:
        return None
    digits = re.sub(r"\D", "", text)
    return digits or None


def normalize_date(value: str | None) -> str | None:
    text = _blank_to_none(value)
    if text is None:
        return None
    if text.casefold() in _PRESENT:
        return "present"
    match = _DATE_RE.match(text)
    if not match:
        return None
    return f"{match.group(1)}-{int(match.group(2)):02d}"


def normalize_skills(skills: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in skills:
        token = item.strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _clean_list(items: list[str]) -> list[str]:
    return [item.strip() for item in items if item.strip()]


def normalize_resume(resume: Resume) -> Resume:
    info = resume.personal_info
    return Resume(
        personal_info=PersonalInfo(
            name=_blank_to_none(info.name),
            phone=normalize_phone(info.phone),
            email=_blank_to_none(info.email),
            location=_blank_to_none(info.location),
        ),
        education=[
            Education(
                school=_blank_to_none(item.school),
                degree=_blank_to_none(item.degree),
                major=_blank_to_none(item.major),
                start_date=normalize_date(item.start_date),
                end_date=normalize_date(item.end_date),
            )
            for item in resume.education
        ],
        work_experience=[
            WorkExperience(
                company=_blank_to_none(item.company),
                title=_blank_to_none(item.title),
                start_date=normalize_date(item.start_date),
                end_date=normalize_date(item.end_date),
                responsibilities=_clean_list(item.responsibilities),
                achievements=_clean_list(item.achievements),
            )
            for item in resume.work_experience
        ],
        projects=[
            Project(
                name=_blank_to_none(item.name),
                role=_blank_to_none(item.role),
                start_date=normalize_date(item.start_date),
                end_date=normalize_date(item.end_date),
                responsibilities=_clean_list(item.responsibilities),
                achievements=_clean_list(item.achievements),
            )
            for item in resume.projects
        ],
        skills=normalize_skills(resume.skills),
        summary=_blank_to_none(resume.summary),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_normalizer.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add job_agent/resume/normalizer.py tests/test_normalizer.py
git commit -m "feat: normalize resume dates, phones, and skills"
```

---

### Task 5: .env 模型配置

**Files:**
- Create: `job_agent/config.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: `ResumeConfigError`；`python-dotenv.load_dotenv`
- Produces: `LLMConfig(api_key: str, base_url: str, model: str)`；`load_llm_config() -> LLMConfig`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import pytest

from job_agent.config import load_llm_config
from job_agent.resume.errors import ResumeConfigError


def test_load_llm_config_requires_all_vars(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with pytest.raises(ResumeConfigError, match="OPENAI_API_KEY"):
        load_llm_config()


def test_load_llm_config_reads_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "deepseek-chat")
    cfg = load_llm_config()
    assert cfg.api_key == "sk-test"
    assert cfg.base_url == "https://api.example.com/v1"
    assert cfg.model == "deepseek-chat"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`

Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
from dataclasses import dataclass
import os

from dotenv import load_dotenv

from job_agent.resume.errors import ResumeConfigError


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str


def load_llm_config() -> LLMConfig:
    load_dotenv()
    missing: list[str] = []
    values: dict[str, str] = {}
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        value = (os.getenv(key) or "").strip()
        if not value:
            missing.append(key)
        else:
            values[key] = value
    if missing:
        raise ResumeConfigError("缺少环境变量：" + "、".join(missing))
    return LLMConfig(
        api_key=values["OPENAI_API_KEY"],
        base_url=values["OPENAI_BASE_URL"],
        model=values["OPENAI_MODEL"],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add job_agent/config.py tests/test_config.py
git commit -m "feat: load OpenAI-compatible LLM config from env"
```

---

### Task 6: Prompt 与一次结构化提取

**Files:**
- Create: `job_agent/resume/prompts.py`
- Create: `job_agent/resume/extractor.py`

**Interfaces:**
- Consumes: `dict[str, str]` 切块、`LLMConfig`、`Resume`
- Produces: `build_extract_prompt() -> ChatPromptTemplate`；`extract_resume(sections: dict[str, str], config: LLMConfig) -> Resume`

本任务不写假 LLM 单测。实现后用 `python -c "from job_agent.resume.prompts import build_extract_prompt; print(build_extract_prompt().input_variables)"` 确认模板变量为 `title, personal_info, skills, work_experience, projects, education, other`。

- [ ] **Step 1: 再查 Context7**

实现前用 Context7 复核：`ChatPromptTemplate.from_messages`、`ChatOpenAI(..., base_url=..., api_key=..., model=...)`、`with_structured_output(Resume, method="json_schema")`。若官方入口变成 `langchain.prompts` 而 `langchain_core.prompts` 仍可用，继续用计划中的 import，不要换供应商 SDK。

- [ ] **Step 2: Write prompts.py**

```python
# job_agent/resume/prompts.py
from langchain_core.prompts import ChatPromptTemplate

SECTION_KEYS = (
    "title",
    "personal_info",
    "skills",
    "work_experience",
    "projects",
    "education",
    "other",
)

SYSTEM_PROMPT = """你是中文简历解析器。只根据用户提供的已切分区块填写 Resume schema，不编造。
规则：
- 原文没有的信息用 null 或空数组。
- 明显占位（XXX、保密）视为缺失。
- location 取期望城市或现居地原文，不要补全省市区。
- 工作经历抽取五元组：company、title、start_date、end_date、responsibilities、achievements。
- 项目抽取 name、role、start_date、end_date、responsibilities、achievements。
- 「个人实践学习」不得写入 work_experience；能对应到项目则进入 projects。
- skills 输出短标签（如 java、langgraph），不要整段能力描述。
- summary 可根据专业能力/求职方向概括，但不得编造未出现的公司、项目或数字。
- 日期尽量写成 YYYY-MM 或 至今；起止时间分到 start_date / end_date，不要写在一个字段里。
"""

HUMAN_PROMPT = """请根据下列已切分的简历区块提取结构化信息。

# 标题区
{title}

# 个人信息
{personal_info}

# 技能 / 专业能力
{skills}

# 工作经历
{work_experience}

# 项目经历
{projects}

# 教育经历
{education}

# 其他
{other}
"""


def escape_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def build_extract_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    )
```

- [ ] **Step 3: Write extractor.py**

```python
from langchain_openai import ChatOpenAI

from job_agent.config import LLMConfig
from job_agent.resume.errors import ResumeExtractError
from job_agent.resume.prompts import SECTION_KEYS, build_extract_prompt, escape_braces
from job_agent.resume.schema import Resume


def extract_resume(sections: dict[str, str], config: LLMConfig) -> Resume:
    prompt = build_extract_prompt()
    llm = ChatOpenAI(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=0,
    )
    chain = prompt | llm.with_structured_output(Resume, method="json_schema")
    payload = {
        key: escape_braces(sections[key]) if key in sections else "（无）"
        for key in SECTION_KEYS
    }
    try:
        result = chain.invoke(payload)
    except Exception as exc:  # noqa: BLE001 — 统一成提取失败
        raise ResumeExtractError(f"简历提取失败：{exc}") from exc
    if not isinstance(result, Resume):
        raise ResumeExtractError("模型未返回 Resume 结构")
    return result
```

extractor 不要再 import `ChatPromptTemplate`（只在 prompts 里构建）。

- [ ] **Step 4: Smoke-check prompt variables**

Run: `python -c "from job_agent.resume.prompts import build_extract_prompt; print(sorted(build_extract_prompt().input_variables))"`

Expected: `['education', 'other', 'personal_info', 'projects', 'skills', 'title', 'work_experience']`

- [ ] **Step 5: Commit**

```bash
git add job_agent/resume/prompts.py job_agent/resume/extractor.py
git commit -m "feat: extract structured resume via ChatPromptTemplate"
```

---

### Task 7: pipeline 与 CLI

**Files:**
- Create: `job_agent/resume/pipeline.py`
- Create: `job_agent/parse.py`
- Modify: `job_agent/resume/__init__.py`

**Interfaces:**
- Consumes: `load_markdown`、`split_sections`、`extract_resume`、`normalize_resume`、`load_llm_config`
- Produces: `parse_resume(path: str | Path) -> Resume`；CLI 退出码 0/1/2

- [ ] **Step 1: Implement pipeline**

```python
from pathlib import Path

from job_agent.config import load_llm_config
from job_agent.resume.extractor import extract_resume
from job_agent.resume.loader import load_markdown
from job_agent.resume.normalizer import normalize_resume
from job_agent.resume.schema import Resume
from job_agent.resume.splitter import split_sections


def parse_resume(path: str | Path) -> Resume:
    markdown = load_markdown(path)
    sections = split_sections(markdown)
    config = load_llm_config()
    extracted = extract_resume(sections, config)
    return normalize_resume(extracted)
```

`job_agent/resume/__init__.py` 导出 `parse_resume`。

- [ ] **Step 2: Implement CLI** `job_agent/parse.py`

```python
from __future__ import annotations

import sys

from job_agent.resume.errors import ResumeConfigError, ResumeExtractError, ResumeFileError
from job_agent.resume.pipeline import parse_resume

USAGE = "用法：python -m job_agent.parse <resume.md>"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(USAGE, file=sys.stderr)
        return 2
    try:
        resume = parse_resume(args[0])
    except (ResumeFileError, ResumeConfigError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except ResumeExtractError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"简历解析失败：{exc}", file=sys.stderr)
        return 1
    print(resume.model_dump_json(indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

`job_agent/resume/__init__.py`：

```python
from job_agent.resume.pipeline import parse_resume

__all__ = ["parse_resume"]
```

- [ ] **Step 3: CLI 参数错误可测**

Run: `python -m job_agent.parse`

Expected: 非 0；stderr 含用法。不要在无 `.env` 时要求成功解析。

- [ ] **Step 4: Commit**

```bash
git add job_agent/resume/pipeline.py job_agent/resume/__init__.py job_agent/parse.py
git commit -m "feat: wire resume parse pipeline and CLI"
```

---

### Task 8: 评估集跑通真简历

**Files:**
- Create: `tests/fixtures/zong_yanyun.gold.json`
- Create: `tests/eval_resume_parse.py`

**Interfaces:**
- Consumes: `parse_resume`；外部 md；gold JSON
- Produces: `python -m pytest tests/eval_resume_parse.py -v` 在配置齐全时通过

- [ ] **Step 1: 手写 gold JSON**

按真实简历填写 `tests/fixtures/zong_yanyun.gold.json`（schema 含 `projects[]`）。硬性字段必须正确：

- `personal_info.name` = `宗艳云`
- `personal_info.location` = `杭州`
- `personal_info.email` = `null`（原文 XXX）
- `education[0]`：东华理工大学 / 本科 / 软件 / `2011-09` / `2015-07`
- `work_experience` 至少 3 条，公司含 丁香园、小黄柜、恒生电子
- `projects` 名称能覆盖：医考智能客服、刷题/看课、题库、公开课（可写简称，评估用子串）
- 日期均为 `YYYY-MM` 或 `present`
- `skills` 为短标签；`summary` 非空

职责/业绩金标写关键事实即可，评估不做逐字相等。

- [ ] **Step 2: Write eval_resume_parse.py**

```python
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from job_agent.resume.pipeline import parse_resume

DEFAULT_SAMPLE = "/Users/yannic/面试/宗艳云简历v9.md"
DATE_RE = re.compile(r"^(\d{4}-\d{2}|present)$")
REQUIRED_KEYS = {
    "personal_info",
    "education",
    "work_experience",
    "projects",
    "skills",
    "summary",
}


def _sample_path() -> Path:
    return Path(os.environ.get("RESUME_SAMPLE_PATH", DEFAULT_SAMPLE))


def _walk_dates(obj, key: str | None = None) -> None:
    if isinstance(obj, dict):
        for child_key, child in obj.items():
            _walk_dates(child, child_key)
        return
    if isinstance(obj, list):
        for child in obj:
            _walk_dates(child, key)
        return
    if key in {"start_date", "end_date"} and obj is not None:
        assert DATE_RE.match(str(obj)), f"bad date {key}={obj}"


def test_parse_zong_yanyun_resume():
    path = _sample_path()
    if not path.is_file():
        pytest.skip(f"sample resume not found: {path}")
    resume = parse_resume(path)
    dumped = resume.model_dump()
    assert REQUIRED_KEYS <= dumped.keys()
    assert dumped["personal_info"]["name"] == "宗艳云"
    companies = " ".join(item.get("company") or "" for item in dumped["work_experience"])
    assert "丁香园" in companies
    assert "小黄柜" in companies
    assert "恒生" in companies
    assert len(dumped["work_experience"]) >= 3
    names = " ".join(item.get("name") or "" for item in dumped["projects"])
    assert "医考智能客服" in names
    assert "题库" in names
    assert "公开课" in names
    _walk_dates(dumped)
    gold_path = Path(__file__).parent / "fixtures" / "zong_yanyun.gold.json"
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    print("gold name", gold["personal_info"]["name"], "got", dumped["personal_info"]["name"])
```

- [ ] **Step 3: 安装依赖、配置 .env、跑单测 + 评估**

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/test_schema.py tests/test_loader.py tests/test_splitter.py tests/test_normalizer.py tests/test_config.py -v
python -m pytest tests/eval_resume_parse.py -v
python -m job_agent.parse "/Users/yannic/面试/宗艳云简历v9.md"
```

Expected: 单测 PASS；评估 PASS（需有效 `.env`）；CLI 退出 0 且 stdout 为 JSON。

若模型 API 与 Context7 不一致：停止改供应商，对照文档微调 import/参数后重跑。

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/zong_yanyun.gold.json tests/eval_resume_parse.py
git commit -m "test: add resume parse eval against annotated sample"
```

- [ ] **Step 5: 更新 `dev-notes/ch01.md`**

每个 Task 完成后立刻追加一段（用户原话「执行」、本任务文件/结论、纠偏、翻车），禁止收尾一次性补记。
