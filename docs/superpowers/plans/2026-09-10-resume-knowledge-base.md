# 简历知识库（第 3 章）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把第 1 章 `Resume` 按条目写入 MySQL 标量 + 切块，Milvus dense 检索；匹配支持文件路径与自增 ID；技能/职责先检索再打分，年限/学历/地点只读父表标量。

**Architecture:** `parse_resume` 只在 ingest 时进内存。父表存 `ResumeProfile` 标量，切块进 `resume_chunks` 再 embed。匹配图节点名不变：`resume_extract` 写入 `resume_id + ResumeProfile`；技能/职责节点 Python 检索后走现有 `score_*` tool；年限/学历/地点只读 profile。

**Tech Stack:** Python 3.11+、pydantic v2、LangChain / LangGraph、SQLAlchemy 2.x async + aiomysql、pymilvus、FlagEmbedding `BGEM3FlagModel`。API 以 Context7 2026-09-10 查询为准：
- `create_async_engine("mysql+aiomysql://user:pass@host:port/db?charset=utf8mb4")`
- `async_sessionmaker(engine, expire_on_commit=False)`；`class Base(AsyncAttrs, DeclarativeBase)`
- `mapped_column`；MySQL JSON 用 `sqlalchemy.JSON`
- PyMilvus：`MilvusClient(uri=..., token=...)`；向量字段 `DataType.FLOAT_VECTOR, dim=1024`；索引 `index_type="FLAT", metric_type="COSINE"`
- BGE-M3：`from FlagEmbedding import BGEM3FlagModel`；`BGEM3FlagModel(model_name)`；`encode(texts)["dense_vecs"]`；dense 维 **1024**
- 只取 dense，不启用 sparse / colbert
- 图执行仍 `await graph.ainvoke`；CLI `asyncio.run`

## Global Constraints

- 不持久化完整 `Resume` JSON；父表无 `payload_json`。
- 每次 ingest 按 `source_path` 复用 `resume_id`，标量覆盖、旧块全删再写，不因 `source_hash` 跳过嵌入。
- 电话/邮箱只进 `resumes.phone` / `email`，不进 `embed_text` / Milvus。
- 技能 Top-K=1、`chunk_type=skills`；职责 Top-K=5、`work_experience`+`project`。
- 召回为空：该维 3 分，证据「未召回到相关条目」。
- 年限/学历/地点禁止为打分拼接全部 chunk。
- 第二参整串十进制数字 → `resume_id`；否则 `.md` 路径。
- ID 不存在退出码 2；`status != vectorized` 退出码 1。
- 不修改第 1 章 `Resume` 字段；不改决策阈值；图节点名不变。
- 确定性代码 TDD；Prompt / 全图 / 真库用 `@pytest.mark.live`，缺配置失败不 skip。
- 真实简历原文不进 git。
- 上列 SDK 与记忆不符时停下来对照 Context7，不换栈。

## File Map

| Path | Responsibility |
| --- | --- |
| `pyproject.toml` / `.env.example` | 依赖与环境变量 |
| `job_agent/config.py` | `load_kb_config()` |
| `job_agent/kb/errors.py` | `KbConfigError`、`KbStoreError` |
| `job_agent/kb/models.py` | `ResumeProfile`、`ResumeChunk`、`ResumeChunkHit`、`ChunkType` |
| `job_agent/kb/profile.py` | `build_resume_profile(resume) -> ResumeProfile` |
| `job_agent/kb/embed_text.py` | 模板拼 `embed_text` |
| `job_agent/kb/chunker.py` | `Resume -> list[ResumeChunk]` |
| `job_agent/kb/stores.py` | MySQL / Milvus / Embedder Protocol |
| `job_agent/kb/mysql_store.py` | SQLAlchemy 实现 |
| `job_agent/kb/milvus_store.py` | PyMilvus 实现 |
| `job_agent/kb/embedder.py` | BGE-M3 |
| `job_agent/kb/pipeline.py` | `ingest_resume` / `load_profile` |
| `job_agent/kb/retrieve.py` | `search_resume_chunks` |
| `job_agent/ingest.py` | CLI |
| `job_agent/matching/scoring.py` | 年限/学历/地点改读 profile |
| `job_agent/matching/state.py` | `resume_id` + `resume_profile` |
| `job_agent/matching/resume_tools.py` | `ingest_resume_file` / `load_resume_by_id` |
| `job_agent/matching/evaluate_tools.py` | years/edu/location 吃 profile JSON |
| `job_agent/matching/nodes.py` | 检索 + 新状态 |
| `job_agent/matching/pipeline.py` / `match.py` | 双入口 |
| `tests/test_kb_*.py` / `tests/eval_ingest_resume.py` / `tests/eval_retrieve_dimensions.py` | 单测与 live |

---

### Task 1: 配置、错误类型、知识库模型

**Files:**
- Modify: `pyproject.toml`、`.env.example`、`job_agent/config.py`
- Create: `job_agent/kb/__init__.py`、`job_agent/kb/errors.py`、`job_agent/kb/models.py`
- Test: `tests/test_kb_models.py`、`tests/test_config.py`（增补）

**Interfaces:**
- Consumes: 现有 `load_llm_config`
- Produces: `KbConfig`、`load_kb_config() -> KbConfig`；`KbConfigError`、`KbStoreError`、`KbNotFoundError`；`ResumeProfile`；`ChunkType`；`ResumeChunk`；`ResumeChunkHit`；`BGE_M3_DIM = 1024`

- [ ] **Step 1: Write the failing test**

```python
from job_agent.kb.models import BGE_M3_DIM, ResumeProfile

def test_resume_profile_optional_scalars():
    profile = ResumeProfile(id=1, source_path="/a.md")
    assert profile.highest_degree is None
    assert profile.experience_months is None

def test_bge_m3_dim_is_1024():
    assert BGE_M3_DIM == 1024
```

`tests/test_config.py` 增补：缺 `MYSQL_HOST` 时 `load_kb_config` 抛 `KbConfigError`，消息含「缺少环境变量」。

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_kb_models.py tests/test_config.py -v`
Expected: FAIL，`job_agent.kb` 不存在

- [ ] **Step 3: Write minimal implementation**

`pyproject.toml` dependencies 增加：`sqlalchemy[asyncio]>=2.0`、`aiomysql`、`pymilvus`、`FlagEmbedding`。

`.env.example` 增补：

```
MYSQL_HOST=
MYSQL_PORT=3306
MYSQL_DB=
MYSQL_USER=
MYSQL_PASSWORD=
MILVUS_URI=
MILVUS_TOKEN=
BGE_M3_MODEL=BAAI/bge-m3
```

`KbConfig`：`mysql_host/port/db/user/password`、`milvus_uri`、`milvus_token: str`（可空）、`bge_m3_model`。`MYSQL_PORT` 缺省按缺变量失败（必须读到），值转 `int`。

`ResumeProfile` 字段：`id: int | None = None`、`source_path: str`、`source_hash: str = ""`、`name/location/phone/email: str | None = None`、`highest_degree: Literal["博士","硕士","本科"] | None = None`、`experience_months: int | None = None`、`status: Literal["pending","vectorized","failed"] = "pending"`。

`ChunkType = Literal["personal_info","skills","education","work_experience","project","summary"]`。

`ResumeChunk`：`id: int | None`、`resume_id: int | None`、`chunk_type: ChunkType`、`chunk_index: int`、`title: str`、`embed_text: str`、`payload: dict[str, object]`、`vector_status`、`vector_id: int | None`。

`ResumeChunkHit`：`id: int`、`chunk_type`、`score: float`、`embed_text: str`、`payload: dict[str, object]`。

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_kb_models.py tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .env.example job_agent/config.py job_agent/kb tests/test_kb_models.py tests/test_config.py
git commit -m "$(cat <<'EOF'
feat: add resume knowledge-base models and config

EOF
)"
```

---

### Task 2: 标量计算、embed 文本、切块

**Files:**
- Create: `job_agent/kb/profile.py`、`job_agent/kb/embed_text.py`、`job_agent/kb/chunker.py`
- Test: `tests/test_kb_chunker.py`

**Interfaces:**
- Consumes: `Resume`、`ResumeProfile`、`ResumeChunk`
- Produces: `build_resume_profile(resume: Resume, *, today: date | None = None) -> ResumeProfile`；`chunk_resume(resume: Resume) -> list[ResumeChunk]`；`render_embed_text(chunk_type, payload) -> str`

- [ ] **Step 1: Write the failing test**

```python
from datetime import date
from job_agent.kb.chunker import chunk_resume
from job_agent.kb.profile import build_resume_profile
from job_agent.resume.schema import (
    Education, PersonalInfo, Project, Resume, WorkExperience,
)

def _sample() -> Resume:
    return Resume(
        personal_info=PersonalInfo(name="张三", phone="13800138000", email="a@b.c", location="杭州"),
        skills=["Python", "LangGraph"],
        education=[Education(school="浙大", degree="硕士", major="CS", start_date="2018-09", end_date="2021-06")],
        work_experience=[WorkExperience(
            company="丁香园", title="后端", start_date="2021-07", end_date="present",
            responsibilities=["做 Agent"], achievements=["上线"],
        )],
        projects=[Project(name="Xagent", role="开发", responsibilities=["编排"])],
        summary="后端",
    )

def test_profile_highest_degree_and_months():
    profile = build_resume_profile(_sample(), today=date(2026, 9, 10))
    assert profile.name == "张三"
    assert profile.highest_degree == "硕士"
    assert profile.experience_months == (2026 - 2021) * 12 + (9 - 7)

def test_chunk_types_and_indices():
    chunks = chunk_resume(_sample())
    types = [c.chunk_type for c in chunks]
    assert types.count("work_experience") == 1
    assert types.count("project") == 1
    assert types.count("education") == 1
    assert "skills" in types
    skills = next(c for c in chunks if c.chunk_type == "skills")
    assert skills.chunk_index == 0
    assert "13800138000" not in skills.embed_text
    assert "a@b.c" not in "".join(c.embed_text for c in chunks)

def test_empty_projects_omits_project_chunk():
    resume = Resume(skills=["Python"])
    types = {c.chunk_type for c in chunk_resume(resume)}
    assert "project" not in types
    assert "personal_info" not in types  # name/location 都空
```

年限月份算法必须与现有 `_calculate_experience_years` 相同：最早 start 到最晚 end 的月份差（不是各段相加）。把 `_parse_year_month` / `_months_between` 抽到 `job_agent/kb/profile.py`（或 scoring 改为调用 profile），禁止两套日期逻辑。

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_kb_chunker.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

`embed_text` 模板（缺字段省略该键）：

```
【技能】Python, LangGraph
【教育】学校=浙大；学历=硕士；专业=CS；时间=2018-09 ~ 2021-06
【工作经历】公司=丁香园；职位=后端；时间=2021-07 ~ present
职责：做 Agent
业绩：上线
【项目】名称=Xagent；角色=开发
【地点】杭州；姓名=张三
【摘要】后端
```

payload 约定：
- `skills`: `{"skills": [...]}`
- `personal_info`: `{"name": ..., "location": ...}` 不含 phone/email
- 其余：对应 Pydantic `model_dump()`

title：技能=`skills`，地点=`location`，工作=公司名或 `work`，项目=项目名或 `project`，教育=学校名或 `education`，摘要=`summary`。

空 skills / 空摘要 / 无名无地点的 personal_info 不建块。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_kb_chunker.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add job_agent/kb/profile.py job_agent/kb/embed_text.py job_agent/kb/chunker.py tests/test_kb_chunker.py job_agent/matching/scoring.py
git commit -m "$(cat <<'EOF'
feat: chunk resumes and derive profile scalars

EOF
)"
```

若抽日期函数改了 `scoring.py`，一并提交，并跑 `pytest tests/test_scoring.py` 保持原断言通过（Task 3 再改签名）。

---

### Task 3: 年限/学历/地点改读 ResumeProfile；空召回 3 分

**Files:**
- Modify: `job_agent/matching/scoring.py`、`job_agent/matching/evaluate_tools.py`
- Test: `tests/test_scoring.py`

**Interfaces:**
- Consumes: `ResumeProfile`
- Produces: `score_years(job, profile: ResumeProfile) -> DimensionScore`；`score_education(job, profile)`；`score_location(job, profile)`；`missing_retrieval_score(dimension: DimensionName) -> DimensionScore`。`score_skills` / `score_responsibilities` 仍吃 `Resume` 切片。

- [ ] **Step 1: Write the failing test**

把 `test_years_meets_requirement`、`test_education_bachelor_required`、`test_both_locations_empty_is_three` 改为构造 `ResumeProfile`，不再传 `work_experience` / `education` 列表。

```python
def test_years_uses_experience_months_only():
    job = JobRequirement(years_required="3年以上")
    profile = ResumeProfile(source_path="x.md", experience_months=48)
    assert score_years(job, profile).score >= 4

def test_missing_retrieval_is_three():
    result = missing_retrieval_score("skills")
    assert result.score == 3
    assert result.evidence == "未召回到相关条目"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scoring.py -v`
Expected: FAIL（旧签名仍要 Resume）

- [ ] **Step 3: Write minimal implementation**

`score_years`：N 仍从 `years_required` 抽第一个整数；`experience_months is None` 或无 N → 3；`Y = experience_months / 12`；阈值不变。

`score_education`：用 `profile.highest_degree` 映射 博士=3/硕士=2/本科=1/None=0；岗位无学历词或列为 None → 3。

`score_location`：用 `profile.location` 替代 `resume.personal_info.location`。

`score_years_tool` / `education` / `location` 的第二参改为 `profile_json`，内部 `ResumeProfile.model_validate_json`。技能/职责 tool 仍 `resume_json`。

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_scoring.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add job_agent/matching/scoring.py job_agent/matching/evaluate_tools.py tests/test_scoring.py
git commit -m "$(cat <<'EOF'
feat: score years education location from profile scalars

EOF
)"
```

---

### Task 4: 存储协议、内存假实现、ingest 流水线

**Files:**
- Create: `job_agent/kb/stores.py`、`job_agent/kb/fake_stores.py`、`job_agent/kb/pipeline.py`
- Test: `tests/test_kb_ingest.py`

**Interfaces:**
- Consumes: `chunk_resume`、`build_resume_profile`、`parse_resume`
- Produces:

```python
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

class ResumeMysqlStore(Protocol):
    async def upsert_resume(self, *, source_path: str, source_hash: str, profile: ResumeProfile) -> int: ...
    async def replace_chunks(self, resume_id: int, chunks: list[ResumeChunk]) -> list[ResumeChunk]: ...
    async def mark_chunk_vectorized(self, chunk_id: int, vector_id: int) -> None: ...
    async def mark_resume_status(self, resume_id: int, status: str) -> None: ...
    async def get_profile(self, resume_id: int) -> ResumeProfile | None: ...
    async def get_chunks_by_ids(self, ids: list[int]) -> list[ResumeChunk]: ...

class ResumeMilvusStore(Protocol):
    async def delete_by_resume_id(self, resume_id: int) -> None: ...
    async def upsert_vectors(self, rows: list[tuple[int, int, str, int, list[float]]]) -> None:
        """(id, resume_id, chunk_type, chunk_index, embedding)"""
    async def search(self, *, resume_id: int, chunk_types: list[str], query_vector: list[float], top_k: int) -> list[tuple[int, float]]: ...

async def ingest_resume(
    path: str,
    *,
    mysql: ResumeMysqlStore,
    milvus: ResumeMilvusStore,
    embedder: Embedder,
    parse: Callable[[str], Resume] = parse_resume,
) -> int: ...

async def load_profile(resume_id: int, *, mysql: ResumeMysqlStore) -> ResumeProfile: ...
```

`load_profile`：找不到 → 抛可映射退出码 2 的错误（`KbNotFoundError`）；`status != "vectorized"` → `KbStoreError`（匹配退出码 1）。

- [ ] **Step 1: Write the failing test**

内存假存储：字典保存 resumes/chunks/vectors。`FakeEmbedder.embed` 返回 `[[0.0]*1024] * len(texts)`。

```python
@pytest.mark.asyncio
async def test_reingest_same_path_keeps_id_and_replaces_chunks(tmp_path):
    path = tmp_path / "r.md"
    path.write_text("# x\n", encoding="utf-8")
    mysql, milvus, embedder = FakeMysql(), FakeMilvus(), FakeEmbedder()
    resume = Resume(skills=["Python"])
    rid1 = await ingest_resume(str(path), mysql=mysql, milvus=milvus, embedder=embedder, parse=lambda _: resume)
    resume2 = Resume(skills=["Python", "Go"], work_experience=[WorkExperience(company="A", start_date="2020-01", end_date="2021-01")])
    rid2 = await ingest_resume(str(path), mysql=mysql, milvus=milvus, embedder=embedder, parse=lambda _: resume2)
    assert rid1 == rid2
    chunks = mysql.chunks_for(rid1)
    assert len([c for c in chunks if c.chunk_type == "skills"]) == 1
    assert milvus.ids_for(rid1) == {c.id for c in chunks}
```

用 `pytest-asyncio` 或 `asyncio.run` 包一层。本仓库尚未依赖 pytest-asyncio：测试里写 `asyncio.run(...)` 即可，不新增插件。

再测：`load_profile` 缺 ID 抛 `KbNotFoundError`。

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_kb_ingest.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

`ingest_resume` 流程：
1. `path` 用现有 `load_markdown` 校验 `.md`
2. `source_hash = sha256(file bytes)`
3. `resume = parse(path)`
4. `profile = build_resume_profile(resume)`
5. `resume_id = await mysql.upsert_resume(...)`（按 `source_path` 找到则 UPDATE 标量并 `status=pending`，否则 INSERT）
6. `await milvus.delete_by_resume_id(resume_id)`
7. `chunks = chunk_resume(resume)`；`await mysql.replace_chunks`（先删该 ID 旧行再 INSERT，返回带数据库 id 的 chunks）
8. 对每条：`vec = embedder.embed([c.embed_text])[0]` → milvus upsert → `mark_chunk_vectorized`
9. 任一条失败：`mark_resume_status(failed)` 后抛 `KbStoreError`
10. 全成功：`mark_resume_status(vectorized)`，返回 `resume_id`

`FakeEmbedder` 只给测试用，放 `tests/fakes.py` 或 `job_agent/kb/fake_stores.py`（测试可 import）。生产代码不要默认走 Fake。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_kb_ingest.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add job_agent/kb/stores.py job_agent/kb/fake_stores.py job_agent/kb/pipeline.py job_agent/kb/errors.py tests/test_kb_ingest.py
git commit -m "$(cat <<'EOF'
feat: ingest resume chunks through swappable stores

EOF
)"
```

---

### Task 5: 检索

**Files:**
- Create: `job_agent/kb/retrieve.py`
- Test: `tests/test_kb_retrieve.py`

**Interfaces:**
- Consumes: `ResumeMilvusStore.search`、`ResumeMysqlStore.get_chunks_by_ids`、`Embedder`
- Produces: `async def search_resume_chunks(resume_id: int, chunk_types: list[str], query: str, top_k: int, *, mysql, milvus, embedder) -> list[ResumeChunkHit]`

- [ ] **Step 1: Write the failing test**

预置两条 work 向量，query embed 与其中一条相同，`chunk_types=["work_experience","project"]`，`top_k=5`，断言命中 id 顺序、`payload` 来自 MySQL。再测 `chunk_types=["skills"]` 不会返回 work。空库返回 `[]`。

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_kb_retrieve.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
async def search_resume_chunks(...) -> list[ResumeChunkHit]:
    if not query.strip() or top_k <= 0:
        return []
    vector = embedder.embed([query])[0]
    pairs = await milvus.search(resume_id=resume_id, chunk_types=chunk_types, query_vector=vector, top_k=top_k)
    if not pairs:
        return []
    chunks = await mysql.get_chunks_by_ids([i for i, _ in pairs])
    by_id = {c.id: c for c in chunks}
    hits = []
    for chunk_id, score in pairs:
        chunk = by_id.get(chunk_id)
        if chunk is None:
            continue
        hits.append(ResumeChunkHit(id=chunk_id, chunk_type=chunk.chunk_type, score=score, embed_text=chunk.embed_text, payload=chunk.payload))
    return hits
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_kb_retrieve.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add job_agent/kb/retrieve.py tests/test_kb_retrieve.py
git commit -m "$(cat <<'EOF'
feat: retrieve resume chunks by dimension filters

EOF
)"
```

---

### Task 6: SQLAlchemy / PyMilvus / BGE-M3 实现

**Files:**
- Create: `job_agent/kb/mysql_store.py`、`job_agent/kb/milvus_store.py`、`job_agent/kb/embedder.py`
- Modify: `job_agent/kb/pipeline.py`（增加 `open_kb()` 默认实现）
- Test: `tests/test_kb_embedder.py`（monkeypatch FlagEmbedding）

**Interfaces:**
- Consumes: `KbConfig`、Task 4 Protocol
- Produces: `SqlAlchemyResumeStore`、`PyMilvusResumeStore`、`BgeM3Embedder`；`async def open_kb(config: KbConfig | None = None) -> KbHandles`

- [ ] **Step 1: Write the failing test**

```python
def test_bge_embedder_reads_dense_vecs(monkeypatch):
    class FakeModel:
        def encode(self, texts, **kwargs):
            return {"dense_vecs": [[1.0] + [0.0] * 1023 for _ in texts]}
    monkeypatch.setattr("job_agent.kb.embedder.BGEM3FlagModel", lambda *a, **k: FakeModel())
    embedder = BgeM3Embedder(model_name="BAAI/bge-m3")
    vectors = embedder.embed(["hello"])
    assert len(vectors[0]) == 1024
    assert vectors[0][0] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_kb_embedder.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

MySQL 模型：

```python
class ResumeRow(Base):
    __tablename__ = "resumes"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    source_path: Mapped[str] = mapped_column(String(1024), unique=True)
    source_hash: Mapped[str] = mapped_column(String(64))
    name: Mapped[str | None]
    location: Mapped[str | None]
    phone: Mapped[str | None]
    email: Mapped[str | None]
    highest_degree: Mapped[str | None]
    experience_months: Mapped[int | None]
    status: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())

class ResumeChunkRow(Base):
    __tablename__ = "resume_chunks"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    resume_id: Mapped[int] = mapped_column(ForeignKey("resumes.id"))
    chunk_type: Mapped[str] = mapped_column(String(32))
    chunk_index: Mapped[int]
    title: Mapped[str] = mapped_column(String(255))
    embed_text: Mapped[str] = mapped_column(Text)
    payload_json: Mapped[dict] = mapped_column(JSON)
    vector_status: Mapped[str] = mapped_column(String(32))
    vector_id: Mapped[int | None]
    UniqueConstraint("resume_id", "chunk_type", "chunk_index")
```

连接：`create_async_engine(f"mysql+aiomysql://{user}:{password}@{host}:{port}/{db}?charset=utf8mb4")`。ingest 开头 `await conn.run_sync(Base.metadata.create_all)`（开发期建表，不做独立 migration 工具）。

Milvus：集合名 `resume_chunks`。字段 `id INT64 PK`、`embedding FLOAT_VECTOR dim=1024`、`resume_id INT64`、`chunk_type VARCHAR(32)`、`chunk_index INT64`。不存在则 `create_collection` + FLAT/COSINE。`search` 的 `filter`/`expr`：`resume_id == {id} and chunk_type in {list}`。删除：`resume_id == {id}`。`id` 等于 MySQL chunk id，`auto_id=False`。

`BgeM3Embedder`：`BGEM3FlagModel(model_name, use_fp16=False)`；`encode(texts)["dense_vecs"]` 转 `list[list[float]]`。长度必须为 1024，否则抛 `KbStoreError`。

`open_kb()` 读 `load_kb_config()`，返回可 `async with` 关闭 engine 的句柄。`ingest_resume(path)` 无 store 参数时走 `open_kb()`。

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_kb_embedder.py tests/test_kb_ingest.py tests/test_kb_retrieve.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add job_agent/kb/mysql_store.py job_agent/kb/milvus_store.py job_agent/kb/embedder.py job_agent/kb/pipeline.py tests/test_kb_embedder.py
git commit -m "$(cat <<'EOF'
feat: persist resume chunks to MySQL and Milvus

EOF
)"
```

---

### Task 7: ingest CLI

**Files:**
- Create: `job_agent/ingest.py`
- Test: `tests/test_ingest_cli.py`

**Interfaces:**
- Consumes: `ingest_resume(path) -> int`
- Produces: `python -m job_agent.ingest <resume.md>` stdout 一行：`resume_id={id}`；stderr 可打块数日志

- [ ] **Step 1: Write the failing test**

无参 → 退出码 2，stderr 含 `用法：python -m job_agent.ingest <resume.md>`。monkeypatch `ingest_resume` 返回 12，临时 `.md` → 退出码 0，stdout 含 `resume_id=12`。`KbNotFoundError` 不适用于 ingest；`ResumeFileError` → 2；`ResumeExtractError` / `KbStoreError` → 1。

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingest_cli.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

仿 `job_agent/parse.py`：校验参数、`asyncio.run(ingest_resume(path))`、映射退出码。缺 KB 配置 `KbConfigError` → 2。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingest_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add job_agent/ingest.py tests/test_ingest_cli.py
git commit -m "$(cat <<'EOF'
feat: add resume ingest CLI

EOF
)"
```

---

### Task 8: 接入匹配图与双入口 CLI

**Files:**
- Modify: `job_agent/matching/state.py`、`resume_tools.py`、`resume_prompts.py`、`nodes.py`、`pipeline.py`、`match.py`
- Test: `tests/test_match_cli.py`、`tests/test_matching_node_logs.py`（若断言完整 Resume 则改 profile）、新 `tests/test_eval_retrieval.py`

**Interfaces:**
- Consumes: `ingest_resume`、`load_profile`、`search_resume_chunks`
- Produces: `MatchingState` 含 `resume_ref: str`、`resume_id`、`resume_profile`；删除状态中的完整 `Resume`
- `is_resume_id_ref(value: str) -> bool`：`value.isdecimal()`
- `run_matching(jd_path: str, resume_ref: str)`
- tools：`ingest_resume_file(path: str) -> str`（profile JSON，含 id）；`load_resume_by_id(resume_id: int) -> str`

- [ ] **Step 1: Write the failing tests**

CLI：`main(["jd.md", "12"])` 不把 `12` 当文件；monkeypatch `run_matching` 断言第二参 `"12"`。用法改为 `python -m job_agent.match <jd.md> <resume.md|resume_id>`。

`tests/test_eval_retrieval.py`：假 `search_resume_chunks` 返回空列表时，调用将用于 `eval_skills_node` 的纯函数 `build_skills_resume_slice(hits) -> Resume | None`，空则 `missing_retrieval_score`。有命中则 `Resume(skills=payload["skills"])`。职责：hits → 按 `chunk_type` 填 `work_experience` / `projects`。

查询文本：技能 `", ".join(must_have + nice_to_have)`；职责 `"\n".join(responsibilities)`。

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_match_cli.py tests/test_eval_retrieval.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

`resume_extract_node`：若 `is_resume_id_ref(resume_ref)` 则只绑 `load_resume_by_id`，否则只绑 `ingest_resume_file`。`output_schema=ResumeProfile`。写回 `resume_id=result.id`、`resume_profile=result`。ID 模式 Human prompt 用 `简历ID：{resume_ref}`。

`eval_years/education/location`：用 `ResumeProfile` 调对应 scorer 与 tool（`profile_json`）。`parse_join` 检查 `resume_profile`。

`eval_skills` / `eval_responsibilities`：Python `await search_resume_chunks(...)`（`open_kb()`）；空 hits → 直接把 `missing_retrieval_score` 写入状态，**仍**跑评估专家但切片用空 `Resume()` 且 scorer 改为返回 missing？（spec：召回为空该维 3 分）。实现锁定：空 hits **不调用** `score_skills`/`score_responsibilities` 的命中规则，节点内直接 `missing_retrieval_score`；评估专家仍调用一次 tool——为保持「必须调 tool」，增加 `missing_retrieval_score_tool(dimension: str)` 或让空切片走 `score_skills` 会得到「岗位未提供必须技能」而非「未召回到」。因此空召回必须走 `missing_retrieval_score`，tool 用 `@tool async def missing_retrieval_score_tool(dimension: str) -> str`。空召回时节点只绑这个 tool。

有命中：技能绑 `score_skills_tool`，职责绑 `score_responsibilities_tool`，切片来自 hits。

`report_node`：`candidate_name = profile.name`。

`match.py`：第二参是数字则不调用 `load_markdown`；否则 `load_markdown`。`KbNotFoundError`/`MatchingFileError`/`ResumeFileError`/`KbConfigError`/`ResumeConfigError` → 2；`KbStoreError`/`MatchingExtractError` → 1。

`run_matching` 初始 state：`jd_path`、`resume_ref`、`dimension_scores: {}`。兼容测试里旧键 `resume_path`：不要保留，改测试。

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_match_cli.py tests/test_eval_retrieval.py tests/test_matching_graph.py tests/test_scoring.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add job_agent/matching job_agent/match.py tests/test_match_cli.py tests/test_eval_retrieval.py tests/test_matching_node_logs.py
git commit -m "$(cat <<'EOF'
feat: match resumes from files or knowledge-base ids

EOF
)"
```

---

### Task 9: Live 评估

**Files:**
- Create: `tests/eval_ingest_resume.py`、`tests/eval_retrieve_dimensions.py`
- Modify: `tests/eval_matching_pipeline.py`、`pyproject.toml` live marker 说明
- Test: 上述 live 文件

**Interfaces:**
- Consumes: 真实 MySQL / Milvus / BGE-M3 / LLM
- Produces: 缺任一 KB 或 LLM 配置时 live **失败**（raise 或 assert），不 `pytest.skip`（样例简历文件不存在仍可 skip，与第 2 章一致）

- [ ] **Step 1: Write the failing live tests**

`eval_ingest_resume.py`：ingest 样例路径；`load_profile` 成功；`experience_months` 或 `highest_degree` 至少一项非空（样例真实简历应有工作/学历）；MySQL chunk 数 ≥ 4；Milvus `search` 不限 query、`top_k=100` 条数等于 vectorized 块数。

`eval_retrieve_dimensions.py`：用 `JobRequirement` gold 或解析 `xagent_jd.md` 的技能/职责字符串（可直接读 fixture 里技术词：`Python`、`Agent`）作 query；技能命中 `chunk_type=="skills"`；职责命中 `work_experience` 或 `project`。换说法：技能 query 用「精通 python 与大模型应用」而非词表原样。

`eval_matching_pipeline.py`：保留文件路径用例；新增数字 ID 用例（先 ingest 取 id 再 `run_matching(jd, str(resume_id))`）。报告含五维中文名与推荐/待定/不推荐。

- [ ] **Step 2: Run (will fail until env ready)**

Run: `pytest -m live tests/eval_ingest_resume.py tests/eval_retrieve_dimensions.py tests/eval_matching_pipeline.py -v`
Expected: 配置齐全则 PASS；缺 `MYSQL_*` / `MILVUS_URI` / `BGE_M3_MODEL` 则 `KbConfigError` 导致 FAIL 而非 skip

- [ ] **Step 3: Fix any integration gaps**

只修 live 暴露的接线问题，不改阈值。

- [ ] **Step 4: Re-run live + 全量非 live**

Run: `pytest -v` 与 `pytest -m live -v`
Expected: 默认（非 live）PASS；live 在有配置的机器 PASS

- [ ] **Step 5: Commit**

```bash
git add tests/eval_ingest_resume.py tests/eval_retrieve_dimensions.py tests/eval_matching_pipeline.py pyproject.toml
git commit -m "$(cat <<'EOF'
test: add live ingest retrieve and matching evals

EOF
)"
```

---

## Spec coverage

| Spec 项 | Task |
| --- | --- |
| 按条目切块、embed 模板、无电话邮箱 | 2 |
| 父表标量、无整份 JSON | 1、2、6 |
| 每次重建、复用 source_path ID | 4 |
| BGE-M3 1024 dense、Milvus FLAT COSINE | 6 |
| ingest CLI | 7 |
| match 文件 / ID | 8 |
| 技能/职责检索 + 空召回 3 分 | 5、8 |
| 年限/学历/地点只读列 | 3、8 |
| live 验收 | 9 |
