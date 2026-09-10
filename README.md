# job-agent

用 LLM 解析中文 Markdown 简历，写入 MySQL + Milvus 知识库，再按岗位 JD 做五维匹配并输出报告。

当前能力：

- 简历解析：本地 `.md` → 结构化 `Resume` JSON
- 知识库：按工作/项目/教育等条目切块，BGE-M3 向量化后双写 MySQL 与 Milvus
- 岗位匹配：解析 JD，技能/职责走向量召回，年限/学历/地点读父表标量，输出推荐 / 待定 / 不推荐

不支持 PDF/Word/OCR、跨简历搜人、混合检索。应用侧 LLM 走 OpenAI 兼容协议，密钥与地址全部来自环境变量。

## 架构设计

三层流水线，入口都是 CLI。

```text
简历.md ──► parse ──► Resume（内存）
                      │
                      ├─► ingest ──► MySQL resumes（标量）
                      │              MySQL resume_chunks（条目原文）
                      │              Milvus resume_chunks（向量）
                      │
JD.md + 简历.md|id ──► match（LangGraph）
                      ├─ jd_parse ∥ resume_extract
                      ├─ 五维并行评估
                      ├─ decide
                      └─ report（stdout Markdown）
```

### 简历解析

`job_agent/resume/`：读 Markdown → 按 `##` 标题切块 → LLM 结构化提取 → 日期/电话/技能规则后处理。

### 知识库

`job_agent/kb/`：不把整份简历 JSON 落库。

| 存储 | 内容 |
| --- | --- |
| MySQL `resumes` | 自增 `resume_id`，姓名、地点、电话、邮箱、最高学历、工龄月数 |
| MySQL `resume_chunks` | 一条工作 / 一个项目 / 一条教育 / 技能列表等，含 `embed_text` |
| Milvus `resume_chunks` | BGE-M3 dense 向量（1024 维，FLAT / COSINE），按 `resume_id` + `chunk_type` 过滤 |

电话、邮箱只进 MySQL 父表，不进向量。同一 `source_path` 再次 ingest 会复用 ID 并整份重建。

### 匹配图

`job_agent/matching/`：LangGraph `StateGraph`，无 Supervisor。每个专家最多 2 次 tool call。

```text
START
  ├─ jd_parse
  └─ resume_extract     文件则先入库；数字 ID 则只读库
        │
  parse_join
        ├─ eval_skills            JD 切片向量检索 Top-K=1
        ├─ eval_responsibilities  检索工作+项目 Top-K=5
        ├─ eval_years             读 experience_months
        ├─ eval_education         读 highest_degree
        └─ eval_location          读 location
        │
      decide   五维算术平均：≥4 推荐，≥3 待定，<3 不推荐
        │
      report
```

召回为空时该维打 3 分，证据为「未召回到相关条目」，匹配继续。

## 项目启动

需要 Python 3.11+、可连的 MySQL、Milvus，以及一台能下 BGE-M3 的机器（首次 embedding 会拉模型）。

### 1. 安装

```bash
cd /path/to/job-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`FlagEmbedding` 会带上 PyTorch，安装时间较长。

### 2. 环境变量

```bash
cp .env.example .env
```

填写：

| 变量 | 用途 |
| --- | --- |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` | LLM（OpenAI 兼容，可接 DeepSeek 等） |
| `MYSQL_HOST` / `MYSQL_PORT` / `MYSQL_DB` / `MYSQL_USER` / `MYSQL_PASSWORD` | 权威库，首次 ingest 会 `create_all` 建表 |
| `MILVUS_URI` / `MILVUS_TOKEN` | 向量库，token 可空 |
| `BGE_M3_MODEL` | 默认 `BAAI/bge-m3` |

可选：`JOB_AGENT_LOG_LEVEL`（默认 `INFO`，打到 stderr）。

### 3. 跑通三条命令

只解析，不连库：

```bash
python -m job_agent.parse /path/to/resume.md
```

写入知识库，stdout 打印 `resume_id=N`：

```bash
python -m job_agent.ingest /path/to/resume.md
```

匹配。第二参可以是简历文件（没有则先入库）或数字 ID（只读库）：

```bash
python -m job_agent.match tests/fixtures/xagent_jd.md /path/to/resume.md
python -m job_agent.match tests/fixtures/xagent_jd.md 12
```

成功时 stdout 是 Markdown 报告；日志在 stderr。退出码：参数/文件/缺配置为 `2`，解析或入库失败为 `1`。

### 4. 测试

```bash
python -m pytest -q
```

默认跳过 `@pytest.mark.live`。有完整 MySQL / Milvus / BGE / LLM 和样例简历时：

```bash
python -m pytest -m live -v
```

样例简历路径可用环境变量 `RESUME_SAMPLE_PATH` 覆盖。真实简历不要提交进 git。

## 目录

```text
job_agent/
  parse.py          # python -m job_agent.parse
  ingest.py         # python -m job_agent.ingest
  match.py          # python -m job_agent.match
  resume/           # 解析流水线
  kb/               # 切块、MySQL、Milvus、BGE-M3
  matching/         # LangGraph 匹配
docs/superpowers/   # 设计 spec 与实现计划
tests/
```
