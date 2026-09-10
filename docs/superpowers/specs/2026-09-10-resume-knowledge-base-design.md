# 简历知识库（第 3 章）设计

日期：2026-09-10  
状态：用户已批准（回复「执行」）  
范围：第 1 章 `Resume` 按结构化条目切块，双写 MySQL + Milvus；父表只存评估用标量，不持久化整份 JSON；匹配 CLI 支持文件路径与自增 `resume_id`；技能 / 职责 Python 先检索再打分。

## 背景与目标

第 1 章把本地中文 Markdown 抽成 `Resume`。第 2 章当场解析，把该维全部字段注入评估节点。第 3 章把简历落入知识库：嵌套经历进切块与向量；年限 / 学历 / 地点在父表落成单独列，打分只读这些列。

验收：样例简历 ingest 后父表标量与块数正确、Milvus 条数与块数一致；用 Xagent JD 的技能 / 职责换说法能召回对应块；`match` 传文件路径或数字 ID 都能打出五维报告。

## 需求

1. 切块按结构化条目：每段工作、每个项目、每条教育各一块；技能列表、姓名+地点、非空摘要各最多一块。输入是 `parse_resume` 的结果，不再切 Markdown。
2. MySQL 两层：`resumes` 存标量；`resume_chunks` 存条目原文。Milvus 只存向量和过滤标量。嵌入用 BGE-M3，只跑 dense 单路。不编 FAQ 式 `questions`。**不持久化完整 `Resume` JSON。**
3. 简历 ID 为 MySQL 自增主键。`ingest` 打印该 ID。
4. 每次 ingest 都整份重建：按 `source_path` 复用同一 `resume_id`，覆盖标量、删旧块和旧向量再写。不因内容哈希未变而跳过嵌入。
5. CLI：`python -m job_agent.ingest <resume.md>`；`python -m job_agent.match <jd.md> <resume.md|resume_id>`。第二参若整串为十进制数字则当 ID（只读库，没有则失败）；否则当文件路径（先入库再匹配）。
6. 技能 / 职责：节点内 Python 按该维 JD 切片检索 Top-K，用命中块的 `payload_json` 拼切片，再交给现有 `score_*` tool。召回为空则该维 3 分，证据写未召回到相关条目，匹配继续。
7. 年限 / 学历 / 地点：不检索、不组装完整 `Resume`。打分只读 `resumes.experience_months` / `highest_degree` / `location`，规则与第 2 章阈值一致。
8. 电话、邮箱只进 `resumes.phone` / `resumes.email`，不进 `embed_text`，不进 Milvus，本章报告不用这两项。
9. 配置全部从环境变量读取。

## 非目标

- 关键词召回、混合检索、重排
- 客服 FAQ 的 `category` / `questions` / `answer` 及手册元数据
- 对 Markdown 再做标题切分或重叠窗口
- 跨简历人才搜索产品化（标量列只是为年限/学历/地点打分和以后可过滤做准备）
- 把工作/项目/教育摊成 `resumes` 宽表列
- FastAPI、聊天页、OCR、PDF/Word
- 修改第 1 章 `Resume` 字段、改决策阈值
- 评估专家自己调检索 tool，或把检索和打分焊成一个 tool
- 把真实简历原文提交进 git

## 已否决的方案

- 按评估五维各切一块、整份简历一条向量
- 只写 Milvus；把 FAQ `questions` 套到简历块上
- 本章只建库、不改评估；或五维评估全部走向量
- 匹配入口只保留文件，或只改成简历 ID
- 用文件哈希或姓名当简历 ID
- 按内容哈希跳过重嵌，或只追加不覆盖
- 评估专家先 search 再 score；检索为空则整次失败或回退全量字段
- **父表只存 `payload_json`，年限 / 学历 / 地点靠反序列化完整 `Resume`**

## 架构

```
.md
  → parse_resume（内存中的 Resume，只用于切块和计算标量）
  → 按 source_path 找到或新建 resumes 行（自增 resume_id）
  → 写入标量：name / location / phone / email / highest_degree / experience_months
  → 删除该 ID 的旧 chunks（MySQL + Milvus）
  → 按条目建新块 → BGE-M3 → Milvus → 回填 vector_id
  → resumes.status=vectorized

match <jd> <resume.md | id>
  → 文件：先 ingest；数字 ID：只读库
  → resume_extract 写入 resume_id + ResumeProfile（标量），不写完整 Resume
  → eval_skills / eval_responsibilities：Python 检索 → 现有 score tool
  → eval_years / eval_education / eval_location：只读 ResumeProfile
  → decide → report（姓名用 profile.name）
```

图节点名不变，不新增向量节点。检索不占模型 tool 次数。

## 切块

空列表、空摘要不建块。`chunk_index` 与 `Resume` 列表下标一致。

| `chunk_type` | 来源 | 条数 | 本章谁查 |
| --- | --- | --- | --- |
| `personal_info` | 姓名 + 地点（不含电话邮箱） | 0–1 | 不查（地点打分用父表列） |
| `skills` | `skills` | 0–1 | 技能维，Top-K=1 |
| `education` | 每条 `Education` | 0–N | 不查（学历打分用父表列） |
| `work_experience` | 每段工作 | 0–N | 职责维（年限打分用父表列，不查这些块） |
| `project` | 每个项目 | 0–N | 职责维 |
| `summary` | 非空摘要 | 0–1 | 不查 |

职责维过滤 `work_experience` 与 `project`，Top-K=5。技能维 query：必须技能与加分技能逗号拼接。职责维 query：职责要点换行拼接。

`embed_text` 用固定模板，缺字段省略，不写 `null`，不 dump 整份简历。该字段落在 **chunk 行**，供向量重算。

## 父表标量怎么算

入库时从内存里的 `Resume` 计算，算法与第 2 章打分函数相同，只是把结果存下来：

| 列 | 计算 |
| --- | --- |
| `name` | `personal_info.name` |
| `location` | `personal_info.location` |
| `phone` / `email` | 对应字段，可空 |
| `highest_degree` | 所有 `education.degree` 里最高档：博士 / 硕士 / 本科；都没有则为 `null` |
| `experience_months` | 全部 `work_experience` 中最早 `start_date` 到最晚 `end_date` 的月份差（`present` 为运行时当前年月）；无可解析日期则为 `null` |

## 年限 / 学历 / 地点怎么评（改读列）

阈值与第 2 章一致，输入改为列而不是完整 JSON：

- **年限**：岗位 `years_required` 抓第一个整数 N；`experience_months` 为空或岗位无数字 → 3 分。Y = `experience_months / 12`。Y≥N+2→5，Y≥N→4，Y≥N-1→3，Y≥N-3→2，否则 1。
- **学历**：岗位原文最高学历词对照 `highest_degree`。岗位无学历词或列为 `null` → 3 分。博士岗：简历博士→5 否则 2；硕士岗：博士 5 / 硕士 4 / 本科 3 / 其余 2；本科岗：本科或以上→4 否则 2。
- **地点**：`JobRequirement.location` 与 `resumes.location` 去空白后子串包含→5；两侧都空→3；否则 2。

需要把 `score_years` / `score_education` / `score_location` 改成接受这些标量（或先做成 `ResumeProfile` 再调用），**禁止**为了打这三维去拼接全部 chunk。

## 表与集合

**`resumes`**

| 字段 | 说明 |
| --- | --- |
| `id` | 自增 PK，即简历 ID |
| `source_path` | 复用 ID 的键 |
| `source_hash` | 只记录，不用于跳过嵌入 |
| `name` / `location` / `phone` / `email` | 可空；phone/email 不进向量 |
| `highest_degree` | `博士` / `硕士` / `本科` / `null` |
| `experience_months` | 非负整数或 `null` |
| `status` | `pending` / `vectorized` / `failed` |
| 时间戳 | |

无 `payload_json`。

**`resume_chunks`：** `id`（Milvus 主键）、`resume_id`、`chunk_type`、`chunk_index`、`title`、`embed_text`、`payload_json`（**该条**对象，不是整份简历）、`vector_status`、`vector_id`、时间戳。唯一约束 `(resume_id, chunk_type, chunk_index)`。

**Milvus `resume_chunks`：** `id`、`embedding`、`resume_id`、`chunk_type`、`chunk_index`。不镜像原文。索引用 FLAT。维度以实现时 BGE-M3 官方输出为准。

## 模块

```
job_agent/ingest.py
job_agent/kb/
  chunker.py
  embed_text.py
  models.py            # ResumeProfile、chunk 模型
  mysql_store.py
  milvus_store.py
  embedder.py
  pipeline.py          # ingest_resume(path) -> resume_id
  retrieve.py          # search_resume_chunks(...)
```

简历专家 tool：`ingest_resume_file`（路径）与 `load_resume_by_id`（ID），返回 `ResumeProfile` JSON，不是完整 `Resume`。

匹配状态：`resume_id` + `ResumeProfile`；删除对完整 `Resume` 的依赖。报告用 `profile.name`。

配置：`MYSQL_HOST` / `MYSQL_PORT` / `MYSQL_DB` / `MYSQL_USER` / `MYSQL_PASSWORD`、`MILVUS_URI`（可选 `MILVUS_TOKEN`）、`BGE_M3_MODEL`。缺一则 CLI 退出码 2。

实现前 Context7 核对 SQLAlchemy 2.x async、Milvus SDK、BGE-M3 加载与维度。不换技术栈。

## 错误处理

| 情况 | 行为 |
| --- | --- |
| 路径不存在、不是 `.md`；数字 ID 在库中不存在 | 退出码 2 |
| 缺相关环境变量 | 退出码 2 |
| 解析失败 | 退出码 1，不留下半截 `resumes` 行 |
| 嵌入或 Milvus 失败 | ingest 退出码 1，`status=failed`；再跑整份重建 |
| ID 对应 `status != vectorized` | 匹配退出码 1，不打半截报告 |
| 技能 / 职责召回为空 | 该维 3 分，匹配继续 |
| `experience_months` / `highest_degree` / `location` 为空 | 该维按上表规则给 3 分（地点仅在岗位侧也空时为 3） |

## 验证

确定性单测（假存储）：切块类型与下标；`embed_text` 不含电话邮箱；同一 `source_path` 再 ingest，`resume_id` 不变且旧块被替换；父表写入 `highest_degree` 与 `experience_months`；年限 / 学历 / 地点 tool 不读取 `work_experience` 列表；空召回时技能 / 职责分为 3。

Live（缺配置失败而非 skip）：外部样例入库，标量非空约定（样例有地点或学历则列有值）、块数与 Milvus 条数一致；Xagent JD 技能 / 职责换说法召回对应 `chunk_type`；`eval_matching_pipeline` 对文件路径与数字 ID 都能打出含五维名称与推荐/待定/不推荐的报告。

```bash
python -m job_agent.ingest "/Users/yannic/面试/宗艳云简历v9.md"
python -m job_agent.match tests/fixtures/xagent_jd.md 12
python -m job_agent.match tests/fixtures/xagent_jd.md "/Users/yannic/面试/宗艳云简历v9.md"
```

## 依赖

现有：Python 3.10+、LangChain、LangGraph、Pydantic v2、python-dotenv。  
新增：SQLAlchemy 2.x（async）、Milvus Python SDK、BGE-M3 嵌入库（包名以 Context7 为准）。  
LLM 仍走 OpenAI 协议；嵌入不走 OpenAI embedding。
