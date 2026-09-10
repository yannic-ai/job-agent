# 简历解析（第 1 章）设计

日期：2026-09-09  
状态：用户已批准（回复「执行」）  
范围：Markdown 简历 → 结构化 JSON。不含岗位分析、岗位匹配、综合打分、工具调用、Agent 循环、多格式/OCR。

## 背景与目标

仓库目前为空。第 1 章只跑通最基本的简历解析：本地读取 Markdown → 按标题切块 → 一次 LLM 结构化提取 → 规则后处理，得到稳定 JSON。

验收：对真实样例 `/Users/yannic/面试/宗艳云简历v9.md` 跑通，stdout 输出符合本文 schema 的 JSON。

## 需求

1. 库 + CLI：`parse_resume(path) -> Resume`；`python -m job_agent.parse <resume.md>` 打印 JSON。
2. 仅中文 Markdown；仅本地文件；不支持 PDF/Word/OCR。
3. 切分功能块后再提取，缩小模型搜索范围。
4. 工作经历抽取「公司-职位-时间-职责-业绩」五元组；项目经历单列，不塞进工作履历。
5. Prompt 用 `ChatPromptTemplate` 模板化；System Prompt 写清简历解析角色与行为约束。
6. 模型走 OpenAI 协议，`base_url` / `api_key` / `model` 全在 `.env`。

## 非目标

- 工具调用、Agent、LangGraph 循环
- 多文件格式、OCR
- 岗位分析 / 匹配 / 打分
- `github`、证书等未在 schema 中的字段
- 把简历原文提交进 git（含真实手机号）

## 架构

单向流水线，无 Agent：

```
本地 .md
  → MarkdownLoader
  → SectionSplitter（## 标题规则切分）
  → ResumeExtractChain（PromptTemplate + ChatOpenAI + 结构化输出）
  → ResumeNormalizer（日期 / 电话 / 技能规则）
  → Resume JSON
```

只有提取步调用模型；切分与后处理为确定性 Python。

## 目录结构

```
job_agent/
  parse.py              # CLI：读路径，打印 JSON，失败非 0 退出
  config.py             # 从 .env 读取 OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
  resume/
    loader.py
    splitter.py
    schema.py
    prompts.py
    extractor.py
    normalizer.py
    pipeline.py         # parse_resume()
tests/
  test_loader.py
  test_splitter.py
  test_normalizer.py
  eval_resume_parse.py  # 对外部样例 + gold JSON 跑评估
  fixtures/
    zong_yanyun.gold.json
.env.example
```

包名 `job_agent`。评估脚本读取外部简历绝对路径（可用环境变量 `RESUME_SAMPLE_PATH` 覆盖，默认即用户提供的路径），gold JSON 入库，简历原文不入库。

## 数据模型

Pydantic 模型即输出契约。字段名英文；缺省：标量 `null`，数组 `[]`。

```text
Resume
  personal_info: PersonalInfo
    name: str | null
    phone: str | null
    email: str | null
    location: str | null
  education: list[Education]
    school, degree, major: str | null
    start_date, end_date: str | null   # 标准化后 YYYY-MM 或 present
  work_experience: list[WorkExperience]
    company, title: str | null
    start_date, end_date: str | null
    responsibilities: list[str]
    achievements: list[str]
  projects: list[Project]
    name, role: str | null
    start_date, end_date: str | null
    responsibilities: list[str]
    achievements: list[str]
  skills: list[str]
  summary: str | null
  target_role: str | null
```

`summary`：原文无独立摘要时，模型根据「专业能力 / 求职方向」概括，不得编造未出现的公司、项目或数字。

`target_role`：求职岗位 / 求职方向 / 期望职位原文。样例简历写在基本信息「求职方向」一行；单独 `## 求职意向` 等标题切到 `target_role` 区块。没有则 `null`。

`location` 取「期望城市」或「现居地」原文，不做省市区补全（简历写「杭州」就输出「杭州」）。

明显占位（如 `XXX`、`保密`）视为缺失，输出 `null`，不把占位符当真实联系方式。

`「个人实践学习」` 不进入 `work_experience`。能对应到项目的并入 `projects`；否则进入切分后的 `other`，提取时可见但不强制建条。

## 切分规则

- 仅按 Markdown ATX 标题：`#` 一行作为 `title`（文档主标题/姓名区）；`##` 作为区块边界。`###` 及更深标题留在所属 `##` 块内，不单独切。
- 标题关键词映射（包含即命中，先匹配更具体的规则）：

| 标题含 | section id |
| --- | --- |
| 基本信息、个人信息、个人资料、基本资料、联系 | `personal_info` |
| 专业能力、专业技能、技术栈、技术能力、技术专长、技能栈、技能 | `skills` |
| 工作履历、工作经历、工作经验、职业经历、从业经历、任职 | `work_experience` |
| 项目经历、项目经验、个人项目、开源项目、项目介绍 | `projects` |
| 教育经历、教育背景、学习经历、毕业院校、学历 | `education` |
| 求职意向、求职方向、求职岗位、求职目标、期望职位、期望岗位、意向岗位、意向职位、目标岗位、应聘职位 | `target_role` |
| （其余 `##`，含个人实践学习） | `other` |

- 未识别标题进 `other`，原文保留并交给提取步，避免漏字段。
- 同一 id 出现多次则按出现顺序拼接。
- 切分结果结构：`{section_id: markdown_text}`，提取 Prompt 按块名注入。

## Prompt 与提取

- `ChatPromptTemplate`：System + Human。
- System：角色为中文简历解析器；只填充给定 schema；不编造；没有的信息用 `null` / `[]`；工作经历抽五元组；项目抽 `name` / `role` / 时间 / 职责 / 业绩；技能输出短标签而非长句。
- Human：逐块填入已切分 Markdown（含 `title` 与 `other`）。
- 使用 LangChain 针对 Pydantic 的结构化输出（实现前用 Context7 核对当前版本 API，例如 `with_structured_output`），禁止手写 JSON 正则解析。
- 一次模型调用完成全部字段；不按块多次调用（第 1 章范围）。

## 模型接入

应用侧只使用 OpenAI 协议客户端（LangChain `ChatOpenAI` 或文档中的等价类）。环境变量：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`

不在代码里写死供应商。GPT / Claude / DeepSeek / Ollama 只要提供 OpenAI 兼容入口即可切换。实现时若官方类名或参数与记忆不符，停下来对照 Context7，不自行换 SDK。

## 后处理

纯规则，不再调用模型：

- 日期：`2015.07`、`2015年07月`、`2015.07 - 至今` 等 → `YYYY-MM`；进行中 / 至今 → `present`。无法解析则保持 `null`（不把垃圾字符串留下充数）。
- 电话：去除空格、横线、中文间隔符，仅保留数字。
- `skills`：trim、小写、去重、去空串，保持稳定顺序（首次出现为准）。
- `responsibilities` / `achievements`：去空白项。

## 错误处理

- 路径不存在、不是文件、后缀不是 `.md`（大小写不敏感）：CLI 打印中文错误，退出码 2。
- `.env` 缺必要变量：启动即失败，退出码 2，提示缺哪一项。
- 模型调用失败或结构化校验失败：退出码 1，打印原因；不输出半截 JSON。
- 切分后无任何 `##` 块：仍把全文作为 `other` 交给提取，不直接失败。

## 验证与验收

可单测（确定性代码，TDD）：

- loader：缺文件、非 md、正常读入
- splitter：标题映射、`###` 不单独切、未知标题进 `other`、同 id 拼接
- normalizer：日期、电话、skills 去重小写

Prompt / 提取：不写假 LLM 单测。评估集：

- 输入：`/Users/yannic/面试/宗艳云简历v9.md`
- 期望：`tests/fixtures/zong_yanyun.gold.json`（含 `projects[]`，按本文 schema 手写）
- 硬性核对：必填键齐全；`work_experience` 至少 3 条且公司能对上丁香园 / 小黄柜 / 恒生电子；`projects` 名称能对上「医考智能客服」等核心项目；日期为 `YYYY-MM` 或 `present`
- 软性：职责/业绩措辞允许与 gold 不完全逐字相同，结构和关键实体必须对

演示命令：

```bash
python -m job_agent.parse "/Users/yannic/面试/宗艳云简历v9.md"
```

退出码 0，stdout 为格式化 JSON。

## 依赖（实现时以 Context7 + 当时锁定版本为准）

- Python 3.11+
- `langchain` / `langchain-openai`（具体包名实现前查文档）
- `pydantic` v2
- `python-dotenv`

不引入 LangGraph、向量库、Langfuse（第 1 章不用）。
