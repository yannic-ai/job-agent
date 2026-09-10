# 简历匹配主流程（第 2 章）设计

日期：2026-09-10  
状态：待用户审阅 spec 文件  
范围：LangGraph 编排「岗位解析 ∥ 简历提取 → 五维并行评估 → 决策 → 报告」。本章把需求解析专家做细；简历提取复用第 1 章；评估 / 决策 / 报告跑通但规则做薄。

## 背景与目标

第 1 章已能把本地中文 Markdown 简历抽成 `Resume` JSON。第 2 章在此之上建立匹配主流程：读岗位 JD + 读简历，多维度评估后给出一份简单报告。

验收：对样例 JD `tests/fixtures/xagent_jd.md` 与第 1 章外部样例简历跑通图，stdout 打印含五维分数、平均分、推荐/待定/不推荐的 Markdown 报告。

## 需求

1. 主流程：解析岗位需求 → 读取简历 → 多维度评估 → 决策汇总 → 生成报告。
2. 角色：需求解析专家、简历提取专家、评估专家（五维并行节点的统称）、决策专家、报告生成专家。
3. Prompt 用 `ChatPromptTemplate` 模板化；每个角色有独立 System Prompt（角色设定 + 行为约束）。
4. 协作：串行流水线 + 并行评估。解析阶段岗位解析与简历提取并行；两者都成功后进入五维并行评估；五维都完成后决策；再报告。
5. 每个专家节点自带 tool list；节点内模型最多 2 次 tool call，到点必须结构化离开，不做成开放式 Agent Loop。
6. 入口：本地文件。`python -m job_agent.match <jd.md> <resume.md>`。
7. 需求解析输出精简匹配集：岗位名称、必须技能、加分技能、年限、学历、地点、职责要点。
8. 评估五维：技能、年限、学历、地点、职责匹配；每维 1–5 分 + 一句证据。
9. 决策：五维算术平均（仍为 1–5）；≥4 推荐，≥3 且 <4 待定，<3 不推荐。无硬门槛一票否决。

## 非目标

- 多轮开放式 Agent Loop、Supervisor 总控
- 向量检索、RAG、异步落向量库（简历提取预留「以后改成读库」的接口注释，图上无向量节点）
- 聊天页、FastAPI、OCR、PDF/Word
- 评估精打、可调权重、人工校准模型
- 修改第 1 章 Resume schema / 提取 Prompt（除非匹配节点调用时发现接口损坏）
- 把真实简历原文提交进 git

## 已否决的方案

- 评估并进决策专家单节点、或评估作为第 5 个串行单节点一次打完五维
- 本章接真实向量库、CLI 第二参改成简历 ID
- Supervisor 调度
- 评估/决策/报告做成纯 Python 节点、没有本角色 tool list
- 决策先卡学历/年限硬门槛

## 架构

LangGraph `StateGraph`，无 Supervisor。每个专家节点：绑定该角色 tool list → 最多 2 次 tool call → 结构化写回图状态 → 离开。

```
START
  ├─ jd_parse          需求解析专家
  └─ resume_extract    简历提取专家
           │ 两者都完成后扇入（job_requirement 与 resume 均已写入）
  ├─ eval_skills
  ├─ eval_years
  ├─ eval_education
  ├─ eval_location
  └─ eval_responsibilities     ← 合称「评估专家」
           │ 五维都完成后扇入
         decide          决策专家
           │
         report          报告生成专家
           │
          END
```

解析扇入必须等两侧都成功再启动五维；任一侧失败则不进入评估。实现时用 Context7 核对当前 LangGraph 的 join 写法（例如 `defer`、条件边或官方推荐的 barrier），禁止「只完成一侧就启动评估」。五维扇出/扇入同样要等五条都完成再决策。

并行节点写状态必须带 reducer：五维分数按 `dimension` 键合并，不得互相覆盖。

## 目录结构

```
job_agent/match.py                 # CLI
job_agent/matching/
  state.py                         # 图状态
  graph.py                         # 编译 StateGraph
  runtime.py                       # 单节点 tool 循环（上限 2）
  schemas.py                       # JobRequirement / DimensionScore / Decision
  jd_prompts.py
  jd_tools.py                      # read_and_split_jd
  jd_splitter.py                   # JD 标题切分（确定性）
  resume_prompts.py
  resume_tools.py                  # parse_resume_file
  evaluate_prompts.py
  evaluate_tools.py                # 五个 score_* tool
  decide_prompts.py
  decide_tools.py                  # average_and_label
  report_prompts.py
  report_tools.py                  # render_report
  nodes.py
tests/fixtures/xagent_jd.md
tests/fixtures/xagent_jd.gold.json
tests/eval_jd_parse.py             # 需求解析 live 评估
tests/eval_matching_pipeline.py    # 主流程 live 评估（报告形态）
tests/test_jd_splitter.py
tests/test_decision.py
tests/test_matching_graph.py
tests/test_runtime_tool_limit.py
```

包名保持 `job_agent`。模型配置复用 `job_agent.config.load_llm_config()`（`OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`）。依赖增加 `langgraph`（实现前用 Context7 锁定导入路径与编译 API），不换供应商 SDK。

## 数据模型

### JobRequirement

需求解析专家的输出契约。字段名英文；标量缺省 `null`，数组缺省 `[]`。

```text
JobRequirement
  title: str | null
  must_have_skills: list[str]
  nice_to_have_skills: list[str]
  years_required: str | null
  education_required: str | null
  location: str | null
  responsibilities: list[str]
```

- `title`：有岗位名称用原文；没有则从职位介绍概括成短标题，不编造公司名。
- `must_have_skills`：只来自「岗位要求」里的技能/能力，短标签，小写；年限、学历、专业不进此列表。
- `nice_to_have_skills`：只来自「加分项」；无该区块则为 `[]`。
- `years_required` / `education_required`：保留原文措辞，不改写成纯数字。
- `location`：原文没有则 `null`，不默认「远程」或「不限」。样例 JD 无地点，gold 为 `null`。
- `responsibilities`：岗位职责一条一项，不把职位介绍整段复制进去。

年限/学历的数字比较放在评估 tool 里做，不进入本 schema。

### DimensionScore

```text
DimensionScore
  dimension: "skills" | "years" | "education" | "location" | "responsibilities"
  score: int            # 仅 1、2、3、4、5
  evidence: str         # 一句中文证据，不得为空
```

### Decision

```text
Decision
  average: float        # 五维算术平均，展示保留 1 位小数
  recommendation: "推荐" | "待定" | "不推荐"
```

比较用精确平均值 `(s1+s2+s3+s4+s5)/5`：≥4.0 推荐；≥3.0 且 <4.0 待定；<3.0 不推荐。

### 图状态

```text
MatchingState
  jd_path: str
  resume_path: str
  job_requirement: JobRequirement | null
  resume: Resume | null                 # 第 1 章模型
  dimension_scores: dict[str, DimensionScore]   # 按 dimension 合并
  decision: Decision | null
  report: str | null
```

节点只写自己负责的字段。

## 需求解析专家（本章做细）

### JD 切分

确定性 Python，由 `read_and_split_jd` 内部调用，不单独占一次 tool call。

- 按 Markdown ATX 识别区块：`#` 或 `##` 标题含下列关键词则映射（先匹配更具体的）：

| 标题含 | section id |
| --- | --- |
| 加分 | `nice_to_have` |
| 岗位要求、任职要求、职位要求 | `requirements` |
| 岗位职责、工作职责 | `responsibilities` |
| 职位介绍、岗位介绍、职位描述 | `intro` |
| （其余） | `other` |

- `#职位介绍` 这种无空格标题视为有效。
- 切不出上述标题时，全文作为 `other` 交给模型，不直接失败。
- 同 id 多次出现则按出现顺序拼接。

### 工具

- `read_and_split_jd(path: str) -> str`：校验文件存在且后缀为 `.md` 或 `.txt`（大小写不敏感）；读取 UTF-8；返回已切分、带区块标题的文本。

本节点典型 1 次 tool call 后结构化输出 `JobRequirement`。第 2 次配额仅用于补读；两次仍无合法结构则节点失败。

### Prompt

`ChatPromptTemplate`：System + Human。输出用 LangChain 针对 Pydantic 的结构化输出（实现前 Context7 核对当前 API；第 1 章对 DeepSeek 实际使用 `method="json_mode"`，本章沿用同一策略，禁止手写 JSON 正则）。

System：中文岗位需求解析专家。只根据工具读到的 JD 原文填 schema；不编造；不把简历信息写入岗位；明显 OCR/笔误可在技能标签里归一（如将 AlAgent 理解为 AI Agent），但不新增原文没有的技术栈；只使用本节点 tool list；最多 2 次 tool call；读完后必须给出完整 `JobRequirement`，不得只回复散文。

Human：给出 `jd_path`，要求先调用 `read_and_split_jd`，再按切开的区块填写。

### 样例与 gold 硬核对

输入：`tests/fixtures/xagent_jd.md`（用户提供的 Xagent / Agent 平台 JD 原文，可入库）。

硬性：

- 七个字段都在
- `location is null`
- `must_have_skills` 含 `python`（小写比较）
- `nice_to_have_skills` 含 `langchain`
- `years_required` 含「3」
- `education_required` 含「本科」
- `responsibilities` 至少 3 条，且能对上编排 / Runtime / 记忆 中至少两项（子串，大小写不敏感）

软性：`title` 能看出 Agent / Xagent / 核心框架；职责措辞允许与 gold 不完全逐字相同。

## 其余专家（做薄）

简历提取、五维评估、决策、报告都有独立 System Prompt 与 tool list，但逻辑从简。

### 简历提取专家

- Tool：`parse_resume_file(path: str) -> Resume` JSON 字符串，内部调用第 1 章 `parse_resume`。
- 失败不 mock，节点失败。
- 代码注释标明：日后可改为从向量库按 ID 读取结构化简历，本章不实现。
- Prompt：要求调用该 tool，将结果作为本节点的 `Resume` 写出；不要改写履历事实。

### 五维评估

Python 把该维所需切片注入 Human Prompt（JD + 简历相关字段）。每个节点一个 tool，模型调用后写出 `DimensionScore`。

简单规则（本章固定，不调参）：

| 维 | 规则 |
| --- | --- |
| skills | `must_have_skills` 与 `resume.skills` 小写精确或子串命中的比例。0%→1，(0,0.25]→2，(0.25,0.5]→3，(0.5,0.9)→4，≥0.9→5。`must_have_skills` 为空→3 |
| years | 从 `years_required` 取第一个整数 N。工作年限 Y = 所有 `work_experience` 中最早 `start_date` 到最晚 `end_date` 的月份差 / 12（`present` 视为运行时当前年月）。缺 N 或无可解析日期→3。Y≥N+2→5，Y≥N→4，Y≥N-1→3，Y≥N-3→2，否则 1 |
| education | 只看 JD 最高学历词：含「博士」则简历学位含博士→5，否则 2；否则含「硕士」则博士→5、硕士→4、本科→3、其余 2；否则含「本科」则本科/硕士/博士→4，其余 2。JD 无学历词或简历 `education` 为空→3 |
| location | 两侧都空→3。任一侧子串包含另一侧（去空白）→5。否则 2 |
| responsibilities | 从每条 JD 职责取出长度≥2 的英文单词（按非字母切）和连续汉字串，与工作/项目 `responsibilities`+`achievements` 拼接文本做不区分大小写子串匹配，命中比例分档同 skills。JD 职责为空→3 |

证据必须引用规则结果（例如「必须技能 4/6 命中」），禁止编造简历里没有的公司或项目。

`score` 若模型改写，节点以 tool 返回值为准再写入状态（模型不得把 1–5 改成 0–100）。

### 决策专家

- Tool：`average_and_label(scores: list[int]) -> Decision`
- 必须用五维分数调用该 tool，以 tool 结果写入状态，模型不自己算平均。

### 报告生成专家

- Tool：`render_report(...)` 生成 Markdown，至少包含：岗位 `title`、候选人姓名、五维（维名、分数、证据）、平均分（1 位小数）、结论（推荐/待定/不推荐）。
- stdout 打印该报告全文。

## 单节点运行时

`matching/runtime.py` 封装，所有专家节点共用：

1. 将该角色 tools 用 LangChain `@tool` 定义并 bind 到聊天模型。
2. 循环：模型若返回 tool_calls，则执行并回填；计数 +1。
3. 计数达到 2 仍存在未完成的 tool_calls：停止调用工具，强制进入结构化输出；若仍失败则节点报错。
4. 禁止使用无限 recursion 的 ReAct Agent。实现前用 Context7 查 LangGraph / LangChain 绑定 tools 与 `with_structured_output` 的当前写法；LLM 调用按仓库规范走 async（CLI 用 `asyncio.run` 包一层）。

第 1 章提取是同步 `chain.invoke`。本章图节点用 async；`parse_resume` 仍同步，在简历 tool 里直接调用即可，不在本章重写成 async。

## 错误处理

| 情况 | 行为 |
| --- | --- |
| CLI 参数不是两个路径 | 中文用法说明，退出码 2 |
| JD 或简历路径不存在、不是文件 | 退出码 2 |
| JD 后缀不是 `.md` / `.txt` | 退出码 2 |
| 简历后缀不是 `.md` | 退出码 2（沿用第 1 章） |
| 缺 `OPENAI_*` | 退出码 2 |
| 简历解析失败、JD 提取失败、超 2 次 tool 仍无结构、schema 校验失败、模型调用失败 | 退出码 1，stderr 打印原因，不输出半截报告 |
| 切分无标准标题 | 全文进 `other`，不直接失败 |

## 验证

确定性单测（TDD）：

- JD splitter：标题映射、无标题则 `other`、`#职位介绍` 无空格可切
- 决策阈值：平均 4.0 / 3.0 / 2.9 分别推荐 / 待定 / 不推荐
- 五维规则：空技能→3；地点双空→3；全命中技能→5
- 报告渲染含五维与结论
- 图编译后节点集合包含 `jd_parse`、`resume_extract`、五个 `eval_*`、`decide`、`report`
- runtime：第 3 次 tool call 不会发出

Prompt / 需求解析 / 主流程：不写假 LLM 单测。评估集：

- `tests/eval_jd_parse.py`：样例 JD + gold，标记 `@pytest.mark.live`
- `tests/eval_matching_pipeline.py`：跑全图，断言报告含「推荐」或「待定」或「不推荐」以及五维名称；简历路径默认 `RESUME_SAMPLE_PATH`（与第 1 章相同）。缺配置时 live 失败而非 skip

演示命令：

```bash
python -m job_agent.match tests/fixtures/xagent_jd.md "/Users/yannic/面试/宗艳云简历v9.md"
```

退出码 0，stdout 为 Markdown 报告。

## 依赖

- 现有：Python 3.11+、langchain / langchain-core / langchain-openai、pydantic v2、python-dotenv
- 新增：`langgraph`（版本以实现时 Context7 + pyproject 锁定为准）
- `@tool` 从 LangChain 官方导出路径导入（Context7 核对，不凭记忆）

应用侧仍只使用 OpenAI 协议客户端。实现时官方类名或参数与记忆不符，停下来对照 Context7，不自行换 SDK。
