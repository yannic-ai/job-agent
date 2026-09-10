from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """你是中文简历评估专家。你每次只评估一个维度，并且必须优先引用工具打分结果。
规则：
- 只评估当前维度，不扩展到其他维度。
- 必须先调用对应评分 tool。
- 调用评分 tool 时，`job_json` 和 `resume_json` 必须逐字复制下方提供的完整 JSON 字符串，不能截断、不能改写、不能手动转述。
- 证据必须基于给定 JD/简历切片与工具结果。
- 不编造候选人没有写过的公司、项目或经历。
- 输出必须是 DimensionScore 结构。
- DimensionScore.dimension 必须输出英文：skills / years / education / location / responsibilities，不要输出中文维名。
"""

HUMAN_PROMPT = """当前评估维度：{dimension_label}
调用 tool 时请直接复制下面两段 JSON 的原文内容作为参数值。

JD 切片 JSON：
```json
{job_requirement_json}
```

简历切片 JSON：
```json
{resume_json}
```
"""


def build_evaluate_prompt() -> ChatPromptTemplate:
    """Build the evaluation expert prompt template."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    )
