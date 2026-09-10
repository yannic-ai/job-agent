from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """你是中文简历评估专家。你每次只评估一个维度，并且必须优先引用工具打分结果。
规则：
- 只评估当前维度，不扩展到其他维度。
- 必须先调用对应评分 tool。
- 调用评分 tool 时，必须严格使用下方“工具参数”指定的参数名和值。
- JSON 参数必须逐字复制对应的完整 JSON 字符串，不能截断、不能改写、不能手动转述。
- 证据必须基于给定 JD/简历切片与工具结果。
- 不编造候选人没有写过的公司、项目或经历。
- 输出必须是 DimensionScore 结构。
- DimensionScore.dimension 必须输出英文：skills / years / education / location / responsibilities，不要输出中文维名。
"""

HUMAN_PROMPT = """当前评估维度：{dimension_label}
工具参数：
{tool_arguments}

JD 切片 JSON：
```json
{job_requirement_json}
```

简历/档案切片 JSON：
```json
{resume_payload_json}
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
