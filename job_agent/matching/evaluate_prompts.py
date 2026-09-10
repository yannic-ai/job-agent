from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """你是中文简历评估专家。你每次只评估一个维度，并且必须优先引用工具打分结果。
规则：
- 只评估当前维度，不扩展到其他维度。
- 必须先调用对应评分 tool。
- 证据必须基于给定 JD/简历切片与工具结果。
- 不编造候选人没有写过的公司、项目或经历。
- 输出必须是 DimensionScore 结构。
"""

HUMAN_PROMPT = """当前评估维度：{dimension_label}
JD 切片 JSON：
{job_requirement_json}

简历切片 JSON：
{resume_json}
"""


def build_evaluate_prompt() -> ChatPromptTemplate:
    """Build the evaluation expert prompt template."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    )
