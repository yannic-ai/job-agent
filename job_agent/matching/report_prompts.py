from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """你是中文匹配报告专家。你只负责把已有评分与结论整理成最终报告。
规则：
- 必须调用 render_report_tool，不能手写一份不同结论的报告。
- 不修改已有维度分数与决策。
- 输出必须包含 report 字段。
"""

HUMAN_PROMPT = """请生成最终报告。
岗位标题：{title}
候选人姓名：{candidate_name}
维度分数 JSON：
{scores_json}
决策 JSON：
{decision_json}
"""


def build_report_prompt() -> ChatPromptTemplate:
    """Build the report expert prompt template."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    )
