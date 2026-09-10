from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """你是中文匹配决策专家。你只根据五维评分做最终决策。
规则：
- 必须调用 average_and_label_tool，不能自己心算平均分。
- 不新增额外维度，不改写已有分数。
- 输出必须符合 Decision 结构。
"""

HUMAN_PROMPT = """请根据以下五维分数做决策。
分数 JSON：
{scores_json}
"""


def build_decide_prompt() -> ChatPromptTemplate:
    """Build the decision expert prompt template."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    )
