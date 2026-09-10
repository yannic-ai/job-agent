from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """你是中文岗位需求解析专家。只根据岗位描述原文提取信息，必须输出完整 JobRequirement。
规则：
- 不编造，原文没有的信息填 null 或空数组。
- 不写简历信息，不要输出候选人经历、能力判断或人选建议。
- 地点没有则 null。
- 技能短标签小写。
- 年限学历保留原文。
- 最多 2 次 tool call。
"""

HUMAN_PROMPT = "JD 文件路径：{jd_path}，要求先调 read_and_split_jd。"


def build_jd_prompt() -> ChatPromptTemplate:
    """Build the JD parsing expert prompt template."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    )
