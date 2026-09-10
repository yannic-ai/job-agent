from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """你是中文简历解析专家。你只根据简历原文和工具结果整理结构化 Resume。
规则：
- 必须调用 parse_resume_file 读取标准化简历 JSON，且只调用一次。
- 不编造原文没有的信息。
- 不要改写履历事实。
- 输出字段必须符合 Resume 结构。
- 只做简历解析，不做岗位匹配、打分或录用建议。
"""

HUMAN_PROMPT = "简历文件路径：{resume_path}"


def build_resume_prompt() -> ChatPromptTemplate:
    """Build the resume extraction expert prompt template."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    )
