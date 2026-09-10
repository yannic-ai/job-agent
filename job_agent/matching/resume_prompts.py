from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """你是中文简历档案读取专家。你只根据工具结果整理结构化 ResumeProfile。
规则：
- 必须调用唯一提供的简历工具读取 ResumeProfile JSON，且只调用一次。
- 不编造原文没有的信息。
- 不要改写履历事实。
- 输出字段必须符合 ResumeProfile 结构。
- 只做简历解析，不做岗位匹配、打分或录用建议。
"""

FILE_HUMAN_PROMPT = "简历文件路径：{resume_ref}"
ID_HUMAN_PROMPT = "简历ID：{resume_ref}"


def build_resume_prompt(*, is_id: bool) -> ChatPromptTemplate:
    """Build the resume extraction expert prompt template."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", ID_HUMAN_PROMPT if is_id else FILE_HUMAN_PROMPT),
        ]
    )
