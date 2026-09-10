from langchain_openai import ChatOpenAI

from job_agent.config import LLMConfig
from job_agent.resume.errors import ResumeExtractError
from job_agent.resume.prompts import SECTION_KEYS, build_extract_prompt, escape_braces
from job_agent.resume.schema import Resume


def extract_resume(sections: dict[str, str], config: LLMConfig) -> Resume:
    prompt = build_extract_prompt()
    llm = ChatOpenAI(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=0,
        max_tokens=8192,
        timeout=180,
    )
    chain = prompt | llm.with_structured_output(Resume, method="json_mode")
    payload = {
        key: escape_braces(sections[key]) if key in sections else "（无）"
        for key in SECTION_KEYS
    }
    try:
        result = chain.invoke(payload)
    except Exception as exc:  # noqa: BLE001
        raise ResumeExtractError(f"简历提取失败：{exc}") from exc
    if not isinstance(result, Resume):
        raise ResumeExtractError("模型未返回 Resume 结构")
    return result
