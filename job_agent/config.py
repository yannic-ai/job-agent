from dataclasses import dataclass, field
import os

from dotenv import load_dotenv

from job_agent.resume.errors import ResumeConfigError


@dataclass(frozen=True)
class LLMConfig:
    api_key: str = field(repr=False)
    base_url: str
    model: str


def load_llm_config() -> LLMConfig:
    load_dotenv()
    missing: list[str] = []
    values: dict[str, str] = {}
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        value = (os.getenv(key) or "").strip()
        if not value:
            missing.append(key)
        else:
            values[key] = value
    if missing:
        raise ResumeConfigError("缺少环境变量：" + "、".join(missing))
    return LLMConfig(
        api_key=values["OPENAI_API_KEY"],
        base_url=values["OPENAI_BASE_URL"],
        model=values["OPENAI_MODEL"],
    )
