from dataclasses import dataclass, field
import os

from dotenv import load_dotenv

from job_agent.kb.errors import KbConfigError
from job_agent.resume.errors import ResumeConfigError


@dataclass(frozen=True)
class LLMConfig:
    api_key: str = field(repr=False)
    base_url: str
    model: str


@dataclass(frozen=True)
class KbConfig:
    """知识库持久化与向量模型配置。"""

    mysql_host: str
    mysql_port: int
    mysql_db: str
    mysql_user: str
    mysql_password: str = field(repr=False)
    milvus_uri: str
    milvus_token: str = field(repr=False)
    bge_m3_model: str


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


def load_kb_config() -> KbConfig:
    """从环境变量加载知识库配置。"""
    load_dotenv()
    required_keys = (
        "MYSQL_HOST",
        "MYSQL_PORT",
        "MYSQL_DB",
        "MYSQL_USER",
        "MYSQL_PASSWORD",
        "MILVUS_URI",
        "BGE_M3_MODEL",
    )
    missing: list[str] = []
    values: dict[str, str] = {}
    for key in required_keys:
        value = (os.getenv(key) or "").strip()
        if not value:
            missing.append(key)
        else:
            values[key] = value
    if missing:
        raise KbConfigError("缺少环境变量：" + "、".join(missing))

    try:
        mysql_port = int(values["MYSQL_PORT"])
    except ValueError as error:
        raise KbConfigError("环境变量 MYSQL_PORT 必须为整数") from error

    return KbConfig(
        mysql_host=values["MYSQL_HOST"],
        mysql_port=mysql_port,
        mysql_db=values["MYSQL_DB"],
        mysql_user=values["MYSQL_USER"],
        mysql_password=values["MYSQL_PASSWORD"],
        milvus_uri=values["MILVUS_URI"],
        milvus_token=(os.getenv("MILVUS_TOKEN") or "").strip(),
        bge_m3_model=values["BGE_M3_MODEL"],
    )
