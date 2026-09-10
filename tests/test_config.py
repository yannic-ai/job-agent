import pytest

from job_agent.config import load_kb_config, load_llm_config
from job_agent.kb.errors import KbConfigError
from job_agent.resume.errors import ResumeConfigError


def test_load_llm_config_requires_all_vars(monkeypatch):
    monkeypatch.setattr("job_agent.config.load_dotenv", lambda: False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with pytest.raises(ResumeConfigError, match="OPENAI_API_KEY"):
        load_llm_config()


def test_load_llm_config_reads_env(monkeypatch):
    monkeypatch.setattr("job_agent.config.load_dotenv", lambda: False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "deepseek-chat")
    cfg = load_llm_config()
    assert cfg.api_key == "sk-test"
    assert cfg.base_url == "https://api.example.com/v1"
    assert cfg.model == "deepseek-chat"


def test_load_kb_config_requires_mysql_host(monkeypatch):
    monkeypatch.setattr("job_agent.config.load_dotenv", lambda: False)
    monkeypatch.delenv("MYSQL_HOST", raising=False)
    monkeypatch.setenv("MYSQL_PORT", "3306")
    monkeypatch.setenv("MYSQL_DB", "job_agent")
    monkeypatch.setenv("MYSQL_USER", "tester")
    monkeypatch.setenv("MYSQL_PASSWORD", "secret")
    monkeypatch.setenv("MILVUS_URI", "http://localhost:19530")
    monkeypatch.setenv("MILVUS_TOKEN", "")
    monkeypatch.setenv("BGE_M3_MODEL", "BAAI/bge-m3")

    with pytest.raises(KbConfigError, match="缺少环境变量.*MYSQL_HOST"):
        load_kb_config()


def test_load_kb_config_reads_env_and_allows_empty_token(monkeypatch):
    monkeypatch.setattr("job_agent.config.load_dotenv", lambda: False)
    monkeypatch.setenv("MYSQL_HOST", "localhost")
    monkeypatch.setenv("MYSQL_PORT", "3306")
    monkeypatch.setenv("MYSQL_DB", "job_agent")
    monkeypatch.setenv("MYSQL_USER", "tester")
    monkeypatch.setenv("MYSQL_PASSWORD", "secret")
    monkeypatch.setenv("MILVUS_URI", "http://localhost:19530")
    monkeypatch.setenv("MILVUS_TOKEN", "")
    monkeypatch.setenv("BGE_M3_MODEL", "BAAI/bge-m3")

    cfg = load_kb_config()

    assert cfg.mysql_host == "localhost"
    assert cfg.mysql_port == 3306
    assert cfg.mysql_db == "job_agent"
    assert cfg.mysql_user == "tester"
    assert cfg.mysql_password == "secret"
    assert cfg.milvus_uri == "http://localhost:19530"
    assert cfg.milvus_token == ""
    assert cfg.bge_m3_model == "BAAI/bge-m3"
