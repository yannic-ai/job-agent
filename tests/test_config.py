import pytest

from job_agent.config import load_llm_config
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
