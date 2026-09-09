from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from job_agent.resume.errors import ResumeConfigError
from job_agent.resume.pipeline import parse_resume

DEFAULT_SAMPLE = "/Users/yannic/面试/宗艳云简历v9.md"
DATE_RE = re.compile(r"^(\d{4}-\d{2}|present)$")
REQUIRED_KEYS = {
    "personal_info",
    "education",
    "work_experience",
    "projects",
    "skills",
    "summary",
}


def _sample_path() -> Path:
    return Path(os.environ.get("RESUME_SAMPLE_PATH", DEFAULT_SAMPLE))


def _walk_dates(obj, key: str | None = None) -> None:
    if isinstance(obj, dict):
        for child_key, child in obj.items():
            _walk_dates(child, child_key)
        return
    if isinstance(obj, list):
        for child in obj:
            _walk_dates(child, key)
        return
    if key in {"start_date", "end_date"} and obj is not None:
        assert DATE_RE.match(str(obj)), f"bad date {key}={obj}"


def test_parse_zong_yanyun_resume():
    path = _sample_path()
    if not path.is_file():
        pytest.skip(f"sample resume not found: {path}")
    try:
        resume = parse_resume(path)
    except ResumeConfigError as exc:
        pytest.skip(str(exc))
    dumped = resume.model_dump()
    assert REQUIRED_KEYS <= dumped.keys()
    assert dumped["personal_info"]["name"] == "宗艳云"
    companies = " ".join(item.get("company") or "" for item in dumped["work_experience"])
    assert "丁香园" in companies
    assert "小黄柜" in companies
    assert "恒生" in companies
    assert len(dumped["work_experience"]) >= 3
    names = " ".join(item.get("name") or "" for item in dumped["projects"])
    assert "医考智能客服" in names
    assert "题库" in names
    assert "公开课" in names
    _walk_dates(dumped)
    gold_path = Path(__file__).parent / "fixtures" / "zong_yanyun.gold.json"
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    print("gold name", gold["personal_info"]["name"], "got", dumped["personal_info"]["name"])
