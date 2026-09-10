from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from job_agent.matching.pipeline import run_matching

DEFAULT_RESUME_SAMPLE_PATH = "/Users/yannic/面试/宗艳云简历v9.md"


def _resume_sample_path() -> Path:
    return Path(os.environ.get("RESUME_SAMPLE_PATH", DEFAULT_RESUME_SAMPLE_PATH))


@pytest.mark.live
def test_matching_pipeline_live_report() -> None:
    jd_path = Path(__file__).parent / "fixtures" / "xagent_jd.md"
    resume_path = _resume_sample_path()
    if not resume_path.is_file():
        pytest.skip(f"sample resume not found: {resume_path}")

    result = asyncio.run(run_matching(str(jd_path), str(resume_path)))
    report = str(result["report"])

    for keyword in ("技能", "年限", "学历", "地点", "职责"):
        assert keyword in report
    assert any(label in report for label in ("推荐", "待定", "不推荐"))
