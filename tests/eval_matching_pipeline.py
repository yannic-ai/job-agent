from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from job_agent.kb.pipeline import ingest_resume
from job_agent.matching.pipeline import run_matching

DEFAULT_RESUME_SAMPLE_PATH = "/Users/yannic/面试/宗艳云简历v9.md"
JD_PATH = Path(__file__).parent / "fixtures" / "xagent_jd.md"


def _resume_sample_path() -> Path:
    return Path(os.environ.get("RESUME_SAMPLE_PATH", DEFAULT_RESUME_SAMPLE_PATH))


def _assert_report(report: str) -> None:
    for keyword in ("技能", "年限", "学历", "地点", "职责"):
        assert keyword in report
    assert any(label in report for label in ("推荐", "待定", "不推荐"))


@pytest.mark.live
def test_matching_pipeline_live_report_by_file() -> None:
    resume_path = _resume_sample_path()
    if not resume_path.is_file():
        pytest.skip(f"sample resume not found: {resume_path}")

    result = asyncio.run(run_matching(str(JD_PATH), str(resume_path)))
    _assert_report(str(result["report"]))


@pytest.mark.live
def test_matching_pipeline_live_report_by_id() -> None:
    resume_path = _resume_sample_path()
    if not resume_path.is_file():
        pytest.skip(f"sample resume not found: {resume_path}")

    async def _run() -> None:
        resume_id = await ingest_resume(str(resume_path))
        result = await run_matching(str(JD_PATH), str(resume_id))
        _assert_report(str(result["report"]))

    asyncio.run(_run())
