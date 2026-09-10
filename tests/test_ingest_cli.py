from __future__ import annotations

from pathlib import Path

import pytest

from job_agent.kb.errors import KbConfigError, KbStoreError
from job_agent.resume.errors import (
    ResumeConfigError,
    ResumeExtractError,
    ResumeFileError,
)


def test_main_requires_one_path(capsys: pytest.CaptureFixture[str]):
    from job_agent.ingest import main

    assert main([]) == 2
    captured = capsys.readouterr()
    assert "用法：python -m job_agent.ingest <resume.md>" in captured.err


def test_main_prints_resume_id_and_returns_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
):
    from job_agent.ingest import main

    resume_path = tmp_path / "resume.md"
    resume_path.write_text("# Resume\n", encoding="utf-8")

    async def fake_ingest_resume(path: str) -> int:
        assert path == str(resume_path)
        return 12

    monkeypatch.setattr("job_agent.ingest.ingest_resume", fake_ingest_resume)

    assert main([str(resume_path)]) == 0
    captured = capsys.readouterr()
    assert "resume_id=12" in captured.out


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (ResumeFileError("简历文件失败"), 2),
        (ResumeConfigError("缺少环境变量"), 2),
        (KbConfigError("缺少环境变量"), 2),
        (ResumeExtractError("简历解析失败"), 1),
        (KbStoreError("入库失败"), 1),
    ],
)
def test_main_maps_ingest_errors_to_exit_codes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    error: Exception,
    expected_code: int,
):
    from job_agent.ingest import main

    resume_path = tmp_path / "resume.md"
    resume_path.write_text("# Resume\n", encoding="utf-8")

    async def fake_ingest_resume(_: str) -> int:
        raise error

    monkeypatch.setattr("job_agent.ingest.ingest_resume", fake_ingest_resume)

    assert main([str(resume_path)]) == expected_code
    captured = capsys.readouterr()
    assert str(error) in captured.err
