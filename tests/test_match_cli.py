from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from job_agent.kb.errors import KbConfigError, KbNotFoundError, KbStoreError
from job_agent.matching.errors import MatchingExtractError, MatchingFileError
from job_agent.resume.errors import (
    ResumeConfigError,
    ResumeExtractError,
    ResumeFileError,
)


def test_run_matching_invokes_graph_with_expected_state(monkeypatch: pytest.MonkeyPatch):
    from job_agent.matching.pipeline import run_matching

    captured: dict[str, object] = {}

    class FakeGraph:
        async def ainvoke(self, state: dict[str, object]) -> dict[str, object]:
            captured["state"] = state
            return {"report": "ok"}

    def fake_build_matching_graph() -> FakeGraph:
        return FakeGraph()

    monkeypatch.setattr(
        "job_agent.matching.pipeline.build_matching_graph",
        fake_build_matching_graph,
    )

    result = asyncio.run(run_matching("job.md", "resume.md"))

    assert result == {"report": "ok"}
    assert captured["state"] == {
        "jd_path": "job.md",
        "resume_ref": "resume.md",
        "dimension_scores": {},
    }


def test_main_requires_two_paths(capsys: pytest.CaptureFixture[str]):
    from job_agent.match import main

    assert main([]) == 2
    captured = capsys.readouterr()
    assert (
        "用法：python -m job_agent.match <jd.md> <resume.md|resume_id>"
        in captured.err
    )


def test_main_passes_numeric_resume_id_without_loading_file(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    from job_agent.match import main

    jd_path = tmp_path / "jd.md"
    jd_path.write_text("# JD\n", encoding="utf-8")
    resume_loads: list[str] = []

    async def fake_run_matching(jd_path_arg: str, resume_ref_arg: str) -> dict[str, object]:
        assert jd_path_arg == str(jd_path)
        assert resume_ref_arg == "12"
        return {"report": "匹配报告"}

    monkeypatch.setattr("job_agent.match.run_matching", fake_run_matching)
    monkeypatch.setattr(
        "job_agent.match.load_markdown",
        lambda path: resume_loads.append(path),
    )

    assert main([str(jd_path), "12"]) == 0
    assert resume_loads == []
    assert capsys.readouterr().out == "匹配报告\n"


def test_main_prints_report_and_returns_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
):
    from job_agent.match import main

    jd_path = tmp_path / "jd.md"
    resume_path = tmp_path / "resume.md"
    jd_path.write_text("# JD\n", encoding="utf-8")
    resume_path.write_text("# Resume\n", encoding="utf-8")

    async def fake_run_matching(jd_path_arg: str, resume_path_arg: str) -> dict[str, object]:
        assert jd_path_arg == str(jd_path)
        assert resume_path_arg == str(resume_path)
        return {"report": "匹配报告"}

    monkeypatch.setattr("job_agent.match.run_matching", fake_run_matching)

    assert main([str(jd_path), str(resume_path)]) == 0
    captured = capsys.readouterr()
    assert captured.out == "匹配报告\n"
    assert captured.err == ""


@pytest.mark.parametrize(
    ("argv", "error_text"),
    [
        (["missing.md", "resume.md"], "不存在"),
        (["job.md", "resume.txt"], "Markdown"),
    ],
)
def test_main_maps_file_validation_errors_to_exit_code_two(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    argv: list[str],
    error_text: str,
):
    from job_agent.match import main

    jd_path = tmp_path / "job.md"
    resume_path = tmp_path / "resume.md"
    wrong_resume_path = tmp_path / "resume.txt"
    jd_path.write_text("# JD\n", encoding="utf-8")
    resume_path.write_text("# Resume\n", encoding="utf-8")
    wrong_resume_path.write_text("# Resume\n", encoding="utf-8")

    resolved_argv = [
        str(jd_path)
        if item == "job.md"
        else str(resume_path)
        if item == "resume.md"
        else str(wrong_resume_path)
        if item == "resume.txt"
        else str(tmp_path / item)
        for item in argv
    ]

    assert main(resolved_argv) == 2
    captured = capsys.readouterr()
    assert error_text in captured.err


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (MatchingFileError("岗位文件失败"), 2),
        (ResumeFileError("简历文件失败"), 2),
        (ResumeConfigError("缺少环境变量"), 2),
        (KbConfigError("知识库配置失败"), 2),
        (KbNotFoundError("简历不存在"), 2),
        (KbStoreError("简历未向量化"), 1),
        (MatchingExtractError("匹配失败"), 1),
        (ResumeExtractError("简历失败"), 1),
    ],
)
def test_main_maps_matching_errors_to_exit_codes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    error: Exception,
    expected_code: int,
):
    from job_agent.match import main

    jd_path = tmp_path / "jd.md"
    resume_path = tmp_path / "resume.md"
    jd_path.write_text("# JD\n", encoding="utf-8")
    resume_path.write_text("# Resume\n", encoding="utf-8")

    async def fake_run_matching(_: str, __: str) -> dict[str, object]:
        raise error

    monkeypatch.setattr("job_agent.match.run_matching", fake_run_matching)

    assert main([str(jd_path), str(resume_path)]) == expected_code
    captured = capsys.readouterr()
    assert str(error) in captured.err


def test_main_returns_one_when_report_is_empty(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
):
    from job_agent.match import main

    jd_path = tmp_path / "jd.md"
    resume_path = tmp_path / "resume.md"
    jd_path.write_text("# JD\n", encoding="utf-8")
    resume_path.write_text("# Resume\n", encoding="utf-8")

    async def fake_run_matching(_: str, __: str) -> dict[str, object]:
        return {"report": ""}

    monkeypatch.setattr("job_agent.match.run_matching", fake_run_matching)

    assert main([str(jd_path), str(resume_path)]) == 1
    captured = capsys.readouterr()
    assert "报告为空" in captured.err
