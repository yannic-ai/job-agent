from pathlib import Path

import pytest

from job_agent.resume.errors import ResumeFileError
from job_agent.resume.loader import load_markdown


def test_load_markdown_reads_utf8(tmp_path: Path):
    path = tmp_path / "resume.md"
    path.write_text("# 张三\n", encoding="utf-8")
    assert load_markdown(path) == "# 张三\n"


def test_load_markdown_missing_file(tmp_path: Path):
    with pytest.raises(ResumeFileError, match="不存在"):
        load_markdown(tmp_path / "nope.md")


def test_load_markdown_rejects_non_md(tmp_path: Path):
    path = tmp_path / "resume.txt"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ResumeFileError, match="Markdown"):
        load_markdown(path)


def test_load_markdown_accepts_uppercase_md(tmp_path: Path):
    path = tmp_path / "resume.MD"
    path.write_text("# A\n", encoding="utf-8")
    assert "# A" in load_markdown(path)
