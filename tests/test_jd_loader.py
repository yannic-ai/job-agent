from pathlib import Path

import pytest

from job_agent.matching.errors import MatchingFileError
from job_agent.matching.jd_loader import load_jd


def test_load_jd_rejects_missing(tmp_path: Path):
    with pytest.raises(MatchingFileError):
        load_jd(tmp_path / "nope.md")


def test_load_jd_rejects_pdf(tmp_path: Path):
    path = tmp_path / "a.pdf"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(MatchingFileError):
        load_jd(path)


def test_load_jd_reads_md(tmp_path: Path):
    path = tmp_path / "jd.md"
    path.write_text("#职位介绍\nhello", encoding="utf-8")
    assert "hello" in load_jd(path)


def test_load_jd_reads_txt(tmp_path: Path):
    path = tmp_path / "jd.txt"
    path.write_text("#职位介绍\nhello", encoding="utf-8")
    assert "hello" in load_jd(path)


def test_load_jd_rejects_invalid_utf8(tmp_path: Path):
    path = tmp_path / "bad.md"
    path.write_bytes(b"\xff\xfe invalid")
    with pytest.raises(MatchingFileError, match="UTF-8"):
        load_jd(path)
