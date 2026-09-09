from pathlib import Path

from job_agent.resume.errors import ResumeFileError


def load_markdown(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise ResumeFileError(f"文件不存在：{file_path}")
    if not file_path.is_file():
        raise ResumeFileError(f"不是文件：{file_path}")
    if file_path.suffix.lower() != ".md":
        raise ResumeFileError(f"仅支持 Markdown（.md）文件：{file_path}")
    return file_path.read_text(encoding="utf-8")
