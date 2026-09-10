from __future__ import annotations

import asyncio
import sys

from job_agent.matching.errors import MatchingExtractError, MatchingFileError
from job_agent.matching.jd_loader import load_jd
from job_agent.matching.pipeline import run_matching
from job_agent.resume.errors import (
    ResumeConfigError,
    ResumeExtractError,
    ResumeFileError,
)
from job_agent.resume.loader import load_markdown

USAGE = "用法：python -m job_agent.match <jd.md> <resume.md>"


def main(argv: list[str] | None = None) -> int:
    """Run the matching CLI and return the process exit code."""

    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print(USAGE, file=sys.stderr)
        return 2

    jd_path, resume_path = args
    try:
        load_jd(jd_path)
        load_markdown(resume_path)
        state = asyncio.run(run_matching(jd_path, resume_path))
    except (MatchingFileError, ResumeFileError, ResumeConfigError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (MatchingExtractError, ResumeExtractError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"匹配流程失败：{exc}", file=sys.stderr)
        return 1

    report = state.get("report")
    if not report:
        print("报告为空", file=sys.stderr)
        return 1

    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
