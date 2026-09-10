from __future__ import annotations

import asyncio
import sys

from job_agent.kb.errors import KbConfigError, KbNotFoundError, KbStoreError
from job_agent.matching.errors import MatchingExtractError, MatchingFileError
from job_agent.matching.jd_loader import load_jd
from job_agent.matching.pipeline import run_matching
from job_agent.matching.resume_tools import is_resume_id_ref
from job_agent.resume.errors import (
    ResumeConfigError,
    ResumeExtractError,
    ResumeFileError,
)
from job_agent.resume.loader import load_markdown

USAGE = "用法：python -m job_agent.match <jd.md> <resume.md|resume_id>"


def main(argv: list[str] | None = None) -> int:
    """Run the matching CLI and return the process exit code."""

    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print(USAGE, file=sys.stderr)
        return 2

    jd_path, resume_ref = args
    try:
        load_jd(jd_path)
        if not is_resume_id_ref(resume_ref):
            load_markdown(resume_ref)
        state = asyncio.run(run_matching(jd_path, resume_ref))
    except (
        KbConfigError,
        KbNotFoundError,
        MatchingFileError,
        ResumeFileError,
        ResumeConfigError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (KbStoreError, MatchingExtractError, ResumeExtractError) as exc:
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
