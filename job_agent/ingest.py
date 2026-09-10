from __future__ import annotations

import asyncio
import logging
import os
import sys

from job_agent.kb.errors import KbConfigError, KbStoreError
from job_agent.kb.pipeline import ingest_resume
from job_agent.resume.errors import (
    ResumeConfigError,
    ResumeExtractError,
    ResumeFileError,
)

USAGE = "用法：python -m job_agent.ingest <resume.md>"


def _configure_logging() -> None:
    """Send INFO logs to stderr so resume_id stays on stdout."""
    if logging.getLogger().handlers:
        return
    level_name = os.environ.get("JOB_AGENT_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the resume ingest CLI and return the process exit code."""
    _configure_logging()

    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(USAGE, file=sys.stderr)
        return 2
    try:
        resume_id = asyncio.run(ingest_resume(args[0]))
    except (KbConfigError, ResumeFileError, ResumeConfigError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (ResumeExtractError, KbStoreError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"简历入库失败：{exc}", file=sys.stderr)
        return 1
    print(f"resume_id={resume_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
