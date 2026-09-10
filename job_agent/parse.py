from __future__ import annotations

import sys

from job_agent.resume.errors import ResumeConfigError, ResumeExtractError, ResumeFileError
from job_agent.resume.pipeline import parse_resume

USAGE = "用法：python -m job_agent.parse <resume.md>"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(USAGE, file=sys.stderr)
        return 2
    try:
        resume = parse_resume(args[0])
    except (ResumeFileError, ResumeConfigError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except ResumeExtractError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"简历解析失败：{exc}", file=sys.stderr)
        return 1
    print(resume.model_dump_json(indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
