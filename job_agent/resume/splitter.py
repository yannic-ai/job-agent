from __future__ import annotations

import re

_H_RE = re.compile(r"^(#{1,6})\s+(.*)$")

_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("项目经历", "项目经验"), "projects"),
    (("工作履历", "工作经历", "任职"), "work_experience"),
    (("教育经历", "教育背景"), "education"),
    (("基本信息", "个人信息", "联系"), "personal_info"),
    (("专业能力", "技能"), "skills"),
)


def _map_heading(title: str) -> str:
    for keywords, section_id in _RULES:
        if any(keyword in title for keyword in keywords):
            return section_id
    return "other"


def _append(sections: dict[str, str], key: str, text: str) -> None:
    body = text.strip()
    if not body:
        return
    if key in sections:
        sections[key] = sections[key] + "\n\n" + body
    else:
        sections[key] = body


def split_sections(markdown: str) -> dict[str, str]:
    lines = markdown.splitlines()
    sections: dict[str, str] = {}
    current_id: str | None = None
    buf: list[str] = []
    saw_heading = False

    def flush() -> None:
        nonlocal buf, current_id
        if current_id is None:
            buf = []
            return
        _append(sections, current_id, "\n".join(buf))
        buf = []

    for line in lines:
        match = _H_RE.match(line)
        if match:
            hashes, title = match.group(1), match.group(2).strip()
            if len(hashes) == 1:
                flush()
                saw_heading = True
                current_id = "title"
                buf = [line]
                continue
            if len(hashes) == 2:
                flush()
                saw_heading = True
                current_id = _map_heading(title)
                buf = [line]
                continue
        buf.append(line)

    flush()
    if not saw_heading:
        body = markdown.strip()
        return {"other": body} if body else {}
    return sections


def heading_name(sections: dict[str, str]) -> str | None:
    for line in (sections.get("title") or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("#") and not stripped.startswith("##"):
            name = stripped.lstrip("#").strip()
            return name or None
    return None
