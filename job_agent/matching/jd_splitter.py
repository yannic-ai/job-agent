from __future__ import annotations

import re

_H_RE = re.compile(r"^(#{1,6})\s*(.+?)\s*$")

_SECTION_ORDER = (
    "intro",
    "responsibilities",
    "requirements",
    "nice_to_have",
    "other",
)

_FORMAT_TITLES = {
    "intro": "职位介绍",
    "responsibilities": "岗位职责",
    "requirements": "岗位要求",
    "nice_to_have": "加分项",
    "other": "其他",
}


def _map_heading(title: str) -> str:
    if "加分" in title:
        return "nice_to_have"
    if any(kw in title for kw in ("岗位要求", "任职要求", "职位要求")):
        return "requirements"
    if any(kw in title for kw in ("岗位职责", "工作职责")):
        return "responsibilities"
    if any(kw in title for kw in ("职位介绍", "岗位介绍", "职位描述")):
        return "intro"
    return "other"


def _append(sections: dict[str, str], key: str, text: str) -> None:
    body = text.strip()
    if not body:
        return
    if key in sections:
        sections[key] = sections[key] + "\n\n" + body
    else:
        sections[key] = body


def split_jd_sections(markdown: str) -> dict[str, str]:
    """Split JD markdown into intro, requirements, and related sections."""
    lines = markdown.splitlines()
    sections: dict[str, str] = {}
    current_id: str | None = None
    buf: list[str] = []
    saw_boundary = False

    def flush() -> None:
        nonlocal buf, current_id
        if current_id is None:
            buf = []
            return
        _append(sections, current_id, "\n".join(buf))
        buf = []

    for line in lines:
        match = _H_RE.match(line)
        if match and len(match.group(1)) <= 2:
            flush()
            saw_boundary = True
            current_id = _map_heading(match.group(2).strip())
            buf = [line]
            continue
        buf.append(line)

    flush()
    if not saw_boundary:
        body = markdown.strip()
        return {"other": body} if body else {}
    return sections


def format_jd_sections(sections: dict[str, str]) -> str:
    """Format section dict into normalized JD markdown with Chinese headings."""
    blocks: list[str] = []
    for section_id in _SECTION_ORDER:
        title = _FORMAT_TITLES[section_id]
        body = sections.get(section_id, "").strip() or "（无）"
        blocks.append(f"# {title}\n{body}")
    return "\n\n".join(blocks)
