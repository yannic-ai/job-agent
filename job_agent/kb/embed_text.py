from __future__ import annotations

from collections.abc import Iterable

from job_agent.kb.models import ChunkType


def render_embed_text(
    chunk_type: ChunkType,
    payload: dict[str, object],
) -> str:
    """Render a stable embedding text for one resume chunk."""
    if chunk_type == "skills":
        skills = _string_list(payload.get("skills"))
        return f"【技能】{', '.join(skills)}" if skills else ""
    if chunk_type == "education":
        return _render_item(
            "教育",
            (
                ("学校", payload.get("school")),
                ("学历", payload.get("degree")),
                ("专业", payload.get("major")),
                ("时间", _date_range(payload)),
            ),
        )
    if chunk_type == "work_experience":
        header = _render_item(
            "工作经历",
            (
                ("公司", payload.get("company")),
                ("职位", payload.get("title")),
                ("时间", _date_range(payload)),
            ),
        )
        return _append_lists(header, payload)
    if chunk_type == "project":
        header = _render_item(
            "项目",
            (
                ("名称", payload.get("name")),
                ("角色", payload.get("role")),
                ("时间", _date_range(payload)),
            ),
        )
        return _append_lists(header, payload)
    if chunk_type == "personal_info":
        return _render_item(
            "地点",
            (
                ("地点", payload.get("location")),
                ("姓名", payload.get("name")),
            ),
            first_value_without_key=True,
        )
    if chunk_type == "summary":
        summary = _text(payload.get("summary"))
        return f"【摘要】{summary}" if summary else ""
    raise ValueError(f"Unsupported chunk type: {chunk_type}")


def _render_item(
    label: str,
    fields: Iterable[tuple[str, object]],
    *,
    first_value_without_key: bool = False,
) -> str:
    rendered_fields: list[str] = []
    for index, (key, value) in enumerate(fields):
        text = _text(value)
        if not text:
            continue
        if first_value_without_key and index == 0:
            rendered_fields.append(text)
        else:
            rendered_fields.append(f"{key}={text}")
    return f"【{label}】{'；'.join(rendered_fields)}"


def _date_range(payload: dict[str, object]) -> str:
    start_date = _text(payload.get("start_date"))
    end_date = _text(payload.get("end_date"))
    if start_date and end_date:
        return f"{start_date} ~ {end_date}"
    return start_date or end_date


def _append_lists(header: str, payload: dict[str, object]) -> str:
    lines = [header]
    responsibilities = _string_list(payload.get("responsibilities"))
    achievements = _string_list(payload.get("achievements"))
    if responsibilities:
        lines.append(f"职责：{'；'.join(responsibilities)}")
    if achievements:
        lines.append(f"业绩：{'；'.join(achievements)}")
    return "\n".join(lines)


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]
