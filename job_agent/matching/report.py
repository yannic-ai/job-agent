from job_agent.matching.schemas import DIMENSIONS, Decision, DimensionScore

DIMENSION_LABELS = {
    "skills": "技能",
    "years": "年限",
    "education": "学历",
    "location": "地点",
    "responsibilities": "职责",
}


def render_report(
    *,
    title: str | None,
    candidate_name: str | None,
    scores: list[DimensionScore],
    decision: Decision,
) -> str:
    """Render a deterministic markdown report for the five-dimension decision."""
    score_by_dimension = {score.dimension: score for score in scores}
    lines = [
        "# 简历匹配报告",
        "",
        f"- 岗位：{title or '未提供'}",
        f"- 候选人：{candidate_name or '未提供'}",
        f"- 平均分：{decision.average:.1f}",
        f"- 结论：{decision.recommendation}",
        "",
        "| 维度 | 分数 | 证据 |",
        "| --- | --- | --- |",
    ]

    for dimension in DIMENSIONS:
        label = DIMENSION_LABELS[dimension]
        item = score_by_dimension.get(dimension)
        if item is None:
            lines.append(f"| {label} | - | 未提供 |")
            continue
        lines.append(f"| {label} | {item.score} | {item.evidence} |")

    return "\n".join(lines)
