from __future__ import annotations

import asyncio
import json
import logging

from langchain.tools import tool

from job_agent.matching.report import render_report
from job_agent.matching.schemas import Decision, DimensionScore

logger = logging.getLogger(__name__)


def _render_report_from_json(
    title: str | None,
    candidate_name: str | None,
    scores_json: str,
    decision_json: str,
) -> str:
    scores = [
        DimensionScore.model_validate(item) for item in json.loads(scores_json)
    ]
    decision = Decision.model_validate_json(decision_json)
    return render_report(
        title=title,
        candidate_name=candidate_name,
        scores=scores,
        decision=decision,
    )


@tool
async def render_report_tool(
    title: str | None,
    candidate_name: str | None,
    scores_json: str,
    decision_json: str,
) -> str:
    """Render the final markdown report from score and decision JSON."""
    logger.info("rendering final report")
    return await asyncio.to_thread(
        _render_report_from_json,
        title,
        candidate_name,
        scores_json,
        decision_json,
    )
