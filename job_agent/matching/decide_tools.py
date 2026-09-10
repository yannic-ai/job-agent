from __future__ import annotations

import asyncio
import logging

from langchain.tools import tool

from job_agent.matching.decision import average_and_label

logger = logging.getLogger(__name__)


@tool
async def average_and_label_tool(scores: list[int]) -> str:
    """Average five dimension scores and return Decision JSON."""
    logger.info("averaging dimension scores scores=%s", scores)
    decision = await asyncio.to_thread(average_and_label, scores)
    return decision.model_dump_json()
