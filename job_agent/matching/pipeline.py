from __future__ import annotations

from typing import cast

from job_agent.matching.graph import build_matching_graph
from job_agent.matching.state import MatchingState


async def run_matching(jd_path: str, resume_ref: str) -> MatchingState:
    """Run the matching graph for one JD and one resume reference."""

    graph = build_matching_graph()
    state = await graph.ainvoke(
        {
            "jd_path": jd_path,
            "resume_ref": resume_ref,
            "dimension_scores": {},
        }
    )
    return cast(MatchingState, state)
