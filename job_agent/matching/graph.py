from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from job_agent.matching.nodes import (
    decide_node,
    eval_education_node,
    eval_location_node,
    eval_responsibilities_node,
    eval_skills_node,
    eval_years_node,
    jd_parse_node,
    parse_join_node,
    report_node,
    resume_extract_node,
)
from job_agent.matching.state import MatchingState


def build_matching_graph() -> CompiledStateGraph:
    """Build and compile the matching workflow graph."""

    graph = StateGraph(MatchingState)
    graph.add_node("jd_parse", jd_parse_node)
    graph.add_node("resume_extract", resume_extract_node)
    graph.add_node("parse_join", parse_join_node)
    graph.add_node("eval_skills", eval_skills_node)
    graph.add_node("eval_years", eval_years_node)
    graph.add_node("eval_education", eval_education_node)
    graph.add_node("eval_location", eval_location_node)
    graph.add_node("eval_responsibilities", eval_responsibilities_node)
    graph.add_node("decide", decide_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "jd_parse")
    graph.add_edge(START, "resume_extract")
    graph.add_edge("jd_parse", "parse_join")
    graph.add_edge("resume_extract", "parse_join")
    graph.add_edge("parse_join", "eval_skills")
    graph.add_edge("parse_join", "eval_years")
    graph.add_edge("parse_join", "eval_education")
    graph.add_edge("parse_join", "eval_location")
    graph.add_edge("parse_join", "eval_responsibilities")
    graph.add_edge("eval_skills", "decide")
    graph.add_edge("eval_years", "decide")
    graph.add_edge("eval_education", "decide")
    graph.add_edge("eval_location", "decide")
    graph.add_edge("eval_responsibilities", "decide")
    graph.add_edge("decide", "report")
    graph.add_edge("report", END)
    return graph.compile()
