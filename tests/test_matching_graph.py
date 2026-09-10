from job_agent.matching.graph import build_matching_graph

REQUIRED = {
    "jd_parse",
    "resume_extract",
    "eval_skills",
    "eval_years",
    "eval_education",
    "eval_location",
    "eval_responsibilities",
    "decide",
    "report",
}


def test_compiled_graph_contains_required_nodes() -> None:
    app = build_matching_graph()
    names = set(app.nodes)
    assert REQUIRED <= names
    assert "parse_join" in app.nodes


def test_compiled_graph_locks_fan_in_and_fan_out_edges() -> None:
    app = build_matching_graph()
    graph = app.get_graph()
    edges = {(edge.source, edge.target) for edge in graph.edges}

    assert ("__start__", "jd_parse") in edges
    assert ("__start__", "resume_extract") in edges
    assert ("jd_parse", "parse_join") in edges
    assert ("resume_extract", "parse_join") in edges
    assert ("parse_join", "eval_skills") in edges
    assert ("parse_join", "eval_years") in edges
    assert ("parse_join", "eval_education") in edges
    assert ("parse_join", "eval_location") in edges
    assert ("parse_join", "eval_responsibilities") in edges
    assert ("eval_skills", "decide") in edges
    assert ("eval_years", "decide") in edges
    assert ("eval_education", "decide") in edges
    assert ("eval_location", "decide") in edges
    assert ("eval_responsibilities", "decide") in edges
    assert ("decide", "report") in edges
