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
