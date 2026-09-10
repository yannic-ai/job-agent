from job_agent.matching.schemas import DIMENSIONS, Decision, JobRequirement, label_from_average


def test_job_requirement_defaults():
    dumped = JobRequirement().model_dump()
    assert dumped["title"] is None
    assert dumped["must_have_skills"] == []
    assert dumped["nice_to_have_skills"] == []
    assert dumped["years_required"] is None
    assert dumped["education_required"] is None
    assert dumped["location"] is None
    assert dumped["responsibilities"] == []


def test_label_from_average_thresholds():
    assert label_from_average(4.0) == "推荐"
    assert label_from_average(3.0) == "待定"
    assert label_from_average(2.9) == "不推荐"


def test_decision_round_trip():
    decision = Decision(average=3.6, recommendation="待定")
    assert decision.recommendation == "待定"
    assert DIMENSIONS[0] == "skills"
