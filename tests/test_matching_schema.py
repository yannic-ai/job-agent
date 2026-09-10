import pytest
from pydantic import ValidationError

from job_agent.matching.schemas import (
    DIMENSIONS,
    Decision,
    DimensionScore,
    JobRequirement,
    label_from_average,
)


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


def test_dimension_score_score_range():
    low = DimensionScore(dimension="skills", score=1, evidence="ok")
    high = DimensionScore(dimension="skills", score=5, evidence="ok")
    assert low.score == 1
    assert high.score == 5

    with pytest.raises(ValidationError):
        DimensionScore(dimension="skills", score=0, evidence="bad")
    with pytest.raises(ValidationError):
        DimensionScore(dimension="skills", score=6, evidence="bad")


def test_dimension_score_evidence_non_empty():
    score = DimensionScore(dimension="skills", score=3, evidence="技能 2/2 命中")
    assert score.evidence == "技能 2/2 命中"

    with pytest.raises(ValidationError):
        DimensionScore(dimension="skills", score=3, evidence="")

    with pytest.raises(ValidationError):
        DimensionScore(dimension="skills", score=3, evidence="   ")
