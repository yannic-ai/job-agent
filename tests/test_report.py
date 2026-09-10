from job_agent.matching.report import render_report
from job_agent.matching.schemas import Decision, DimensionScore


def test_report_contains_dimensions_and_verdict():
    scores = [
        DimensionScore(dimension="skills", score=4, evidence="技能 2/2 命中"),
        DimensionScore(dimension="years", score=5, evidence="年限充足"),
        DimensionScore(dimension="education", score=4, evidence="本科学历"),
        DimensionScore(dimension="location", score=3, evidence="地点缺失"),
        DimensionScore(dimension="responsibilities", score=4, evidence="职责 3/5"),
    ]
    text = render_report(
        title="Agent 平台核心开发",
        candidate_name="宗艳云",
        scores=scores,
        decision=Decision(average=4.0, recommendation="推荐"),
    )
    for word in ("技能", "年限", "学历", "地点", "职责", "推荐", "宗艳云", "Agent"):
        assert word in text
    assert "4.0" in text
