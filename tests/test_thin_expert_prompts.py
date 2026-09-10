from job_agent.matching.decide_prompts import SYSTEM_PROMPT as DECIDE
from job_agent.matching.evaluate_prompts import SYSTEM_PROMPT as EVAL
from job_agent.matching.report_prompts import SYSTEM_PROMPT as REPORT
from job_agent.matching.resume_prompts import SYSTEM_PROMPT as RESUME


def test_role_prompts_exist() -> None:
    assert "简历" in RESUME
    assert "评估" in EVAL or "打分" in EVAL
    assert "决策" in DECIDE
    assert "报告" in REPORT
