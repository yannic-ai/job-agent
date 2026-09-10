from job_agent.matching.jd_prompts import SYSTEM_PROMPT, build_jd_prompt


def test_jd_prompt_has_path_variable_and_required_rules():
    prompt = build_jd_prompt()

    assert "jd_path" in prompt.input_variables
    assert "岗位需求解析专家" in SYSTEM_PROMPT
    assert "不编造" in SYSTEM_PROMPT
    assert "不写简历信息" in SYSTEM_PROMPT
