from job_agent.matching.jd_splitter import format_jd_sections, split_jd_sections

SAMPLE = """#职位介绍
我们正在构建 Agent 平台。

#岗位职责
参与编排系统

#岗位要求
精通 Python

##加分项
有 LangChain 经验
"""


def test_split_maps_hash_headings_without_space():
    sections = split_jd_sections(SAMPLE)
    assert "Agent 平台" in sections["intro"]
    assert "编排" in sections["responsibilities"]
    assert "Python" in sections["requirements"]
    assert "LangChain" in sections["nice_to_have"]


def test_split_without_headings_goes_to_other():
    assert split_jd_sections("纯文本岗位") == {"other": "纯文本岗位"}


def test_format_includes_section_titles():
    text = format_jd_sections({"intro": "hello", "other": "x"})
    assert "职位介绍" in text or "intro" in text
    assert "hello" in text
