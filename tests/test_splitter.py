import pytest

from job_agent.resume.splitter import split_sections

SAMPLE = """# 张三

## 基本信息
电话：123

## 专业能力
Python

## 工作履历
A 公司

### 不该单独成块
细节

## 项目经历
项目甲

## 教育经历
某大学

## 个人实践学习
自学

## 奇怪标题
额外
"""


def test_split_maps_known_headings_and_keeps_h3():
    sections = split_sections(SAMPLE)
    assert "张三" in sections["title"]
    assert "电话：123" in sections["personal_info"]
    assert "Python" in sections["skills"]
    assert "A 公司" in sections["work_experience"]
    assert "### 不该单独成块" in sections["work_experience"]
    assert "项目甲" in sections["projects"]
    assert "某大学" in sections["education"]
    assert "自学" in sections["other"]
    assert "额外" in sections["other"]


def test_split_maps_job_intention_heading():
    sections = split_sections("## 求职意向\nJava 后端\n")
    assert "Java 后端" in sections["target_role"]


def test_split_without_headings_goes_to_other():
    sections = split_sections("纯文本简历")
    assert sections == {"other": "纯文本简历"}


def test_split_concatenates_same_id():
    text = "## 技能\nA\n\n## 专业能力\nB\n"
    sections = split_sections(text)
    assert "A" in sections["skills"]
    assert "B" in sections["skills"]


@pytest.mark.parametrize(
    "heading",
    ("专业技能", "技术栈", "技术能力", "技术专长", "技能栈"),
)
def test_split_maps_skill_heading_aliases(heading: str) -> None:
    sections = split_sections(f"## {heading}\nPython\n")
    assert "Python" in sections["skills"]
    assert "other" not in sections


@pytest.mark.parametrize(
    "heading",
    ("个人资料", "基本资料", "联系方式"),
)
def test_split_maps_personal_info_heading_aliases(heading: str) -> None:
    sections = split_sections(f"## {heading}\n杭州\n")
    assert "杭州" in sections["personal_info"]
    assert "other" not in sections


@pytest.mark.parametrize(
    "heading",
    ("工作经验", "职业经历", "从业经历"),
)
def test_split_maps_work_experience_heading_aliases(heading: str) -> None:
    sections = split_sections(f"## {heading}\nA 公司\n")
    assert "A 公司" in sections["work_experience"]
    assert "other" not in sections


@pytest.mark.parametrize(
    "heading",
    ("个人项目", "开源项目", "项目介绍"),
)
def test_split_maps_project_heading_aliases(heading: str) -> None:
    sections = split_sections(f"## {heading}\n项目甲\n")
    assert "项目甲" in sections["projects"]
    assert "other" not in sections


@pytest.mark.parametrize(
    "heading",
    ("学历", "学习经历", "毕业院校"),
)
def test_split_maps_education_heading_aliases(heading: str) -> None:
    sections = split_sections(f"## {heading}\n某大学\n")
    assert "某大学" in sections["education"]
    assert "other" not in sections


@pytest.mark.parametrize(
    "heading",
    ("求职目标", "期望岗位", "应聘职位", "意向职位"),
)
def test_split_maps_target_role_heading_aliases(heading: str) -> None:
    sections = split_sections(f"## {heading}\nJava 后端\n")
    assert "Java 后端" in sections["target_role"]
    assert "other" not in sections


def test_heading_name_from_title_section():
    from job_agent.resume.pipeline import _fill_personal_info
    from job_agent.resume.schema import Resume
    from job_agent.resume.splitter import heading_name, split_sections

    sections = split_sections("# 宗艳云\n\n## 基本信息\n杭州\n")
    assert heading_name(sections) == "宗艳云"

    filled = _fill_personal_info(
        Resume(),
        split_sections("# 宗艳云\n\n## 基本信息\n- 期望城市：杭州\n- 电话：138-0000-1234\n"),
    )
    assert filled.personal_info.name == "宗艳云"
    assert filled.personal_info.location == "杭州"
    assert filled.personal_info.phone == "138-0000-1234"


def test_fill_target_role_from_personal_info():
    from job_agent.resume.pipeline import _fill_target_role
    from job_agent.resume.schema import Resume
    from job_agent.resume.splitter import split_sections

    filled = _fill_target_role(
        Resume(),
        split_sections(
            "# 宗艳云\n\n## 基本信息\n- 求职方向：AI Agent / Java 全栈开发\n"
        ),
    )
    assert filled.target_role == "AI Agent / Java 全栈开发"
