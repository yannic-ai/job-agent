from datetime import date

from job_agent.kb.chunker import chunk_resume
from job_agent.kb.profile import build_resume_profile
from job_agent.resume.schema import (
    Education,
    PersonalInfo,
    Project,
    Resume,
    WorkExperience,
)


def _sample() -> Resume:
    return Resume(
        personal_info=PersonalInfo(
            name="张三",
            phone="13800138000",
            email="a@b.c",
            location="杭州",
        ),
        skills=["Python", "LangGraph"],
        education=[
            Education(
                school="浙大",
                degree="硕士",
                major="CS",
                start_date="2018-09",
                end_date="2021-06",
            )
        ],
        work_experience=[
            WorkExperience(
                company="丁香园",
                title="后端",
                start_date="2021-07",
                end_date="present",
                responsibilities=["做 Agent"],
                achievements=["上线"],
            )
        ],
        projects=[
            Project(
                name="Xagent",
                role="开发",
                responsibilities=["编排"],
            )
        ],
        summary="后端",
    )


def test_profile_highest_degree_and_months() -> None:
    profile = build_resume_profile(_sample(), today=date(2026, 9, 10))
    assert profile.name == "张三"
    assert profile.highest_degree == "硕士"
    assert profile.experience_months == (2026 - 2021) * 12 + (9 - 7)


def test_chunk_types_and_indices() -> None:
    chunks = chunk_resume(_sample())
    types = [chunk.chunk_type for chunk in chunks]
    assert types.count("work_experience") == 1
    assert types.count("project") == 1
    assert types.count("education") == 1
    assert "skills" in types
    skills = next(chunk for chunk in chunks if chunk.chunk_type == "skills")
    assert skills.chunk_index == 0
    assert "13800138000" not in skills.embed_text
    assert "a@b.c" not in "".join(chunk.embed_text for chunk in chunks)


def test_empty_projects_omits_project_chunk() -> None:
    resume = Resume(skills=["Python"])
    types = {chunk.chunk_type for chunk in chunk_resume(resume)}
    assert "project" not in types
    assert "personal_info" not in types


def test_embed_text_uses_required_templates() -> None:
    chunks = chunk_resume(_sample())
    texts = {chunk.chunk_type: chunk.embed_text for chunk in chunks}
    assert texts["skills"] == "【技能】Python, LangGraph"
    assert (
        texts["education"]
        == "【教育】学校=浙大；学历=硕士；专业=CS；时间=2018-09 ~ 2021-06"
    )
    assert texts["work_experience"] == (
        "【工作经历】公司=丁香园；职位=后端；时间=2021-07 ~ present\n"
        "职责：做 Agent\n"
        "业绩：上线"
    )
    assert texts["project"] == "【项目】名称=Xagent；角色=开发\n职责：编排"
    assert texts["personal_info"] == "【地点】杭州；姓名=张三"
    assert texts["summary"] == "【摘要】后端"


def test_blank_scalar_fields_omit_chunks() -> None:
    resume = Resume(
        personal_info=PersonalInfo(name=" ", location=""),
        summary=" ",
    )
    assert chunk_resume(resume) == []


def test_experience_months_uses_outer_span_not_segment_sum() -> None:
    resume = Resume(
        work_experience=[
            WorkExperience(start_date="2020-01", end_date="2020-06"),
            WorkExperience(start_date="2022-01", end_date="2022-06"),
        ]
    )
    profile = build_resume_profile(resume)
    assert profile.experience_months == 29


def test_contact_values_are_removed_from_all_embed_text() -> None:
    resume = Resume(
        personal_info=PersonalInfo(phone="13800138000", email="a@b.c"),
        summary="联系 13800138000 或 a@b.c",
    )
    chunks = chunk_resume(resume)
    assert chunks[0].embed_text == "【摘要】联系  或 "
