from job_agent.resume.normalizer import normalize_resume
from job_agent.resume.schema import PersonalInfo, Resume, WorkExperience


def test_normalize_dates_phone_skills_and_placeholders():
    resume = Resume(
        personal_info=PersonalInfo(
            name="XXX",
            phone="138-0000-1234",
            email="保密",
            location="杭州",
        ),
        work_experience=[
            WorkExperience(
                company="丁香园",
                start_date="2021.08",
                end_date="至今",
                responsibilities=["  ", "做题"],
                achievements=["TP99 下降"],
            )
        ],
        skills=["Python", "python", " Redis ", ""],
        summary="后端",
    )
    out = normalize_resume(resume)
    assert out.personal_info.name is None
    assert out.personal_info.phone == "13800001234"
    assert out.personal_info.email is None
    assert out.personal_info.location == "杭州"
    assert out.work_experience[0].start_date == "2021-08"
    assert out.work_experience[0].end_date == "present"
    assert out.work_experience[0].responsibilities == ["做题"]
    assert out.skills == ["python", "redis"]


def test_normalize_chinese_month_and_garbage_date():
    resume = Resume(
        work_experience=[
            WorkExperience(start_date="2015年07月", end_date="sometime"),
        ]
    )
    out = normalize_resume(resume)
    assert out.work_experience[0].start_date == "2015-07"
    assert out.work_experience[0].end_date is None
