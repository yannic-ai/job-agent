from job_agent.resume.schema import Resume, WorkExperience


def test_resume_defaults_and_nulls():
    resume = Resume()
    dumped = resume.model_dump()
    assert dumped["personal_info"]["name"] is None
    assert dumped["education"] == []
    assert dumped["work_experience"] == []
    assert dumped["projects"] == []
    assert dumped["skills"] == []
    assert dumped["summary"] is None
    assert dumped["target_role"] is None


def test_work_experience_coerces_string_lists():
    item = WorkExperience.model_validate(
        {
            "company": "丁香园",
            "responsibilities": "负责答题主路径",
            "achievements": None,
        }
    )
    assert item.responsibilities == ["负责答题主路径"]
    assert item.achievements == []


def test_resume_coerces_single_object_lists():
    resume = Resume.model_validate(
        {
            "education": {
                "school": "东华理工大学",
                "major": "软件专业",
                "degree": "本科",
                "start_date": "2011-09",
                "end_date": "2015-07",
            },
            "work_experience": {"company": "丁香园"},
            "projects": {"name": "医考智能客服助理"},
        }
    )
    assert len(resume.education) == 1
    assert resume.education[0].school == "东华理工大学"
    assert resume.work_experience[0].company == "丁香园"
    assert resume.projects[0].name == "医考智能客服助理"
