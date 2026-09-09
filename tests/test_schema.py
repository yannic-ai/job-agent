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
