from job_agent.resume.schema import Resume


def test_resume_defaults_and_nulls():
    resume = Resume()
    dumped = resume.model_dump()
    assert dumped["personal_info"]["name"] is None
    assert dumped["education"] == []
    assert dumped["work_experience"] == []
    assert dumped["projects"] == []
    assert dumped["skills"] == []
    assert dumped["summary"] is None
