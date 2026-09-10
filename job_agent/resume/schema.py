from pydantic import BaseModel, Field, field_validator


def _coerce_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _coerce_item_list(value: object) -> object:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    return value


class PersonalInfo(BaseModel):
    name: str | None = None
    phone: str | None = None
    email: str | None = None
    location: str | None = None


class Education(BaseModel):
    school: str | None = None
    degree: str | None = None
    major: str | None = None
    start_date: str | None = None
    end_date: str | None = None


class WorkExperience(BaseModel):
    company: str | None = None
    title: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    responsibilities: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)

    @field_validator("responsibilities", "achievements", mode="before")
    @classmethod
    def _lists(cls, value: object) -> list[str]:
        return _coerce_str_list(value)


class Project(BaseModel):
    name: str | None = None
    role: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    responsibilities: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)

    @field_validator("responsibilities", "achievements", mode="before")
    @classmethod
    def _lists(cls, value: object) -> list[str]:
        return _coerce_str_list(value)


class Resume(BaseModel):
    personal_info: PersonalInfo = Field(default_factory=PersonalInfo)
    education: list[Education] = Field(default_factory=list)
    work_experience: list[WorkExperience] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    summary: str | None = None

    @field_validator("education", "work_experience", "projects", mode="before")
    @classmethod
    def _item_lists(cls, value: object) -> object:
        return _coerce_item_list(value)

    @field_validator("skills", mode="before")
    @classmethod
    def _skills(cls, value: object) -> list[str]:
        return _coerce_str_list(value)
