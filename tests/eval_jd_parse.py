from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from job_agent.matching.nodes import jd_parse_node

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _count_responsibility_hits(responsibilities: list[str]) -> int:
    text = " ".join(responsibilities).lower()
    keywords = ("编排", "runtime", "记忆")
    return sum(1 for keyword in keywords if keyword in text)


@pytest.mark.live
def test_parse_xagent_jd_live() -> None:
    jd_path = FIXTURES_DIR / "xagent_jd.md"
    gold_path = FIXTURES_DIR / "xagent_jd.gold.json"
    gold = json.loads(gold_path.read_text(encoding="utf-8"))

    result = asyncio.run(jd_parse_node({"jd_path": str(jd_path)}))
    job = result["job_requirement"]
    dumped = job.model_dump()

    assert set(dumped.keys()) == {
        "title",
        "must_have_skills",
        "nice_to_have_skills",
        "years_required",
        "education_required",
        "location",
        "responsibilities",
    }
    assert job.location is None
    assert gold["must_have_skills"][0] in {skill.lower() for skill in job.must_have_skills}
    assert gold["nice_to_have_skills"][0] in {
        skill.lower() for skill in job.nice_to_have_skills
    }
    assert "3" in (job.years_required or "")
    assert "本科" in (job.education_required or "")
    assert len(job.responsibilities) >= 3
    assert _count_responsibility_hits(job.responsibilities) >= 2
