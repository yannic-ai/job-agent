from pathlib import Path


def test_xagent_jd_fixture_exists() -> None:
    text = (Path(__file__).parent / "fixtures" / "xagent_jd.md").read_text(
        encoding="utf-8"
    )
    assert "职位介绍" in text
    assert "加分项" in text
