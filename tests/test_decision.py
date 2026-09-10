import pytest

from job_agent.matching.decision import average_and_label


def test_average_thresholds():
    assert average_and_label([5, 5, 5, 5, 5]).recommendation == "推荐"
    assert average_and_label([4, 4, 4, 4, 4]).recommendation == "推荐"
    assert average_and_label([3, 3, 3, 3, 3]).recommendation == "待定"
    assert average_and_label([2, 3, 3, 3, 4]).recommendation == "待定"
    assert average_and_label([1, 1, 1, 1, 1]).recommendation == "不推荐"


def test_requires_five_scores():
    with pytest.raises(ValueError):
        average_and_label([1, 2, 3, 4])
