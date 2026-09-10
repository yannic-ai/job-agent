from job_agent.matching.schemas import Decision, label_from_average


def average_and_label(scores: list[int]) -> Decision:
    """Return the average score and recommendation for exactly five dimensions."""
    if len(scores) != 5:
        raise ValueError("scores must contain exactly 5 items")
    if any(score not in {1, 2, 3, 4, 5} for score in scores):
        raise ValueError("scores must all be between 1 and 5")

    average = sum(scores) / 5
    return Decision(average=average, recommendation=label_from_average(average))
