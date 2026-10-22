"""Codabench entry point for the logistic-regression robustness classifier."""

from logistic_inference import predict_robustness


def are_robust(
    model_id: str,
    reasoning_effort: str,
    problems: list[str],
) -> list[bool]:
    """Return one native robustness boolean per problem, preserving order."""

    return predict_robustness(
        model_id=model_id,
        reasoning_effort=reasoning_effort,
        problem_texts=problems,
    )
