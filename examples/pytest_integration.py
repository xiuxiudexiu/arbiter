"""Pytest integration example for Arbiter.

This example shows how to:
1. Use Arbiter as test assertions
2. Create reusable fixtures
3. Test LLM output quality
4. Set quality thresholds

The tests are skipped when OPENAI_API_KEY is not configured. Arbiter uses
temperature 0.0 for evaluations, but LLM scores can still vary slightly; choose
thresholds that allow acceptable variation rather than expecting exact scores.

Run with:
    pytest examples/pytest_integration.py -v
"""

import os
from typing import Optional

import pytest

from arbiter_ai import EvaluationResult, evaluate

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY is required for live LLM evaluations",
)


@pytest.fixture
def evaluator_model() -> str:
    """Model to use for evaluation."""
    return os.getenv("ARBITER_MODEL", "gpt-4o-mini")


async def assert_quality(
    output: str,
    reference: Optional[str] = None,
    criteria: Optional[str] = None,
    threshold: float = 0.8,
    model: str = "gpt-4o-mini",
) -> EvaluationResult:
    """Assert that output meets quality threshold.

    Args:
        output: The LLM output to evaluate.
        reference: Optional reference text for semantic comparison.
        criteria: Optional criteria for reference-free evaluation.
        threshold: Minimum score required to pass.
        model: Model to use for the evaluation.

    Returns:
        The evaluation result when the quality assertion passes.
    """
    result = await evaluate(
        output=output,
        reference=reference,
        criteria=criteria,
        evaluators=["semantic"] if reference else ["custom_criteria"],
        model=model,
        threshold=threshold,
    )

    assert result.passed, (
        f"Quality check failed: score={result.overall_score:.2f} "
        f"< threshold={threshold:.2f}"
    )
    return result


@pytest.mark.asyncio
async def test_translation_quality(evaluator_model: str) -> None:
    """Test that translation is semantically similar to reference."""
    translation = "Paris is the capital of France."
    reference = "The capital of France is Paris."

    await assert_quality(
        output=translation,
        reference=reference,
        threshold=0.8,
        model=evaluator_model,
    )


@pytest.mark.asyncio
async def test_response_follows_guidelines(evaluator_model: str) -> None:
    """Test that response follows guidelines."""
    response = "Thank you for contacting us. I understand your concern."

    await assert_quality(
        output=response,
        criteria="Professional tone, empathetic, helpful",
        threshold=0.75,
        model=evaluator_model,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("output", "minimum_score"),
    [
        ("Clear, professional response.", 0.7),
        ("um yeah so like...", 0.3),
    ],
)
async def test_quality_levels(
    output: str, minimum_score: float, evaluator_model: str
) -> None:
    """Test various response quality levels."""
    await assert_quality(
        output=output,
        criteria="clarity, professionalism",
        threshold=minimum_score,
        model=evaluator_model,
    )
