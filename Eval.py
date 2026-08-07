from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase


def evaluate_rag(
    query: str,
    answer: str,
    retrieved_chunks: list[str],
):
    """
    Evaluate one RAG response.
    """

    test_case = LLMTestCase(
        input=query,
        actual_output=answer,
        retrieval_context=retrieved_chunks,
    )

    results = evaluate(
        test_cases=[test_case],
        metrics=[
            FaithfulnessMetric(threshold=0.7),
            AnswerRelevancyMetric(threshold=0.7),
            # ContextualPrecisionMetric(threshold=0.7),
        ],
    )

    return results
