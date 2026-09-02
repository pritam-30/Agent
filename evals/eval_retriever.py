from src.retrieval_strategies import similarity_search
from test_cases.retriever_eval import QUESTIONS
from deepeval import evaluate
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric
)
from deepeval.test_case import LLMTestCase


def evaluate_retrieval(query: str,
                       retrieved_chunks: list[str] | None,
                       expected_output: str):

    test_case = LLMTestCase(
        input=query,
        retrieval_context=retrieved_chunks,
        expected_output=expected_output,
    )

    metrics = [ContextualPrecisionMetric(
        threshold=0.7), ContextualRecallMetric(threshold=0.7)]

    return evaluate(test_cases=[test_case], metrics=metrics)


for question in QUESTIONS:
    query = question["question"]
    expected_output = question["ideal_answer"]

    documents, metadatas = similarity_search(
        query=query,
        candidate_k=15,
        k=5,
        rerank=True,
    )

    results = evaluate_retrieval(
        query=query,
        retrieved_chunks=documents,
        expected_output=expected_output,
    )
