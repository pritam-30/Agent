from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from test import test_cases
from app import run_agent

import json
import time


correctness_metric = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct and complete based on the expected output.",
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.7,
)


def evaluate_rag(
    query: str,
    answer: str,
    retrieved_chunks: list[str] | None,
    expected_output: str,
):
    """
    Evaluate one RAG response.
    """

    test_case = LLMTestCase(
        input=query,
        actual_output=answer,
        retrieval_context=retrieved_chunks,
        expected_output=expected_output,
    )

    metrics = [AnswerRelevancyMetric(threshold=0.7), correctness_metric]

    # Faithfulness and Contextual Precision both need retrieval_context.
    # Skip them if this question never triggered retrieval (e.g. a greeting),
    # so a no-retrieval test case doesn't crash the whole run.
    if retrieved_chunks:
        metrics.append(FaithfulnessMetric(threshold=0.7))
        metrics.append(ContextualPrecisionMetric(threshold=0.7))

    return evaluate(test_cases=[test_case], metrics=metrics)


def save_result(query: str, results):
    """
    Append this test case's scores to a running results file,
    so results from every run — across every session — end up in one place.
    """

    scores = {
        metric.name: metric.score
        for metric in results.test_results[0].metrics_data
    }

    with open("eval_results.json", "a") as f:
        f.write(json.dumps({"query": query, **scores}) + "\n")


for test_case in test_cases:
    query = test_case["input"]
    expected_output = test_case["expected_output"]

    for attempt in range(3):
        try:
            answer, retrieved_chunks = run_agent(query, None)
            break
        except Exception as e:
            print(f"Retrying after error: {e}")
            time.sleep(20)

    print(f"\nQuery: {query}")
    print(f"Expected Output: {expected_output}")
    print(f"Actual Output: {answer}")
    print(f"Retrieved Chunks: {retrieved_chunks}")

    time.sleep(8)

    results = evaluate_rag(
        query=query,
        answer=answer,
        retrieved_chunks=retrieved_chunks,
        expected_output=expected_output,
    )

    save_result(query, results)

    print("\nEvaluation Results:")
    print(results)
