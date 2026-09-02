from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric
)
from deepeval.test_case import LLMTestCase
import json
from pathlib import Path
from src.generator import generate_answer

with open("test_cases/generator_evalset.json", "r") as f:
    generator_evalset = json.load(f)


def evaluate_generator(
    query: str,
    answer: str,
    ideal_context: list[str] | None,
):

    test_case = LLMTestCase(
        input=query,
        actual_output=answer,
        retrieval_context=ideal_context,
    )

    metrics = [AnswerRelevancyMetric(
        threshold=0.7), FaithfulnessMetric(threshold=0.7)]

    return evaluate(test_cases=[test_case], metrics=metrics)


RESULTS_FILE = Path(
    "/mnt/sdb1/Programming/DS_Projects/Rag/results/generator_results.json"
)


def save_generator_result(question_id: int, query: str, results):
    scores = {
        metric.name: metric.score
        for metric in results.test_results[0].metrics_data
    }

    result = {
        "question_id": question_id,
        "query": query,
        "scores": scores,
    }

    # Load existing results
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []

    # Prevent duplicate questions if you have to rerun one
    all_results = [
        item
        for item in all_results
        if item["question_id"] != question_id
    ]

    all_results.append(result)

    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=4)


for question in generator_evalset:

    query = question["query"]
    context = [
        item["chunk"]
        for item in question["ideal_context"]
    ]

    answer = generate_answer(
        query=query,
        context=context
    )

    results = evaluate_generator(
        query=query,
        answer=answer,
        ideal_context=context,
    )

    save_generator_result(
        question_id=question["id"],
        query=query,
        results=results,
    )
