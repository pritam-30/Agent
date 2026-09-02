from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    GEval
)
from src.retrieval_strategies import similarity_search
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics.g_eval import Rubric
from src.generator import generate_answer
import json
from pathlib import Path


correctness = GEval(
    name="Correctness",
    evaluation_steps=[
        "Determine whether the actual output correctly answers the input question.",
        "Compare the factual claims in the actual output against the expected output.",
        "Penalize factual contradictions or incorrect claims.",
        "Penalize important omissions when the omitted information is necessary to correctly answer the question.",
        "Do not penalize differences in wording, phrasing, or answer structure.",
        "Do not penalize vague language unless it makes the answer factually incorrect."
    ],
    rubric=[
        Rubric(
            score_range=(9, 10),
            expected_outcome=(
                "The answer is factually correct, contains no contradictions, "
                "and covers the important information necessary to answer the question. "
                "Minor differences in wording or level of detail are acceptable."
            )
        ),
        Rubric(
            score_range=(5, 8),
            expected_outcome=(
                "The answer is substantially correct but contains a minor factual "
                "inaccuracy or omits an important detail needed for a complete answer."
            )
        ),
        Rubric(
            score_range=(0, 4),
            expected_outcome=(
                "The answer contains a major factual error, contradicts the expected "
                "answer, or fails to provide the information necessary to answer the question."
            )
        ),
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    threshold=0.7,
    strict_mode=False
)

completeness = GEval(
    name="Completeness",
    evaluation_steps=[
        "Identify the key points contained in the expected output.",
        "Check how many of those key points are addressed in the actual output.",
        "Penalize the actual output for each important key point from the expected output that it omits or only partially covers.",
        "Judge coverage only. Do not lower the score because a covered point is stated incorrectly; factual correctness is judged separately.",
        "Do not penalize the actual output merely because it contains additional information. Evaluate completeness based on coverage of the expected key points."
    ],
    rubric=[
        Rubric(
            score_range=(9, 10),
            expected_outcome=(
                "Covers essentially all key points necessary to answer the question. "
                "Only negligible details may be omitted."
            )
        ),
        Rubric(
            score_range=(5, 8),
            expected_outcome=(
                "Covers the main answer but omits one or more important supporting "
                "points or partially covers an important point."
            )
        ),
        Rubric(
            score_range=(0, 4),
            expected_outcome=(
                "Omits several important points or covers only a small portion "
                "of the information necessary to answer the question."
            )
        ),
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    threshold=0.7,
    strict_mode=False
)

with open("test_cases/pipeline_eval.json", "r") as f:
    pipeline_evalset = json.load(f)


def evaluate_rag(
    query: str,
    answer: str,
    expected_output: str,
    retrieved_chunks: list[str] | None,
):

    test_case = LLMTestCase(
        input=query,
        actual_output=answer,
        expected_output=expected_output,
        retrieval_context=retrieved_chunks,
    )

    metrics = [AnswerRelevancyMetric(threshold=0.7, include_reason=True), FaithfulnessMetric(
        threshold=0.7, include_reason=True), correctness, completeness]

    return evaluate(test_cases=[test_case], metrics=metrics)


RESULTS_FILE = Path(
    "/mnt/sdb1/Programming/DS_Projects/Rag/results/prompt_v2/pipeline_results.json"
)


def save_result(question_id: int, query: str, results):

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


for question in pipeline_evalset[:2]:

    query = question["query"]
    expected_output = question["reference_answer"]

    documents, metadata = similarity_search(
        query, candidate_k=15, k=5, rerank=True)

    answer = generate_answer(
        query=query,
        context=documents
    )

    results = evaluate_rag(
        query=query,
        answer=answer,
        expected_output=expected_output,
        retrieved_chunks=documents,
    )

    save_result(question_id=question["id"], query=query, results=results)
