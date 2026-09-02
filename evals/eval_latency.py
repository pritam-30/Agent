# ============================================================
# 1. IMPORTS
# ============================================================
import math
import time

from src.retrieval_strategies import similarity_search
# generate_stream: the streaming twin
from src.generator import generate_answer, generate_stream


QUESTIONS = [
    # "Why can machine learning automatically adapt when the patterns in spam change?",
    # "How does online learning differ from batch learning when new data arrives?",
    # "How does regularization affect the risk of overfitting?",
    "What are the two main problems with hard-margin SVM classification, and how does soft-margin classification address them?",
]

# measured runs PER question -> total samples = len(QUESTIONS) * REPEATS
REPEATS = 4
WARMUP_RUNS = 2       # throwaway calls before measuring (cold start)

# stream generation and clock time-to-first-token (perceived latency)
MEASURE_TTFT = True
# split retrieval vs generation (ignored/implied when MEASURE_TTFT is on)
STAGE_LEVEL = True

# SLOs / budgets. A latency number is meaningless without a target to pass/fail against.
SLO_P95_MS = 3000        # end-to-end: full answer p95 under 3s
SLO_TTFT_P95_MS = 1200   # perceived: first visible token p95 under 1.2s

# ============================================================
# 3. PIPELINE ADAPTERS  (the ONE place you edit to match your API)
# ============================================================
# End-to-end: retrieve documents and generate the complete answer.


def run_end_to_end(question):
    docs, metadata = similarity_search(
        query=question, candidate_k=15, k=5, rerank=True)
    answer = generate_answer(query=question, context=docs)
    return answer

# Stage-level (non-streaming): reuse the pipeline's retriever + the generator,
# timing each leg. "retrieval" bundles query-embedding + vector search + the
# cross-encoder rerank pass (over-fetch fetch_k, rerank to top_k).


def run_stages(question):
    t0 = time.perf_counter()
    docs, metadata = similarity_search(
        query=question, candidate_k=15, k=5, rerank=True)
    t1 = time.perf_counter()
    answer = generate_answer(query=question, context=docs)
    t2 = time.perf_counter()
    return answer, {"retrieval": (t1 - t0) * 1000, "generation": (t2 - t1) * 1000}

# Stage-level (streaming): same retrieval, but stream generation and record the
# clock the instant the FIRST content token arrives.
#   ttft      = query submitted (t0) -> first visible token   (what the user feels)
#   generation= generation start (t1) -> last token
# TTFT includes retrieval on purpose: the user waits through retrieval before the
# first token can stream, so perceived latency = retrieval + generation-prefill.


def run_stages_streaming(question):
    t0 = time.perf_counter()
    docs, metadata = similarity_search(
        query=question, candidate_k=15, k=5, rerank=True)
    t1 = time.perf_counter()

    first_token_t = None
    pieces = []
    for piece in generate_stream(query=question, context=docs):
        if first_token_t is None:
            # clock the first non-empty chunk
            first_token_t = time.perf_counter()
        pieces.append(piece)
    t2 = time.perf_counter()

    answer = "".join(pieces)
    ttft_ms = (first_token_t - t1) * 1000 if first_token_t else float("nan")
    return answer, {
        "retrieval": (t1 - t0) * 1000,
        "generation": (t2 - t1) * 1000,
        "ttft": ttft_ms,
    }

# ============================================================
# 4. PERCENTILE HELPER
# ============================================================
# Linear-interpolation percentile -- same result as numpy.percentile, but written
# out so it is not a black box when you explain the tail in class.


def percentile(values, p):
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)

# ============================================================
# 5. BENCHMARK LOOP
# ============================================================


def benchmark():
    # --- 5a. Warmup: run and DISCARD, so cold start does not pollute stats ---
    print(f"Warming up ({WARMUP_RUNS} runs, discarded)...")
    for i in range(WARMUP_RUNS):
        run_end_to_end(QUESTIONS[i % len(QUESTIONS)])

    total_ms, retrieval_ms, generation_ms, ttft_ms = [], [], [], []
    answer_lengths = []

    # --- 5b. Measured runs: each question REPEATS times, pool all samples ---
    print("Measuring...")
    for question in QUESTIONS:
        for _ in range(REPEATS):
            start = time.perf_counter()
            if MEASURE_TTFT:
                answer, stage = run_stages_streaming(question)
                retrieval_ms.append(stage["retrieval"])
                generation_ms.append(stage["generation"])
                ttft_ms.append(stage["ttft"])
            elif STAGE_LEVEL:
                answer, stage = run_stages(question)
                retrieval_ms.append(stage["retrieval"])
                generation_ms.append(stage["generation"])
            else:
                answer = run_end_to_end(question)
            elapsed_ms = (time.perf_counter() - start) * 1000

            total_ms.append(elapsed_ms)
            answer_lengths.append(len(answer or ""))

    return {
        "total": total_ms,
        "retrieval": retrieval_ms,
        "generation": generation_ms,
        "ttft": ttft_ms,
        "answer_len": answer_lengths,
    }

# ============================================================
# 6. AGGREGATE + REPORT
# ============================================================


def summarize(samples):
    clean = [s for s in samples if not math.isnan(s)]
    return {
        "n": len(clean),
        "mean": sum(clean) / len(clean),
        "p50": percentile(clean, 50),
        "p95": percentile(clean, 95),
        "p99": percentile(clean, 99),
        "min": min(clean),
        "max": max(clean),
    }


def print_row(label, s):
    print(f"{label:<12} | n={s['n']:<3} "
          f"mean={s['mean']:7.1f}  p50={s['p50']:7.1f}  "
          f"p95={s['p95']:7.1f}  p99={s['p99']:7.1f}  "
          f"min={s['min']:7.1f}  max={s['max']:7.1f}")


def slo_line(label, p95, budget):
    verdict = "PASS" if p95 <= budget else "FAIL"
    print(
        f"SLO: {label:<22} p95 <= {budget:>5} ms  ->  p95 = {p95:7.0f} ms   [{verdict}]")


def report(results):
    print("\n" + "=" * 78)
    print("LATENCY (milliseconds)")
    print("=" * 78)
    print(f"{'stage':<12} | {'samples':<5} {'mean':>11} {'p50':>11} "
          f"{'p95':>11} {'p99':>11} {'min':>11} {'max':>11}")
    print("-" * 78)

    total = summarize(results["total"])
    print_row("end-to-end", total)
    if results["ttft"]:
        # perceived: query -> first token
        print_row("ttft", summarize(results["ttft"]))
    if results["retrieval"]:
        print_row("retrieval", summarize(results["retrieval"]))
        print_row("generation", summarize(results["generation"]))

    avg_len = sum(results["answer_len"]) / len(results["answer_len"])
    print("-" * 78)
    print(f"avg answer length: {avg_len:.0f} chars "
          f"(latency scales with output length -- keep in mind when comparing configs)")

    # --- SLO verdicts: the teaching contrast lives here ---
    print("=" * 78)
    slo_line("full answer", total["p95"], SLO_P95_MS)
    if results["ttft"]:
        slo_line("first token (perceived)", summarize(
            results["ttft"])["p95"], SLO_TTFT_P95_MS)
    print("=" * 78)

# ============================================================
# 7. ENTRYPOINT
# ============================================================


def main():
    results = benchmark()
    report(results)


if __name__ == "__main__":
    main()
