import os

from google import genai
from google.genai import types
from dotenv import load_dotenv


load_dotenv()


# ============================================================
# Gemini Client
# ============================================================

gen_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# ============================================================
# Query Rewriter Prompt
# ============================================================
QUERY_REWRITER_PROMPT = """
You are a query rewriting component for a Retrieval-Augmented
Generation (RAG) system.

The knowledge base contains technical material primarily about:

- Machine Learning
- Deep Learning
- Neural Networks
- TensorFlow
- Keras
- Scikit-Learn
- Optimization
- Statistical and mathematical concepts used in machine learning

Your job is to rewrite the user's question into ONE concise,
semantically rich search query for this knowledge base.

Rules:

1. Preserve the user's original intent.

2. Interpret ambiguous terms according to the knowledge-base domain
   when the context strongly suggests a domain-specific meaning.

3. Preserve important technical terms, names, acronyms, and concepts.

4. Do not answer the question.

5. Do not introduce unrelated domains.

6. Do not invent specific concepts that are not reasonably implied
   by the query.

7. If the original query is already a good search query, keep it
   close to the original instead of unnecessarily rewriting it.

8. Remove conversational filler.

9. Return exactly ONE search query.

10. Return ONLY the rewritten query.
"""


# ============================================================
# Query Rewriter
# ============================================================

def rewrite_query(query: str) -> str:
    """
    Rewrite a user query into a retrieval-optimized query.

    Args:
        query: Original user query.

    Returns:
        A single rewritten query suitable for vector retrieval.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    response = gen_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=query.strip()
                    )
                ],
            )
        ],
        config=types.GenerateContentConfig(
            system_instruction=QUERY_REWRITER_PROMPT,
            temperature=0.0,
        ),
    )

    rewritten_query = response.text.strip()

    if not rewritten_query:
        raise RuntimeError(
            "Query rewriter returned an empty query."
        )

    return rewritten_query


if __name__ == "__main__":

    queries = [
        "Neural nets"
    ]

    for query in queries:

        rewritten = rewrite_query(query)

        print("=" * 80)
        print("Original:")
        print(query)

        print("\nRewritten:")
        print(rewritten)
