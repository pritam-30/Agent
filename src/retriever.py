import os
from .retrieval_strategies import (
    similarity_search,
    mmr_search,
    hybrid_search
)

from .query_rewriter import rewrite_query

RETRIEVAL_METHOD = os.getenv(
    "RETRIEVAL_METHOD",
    "similarity",
)


def retrieve(
    query: str,
    k: int = 3,
):
    """
    Dispatch retrieval to the selected strategy.
    """

    if RETRIEVAL_METHOD == "similarity":
        query = rewrite_query(query)
        print(f"Rewritten query: {query}")
        documents, metadatas = similarity_search(
            query=query,
            k=k,
        )
        return documents, metadatas

    elif RETRIEVAL_METHOD == "mmr":
        query = rewrite_query(query)
        print(f"Rewritten query: {query}")
        documents, metadatas = mmr_search(
            query=query,
            k=k,
        )
        return documents, metadatas

    elif RETRIEVAL_METHOD == "hybrid":
        query = rewrite_query(query)
        print(f"Rewritten query: {query}")
        documents, metadatas = hybrid_search(query=query, k=k)
        return documents, metadatas

    else:
        raise ValueError(
            f"Unknown retrieval method: {RETRIEVAL_METHOD}"
        )


# ==========================
# Tool Registry
# ==========================

TOOLS = {
    "retrieve_documents": retrieve
}
