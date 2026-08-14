from sentence_transformers import CrossEncoder


# ============================================================
# Reranker Model
# ============================================================

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

reranker = CrossEncoder(RERANKER_MODEL)


# ============================================================
# Rerank Documents
# ============================================================

def rerank_documents(
    query: str,
    documents: list[str],
    metadatas: list[dict],
    top_k: int = 3,
):
    """
    Rerank retrieved documents using a cross-encoder.

    Args:
        query: Original user query.
        documents: Candidate documents retrieved from the vector store.
        metadatas: Metadata corresponding to each document.
        top_k: Number of documents to return after reranking.

    Returns:
        Reranked documents and their corresponding metadata.
    """

    if not documents:
        return [], []

    # --------------------------------------------------------
    # Create query-document pairs
    # --------------------------------------------------------

    pairs = [
        (query, document)
        for document in documents
    ]

    # --------------------------------------------------------
    # Score each query-document pair
    # --------------------------------------------------------

    scores = reranker.predict(pairs)

    # --------------------------------------------------------
    # Keep document + metadata + score together
    # --------------------------------------------------------

    ranked = sorted(
        zip(documents, metadatas, scores),
        key=lambda x: x[2],
        reverse=True,
    )

    # --------------------------------------------------------
    # Select top-k
    # --------------------------------------------------------

    ranked = ranked[:top_k]

    reranked_documents = [
        document
        for document, metadata, score in ranked
    ]

    reranked_metadatas = [
        metadata
        for document, metadata, score in ranked
    ]

    return reranked_documents, reranked_metadatas
