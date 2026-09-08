from knowledgeBase.collection import get_collection
from utils import embedding_model
from .reranker import rerank_documents

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from rank_bm25 import BM25Okapi
import re


# ============================================================
# Chroma Collection
# ============================================================

collection = get_collection()


# ============================================================
# LangChain VectorStore
# ============================================================

embedding_function = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    encode_kwargs={
        "normalize_embeddings": True,
    },
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    collection_name="documents",
    embedding_function=embedding_function,
)


# ============================================================
# BM25 Corpus
# ============================================================

def _tokenize(text: str) -> list[str]:
    """
    Simple tokenizer for BM25.
    """

    return re.findall(r"\b\w+\b", text.lower())


bm25_data = collection.get(
    include=[
        "documents",
        "metadatas",
    ],
)

bm25_documents = bm25_data["documents"]
bm25_metadatas = bm25_data["metadatas"]

bm25_tokenized_documents = [
    _tokenize(document)
    for document in bm25_documents
]

bm25 = BM25Okapi(bm25_tokenized_documents)


# ============================================================
# Similarity Search
# ============================================================

def similarity_search(
    query: str,
    k: int = 5,
    candidate_k: int = 10,
    rerank: bool = True,
):
    """
    Dense vector similarity search using ChromaDB.
    """

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    if rerank:
        documents, metadatas = rerank_documents(
            query=query,
            documents=documents,
            metadatas=metadatas,
            top_k=k,
        )

    return documents, metadatas


# ============================================================
# MMR Search
# ============================================================

def mmr_search(
    query: str,
    k: int = 5,
    fetch_k: int = 10,
    lambda_mult: float = 0.5,
    rerank: bool = True,
):
    """
    Max Marginal Relevance retrieval.
    """

    docs = vectorstore.max_marginal_relevance_search(
        query=query,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
    )

    documents = []
    metadatas = []

    for doc in docs:
        documents.append(doc.page_content)
        metadatas.append(doc.metadata)

    if rerank:
        documents, metadatas = rerank_documents(
            query=query,
            documents=documents,
            metadatas=metadatas,
            top_k=k,
        )

    return documents, metadatas


# ============================================================
# BM25 Search
# ============================================================

def bm25_search(
    query: str,
    k: int = 10,
):
    """
    BM25 lexical retrieval over the entire knowledge base.
    """

    query_tokens = _tokenize(query)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:k]

    documents = [
        bm25_documents[i]
        for i in ranked_indices
    ]

    metadatas = [
        bm25_metadatas[i]
        for i in ranked_indices
    ]

    return documents, metadatas


# ============================================================
# Reciprocal Rank Fusion
# ============================================================

def reciprocal_rank_fusion(
    ranked_results: list[tuple[list[str], list[dict]]],
    rrf_k: int = 60,
):
    """
    Combine multiple ranked result lists using
    Reciprocal Rank Fusion (RRF).
    """

    scores = {}
    result_lookup = {}

    for documents, metadatas in ranked_results:

        for rank, (document, metadata) in enumerate(
            zip(documents, metadatas),
            start=1,
        ):

            # Identify the chunk using its metadata.
            chunk_id = (
                metadata.get("source"),
                metadata.get("chunk_index"),
            )

            scores[chunk_id] = (
                scores.get(chunk_id, 0.0)
                + 1 / (rrf_k + rank)
            )

            result_lookup[chunk_id] = (
                document,
                metadata,
            )

    ranked_chunks = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    documents = []
    metadatas = []

    for chunk_id in ranked_chunks:

        document, metadata = result_lookup[chunk_id]

        documents.append(document)
        metadatas.append(metadata)

    return documents, metadatas


# ============================================================
# Hybrid Search
# ============================================================

def hybrid_search(
    query: str,
    k: int = 5,
    candidate_k: int = 10,
    rrf_k: int = 60,
    rerank: bool = True,
):
    """
    Hybrid retrieval combining:

    1. Dense vector similarity
    2. BM25 lexical search
    3. Reciprocal Rank Fusion
    4. Cross-encoder reranking
    """

    # --------------------------------------------------------
    # Dense retrieval
    # --------------------------------------------------------

    vector_documents, vector_metadatas = similarity_search(
        query=query,
        k=k,
        candidate_k=candidate_k,
        rerank=False,
    )

    # --------------------------------------------------------
    # BM25 retrieval
    # --------------------------------------------------------

    bm25_documents_result, bm25_metadatas_result = bm25_search(
        query=query,
        k=candidate_k,
    )

    # --------------------------------------------------------
    # Reciprocal Rank Fusion
    # --------------------------------------------------------

    fused_documents, fused_metadatas = reciprocal_rank_fusion(
        [
            (
                vector_documents,
                vector_metadatas,
            ),
            (
                bm25_documents_result,
                bm25_metadatas_result,
            ),
        ],
        rrf_k=rrf_k,
    )

    # --------------------------------------------------------
    # Cross-encoder reranking
    # --------------------------------------------------------

    if rerank:
        documents, metadatas = rerank_documents(
            query=query,
            documents=fused_documents,
            metadatas=fused_metadatas,
            top_k=k,
        )

    return documents, metadatas
