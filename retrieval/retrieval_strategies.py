from knowledgeBase.collection import get_collection
from utils import embedding_model

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


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
# Similarity Search
# ============================================================

def similarity_search(
    query: str,
    k: int = 3,
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
        n_results=k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return documents, metadatas, distances


# ============================================================
# MMR Search
# ============================================================

def mmr_search(
    query: str,
    k: int = 3,
    fetch_k: int = 10,
    lambda_mult: float = 0.5,
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

    return documents, metadatas
