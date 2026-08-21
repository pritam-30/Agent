import os
from .retrieval_strategies import (
    similarity_search,
    mmr_search,
)
from utils import notes
from datetime import datetime
from .query_rewriter import rewrite_query
from .retrieval_strategies import hybrid_search

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
# Tool Implementations
# ==========================

def save_note(title: str, content: str, tags: list[str] | None = None):
    """
    Save a structured note.

    Args:
        title: Title of the note.
        content: Main note content.
        tags: Optional list of tags.
    """

    if not content.strip():
        return {
            "success": False,
            "error": "Missing content",
            "message": "Please tell me what you want me to save."
        }

    note = {
        "title": title.strip() if title else "Untitled",
        "content": content.strip(),
        "tags": tags or [],
        "created_at": datetime.now().isoformat()
    }

    notes.append(note)

    return {
        "success": True,
        "note": note
    }


# ==========================
# Tool Registry
# ==========================


TOOLS = {
    "save_note": save_note,
    "retrieve_documents": retrieve
}
