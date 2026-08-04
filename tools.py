from pprint import pprint
from google.genai import types
from utils import embedding_model, notes
from knowledgeBase.collection import collection
from datetime import datetime


# ==========================
# Tool Declarations
# ==========================

note_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="save_note",
            description=(
                "Save a structured note."
                " Create a concise title, detailed content, and relevant tags."
                " Use tags only when they add meaningful organization."
            ),
            parameters={
                "type": "OBJECT",
                "properties": {
                    "title": {
                        "type": "STRING",
                        "description": "A short title for the note."
                    },
                    "content": {
                        "type": "STRING",
                        "description": "The main content of the note."
                    },
                    "tags": {
                        "type": "ARRAY",
                        "items": {
                            "type": "STRING"
                        },
                        "description": "Optional tags for organizing the note."
                    }
                },
                "required": ["title", "content"]
            }
        )
    ]
)


rag_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="retrieve_documents",
            description=(
                "Search the user's indexed documents and return the most "
                "relevant information needed to answer the user's question."
            ),
            parameters={
                "type": "OBJECT",
                "properties": {
                    "query": {
                        "type": "STRING",
                        "description": (
                            "A concise semantic search query derived from the user's request."
                        )
                    }
                },
                "required": ["query"]
            }
        )
    ]
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


def retrieve_documents(query: str):
    """
    Retrieve the most relevant document chunks from ChromaDB.
    """

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    )

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3
    )

    return results["documents"][0]


# ==========================
# Utility Functions
# ==========================

def show_notes():
    print("\n========== Notes ==========")
    pprint(notes)
    print("===========================\n")

# ==========================
# Tool Registry
# ==========================


TOOLS = {
    "save_note": save_note,
    "retrieve_documents": retrieve_documents,
}
