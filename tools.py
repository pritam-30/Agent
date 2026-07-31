from google.genai import types
from utils import embedding_model
from knowlegdeBase.collection import collection


# ==========================
# Tool Declarations
# ==========================

note_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="save_note",
            description=(
                "Save a note or reminder. "
                "If the user does not provide enough information, "
                "pass an empty string and let the function handle validation."
            ),
            parameters={
                "type": "OBJECT",
                "properties": {
                    "text": {
                        "type": "STRING",
                        "description": "The note or reminder to save."
                    }
                }
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

notes = []


def save_note(text: str):
    """
    Save a note or reminder.
    """

    if not text.strip():
        return {
            "success": False,
            "error": "Missing text",
            "message": "Please tell me what you want me to add."
        }

    notes.append(text)

    return {
        "success": True,
        "note": text
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
# Tool Registry
# ==========================

TOOLS = {
    "save_note": save_note,
    "retrieve_documents": retrieve_documents,
}
