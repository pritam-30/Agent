from google.genai import types

# ==========================
# Tool Declarations
# ==========================


rag_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="retrieve_documents",
            description=(
                "Search the user's indexed knowledge base using semantic search and "
                "retrieve the most relevant document passages for answering the user's "
                "question. Use this tool whenever the answer depends on information "
                "contained in the user's indexed documents. The tool returns only the "
                "most relevant document chunks, not a final answer."
            ),
            parameters={
                "type": "OBJECT",
                "properties": {
                    "query": {
                        "type": "STRING",
                        "description": (
                            "A concise search query that captures the user's information need. "
                            "Preserve important technical terms, names, acronyms, and keywords. "
                            "Do not add unnecessary words or assumptions."
                        )
                    },
                },
                "required": ["query"]
            }
        )
    ]
)
