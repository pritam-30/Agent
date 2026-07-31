# Agentic RAG Assistant

A lightweight CLI assistant that combines Gemini function-calling with a document retrieval pipeline backed by ChromaDB. The agent can answer directly, retrieve relevant information from indexed documents, or save notes/reminders depending on the user request.

This project is intentionally simple and explicit: there is no orchestration framework involved. The tool selection logic, agent loop, and conversation state handling are all implemented directly in Python.

## What it does

You can ask the assistant questions in plain English, and it will decide whether it needs to:

- answer directly from its own knowledge,
- retrieve relevant content from your indexed documents, or
- save a note/reminder for later.

It can also call multiple tools in the same turn when needed. For example, a request like “check my docs and save a note about it” can trigger retrieval and note-saving in one round trip.

## How it works

1. The user sends a message.
2. Gemini decides whether a tool is needed.
3. If needed, the assistant calls one or more tools.
4. The tool results are appended back into the conversation history.
5. The loop continues until the assistant can produce a final answer.

## Project structure

```text
RAG/
├── app.py                  # CLI loop and agent loop
├── tools.py                # Tool schemas, implementations, and tool registry
├── utils.py                # Shared utilities such as tokenizer and embedding model
├── buildDB.py              # Build/rebuild the vector database
├── operations/
│   ├── parser.py           # PDF parsing
│   ├── chunker.py          # Chunking logic
│   └── embedding.py        # Embedding generation
├── knowlegdeBase/
│   └── collection.py       # ChromaDB collection setup and access
├── chroma_db/              # Persisted vector store data
├── .env                    # GEMINI_API_KEY (not committed)
└── rag_env/                # Local virtual environment
```

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv rag_env
source rag_env/bin/activate
```

2. Install the required packages:

```bash
pip install python-dotenv google-genai langchain-community langchain-text-splitters sentence-transformers transformers chromadb pymupdf
```

3. Create a `.env` file in the project root with your Gemini API key:

```bash
GEMINI_API_KEY=your-key-here
```

4. Place a PDF file in the project root (or update `PDF_PATH` in `buildDB.py`) and build the vector database:

```bash
python buildDB.py
```

5. Run the assistant:

```bash
python app.py
```

## Example

```text
You: What does my guide say about bagging vs boosting?
Assistant: According to your indexed document, bagging trains models in parallel and reduces variance, while boosting trains sequentially and focuses on correcting earlier mistakes.

You: Save a note to review this before my interview.
Assistant: Saved.
```

## Notes on implementation

The most important implementation details are straightforward but easy to get wrong:

- The conversation history must include both the model’s function-call turn and the corresponding tool responses.
- Tool schemas and Python function signatures need to stay aligned so the model can call them correctly.
- The loop should collect all function calls returned in a single turn rather than only the first one.

## Limitations and roadmap

- The current setup is a local CLI only.
- Retrieval is currently based on vector similarity and does not yet include hybrid search or reranking.
- There is no automated evaluation harness yet.
- Multi-turn memory within a single CLI session is still fairly simple.

## Next steps

Possible next steps include:

- adding an evaluation pipeline for retrieval and answer quality,
- improving retrieval with hybrid search or reranking,
- exposing the assistant through a web API, and
- containerizing the app for easier deployment.
# Agent
