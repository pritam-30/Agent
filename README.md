# Agentic RAG Assistant

A lightweight command-line AI assistant that combines **Gemini function calling** with a **Retrieval-Augmented Generation (RAG)** pipeline powered by **ChromaDB**.

The assistant decides whether to answer directly, retrieve information from indexed documents, or save structured notes based on the user's request. It supports multi-tool execution, maintains conversation context during a session, and is built without any orchestration frameworks—all planning, tool execution, and conversation management are implemented directly in Python.

---

# Features

- 🤖 Agentic tool-calling using Gemini Function Calling
- 📄 Retrieval-Augmented Generation (RAG) with ChromaDB
- 🔍 Semantic document search using BAAI/bge-base-en-v1.5 embeddings
- 📚 PDF ingestion, chunking, embedding generation, and indexing
- 📝 Structured note creation (title, content, tags, timestamp)
- 🔧 Multiple tool calls within a single planning step when appropriate
- 💬 Multi-turn conversation state within a CLI session
- ⏱ Optional latency profiling for planner, tools, and total request execution
- 🧩 Pure Python implementation without LangChain agents or orchestration frameworks

---

# What the Assistant Can Do

The assistant accepts natural language requests and determines whether it should:

- answer directly using its own knowledge,
- retrieve relevant information from indexed documents,
- create and save structured notes,
- or execute multiple independent tools within the same planning cycle.

For example:

- "What does my interview guide say about bagging vs boosting?"
- "Summarize my documents and save the summary."
- "Check my docs and save a note about them."

---

# Architecture

The overall workflow is:

```text
User
   │
   ▼
Gemini Planner
   │
   ├── No tool required
   │       │
   │       ▼
   │   Final Answer
   │
   └── Tool(s) Required
           │
           ▼
   Execute One or More Tools
           │
           ▼
 Append Tool Results to Conversation
           │
           ▼
 Gemini Planner
           │
           ├── More dependent tools required
           │
           └── Final Answer
```

Independent tools may be executed in a single planning step, while dependent tools are executed only after the required information becomes available.

---

# Project Structure

```text
RAG/
├── app.py                  # CLI application and agent loop
├── tools.py                # Tool schemas, implementations, and registry
├── utils.py                # Shared utilities, embeddings, tokenizer, timers
├── buildDB.py              # Builds the ChromaDB vector database
├── operations/
│   ├── parser.py           # PDF parsing
│   ├── chunker.py          # Document chunking
│   └── embedding.py        # Embedding generation
├── knowledgeBase/
│   └── collection.py       # ChromaDB collection setup
├── chroma_db/              # Persisted vector database
├── .env                    # Environment variables (not committed)
└── rag_env/                # Local virtual environment
```

---

# Setup

## 1. Create a virtual environment

```bash
python -m venv rag_env
```

Linux / macOS

```bash
source rag_env/bin/activate
```

Windows

```bash
rag_env\Scripts\activate
```

---

## 2. Install dependencies

```bash
pip install python-dotenv google-genai langchain-community langchain-text-splitters sentence-transformers transformers chromadb pymupdf
```

---

## 3. Configure environment variables

Create a `.env` file:

```env
GEMINI_API_KEY=your-api-key
ENABLE_TIMING=False
```

Set `ENABLE_TIMING=True` if you want to profile planner, tool, and total request latency.

---

## 4. Build the vector database

Place your PDF(s) in the project (or update the path in `buildDB.py`) and run:

```bash
python buildDB.py
```

---

## 5. Run the assistant

```bash
python app.py
```

---

# Example

```text
You: What does my guide say about bagging vs boosting?

Assistant:
According to your indexed documents, bagging trains multiple models independently in parallel to reduce variance, while boosting trains models sequentially, allowing each model to correct errors made by previous ones.

You: Summarize my documents and save the summary.

Assistant:
✓ Retrieved relevant documents
✓ Generated a structured summary
✓ Saved a structured note

Title:
ML Interview Guide Summary
```

---

# Structured Notes

Notes are stored as structured objects rather than plain text.

Example:

```json
{
  "title": "ML Interview Guide Summary",
  "content": "Summary of the retrieved interview guide...",
  "tags": ["ml", "interview", "summary"],
  "created_at": "2026-08-04T15:40:21"
}
```

This structure makes future searching, filtering, and management significantly easier.

---

# Latency Profiling

Optional latency instrumentation is available for development and debugging.

When enabled (`ENABLE_TIMING=True`), the assistant reports execution time for:

- Planner (Gemini API)
- Individual tool calls
- Total end-to-end request latency

Example:

```text
Planner: 1.32 s
retrieve_documents: 2.11 s
save_note: 0.00 s
Planner: 1.47 s
Total latency: 4.92 s
```

---

# Implementation Notes

Some important implementation details:

- The complete conversation history includes both Gemini's function-call messages and the corresponding tool responses.
- Tool schemas must remain synchronized with their Python function signatures.
- Multiple tool calls returned within a single planner response are executed sequentially before returning control to Gemini.
- The assistant avoids redundant retrieval when the required information already exists in the current conversation history.

---

# Current Limitations

- CLI-only interface
- Notes are currently stored in memory and are not persisted across sessions
- Retrieval uses vector similarity only (no reranking or hybrid retrieval)
- Conversation memory exists only for the current session
- No automated evaluation pipeline yet

---

# Future Improvements

Possible future enhancements include:

- Persistent note storage (SQLite or another database)
- Note retrieval, editing, and deletion
- Hybrid retrieval and reranking
- Retrieval quality evaluation
- Streaming responses
- REST API or web interface
- Docker containerization
- Long-term memory management for extended conversations

---
