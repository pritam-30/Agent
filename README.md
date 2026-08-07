# Agentic RAG Assistant

A lightweight command-line AI assistant that combines **Gemini Function Calling** with a **Retrieval-Augmented Generation (RAG)** pipeline powered by **ChromaDB**.

The assistant decides whether to answer directly, retrieve information from indexed documents, or save structured notes based on the user's request. It supports multi-tool execution, maintains conversation context during a session, and is built without orchestration frameworks—all planning, tool execution, retrieval, and conversation management are implemented directly in Python.

---

# Features

- 🤖 Agentic tool-calling using Gemini Function Calling
- 📄 Retrieval-Augmented Generation (RAG) with ChromaDB
- 🔍 Multiple retrieval strategies
  - Dense Similarity Search
  - Max Marginal Relevance (MMR)
- 🔄 Configurable retrieval strategy through an environment variable
- 🧠 Semantic search using **BAAI/bge-base-en-v1.5** embeddings
- 📚 PDF parsing, chunking, embedding generation, and indexing
- 🧹 Cleaned knowledge base by removing index and low-information pages before indexing
- 📊 Optional automatic RAG evaluation using DeepEval
- 📝 Structured note creation (title, content, tags, timestamp)
- 🔧 Multiple tool calls within a single planning step when appropriate
- 💬 Multi-turn conversation state during a CLI session
- ⏱ Optional latency profiling for planner, retrieval, and total request execution
- 🧩 Pure Python implementation without LangChain Agents or orchestration frameworks

---

# What the Assistant Can Do

The assistant accepts natural language requests and determines whether it should:

- answer directly using its own knowledge,
- retrieve relevant information from indexed documents,
- create and save structured notes,
- execute multiple independent tools within the same planning cycle.

For example:

- "What does my interview guide say about bagging vs boosting?"
- "Summarize my documents and save the summary."
- "Check my notes and save a reminder."

---

# Architecture

```text
                     User
                       │
                       ▼
                Gemini Planner
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
  No Tool Required            Tool(s) Required
         │                           │
         ▼                           ▼
   Final Response          Execute Tool(s)
                                     │
                                     ▼
                     Append Tool Results to Conversation
                                     │
                                     ▼
                             Gemini Planner
                                     │
                        ┌────────────┴────────────┐
                        │                         │
                        ▼                         ▼
               Additional Tool?            Final Answer
```

Independent tools may be executed within the same planning step, while dependent tools execute only after their required information becomes available.

---

# Retrieval Pipeline

The retrieval layer is modular and designed so that new retrieval algorithms can be added without modifying the agent loop.

```text
retrieve()
      │
      ├──────── Similarity Search
      │
      ├──────── MMR Search
      │
      └──────── Future Retrieval Methods
```

Current retrieval strategies include:

## Dense Similarity Search

- SentenceTransformer embeddings
- Normalized embeddings
- ChromaDB vector similarity
- Fast baseline retrieval

## Max Marginal Relevance (MMR)

- Uses the same ChromaDB collection
- Implemented using LangChain's Chroma wrapper
- Retrieves more diverse yet relevant chunks
- Reduces redundant context
- Configurable using:
  - `k`
  - `fetch_k`
  - `lambda_mult`

The retrieval strategy is selected using:

```env
RETRIEVAL_METHOD=similarity
```

or

```env
RETRIEVAL_METHOD=mmr
```

No changes to the agent loop are required when switching retrieval methods.

---

# Project Structure

```text
RAG/
├── app.py                          # Agent loop and CLI
├── tools.py                        # Tool schemas
├── Eval.py                         # DeepEval integration
├── buildDB.py                      # Build ChromaDB
├── utils.py                        # Shared utilities
│
├── retrieval/
│   ├── retrieve.py                 # Retrieval dispatcher
│   └── retrieval_strategies.py     # Similarity & MMR retrieval
│
├── operations/
│   ├── parser.py                   # PDF parsing
│   ├── chunker.py                  # Chunk generation
│   └── embedding.py                # Embedding generation
│
├── knowledgeBase/
│   └── collection.py               # ChromaDB collection
│
├── chroma_db/
├── .env
└── rag_env/
```

---

# Setup

## 1. Create a virtual environment

```bash
python -m venv rag_env
```

Linux/macOS

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
pip install python-dotenv google-genai sentence-transformers transformers chromadb pymupdf langchain-chroma langchain-huggingface deepeval
```

---

## 3. Configure Environment Variables

Create a `.env` file.

```env
GEMINI_API_KEY=your_api_key

ENABLE_TIMING=False
ENABLE_EVALUATION=False

RETRIEVAL_METHOD=similarity
# similarity | mmr
```

Available retrieval methods

- `similarity`
- `mmr`

Enable latency profiling by setting

```env
ENABLE_TIMING=True
```

Enable automatic evaluation by setting

```env
ENABLE_EVALUATION=True
```

---

## 4. Prepare the Knowledge Base

Place your PDF(s) inside the project and run

```bash
python buildDB.py
```

### Knowledge Base Preparation

Before rebuilding the vector database, the source documents were cleaned to improve retrieval quality.

The preprocessing removes low-information pages such as:

- Index pages
- Reference pages
- Other noisy sections

This reduced the knowledge base from approximately

```
1381 chunks
```

to roughly

```
1229 chunks
```

which significantly reduced retrieval noise and improved grounding quality.

---

## 5. Run the Assistant

```bash
python app.py
```

---

# Example

```text
You:
What does my interview guide say about bagging vs boosting?

Assistant:

According to your indexed documents...

Bagging trains multiple models independently to reduce variance, while boosting trains models sequentially so that each model learns from the mistakes of the previous one.
```

---

```text
You:
Summarize my documents and save the summary.

Assistant:

✓ Retrieved relevant documents
✓ Generated summary
✓ Saved structured note

Title:
ML Interview Guide Summary
```

---

# Structured Notes

Notes are stored as structured objects.

Example

```json
{
  "title": "ML Interview Guide Summary",
  "content": "...",
  "tags": ["ml", "summary"],
  "created_at": "2026-08-04T15:40:21"
}
```

This makes future searching, filtering, editing, and persistence significantly easier.

---

# Automatic Evaluation

The assistant supports optional automatic evaluation of RAG responses using **DeepEval**.

Current metrics include

- Faithfulness
- Answer Relevancy

Evaluation runs automatically after retrieval and final response generation.

It can be enabled with

```env
ENABLE_EVALUATION=True
```

Evaluation failures never interrupt the assistant and are reported only for debugging purposes.

---

# Latency Profiling

Optional latency instrumentation is available.

When enabled (`ENABLE_TIMING=True`) the assistant reports

- Planner latency
- Retrieval latency
- Individual tool latency
- Total request latency

Example

```text
Planner: 1.42 s
retrieve_documents: 2.08 s
Planner: 1.31 s

Total latency: 4.87 s
```

---

# Implementation Notes

Some important implementation details

- Gemini function calls and tool responses are both preserved inside the conversation history.
- Tool schemas remain synchronized with their Python implementations.
- Multiple independent tool calls are executed within a single planning iteration.
- Retrieval strategies are isolated behind a dispatcher, making experimentation simple.
- Embedding normalization is applied consistently during indexing and querying.
- Retrieved document content is treated as the primary source of truth whenever retrieval occurs.
- Previous tool outputs are reused whenever possible to avoid unnecessary retrieval.
- Automatic evaluation runs only after successful retrieval and response generation.

---

# Current Limitations

- CLI-only interface
- Notes are stored only in memory
- Conversation memory exists only during the current session
- Retrieval currently supports dense similarity search and MMR only
- No hybrid retrieval
- No reranking
- DeepEval currently measures Faithfulness and Answer Relevancy only
- Automatic evaluation depends on external LLM calls and API limits

---

# Future Improvements

Potential future enhancements include

- Hybrid Retrieval (BM25 + Vector Search)
- Cross-Encoder Reranking
- Query Rewriting
- Query Expansion
- Hierarchical Document Summarization
- Additional DeepEval metrics
- Persistent note storage (SQLite/PostgreSQL)
- Note retrieval, editing, and deletion
- Streaming responses
- REST API
- Web interface
- Docker support
- Long-term conversational memory

---

# Tech Stack

- Python
- Google Gemini
- ChromaDB
- Sentence Transformers
- BAAI/bge-base-en-v1.5
- LangChain (retrieval utilities only)
- DeepEval
- PyMuPDF

---
