# Agentic RAG Assistant

A lightweight command-line AI assistant that combines **Gemini Function Calling** with a **Retrieval-Augmented Generation (RAG)** pipeline powered by **ChromaDB**.

The assistant decides whether to answer directly, retrieve information from indexed documents, or save structured notes based on the user's request. It supports multi-tool execution, maintains conversation context during a session, and is built without orchestration frameworks—all planning, tool execution, retrieval, and conversation management are implemented directly in Python.

---

# Features

- 🤖 Agentic tool-calling using Gemini Function Calling
- 📄 Retrieval-Augmented Generation (RAG) with ChromaDB
- 🔍 Multiple retrieval strategies
  - Dense Similarity Search
  - Hybrid Search
  - Max Marginal Relevance (MMR)
- 🔄 Configurable retrieval strategy through an environment variable
- 🧠 Semantic search using **BAAI/bge-base-en-v1.5** embeddings
- 📚 PDF parsing, chunking, embedding generation, and indexing
- 🧹 Cleaned knowledge base by removing index and low-information pages before indexing
- ✏️ Query rewriting for retrieval-oriented search queries
- 🎯 Cross-encoder reranking of retrieved candidates
- 📊 Optional automatic RAG evaluation using DeepEval
- 📝 Structured note creation (title, content, tags, timestamp)
- 🔧 Multiple tool calls within a single planning step when appropriate
- 💬 Multi-turn conversation state during a CLI session
- ⏱ Optional latency profiling for planner, retrieval, and total request execution
- 🧩 Pure Python implementation without LangChain Agents or orchestration frameworks

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
User Query
    │
    ▼
Query Rewriting
    │
    ▼
Candidate Retrieval
    │
    ├── Dense Similarity Search
    │
    └── MMR Search
    │
    ├── Hybrid Search
    │
    │
    ▼
Cross-Encoder Reranking
    │
    ▼
Top-K Context
    │
    ▼
Gemini Planner
```

Current retrieval strategies include:

## Dense Similarity Search

- SentenceTransformer embeddings
- Normalized embeddings
- ChromaDB vector similarity
- Fast baseline retrieval

## Hybrid

- Combines dense vector similarity and BM25 lexical retrieval.
- Uses Reciprocal Rank Fusion (RRF) to merge ranked lists from both methods.
- Applies a cross-encoder reranker to produce the final Top-K context.
- Configurable parameters: `k`, `candidate_k`, and `rrf_k`.

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

or

```env
RETRIEVAL_METHOD=hybrid
```

No changes to the agent loop are required when switching retrieval methods.

---

# Project Structure

```text
RAG/
├── .deepeval/
├── .env
├── .git/
├── .gitignore
├── .vscode/
├── Eval.py
├── README.md
├── app.py
├── buildDB.py
├── chroma_db/
│   ├── chroma.sqlite3
│   └── 6e3e3cae-19ce-4fd7-90ef-195ad60a45b1/
├── debug_DB.py
├── eval_results.jsonl
├── requirements.txt
├── knowledgeBase/
│   └── collection.py
├── operations/
│   ├── chunker.py
│   ├── embedding.py
│   └── parser.py
├── retrieval/
│   ├── query_rewriter.py
│   ├── reranker.py
│   ├── retrieval_strategies.py
│   └── retrieve.py
├── rag_env/            # virtual environment (ignored)
├── test.py
├── tools.py
└── utils.py
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
- `hybrid`

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

The RAG pipeline was evaluated using **DeepEval** on 23 question-answering
test cases based on the indexed book.

Metrics:

- Faithfulness
- Answer Relevancy
- Correctness
- Contextual Precision

## Evaluation Results

| Metric                   | Average Score |  Pass Rate |
| ------------------------ | ------------: | ---------: |
| **Faithfulness**         |     **0.993** | **100.0%** |
| **Correctness**          |     **0.935** |  **91.3%** |
| **Answer Relevancy**     |     **0.898** |  **82.6%** |
| **Contextual Precision** |     **0.812** |  **78.3%** |

The evaluation threshold was **0.70**.

The evaluation helped identify two main areas for improvement:

- **Contextual Precision:** closely related concepts can sometimes result in less precise retrieval rankings.
- **Answer Relevancy:** the generation model can occasionally include additional information beyond the scope of a narrowly phrased question.

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
- Query rewriting is performed before retrieval when appropriate.
- MMR can be used to improve diversity among retrieved candidates.
- Retrieved candidates can be reranked using a cross-encoder before being passed to the generation model.
- Embedding normalization is applied consistently during indexing and querying.
- Retrieved document content is treated as the primary source of truth whenever retrieval occurs.
- Previous tool outputs are reused whenever possible to avoid unnecessary retrieval.
- Automatic evaluation runs only after successful retrieval and response generation.

---

# Current Limitations

- CLI-only interface
- Notes are stored only in memory
- Conversation memory exists only during the current session
- Automatic evaluation depends on external LLM calls and API limits

---

# Future Improvements

Potential future enhancements include

- Hierarchical Document Summarization
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
- cross-encoder/ms-marco-MiniLM-L-6-v2
- LangChain (retrieval utilities only)
- DeepEval
- PyMuPDF

---
