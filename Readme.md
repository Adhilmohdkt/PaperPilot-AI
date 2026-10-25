# PaperPilot AI 🚀

PaperPilot AI is an advanced, production-grade **Agentic AI Research Assistant** built with **LangGraph**, **FastAPI**, and **React**. It seamlessly unifies **Local Document RAG** with **Live Academic Literature Discovery** (arXiv & OpenAlex), delivering grounded, verified research answers with precise citation attribution.

---

## 🌟 Key Capabilities

- 🤖 **Agentic Multi-Step Workflow (LangGraph)**:
  - **Query Understanding**: Automatic classification of user intent (`academic_research`, `local_rag`, `general_answer`), topic extraction, and recency detection.
  - **Dynamic Routing**: Conditionally routes requests between local vector knowledge bases, external academic paper providers, or conversational synthesis.
  - **Deterministic Ranking & Recency Scoring**: Balances lexical relevance, publication year decay, and citation count to surface the most pertinent and latest (2024–2026) research.
  - **Citation Validation**: Grounded citation tracking distinguishing between user-uploaded papers (`[Source N]`) and external academic literature (`[Paper N]`).

- 📚 **Live Academic Discovery (arXiv & OpenAlex)**:
  - Searches millions of open-access papers in real time.
  - Automatically fetches metadata, abstracts, DOIs, and direct PDF links without permanently cluttering local vector storage.

- 📑 **Credit-Safe Local Document RAG**:
  - Ingests uploaded PDF papers using PyMuPDF and creates semantic embeddings.
  - Hybrid retrieval (BM25 keyword search + Dense Vector Search via Weaviate) with reranking to eliminate hallucinations.
  - Guardrails on document size, chunk count, and page limits to ensure fast, cost-effective vector search.

- 💻 **Modern React UI**:
  - Real-time Server-Sent Events (SSE) token streaming.
  - Full GitHub-Flavored Markdown (GFM) rendering with rich dark-mode tables, blockquotes, code blocks, and structured lists.
  - Persistent conversation management (SQLite backed) with newest-first sidebar ordering.
  - Document focus selector allowing users to target queries to specific uploaded papers.

---

## 🏗️ Architecture & Pipeline Flow

```
                                 ┌─────────────────────────┐
                                 │   User Query / Web UI   │
                                 └────────────┬────────────┘
                                              │ (SSE Stream)
                                 ┌────────────▼────────────┐
                                 │   Query Understanding   │
                                 │   & Intent Classifier   │
                                 └────────────┬────────────┘
                                              │
                      ┌───────────────────────┼───────────────────────┐
                      ▼                       ▼                       ▼
            ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
            │  Academic Search │    │    Local RAG     │    │  General Answer  │
            │ (arXiv/OpenAlex) │    │(Weaviate Hybrid) │    │  (Conversational)│
            └─────────┬────────┘    └─────────┬────────┘    └─────────┬────────┘
                      │                       │                       │
            ┌─────────▼────────┐    ┌─────────▼────────┐              │
            │  Normalize/Rank  │    │ Relevance Check  │              │
            │  (Recency Boost) │    └─────────┬────────┘              │
            └─────────┬────────┘              │                       │
                      │                       │                       │
                      └───────────────────────┼───────────────────────┘
                                              ▼
                                 ┌─────────────────────────┐
                                 │     LLM Generation      │
                                 │    (Groq / Llama 3)     │
                                 └────────────┬────────────┘
                                              │
                                 ┌────────────▼────────────┐
                                 │   Citation Validation   │
                                 └────────────┬────────────┘
                                              │
                                 ┌────────────▼────────────┐
                                 │   Streaming Output UI   │
                                 └─────────────────────────┘
```

---

## 📊 Benchmark & Evaluation Results

PaperPilot AI has been rigorously evaluated across a golden benchmark covering end-to-end agentic routing, live academic search, local vector RAG, and LLM answer quality.

### Summary Metrics

| Evaluation Dimension | Benchmark Metric | Result | Benchmark Scope / Conditions |
| :--- | :--- | :--- | :--- |
| **Local RAG Retrieval** | **Hit@5** | **1.00 (100%)** | 10 realistic RAG queries, 10/10 hits on relevant papers |
| **Local RAG Retrieval** | **Recall@5** | **0.833 (83.3%)** | 9-PDF corpus (555 indexed chunks in `PaperChunk_v2`) |
| **Academic Retrieval** | **Hit@5** | **1.00 (100%)** | 5 multi-domain academic queries across arXiv & OpenAlex |
| **Academic Retrieval** | **Recall@5** | **1.00 (100%)** | Canonical literature ground-truth matching |
| **Structural Agent Routing** | **Check Pass Rate** | **59 / 60 (98.3%)** | 10 multi-category golden test cases (intent, nodes, citations) |
| **Answer Relevancy (DeepEval)** | **Grounded Relevancy** | **1.00 (100%)** | Measured using Groq-backed LLM evaluator |
| **Faithfulness (DeepEval)** | **Hallucination Resistance** | **0.67 (66.7%)** | Factual consistency against retrieved context |
| **Contextual Relevancy (DeepEval)** | **Information Density** | **0.19 (19.0%)** | Strict chunk signal-to-noise ratio over multi-page papers |
| **Unit Test Suite** | **Pytest Structural** | **5 / 5 (100%)** | State machines, node isolation, and router edge cases |

Full reproducible report data is archived in [`backend/tests/evaluation_report.json`](backend/tests/evaluation_report.json).

---

## 🛠️ Tech Stack & Key Enhancements

### Frontend
- **Framework**: React 19
- **Markdown & Tables**: `react-markdown`, `remark-gfm`, `rehype-raw`
- **Styling**: Modern dark-mode Glassmorphism CSS design system
- **Communication**: Native `fetch` with `ReadableStream` for real-time SSE

### Backend & Agent Workflow
- **Framework**: FastAPI, Uvicorn, Pydantic v2
- **Agent Orchestration**: LangGraph, LangChain Core
- **LLM Inference**: Groq API (`openai/gpt-oss-120b`, `llama-3.1-8b-instant`)
- **Resilient Multi-Tier Embeddings**: Google Gemini Embeddings with automatic fallback to local HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors) for 100% offline-tolerant retrieval.
- **Academic Search**: Official `arxiv` client, OpenAlex REST API with query sanitization (stripping arXiv syntax to prevent HTTP 400s) and polite-pool compliance (`User-Agent: mailto:...`).
- **Concurrent PDF Ingestion**: Asynchronous multi-paper downloading via `asyncio.gather`.
- **Vector Database**: Weaviate (BM25 + Semantic Hybrid Search, active staging collection `PaperChunk_v2`).
- **Persistence**: SQLite conversation store

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm
- Docker Desktop (for Weaviate vector database)
- Groq API Key

### 1. Vector Database Setup
Run Weaviate using Docker Compose:
```bash
docker-compose up -d weaviate
```

### 2. Backend Setup
```bash
cd backend

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
# Create a .env file in backend/ with:
# GROQ_API_KEY=your_groq_api_key_here
# WEAVIATE_URL=http://localhost:8080
# OPENALEX_EMAIL=your_email@example.com (optional, for polite pool)

# (Optional) Re-index local PDFs with 384-dim all-MiniLM-L6-v2 embeddings
python scripts/reindex_v2.py

# Start the FastAPI server
python -m uvicorn main:app --reload --port 8000
```

### 3. Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm start
```
Open your browser at **http://localhost:3000**.

---

## 🧪 Testing & Evaluation

### Run Structural Pytest Suite
```bash
cd backend
python -m pytest tests/structural/
```

### Run Full End-to-End Evaluation Benchmark
Runs all 60 structural checks, 10-query Local RAG benchmark, academic retrieval, and DeepEval metrics:
```bash
cd backend
python tests/run_full_evaluation.py
```
Results will update `backend/tests/evaluation_report.json`.

---

## ⚠️ Known Limitations

1. **Groq Free-Tier Token Limits**: Groq's free tier imposes a 200,000 Tokens Per Day (TPD) ceiling. Heavy multi-turn benchmark suites or consecutive DeepEval runs may encounter temporary HTTP 429 rate limits.
2. **Cold-Start Embedding Latency**: The first execution of local `all-MiniLM-L6-v2` embeddings incurs a 2–4 second warm-up time while the HuggingFace model weights load into CPU memory.
3. **Contextual Relevancy in DeepEval**: Academic research papers frequently contain extensive mathematical proofs and bibliography sections, which lower the strict sentence-level contextual relevancy metric despite successful overall grounding.

---

## 📄 License

MIT License. Designed and built for transparent, reproducible academic research.
