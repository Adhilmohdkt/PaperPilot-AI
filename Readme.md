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

## 🛠️ Tech Stack

### Frontend
- **Framework**: React 19
- **Markdown & Tables**: `react-markdown`, `remark-gfm`, `rehype-raw`
- **Styling**: Modern dark-mode Glassmorphism CSS design system
- **Communication**: Native `fetch` with `ReadableStream` for real-time SSE

### Backend & Agent Workflow
- **Framework**: FastAPI, Uvicorn, Pydantic v2
- **Agent Orchestration**: LangGraph, LangChain Core
- **LLM Inference**: Groq API (`openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `llama-3.1`)
- **Academic Search**: Official `arxiv` client, OpenAlex REST API
- **Document Processing**: PyMuPDF (`fitz`), Sentence-Transformers
- **Vector Database**: Weaviate (BM25 + Semantic Hybrid Search)
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

## 🧪 Testing

Run backend structural and workflow tests:
```bash
cd backend
python -m pytest tests/structural
```

---

## 📄 License

MIT License. Designed and built for transparent, reproducible academic research.
