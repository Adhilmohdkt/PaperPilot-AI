# PaperPilot AI 🚀

PaperPilot AI is a production-ready, full-stack **Retrieval-Augmented Generation (RAG)** application designed to serve as a highly intelligent, interactive Research Assistant. 

It allows users to upload PDF research papers and ask complex questions about them. The system responds with high-quality synthesized answers and provides inline source citations, ensuring transparency and accuracy.

---

## 🏗️ Overall Architecture

The application is built on a decoupled, microservice-inspired architecture using **Docker Compose** to orchestrate three main services:

1. **Frontend Application**: A responsive, interactive user interface that streams AI responses in real-time.
2. **Backend Engine**: A high-performance Python API that handles file ingestion, advanced hybrid retrieval, and LLM text generation.
3. **Vector Database**: A local instance of Weaviate that securely stores and indexes document chunks for semantic search.

### The Pipeline Flow

1. **Ingestion (`POST /upload`)**: 
   When a PDF is uploaded, the backend reads it using `PyMuPDF`, cleans the text, and chunks it into manageable pieces. Each chunk is passed through a local `Sentence-Transformers` model to generate mathematical embeddings. Both the text (for keyword search) and vectors (for semantic search) are inserted into Weaviate.

2. **Hybrid Retrieval (`POST /retrieve`)**: 
   When a query is asked, the backend converts the query to a vector. It queries Weaviate using a **Hybrid Search** approach (BM25 + vector search) to fetch the top 15 most relevant chunks. It also respects the `source_filter` if a specific document is targeted.

3. **Reranking**:
   The initial chunks are passed to the **Cohere Reranker** (v3.0), which acts as a precision filter. It re-orders the chunks and selects the top 5 most contextually relevant pieces of text to reduce AI hallucinations.

4. **Generation & Streaming (`POST /stream`)**: 
   The retrieved chunks and the user's conversational history are packaged into a well-crafted prompt. This prompt is sent to the LLM (Llama 3.1 via Groq) which acts as the synthesis engine. The generator outputs text token-by-token directly to the frontend using **Server-Sent Events (SSE)**.

---

## 💻 Tech Stack

### Frontend (User Interface)
- **Framework**: React.js 
- **Styling**: Vanilla CSS featuring a premium, "Glassmorphism" design system (blurred panels, vibrant gradients).
- **Communication**: Native `fetch` API for REST calls and `ReadableStream` for parsing continuous Server-Sent Events (SSE).

### Backend (Retrieval & Generation)
- **Framework**: FastAPI (Python) & Uvicorn (ASGI web server).
- **Extraction**: `PyMuPDF` (fitz) for reliable text extraction from complex PDFs.
- **Embeddings Model**: `sentence-transformers` (`all-MiniLM-L6-v2`) running locally for fast, privacy-first vector generation.
- **Generation Model**: `Groq` API running `llama-3.1-8b-instant` for blisteringly fast token generation.
- **Reranker Engine**: `Cohere` API (`rerank-english-v3.0`) for boosting search accuracy.

### Storage & Infrastructure
- **Vector Database**: Weaviate `v1.37.0` (Dockerized). Supports exact keyword match (BM25) and semantic vector search in a single hybrid query.
- **Orchestration**: `docker-compose` managing network bridges and volume mounts.
- **Web Server (Frontend)**: `Nginx` (Alpine) serving static React build files.

### Evaluation & Testing
- **Framework**: `ragas` (Retrieval-Augmented Generation Assessment).
- **Purpose**: An automated script (`evaluate.py`) that scores the pipeline on metrics such as *Faithfulness*, *Context Precision*, and *Answer Relevancy* to ensure the architecture doesn't degrade over time.

---

## ✨ Core Features

* **Dynamic Document Uploading**: Directly drag and drop PDFs into the UI to ingest new knowledge instantly.
* **Query Focus (Document Filtering)**: Seamlessly toggle between searching your *entire* knowledge base or locking the AI's attention onto a single, specific research paper to avoid context bleeding.
* **Real-time Token Streaming**: Get instant gratification with ChatGPT-like streaming answers utilizing blazing-fast Llama-3 models.
* **Inline PDF Citations**: Every generated answer is paired with a list of the exact document chunks it read to form its logic, complete with clickable links to download or view the unedited PDF.
* **Conversational Memory**: The system remembers your previous chat interactions within a session, allowing for natural follow-up questions.

---

## 🚀 Getting Started

To spin up the entire application:

1. Ensure you have Docker Desktop running.
2. In the root directory, run:
```bash
docker-compose up --build
```
3. Once the terminal indicates that Uvicorn and Weaviate are running, open your browser and navigate to **http://localhost:3000**.
4. Upload a document using the left sidebar and start researching!
