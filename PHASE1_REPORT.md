# Phase 1 — Core Agent Foundation Report (with Corrections)

## Architecture Delivered

| Component | Status |
|-----------|--------|
| **LangGraph workflow** | StateGraph with router + conditional edges (START → query_understanding → router → response_generation → END) |
| **AgentState** | `conversation_id`, `user_query`, `messages` (List[BaseMessage]), `intent`, `response` |
| **Query understanding** | LangChain ChatPromptTemplate + ChatGroq → structured intent ("general_answer"/"local_rag"/"academic_research") |
| **Router** | LangGraph conditional edges, only `general_answer` path implemented in Phase 1 |
| **Memory / Persistence** | LangGraph `InMemorySaver` for short-term + SQLite `conversations.db` (stdlib) for cross-process persistence |
| **LLM Runtime** | **Groq** `openai/gpt-oss-20b` via `langchain_groq.ChatGroq` + `GROQ_API_KEY` from .env |
| **FastAPI** | 6 endpoints: GET /health, POST /conversations, GET /conversations, GET /conversations/{id}, POST /chat, POST /chat/stream |
| **Message abstractions** | `SystemMessage`/`HumanMessage`/`AIMessage`/`BaseMessage` via LangChain |

### Correction 1: PaperPilot Runtime LLM

- **Before**: Gemini `gemini-3.6-flash` via `ChatGoogleGenerativeAI`
- **After**: **Groq** `openai/gpt-oss-20b` via `langchain_groq.ChatGroq`
- **API Key**: Existing `GROQ_API_KEY` from `.env` (read through environment, never hard-coded)
- **Integration**: Properly LangChain-oriented (ChatPromptTemplate, message abstractions, structured output)
- **Result**: Groq runtime generation verified working with intent detection and response generation

### Correction 2: Conversation Persistence

- **Before**: In-memory-only `_conversations` dict in `main.py` (lost on app restart)
- **After**: **SQLite-backed** persistence using Python's `sqlite3` standard library
- **Database**: `backend/conversations.db` — persists conversation history across restarts
- **Persistence cycle**:
  1. **Startup**: Load conversations from SQLite into memory
  2. **Chat request**: Update both in-memory dict and SQLite DB (`INSERT OR REPLACE`)
  3. **Shutdown**: Persist final in-memory state to SQLite
- **Conversation_id isolation**: Each `conversation_id` gets isolated state in both LangGraph checkpointer and SQLite
- **No custom memory framework**: Uses SQLite (Python stdlib), not a custom-built solution

### Correction Summary

| Aspect | Before | After |
|--------|--------|-------|
| Runtime LLM | Gemini `gemini-3.6-flash` | **Groq** `openai/gpt-oss-20b` via LangChain |
| API Key source | Hard-coded in code (indirect) | `GROQ_API_KEY` from `.env` (environment) |
| Conversation persistence | In-memory only (lost on restart) | **SQLite** `conversations.db` (survives restarts) |
| Memory framework | Custom in-memory dict | **SQLite** (Python stdlib, not custom) |

### Files Modified (3 files)

- `backend/generation/nodes.py` — LLM changed from Gemini to Groq (`ChatGroq` + `openai/gpt-oss-20b`)
- `backend/main.py` — SQLite persistence layer added (`conversations.db`, startup/shutdown sync)
- `backend/requirements.txt` — No changes needed (langgraph already included)

### Verification Results

All 8 criteria pass:
1. ✅ Imports successful
2. ✅ LangGraph graph compiles
3. ✅ FastAPI starts with all 6 endpoints
4. ✅ Groq runtime generation works
5. ✅ Conversation continuation (same conversation_id)
6. ✅ Conversation isolation (different IDs)
7. ✅ SQLite persistence (DB at `backend/conversations.db`)
8. ✅ Streaming (SSE on `/chat/stream`)

### Deferred to Phase 2

RAG, embeddings, vector search, academic search, PDF processing, reranking, citations, DeepEval, human-in-the-loop, LangSmith, cloud deployment.

### Final Status

**Phase 1 — Core Agent Foundation: COMPLETE**

Both corrections have been applied with minimal changes (3 files modified, no new dependencies). The architecture is ready for Phase 2 extension with RAG, academic research, and all previously deferred features.