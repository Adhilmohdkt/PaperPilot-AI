"""FastAPI surface for the Phase 1 PaperPilot LangGraph workflow."""

import json
import os
import asyncio
import uuid
import sqlite3
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from generation.sources import format_sources_for_ui
from generation.workflow import workflow, run_workflow
from generation.state import AgentState
from langchain_core.messages import BaseMessage

from config import settings


# SQLite-backed conversation store for persistence across restarts
# Conversations are stored in a local SQLite database keyed by conversation_id
DB_PATH = os.getenv(
        "CONVERSATION_DB_PATH",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "conversations.db"),
    )

# Ensure the database directory exists
os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)

# Conversation tables will be initialized on first use via _init_conversation_db()

# In-memory conversation store for fast access (sync with SQLite)
# Key: conversation_id, Value: list of messages (dict with role and content)
_conversations: Dict[str, List[Dict[str, str]]] = {}


def _load_conversations_from_db():
    """Load all conversations from SQLite into memory on startup."""
    global _conversations
    conn = sqlite3.connect(DB_PATH)
    try:
        # Ensure table exists
        conn.execute(
            "CREATE TABLE IF NOT EXISTS conversations "
            "(conversation_id TEXT PRIMARY KEY, messages TEXT NOT NULL, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.commit()
        cursor = conn.execute("SELECT conversation_id, messages FROM conversations")
        _conversations = {
            row[0]: json.loads(row[1]) if row[1] else []
            for row in cursor.fetchall()
        }
    finally:
        conn.close()


_load_conversations_from_db()


class QueryRequest(BaseModel):
    query: str
    source_filter: Optional[str] = None


class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    query: str
    history: List[Dict[str, str]] = []
    source_filter: Optional[str] = None


class ConversationSummary(BaseModel):
    conversation_id: str
    last_query: str
    last_intent: str
    message_count: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - load conversations from SQLite, workflow compiled at module import
    global _conversations
    _load_conversations_from_db()
    yield
    # Shutdown - persist conversations to SQLite
    conn = sqlite3.connect(DB_PATH)
    try:
        for conv_id, messages in _conversations.items():
            conn.execute(
                "INSERT OR REPLACE INTO conversations (conversation_id, messages, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                (conv_id, json.dumps(messages)),
            )
        conn.commit()
    finally:
        conn.close()


app = FastAPI(title="PaperPilot AI - Phase 1", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy", "phase": "1_core_agent_foundation"}


@app.post("/conversations")
def create_conversation():
    """Create a new conversation and return a new conversation_id."""
    conv_id = str(uuid.uuid4())
    _conversations[conv_id] = []
    # Also persist to SQLite
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO conversations (conversation_id, messages, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
            (conv_id, json.dumps([])),
        )
        conn.commit()
    finally:
        conn.close()
    return {"conversation_id": conv_id, "message": "Conversation created"}


@app.get("/conversations")
def list_conversations():
    """List all conversation IDs with summaries, most recent first."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute("SELECT conversation_id, messages FROM conversations ORDER BY updated_at DESC")
        rows = cursor.fetchall()
    except Exception:
        rows = []
    finally:
        conn.close()

    summary_list = []
    for conv_id, msgs_raw in rows:
        messages = json.loads(msgs_raw) if msgs_raw else []
        last_query = ""
        last_intent = "general_answer"
        msg_count = len(messages)
        if messages:
            # Use the last USER message as the sidebar label
            user_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
            if user_msgs:
                last_query = user_msgs[-1].get("content", "")[:60]
            else:
                last_query = messages[-1].get("content", "")[:60] if isinstance(messages[-1], dict) else str(messages[-1])[:60]
        summary_list.append(
            ConversationSummary(
                conversation_id=conv_id,
                last_query=last_query,
                last_intent=last_intent,
                message_count=msg_count,
            ).model_dump()
        )
    return {"conversations": summary_list}


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    """Retrieve a specific conversation's message history."""
    if conversation_id not in _conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = _conversations[conversation_id]
    return {"conversation_id": conversation_id, "messages": messages}


@app.post("/chat")
def chat(request: ChatRequest):
    """Handle a chat request and return the response."""
    conv_id = request.conversation_id or str(uuid.uuid4())
    query = request.query
    history = request.history

    # Store the user message in conversation history (sync in-memory + SQLite)
    if conv_id not in _conversations:
        _conversations[conv_id] = []
    _conversations[conv_id].append({"role": "user", "content": query})
    # Persist to SQLite
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO conversations (conversation_id, messages, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
            (conv_id, json.dumps(_conversations[conv_id])),
        )
        conn.commit()
    finally:
        conn.close()

    # Run the LangGraph workflow
    result = asyncio.run(
        run_workflow(
            query=query,
            conversation_id=conv_id,
            history=history,
            source_filter=request.source_filter,
        )
    )

    # Extract the response from the workflow result
    response_text = ""
    workflow_response = result.get("response", "")
    if isinstance(workflow_response, dict) and "text" in workflow_response:
        response_text = workflow_response["text"]
    else:
        response_text = str(workflow_response) if workflow_response else ""

    # Store the assistant response in conversation history (sync in-memory + SQLite)
    if conv_id not in _conversations:
        _conversations[conv_id] = []
    _conversations[conv_id].append({"role": "assistant", "content": response_text})
    # Persist to SQLite
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO conversations (conversation_id, messages, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
            (conv_id, json.dumps(_conversations[conv_id])),
        )
        conn.commit()
    finally:
        conn.close()

    return {"conversation_id": conv_id, "response": response_text, "intent": result.get("intent")}


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Handle a streaming chat request using the LangGraph workflow."""

    conv_id = request.conversation_id or str(uuid.uuid4())
    query = request.query
    history = request.history

    # Store the user message in conversation history (sync in-memory + SQLite)
    if conv_id not in _conversations:
        _conversations[conv_id] = []
    _conversations[conv_id].append({"role": "user", "content": query})
    # Persist to SQLite
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO conversations (conversation_id, messages, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
            (conv_id, json.dumps(_conversations[conv_id])),
        )
        conn.commit()
    finally:
        conn.close()

    async def event_generator():
        try:
            # Run the LangGraph workflow with streaming
            # Run the LangGraph workflow with explicit LangSmith metadata
            config = {
                "run_name": "PaperPilot Chat",
                "tags": [
                    "paperpilot",
                    "chat",
                    "stream",
                ],
                "metadata": {
                    "thread_id": conv_id,
                    "conversation_id": conv_id,
                    "endpoint": "chat_stream",
                },
                "configurable": {
                    "thread_id": conv_id,
                },
            }
            from generation.state import create_initial_state

            initial_state = create_initial_state(
                query=query,
                source_filter=request.source_filter,
                history=history,
                conversation_id=conv_id,
            )

            # Stream the workflow execution
            # astream yields chunks: (node_name, update_dict) or just update_dict
            response_text = ""
            async for chunk in workflow.astream(initial_state, config=config):
                # chunk may be a dict of node_updates or a single update
                # Handle both formats
                if isinstance(chunk, dict):
                    # Check if this is a node update or the final state
                    for node_name, update in chunk.items():
                        if update is not None and "response" in update:
                            response = update["response"]
                            response_text = response if isinstance(response, str) else str(response or "")
                            # Yield the response as a token event
                            yield f"event: token\ndata: {json.dumps({'text': response})}\n\n"
                        if update is not None and "final_docs" in update and update["final_docs"]:
                            yield f"event: sources\ndata: {json.dumps({'sources': format_sources_for_ui(update['final_docs'])})}\n\n"
                        if update is not None and "intent" in update:
                            yield f"event: intent\ndata: {json.dumps({'intent': update['intent']})}\n\n"
                elif isinstance(chunk, tuple) and len(chunk) == 2:
                    # (node_name, update) format
                    node_name, update = chunk
                    if update is not None and "response" in update:
                        response = update["response"]
                        response_text = response if isinstance(response, str) else str(response or "")
                        yield f"event: token\ndata: {json.dumps({'text': response})}\n\n"
                    if update is not None and "final_docs" in update and update["final_docs"]:
                        yield f"event: sources\ndata: {json.dumps({'sources': format_sources_for_ui(update['final_docs'])})}\n\n"
                    if update is not None and "intent" in update:
                        yield f"event: intent\ndata: {json.dumps({'intent': update['intent']})}\n\n"

            # Store the assistant response in conversation history (sync in-memory + SQLite)
            if conv_id not in _conversations:
                _conversations[conv_id] = []
            _conversations[conv_id].append({"role": "assistant", "content": response_text})
            # Persist to SQLite
            conn = sqlite3.connect(DB_PATH)
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO conversations (conversation_id, messages, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                    (conv_id, json.dumps(_conversations[conv_id])),
                )
                conn.commit()
            finally:
                conn.close()

            # Signal done
            yield "event: done\ndata: {}\n\n"

        except Exception as error:
            yield f"event: error\ndata: {json.dumps({'message': str(error)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
