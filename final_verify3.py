import os
os.environ['GROQ_API_KEY'] = 'REDACTED_GROQ_KEY'
os.environ['GEMINI_API_KEY'] = 'AIzaSyCyXIg5jnowM6HMLcff400feIwUc5gCzIs'

import sys
sys.path.insert(0, r'C:\Users\ASUS\Desktop\PaperPilot AI\backend')

# Verify all key components
print('=== Phase 3 Verification ===')
print()

# 1. Imports
from generation.workflow import workflow
from generation.nodes import (
    query_understanding_node, router_node, generate_node,
    citation_validation_node, error_handler_node
)
from generation.routing import route_query, route_after_relevance
from generation.state import AgentState, create_initial_state
from main import app, _conversations, DB_PATH, create_conversation, list_conversations, get_conversation, chat, chat_stream
from retrieval.retriever import retrieve
from config import settings
print('1. All imports: OK')

# 2. Workflow compilation
print(f'2. Workflow compiled: {workflow is not None}')

# 3. Routing tests
import asyncio
async def test_routing():
    from generation.nodes import query_understanding_node, router_node
    queries = ['Hello', 'What is RAG?', 'Find papers about AI']
    for q in queries:
        state = {'conversation_id': 'test', 'user_query': q, 'messages': [], 'intent': 'general_answer', 'response': ''}
        after_qu = await query_understanding_node(state)
        after_router = await router_node({**state, **after_qu})
        i = after_qu.get('intent')
        r = after_router.get('route')
        print(f'   {q!r:30s} -> intent: {i!s:25s} -> route: {r!s}')

asyncio.run(test_routing())
print('3. Routing: OK')

# 4. Workflow execution
async def test_workflow():
    from generation.workflow import workflow
    queries = ['Hello', 'What is RAG?']
    for q in queries:
        initial_state = {'conversation_id': 'test', 'user_query': q, 'messages': [], 'intent': 'general_answer', 'response': ''}
        result = await workflow.ainvoke(initial_state, config={'configurable': {'thread_id': 'test'}})
        response = str(result.get('response', ''))[:60]
        print(f'   {q!r:30s} -> response: {response!s}')

asyncio.run(test_workflow())
print('4. Workflow execution: OK')

# 5. Chat endpoint
from fastapi.testclient import TestClient
from main import app
client = TestClient(app)
chat_result = client.post('/chat', json={'conversation_id': 'test-001', 'query': 'Hello'})
resp = chat_result.json()['response'][:50]
intent = chat_result.json()['intent']
print(f'5. Chat endpoint: response={resp!r}, intent={intent}')

# 6. Streaming endpoint
chat_stream_result = client.post('/chat/stream', json={'conversation_id': 'test-002', 'query': 'What is RAG?'})
events = []
for chunk in chat_stream_result.iter_lines():
    if chunk:
        line = chunk.decode('utf-8') if isinstance(chunk, bytes) else chunk
        events.append(line)
print(f'6. Streaming endpoint: {len(events)} events')

# 7. LangSmith config
print(f'7. LangSmith tracing: {settings.langsmith_tracing}')
print(f'   project: {settings.langsmith_project}')

# 8. Conversation persistence
create_result = create_conversation()
list_result = list_conversations()
conversations = list_result.json().get('conversations', [])
print(f'8. Conversation creation: {create_result}')
print(f'   List conversations: {len(conversations)} conversations')

print()
print('=== Phase 3 Verification Complete ===')
"