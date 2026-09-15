"""
Golden test evaluation dataset for PaperPilot Phase 4.

10 representative cases covering different query types and expected
workflow behavior. Each case defines the user query, expected structural
behavior, and contextual information for evaluation.
"""

from typing import Dict, Any, List
from generation.state import create_initial_state, AgentState

golden_cases: List[Dict[str, Any]] = [
    {
        "name": "casual_conversation",
        "query": "Hello! How are you doing today?",
        "expected_intent": "general_answer",
        "expected_route": "chat",
        "description": "Casual greeting should be routed to general conversation",
        "should_not_trigger": ["academic_search", "local_retrieve"],
    },
    {
        "name": "general_knowledge",
        "query": "What is the capital of France?",
        "expected_intent": "general_answer",
        "expected_route": "chat",
        "description": "Simple factual question should get a general answer",
        "should_not_trigger": ["academic_search", "local_retrieve"],
    },
    {
        "name": "general_technical",
        "query": "Explain how transformer attention mechanisms work",
        "expected_intent": "general_answer",
        "expected_route": "chat",
        "description": "General technical explanation question",
        "should_not_trigger": ["academic_search", "local_retrieve"],
    },
    {
        "name": "local_rag_query",
        "query": "What do the uploaded papers say about RAG evaluation?",
        "expected_intent": "local_rag",
        "expected_route": "local_rag",
        "description": "Query against the user's local PDF library",
        "should_not_trigger": ["academic_search"],
    },
    {
        "name": "local_rag_retrieval",
        "query": "What are the key challenges mentioned in the local papers?",
        "expected_intent": "local_rag",
        "expected_route": "local_rag",
        "description": "Local RAG query that requires document retrieval",
        "should_not_trigger": ["academic_search"],
    },
    {
        "name": "academic_research",
        "query": "Find recent papers about RAG evaluation",
        "expected_intent": "academic_research",
        "expected_route": "academic_research",
        "description": "Academic search query should route to arXiv/OpenAlex",
        "should_trigger": ["academic_search"],
    },
    {
        "name": "recent_academic_research",
        "query": "Find recent 2024 papers about multimodal RAG",
        "expected_intent": "academic_research",
        "expected_route": "academic_research",
        "description": "Recent academic search with year filter",
        "should_trigger": ["academic_search"],
    },
    {
        "name": "paper_methodology_question",
        "query": "What methodology do the uploaded papers use for RAG evaluation?",
        "expected_intent": "local_rag",
        "expected_route": "local_rag",
        "description": "Question about methodology of local PDF papers",
        "should_not_trigger": ["academic_search"],
    },
    {
        "name": "insufficient_local_evidence",
        "query": "Find papers about the specific technical details of attention bottleneck mitigation in RAG",
        "expected_intent": "academic_research",
        "expected_route": "academic_research",
        "description": "Local library has insufficient evidence, should fall back to academic search",
        "should_trigger": ["academic_search"],
        "should_not_trigger": ["local_retrieve_with_evidence"],
    },
    {
        "name": "multi_turn_followup",
        "query": "Can you tell me more about the first paper you mentioned?",
        "expected_intent": "general_answer",
        "expected_route": "chat",
        "description": "Follow-up conversation turn",
        "should_not_trigger": ["academic_search", "local_retrieve"],
    },
]

# Validate all golden cases have consistent structure
for i, case in enumerate(golden_cases):
    required_keys = ['name', 'query', 'expected_intent', 'expected_route', 'description']
    for key in required_keys:
        assert key in case, f'Golden case {i} missing required key: {key}'
    assert case['expected_intent'] in ['general_answer', 'local_rag', 'academic_research'], \
        f"Golden case {i} has invalid intent: {case['expected_intent']}"
    assert case['expected_route'] in ['chat', 'local_rag', 'academic_research'], \
        f"Golden case {i} has invalid route: {case['expected_route']}"

print(f'Loaded {len(golden_cases)} golden test cases')
print('All cases validated successfully')
