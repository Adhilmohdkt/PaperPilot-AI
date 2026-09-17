import asyncio

from generation.nodes import query_understanding_node
from generation.state import create_initial_state


TEST_CASES = [
    # ============================================================
    # 1. GENERAL QUESTIONS
    # ============================================================

    "What is RAG?",
    "How does BM25 work?",
    "Explain transformers in simple terms.",
    "What is the difference between embeddings and vector databases?",
    "Why do LLMs hallucinate?",

    # ============================================================
    # 2. EXPLICIT ACADEMIC RESEARCH
    # ============================================================

    "Find papers about BM25 retrieval.",
    "Show me research papers about RAG evaluation.",
    "Find studies on hallucination detection in large language models.",
    "Can you find recent papers about agentic AI?",
    "Give me papers about long-term memory in LLMs.",

    # ============================================================
    # 3. DETAILED / COMPLEX RESEARCH QUESTIONS
    # ============================================================

    "I'm looking for papers that compare sparse retrieval with dense retrieval.",
    "Find research on how long-term memory is implemented in LLM agents.",
    "Find papers evaluating the faithfulness of retrieval augmented generation systems.",
    "Show me research on methods for reducing hallucinations in large language models.",
    "Find papers about using reinforcement learning for robotic manipulation.",

    # ============================================================
    # 4. COMPARISON QUERIES
    # ============================================================

    "Find papers comparing BM25 and dense vector retrieval.",
    "Find research comparing RAG with long-context LLMs.",
    "Show me papers comparing different LLM memory architectures.",
    "Find studies comparing agentic RAG approaches.",

    # ============================================================
    # 5. METHODOLOGY / TECHNICAL RESEARCH
    # ============================================================

    "Find papers that propose new methods for RAG retrieval.",
    "Find research on reranking methods for information retrieval.",
    "Show me papers about hybrid search using BM25 and vector retrieval.",
    "Find papers about efficient transformer architectures.",
    "Find research on reducing the computational cost of LLM inference.",

    # ============================================================
    # 6. EVALUATION / BENCHMARKS
    # ============================================================

    "Find papers about benchmarks for evaluating RAG systems.",
    "Show me research on LLM evaluation benchmarks.",
    "Find papers about evaluating retrieval quality in RAG.",
    "Find studies measuring hallucination rates in LLMs.",

    # ============================================================
    # 7. RECENCY
    # ============================================================

    "Find recent papers about RAG.",
    "Show me the latest research on LLM agents.",
    "Find the newest papers about multimodal LLMs.",
    "What are the recent papers on LLM memory?",
    "Find current research on AI agents.",

    # ============================================================
    # 8. SPELLING ERRORS / INFORMAL LANGUAGE
    # ============================================================

    "find pappers about bm25 retieval stratergy",
    "show me some papers about persistance or memmory in llm models",
    "can you retrive research about hallucination detecton in llms",
    "give me pappers about agentic ai and rag",
    "find relevent studies about vector databeses",

    # ============================================================
    # 9. ACRONYMS / AMBIGUOUS TERMS
    # ============================================================

    "Find papers about ANN.",
    "Find papers about ANN retrieval.",
    "Find papers about ANN in deep learning.",
    "Show me research about RRF.",
    "Find papers about ColBERT.",
    "Find research about MCP for AI agents.",

    # ============================================================
    # 10. LOCAL DOCUMENT REQUESTS
    # ============================================================

    "What does my uploaded paper say about RAG?",
    "Explain the paper in my library.",
    "What methodology is used in my uploaded PDF?",
    "Compare the papers I uploaded.",
    "What does my local document say about BM25?",

    # ============================================================
    # 11. FOLLOW-UP / CONVERSATIONAL RESEARCH
    # ============================================================

    "Find papers about RAG.",
    "Show me more papers about this.",
    "Find newer papers on this topic.",
    "Can you find research similar to these papers?",
    "Compare these papers.",
]


async def main():
    print("=" * 100)
    print("PAPERPILOT QUERY UNDERSTANDING TEST")
    print("Model: GPT-OSS-120B")
    print("=" * 100)

    for i, query in enumerate(TEST_CASES, start=1):
        state = create_initial_state(query=query)

        try:
            result = await query_understanding_node(state)

            print(f"\n{'-' * 100}")
            print(f"TEST {i}")
            print(f"QUERY: {query}")
            print(f"INTENT: {result.get('intent')}")
            print(f"ROUTE: {result.get('route')}")
            print(f"TOPIC: {result.get('query_topic')}")
            print(f"SEARCH QUERY: {result.get('search_query')}")
            print(f"RECENCY: {result.get('recency_requested')}")

        except Exception as e:
            print(f"\n{'-' * 100}")
            print(f"TEST {i}")
            print(f"QUERY: {query}")
            print(f"ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())