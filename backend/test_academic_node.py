import asyncio

from generation.nodes import academic_search_node
from retrieval.academic.ranker import normalize_and_rank


async def main():
    state = {
        "user_query": "can you retrieve most relevant papers about agentic ai",
        "query": "can you retrieve most relevant papers about agentic ai",
        "search_query": "agentic AI",
    }

    # 1. Search ArXiv + OpenAlex
    result = await academic_search_node(state)

    papers = result.get("academic_papers", [])

    print("\nRAW PROVIDER RESULTS")
    print("====================")
    print("Total papers:", len(papers))
    print("Provider counts:", result.get("provider_result_counts"))

    # 2. Normalize + rank
    ranked = normalize_and_rank(
        query=state["search_query"],
        papers=papers,
        top_k=5,
    )

    # 3. Display ranked results
    print("\nTOP 5 RANKED PAPERS")
    print("===================")

    for i, paper in enumerate(ranked, 1):
        print(f"\n{i}. {paper.title}")
        print(f"   Provider: {paper.provider}")
        print(f"   Relevance score: {paper.relevance_score}")
        print(f"   Year: {paper.year}")
        print(f"   Citations: {paper.citation_count}")
        print(f"   URL: {paper.paper_url}")


asyncio.run(main())