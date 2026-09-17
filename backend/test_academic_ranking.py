import asyncio
from generation.nodes import academic_search_node, normalize_papers_node
from generation.state import create_initial_state

async def main():
    state = create_initial_state("find papers about agentic AI")

    search_result = await academic_search_node(state)

    print("SEARCH PAPERS:", len(search_result.get("academic_papers", [])))

    state.update(search_result)

    normalized_result = await normalize_papers_node(state)

    print("NORMALIZED:", len(normalized_result.get("normalized_papers", [])))
    print("FINAL DOCS:", len(normalized_result.get("final_docs", [])))

    print("\nRANKED PAPERS:")
    for paper in normalized_result.get("normalized_papers", []):
        print(
            f"- {paper.get('title')} "
            f"[score={paper.get('relevance_score')}]"
        )

asyncio.run(main())
