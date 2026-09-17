import asyncio
from generation.workflow import workflow
from generation.state import create_initial_state

async def main():
    query = "can you retrive most relevant pappers about agntic ai"

    state = create_initial_state(query)

    final_state = None

    async for state_update in workflow.astream(
        state,
        stream_mode="values",
    ):
        final_state = state_update

        print(
            "INTENT:", state_update.get("intent"),
            "| ROUTE:", state_update.get("route"),
            "| SEARCH:", state_update.get("search_query"),
            "| PAPERS:", len(state_update.get("academic_papers", [])),
            "| DOCS:", len(state_update.get("final_docs", [])),
        )

    print("\n===== FINAL =====")

    if final_state:
        print("INTENT:", final_state.get("intent"))
        print("ROUTE:", final_state.get("route"))
        print("SEARCH QUERY:", final_state.get("search_query"))
        print("ACADEMIC PAPERS:", len(final_state.get("academic_papers", [])))
        print("FINAL DOCS:", len(final_state.get("final_docs", [])))
        print("RESPONSE:", final_state.get("response", "")[:1000])
    else:
        print("No final state received.")

asyncio.run(main())
