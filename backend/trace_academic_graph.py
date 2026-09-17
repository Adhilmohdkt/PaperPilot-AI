import asyncio
from generation.workflow import workflow
from generation.state import create_initial_state

async def main():
    state = create_initial_state(
        "can you retrive most relevant pappers about agntic ai"
    )

    print("\n=== FULL GRAPH STATE TRACE ===")

    async for event in workflow.astream(
        state,
        stream_mode="values",
    ):
        print("\n--- STATE ---")
        print("intent:", event.get("intent"))
        print("route:", event.get("route"))
        print("academic_papers:", len(event.get("academic_papers", [])))
        print("normalized_papers:", len(event.get("normalized_papers", [])))
        print("final_docs:", len(event.get("final_docs", [])))

asyncio.run(main())
