import asyncio
from generation.workflow import run_workflow

async def main():
    result = await run_workflow(
        "can you retrive most relevant pappers about agntic ai"
    )

    print("INTENT:", result.get("intent"))
    print("ROUTE:", result.get("route"))
    print("PAPERS:", len(result.get("academic_papers", [])))
    print("FINAL DOCS:", len(result.get("final_docs", [])))

asyncio.run(main())
