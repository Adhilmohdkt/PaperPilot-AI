import asyncio
from generation.workflow import run_workflow

async def main():
    query = "can you retrive most relevant pappers about agntic ai"

    result = await run_workflow(query)

    print("\n=== WORKFLOW RESULT ===")
    print("INTENT:", result.get("intent"))
    print("ROUTE:", result.get("route"))
    print("ACADEMIC PAPERS:", len(result.get("academic_papers", [])))
    print("NORMALIZED PAPERS:", len(result.get("normalized_papers", [])))
    print("FINAL DOCS:", len(result.get("final_docs", [])))

    print("\n=== RESPONSE ===")
    print(result.get("response", ""))

    print("\n=== CITATIONS ===")
    for citation in result.get("citations", []):
        print(citation)

asyncio.run(main())
