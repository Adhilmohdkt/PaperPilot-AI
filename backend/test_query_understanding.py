import asyncio
from generation.nodes import query_understanding_node

async def main():
    state = {
        "user_query": "can you retrive most relevant pappers about agntic ai",
        "query": "can you retrive most relevant pappers about agntic ai",
    }

    result = await query_understanding_node(state)

    print("INTENT:", result.get("intent"))
    print("ROUTE:", result.get("route"))
    print("TOPIC:", result.get("query_topic"))
    print("SEARCH QUERY:", result.get("search_query"))
    print("RECENCY:", result.get("recency_requested"))

asyncio.run(main())
