import weaviate
import os
from weaviate.classes.query import Filter

def get_client():
    weaviate_host = os.getenv("WEAVIATE_HOST", "localhost")
    if weaviate_host == "localhost":
        client = weaviate.connect_to_local()
    else:
        client = weaviate.connect_to_custom(
            http_host=weaviate_host,
            http_port=8080,
            http_secure=False,
            grpc_host=weaviate_host,
            grpc_port=50051,
            grpc_secure=False,
        )
    return client

def search(query_text: str, query_vector: list, limit=15, source_filter: str = None):
    client = get_client()
    collection = client.collections.get("PaperChunk")

    filters = None
    if source_filter:
        filters = Filter.by_property("source").equal(source_filter)

    response = collection.query.hybrid(
        query=query_text,
        vector=query_vector,
        alpha=0.5,
        filters=filters,
        limit=limit
    )

    return response