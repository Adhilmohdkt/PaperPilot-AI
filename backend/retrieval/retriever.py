import os
import cohere
from embeddings.embedder import embed_query
from db.weaviate_client import search

cohere_api_key = os.getenv("COHERE_API_KEY", "YOUR_COHERE_API_KEY")

def retrieve(query: str, source_filter: str = None):
    
    query_vector = embed_query(query)

    # 1. Hybrid Search (BM25 + Vector) fetches 15 chunks
    results = search(query_text=query, query_vector=query_vector, limit=15, source_filter=source_filter)

    chunks = []
    for obj in results.objects:
        chunks.append({
            "text": obj.properties.get("text", ""),
            "source": obj.properties.get("source", "Unknown")
        })
        
    if not chunks:
        return []

    try:
        co = cohere.Client(cohere_api_key)
        
        # Cohere reranker requires a simple list of strings
        texts_to_rerank = [c["text"] for c in chunks]
        
        response = co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=texts_to_rerank,
            top_n=5
        )
        
        reranked_chunks = []
        for result in response.results:
            reranked_chunks.append(chunks[result.index])
            
        return reranked_chunks
        
    except Exception as e:
        print(f"Reranking failed: {e}")
        # Fallback to just the top 5 from Weaviate hybrid search
        return chunks[:5]