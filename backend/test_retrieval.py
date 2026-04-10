from retrieval.retriever import retrieve
from generation.generator import generate_answer

query = "What is transformer architecture?"

chunks = retrieve(query)

answer = generate_answer(query, chunks)

print("\nFinal Answer:\n")
print(answer)