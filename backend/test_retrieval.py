import pytest

if __name__ != "__main__":
    pytest.skip("Manual retrieval diagnostic; not a pytest test module.", allow_module_level=True)

from retrieval.retriever import retrieve
from generation.generator import generate_answer

query = "What is transformer architecture?"

chunks = retrieve(query)

answer = generate_answer(query, chunks)

print("\nFinal Answer:\n")
print(answer)
