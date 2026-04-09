from ingestion.loader import load_pdf
from ingestion.cleaner import clean_text
from ingestion.chunker import chunk_text
from embeddings.embedder import get_embedding


#  Load PDF
pdf_path = "../Data/sample.pdf"
text = load_pdf(pdf_path)

#  Clean text
cleaned = clean_text(text)

#  Chunk text
chunks = chunk_text(cleaned)

print("Total chunks:", len(chunks))


#  Generate embeddings
all_chunks = []

for chunk in chunks[:2]: 
    emb = get_embedding(chunk)

    data = {
        "text": chunk,
        "embedding": emb
    }

    all_chunks.append(data)

    print("\nCHUNK:", chunk[:100])
    print("EMBEDDING LENGTH:", len(emb))
    print("----------")


#  Check stored structure
print("\nSample stored object:\n")
print(all_chunks[0])