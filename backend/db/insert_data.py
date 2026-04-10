from db.weaviate_client import get_client
from ingestion.loader import load_pdf
from ingestion.cleaner import clean_text
from ingestion.chunker import chunk_text
from embeddings.embedder import get_embedding

import os

def insert_data():
    client = get_client()

    data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Data"))
    
    if not os.path.exists(data_folder):
        print(f"Data folder {data_folder} does not exist.")
        return

    collection = client.collections.get("PaperChunk")
    
    inserted_count = 0

    for filename in os.listdir(data_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(data_folder, filename)
            print(f"Processing: {filename}...")
            
            text = load_pdf(pdf_path)
            cleaned = clean_text(text)
            chunks = chunk_text(cleaned)

            for chunk in chunks:
                embedding = get_embedding(chunk)

                collection.data.insert(
                    properties={
                        "text": chunk,
                        "source": filename
                    },
                    vector=embedding
                )
            inserted_count += 1

    print(f"Data for {inserted_count} PDFs inserted successfully")


if __name__ == "__main__":
    insert_data()
    