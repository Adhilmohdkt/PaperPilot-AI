"""Safe re-indexing script for PaperPilot AI Local RAG.

Indexes all PDFs from Data/ into a new Weaviate collection 'PaperChunk_v2'
using 'all-MiniLM-L6-v2' (384 dimensions).
Keeps the original 'PaperChunk' collection completely untouched as backup.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Add backend directory to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import weaviate
from weaviate.classes.config import DataType, Property

from config import settings
from db.weaviate_client import get_client
from embeddings.embedder import get_embedding, embed_query
from ingestion.loader import load_pdf
from ingestion.chunker import chunk_documents
from ingestion.library import document_hash

NEW_COLLECTION = "PaperChunk_v2"
ORIGINAL_COLLECTION = "PaperChunk"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

SCHEMA_PROPERTIES = [
    Property(name="text", data_type=DataType.TEXT),
    Property(name="source", data_type=DataType.TEXT),
    Property(name="content_hash", data_type=DataType.TEXT),
    Property(name="chunk_id", data_type=DataType.TEXT),
    Property(name="page", data_type=DataType.INT),
    Property(name="embedding_model", data_type=DataType.TEXT),
]


def run_reindex():
    print("=" * 70, flush=True)
    print("PaperPilot AI — Safe Re-Indexing into PaperChunk_v2", flush=True)
    print("=" * 70, flush=True)

    client = get_client()

    try:
        # 1. Verify original collection is preserved
        print(f"\n[Step 1] Verifying original collection '{ORIGINAL_COLLECTION}'...", flush=True)
        if client.collections.exists(ORIGINAL_COLLECTION):
            orig_col = client.collections.get(ORIGINAL_COLLECTION)
            orig_count = len(orig_col)
            print(f"  [OK] Original '{ORIGINAL_COLLECTION}' exists with {orig_count} chunks. Untouched.", flush=True)
        else:
            print(f"  [WARN] Original '{ORIGINAL_COLLECTION}' not found.", flush=True)

        # 2. Setup PaperChunk_v2
        print(f"\n[Step 2] Initializing '{NEW_COLLECTION}' in Weaviate...", flush=True)
        if not client.collections.exists(NEW_COLLECTION):
            client.collections.create(
                name=NEW_COLLECTION,
                properties=SCHEMA_PROPERTIES,
            )
            print(f"  [OK] Created new collection '{NEW_COLLECTION}'.", flush=True)
        else:
            print(f"  [INFO] Collection '{NEW_COLLECTION}' already exists.", flush=True)

        new_col = client.collections.get(NEW_COLLECTION)

        # 3. Locate Data directory
        project_root = Path(backend_dir).parent
        data_dir = project_root / "Data"
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found at: {data_dir}")

        pdf_files = sorted(list(data_dir.glob("*.pdf")))
        print(f"\n[Step 3] Found {len(pdf_files)} PDF files in {data_dir}:", flush=True)
        for idx, pdf in enumerate(pdf_files, start=1):
            print(f"  {idx}. {pdf.name} ({pdf.stat().st_size / 1024:.1f} KB)", flush=True)

        # 4. Ingest each PDF into PaperChunk_v2
        print(f"\n[Step 4] Ingesting and embedding with {EMBEDDING_MODEL_NAME} (384 dims)...", flush=True)
        total_chunks_inserted = 0
        file_results = []

        for idx, pdf_path in enumerate(pdf_files, start=1):
            source = pdf_path.name
            c_hash = document_hash(pdf_path)
            t0 = time.time()
            print(f"\n  [{idx}/{len(pdf_files)}] Processing: {source}...", flush=True)

            pages = load_pdf(str(pdf_path))
            pages = pages[: settings.max_pdf_pages]

            raw_docs = []
            for page in pages:
                text = page.page_content[: settings.max_chars_per_page]
                if text.strip():
                    from langchain_core.documents import Document
                    raw_docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": source,
                            "content_hash": c_hash,
                            "page": int(page.metadata.get("page", 0)) + 1,
                        }
                    ))

            chunks = chunk_documents(raw_docs)[: settings.max_chunks_per_document]
            if not chunks:
                print(f"    [WARN] No chunks generated for {source}!", flush=True)
                file_results.append({"file": source, "chunks": 0, "status": "empty"})
                continue

            # Batch insert with 384-dim embeddings
            with new_col.batch.dynamic() as batch:
                for c_idx, chunk in enumerate(chunks):
                    vec = get_embedding(chunk.page_content)
                    batch.add_object(
                        properties={
                            "text": chunk.page_content,
                            "source": source,
                            "content_hash": c_hash,
                            "chunk_id": f"{c_hash}:{c_idx}",
                            "page": int(chunk.metadata.get("page", 1)),
                            "embedding_model": EMBEDDING_MODEL_NAME,
                        },
                        vector=vec,
                    )

            if new_col.batch.failed_objects:
                err_msg = f"Weaviate rejected {len(new_col.batch.failed_objects)} chunks for {source}"
                print(f"    [ERROR] {err_msg}", flush=True)
                file_results.append({"file": source, "chunks": 0, "status": "failed", "error": err_msg})
            else:
                elapsed = time.time() - t0
                print(f"    [SUCCESS] Inserted {len(chunks)} chunks in {elapsed:.1f}s.", flush=True)
                total_chunks_inserted += len(chunks)
                file_results.append({"file": source, "chunks": len(chunks), "status": "success"})

        # 5. Verification checks
        print("\n" + "=" * 70, flush=True)
        print("[Step 5] Detailed Index Verification Checks", flush=True)
        print("=" * 70, flush=True)

        final_count = len(new_col)
        print(f"  Total chunks in '{NEW_COLLECTION}': {final_count}", flush=True)

        # Sample chunk vector dimension
        sample_obj = next(iter(new_col.iterator(include_vector=True)))
        vec = sample_obj.vector.get("default") if isinstance(sample_obj.vector, dict) else sample_obj.vector
        vec_dim = len(vec) if vec is not None else 0
        print(f"  Sample Chunk ID: {sample_obj.properties.get('chunk_id')}", flush=True)
        print(f"  Sample Source: {sample_obj.properties.get('source')}", flush=True)
        print(f"  Sample Embedding Model: {sample_obj.properties.get('embedding_model')}", flush=True)
        print(f"  Sample Vector Dimension: {vec_dim} (Target: 384)", flush=True)

        # Unique sources in collection
        sources_found = set()
        for obj in new_col.iterator():
            sources_found.add(obj.properties.get("source"))
        print(f"  Source coverage: {len(sources_found)}/{len(pdf_files)} PDF files present in index.", flush=True)

        # Query vector dimension
        query_vec = embed_query("test query")
        print(f"  Query Vector Dimension: {len(query_vec)} (Target: 384)", flush=True)

        success = (
            final_count > 0
            and vec_dim == 384
            and len(query_vec) == 384
            and len(sources_found) == len(pdf_files)
        )

        print("\nVerification Summary:", flush=True)
        print(f"  - Chunk count > 0: {'PASSED' if final_count > 0 else 'FAILED'} ({final_count})", flush=True)
        print(f"  - Vector dimension == 384: {'PASSED' if vec_dim == 384 else 'FAILED'} ({vec_dim})", flush=True)
        print(f"  - Query vector dimension == 384: {'PASSED' if len(query_vec) == 384 else 'FAILED'}", flush=True)
        print(f"  - Full source coverage: {'PASSED' if len(sources_found) == len(pdf_files) else 'FAILED'} ({len(sources_found)}/{len(pdf_files)})", flush=True)
        print(f"  - Original collection preserved: {'PASSED' if len(orig_col) == 308 else 'FAILED'} ({len(orig_col)} chunks)", flush=True)

        if success:
            print("\n>>> ALL VERIFICATION CHECKS PASSED SUCCESSFULLY. <<<", flush=True)
        else:
            print("\n>>> VERIFICATION FAILED. DO NOT SWITCH COLLECTIONS. <<<", flush=True)

        return success

    finally:
        client.close()


if __name__ == "__main__":
    success = run_reindex()
    sys.exit(0 if success else 1)
