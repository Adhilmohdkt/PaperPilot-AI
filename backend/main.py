from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from retrieval.retriever import retrieve
from generation.generator import generate_answer_stream

app = FastAPI()

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Data"))
if os.path.exists(data_dir):
    app.mount("/pdfs", StaticFiles(directory=data_dir), name="pdfs")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional

class QueryRequest(BaseModel):
    query: str
    source_filter: Optional[str] = None

class StreamRequest(BaseModel):
    query: str
    source_filter: Optional[str] = None
    chunks: List[Any] = []
    history: List[Dict[str, str]] = []

@app.get("/")
def home():
    return {"message": "PaperPilot AI is running 🚀"}

@app.post("/retrieve")
def get_documents(request: QueryRequest):
    chunks = retrieve(request.query, source_filter=request.source_filter)
    return {"sources": chunks}

@app.post("/stream")
def stream_answer(request: StreamRequest):
    return StreamingResponse(
        generate_answer_stream(request.query, request.chunks, request.history), 
        media_type="text/event-stream"
    )

import shutil
from db.weaviate_client import get_client
from ingestion.loader import load_pdf
from ingestion.cleaner import clean_text
from ingestion.chunker import chunk_text
from embeddings.embedder import get_embedding

@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Only PDF files are supported"}
        
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    file_path = os.path.join(data_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        text = load_pdf(file_path)
        cleaned = clean_text(text)
        chunks = chunk_text(cleaned)
        
        client = get_client()
        collection = client.collections.get("PaperChunk")
        
        for chunk in chunks:
            embedding = get_embedding(chunk)
            collection.data.insert(
                properties={
                    "text": chunk,
                    "source": file.filename
                },
                vector=embedding
            )
            
        return {"message": "Success", "filename": file.filename, "chunks_inserted": len(chunks)}
    except Exception as e:
        # cleanup if failed
        if os.path.exists(file_path):
            os.remove(file_path)
        return {"error": str(e)}