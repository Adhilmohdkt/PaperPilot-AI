"""Chunk LangChain Documents while preserving their citation metadata."""
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents: List) -> List:
    """Create deterministic chunks without making extra embedding API calls."""
    return RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        add_start_index=True,
    ).split_documents(documents)
