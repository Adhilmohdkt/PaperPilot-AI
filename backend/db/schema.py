"""Weaviate collection schema for PaperPilot local RAG."""

from __future__ import annotations

from weaviate.classes.config import DataType, Property

from config import settings
from db.weaviate_client import get_client


EMBEDDING_MODEL = "gemini-embedding-2"


SCHEMA_PROPERTIES = [
    Property(name="text", data_type=DataType.TEXT),
    Property(name="source", data_type=DataType.TEXT),
    Property(name="content_hash", data_type=DataType.TEXT),
    Property(name="chunk_id", data_type=DataType.TEXT),
    Property(name="page", data_type=DataType.INT),
    Property(name="embedding_model", data_type=DataType.TEXT),
]


def ensure_schema(client=None):
    """Create or safely extend the collection without deleting stored vectors."""
    owns_client = client is None
    client = client or get_client()

    try:
        if not client.collections.exists(settings.collection_name):
            client.collections.create(
                name=settings.collection_name,
                properties=SCHEMA_PROPERTIES,
            )
            return client.collections.get(settings.collection_name)

        collection = client.collections.get(settings.collection_name)

        existing = {
            prop.name
            for prop in collection.config.get().properties
        }

        for prop in SCHEMA_PROPERTIES:
            if prop.name not in existing:
                collection.config.add_property(prop)

        return collection

    finally:
        if owns_client:
            client.close()


def create_schema():
    """Create or update the PaperPilot local RAG schema."""
    ensure_schema()
    print("PaperPilot Weaviate schema is ready.")


if __name__ == "__main__":
    create_schema()