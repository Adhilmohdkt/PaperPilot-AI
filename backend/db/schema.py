from weaviate.classes.config import Property, DataType
from config import settings
from db.weaviate_client import get_client

SCHEMA_PROPERTIES = [
    Property(name="text", data_type=DataType.TEXT),
    Property(name="source", data_type=DataType.TEXT),
    Property(name="content_hash", data_type=DataType.TEXT),
    Property(name="chunk_id", data_type=DataType.TEXT),
    Property(name="page", data_type=DataType.INT),
]


def ensure_schema(client=None):
    """Create or safely extend the collection without discarding stored vectors."""
    client = client or get_client()
    if not client.collections.exists(settings.collection_name):
        client.collections.create(name=settings.collection_name, properties=SCHEMA_PROPERTIES)
        return client.collections.get(settings.collection_name)

    collection = client.collections.get(settings.collection_name)
    existing = {property.name for property in collection.config.get().properties}
    for property in SCHEMA_PROPERTIES:
        if property.name not in existing:
            collection.config.add_property(property)
    return collection


def create_schema():
    """Backward-compatible non-destructive schema setup entry point."""
    ensure_schema()
    print("PaperChunk schema is ready.")


if __name__ == "__main__":
    create_schema()
