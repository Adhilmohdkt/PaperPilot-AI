from weaviate.classes.config import Property, DataType
from db.weaviate_client import get_client

def create_schema():
    client = get_client()

    if client.collections.exists("PaperChunk"):
        print("Deleting existing PaperChunk schema to upgrade layout...")
        client.collections.delete("PaperChunk")

    client.collections.create(
        name="PaperChunk",
        properties=[
            Property(
                name="text",
                data_type=DataType.TEXT   
            ),
            Property(
                name="source",
                data_type=DataType.TEXT
            )
        ]
    )

    print("New Schema created successfully!")


if __name__ == "__main__":
    create_schema()