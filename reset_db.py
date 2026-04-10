import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.db.weaviate_client import get_client
from backend.db.schema import create_schema
from backend.db.insert_data import insert_data

client = get_client()
print("Dropping existing PaperChunk collection...")
if client.collections.exists("PaperChunk"):
    client.collections.delete("PaperChunk")

print("Recreating schema...")
create_schema()

print("Re-ingesting all data...")
insert_data()
print("Done!")
