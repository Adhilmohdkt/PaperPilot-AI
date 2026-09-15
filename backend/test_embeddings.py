import os
import pytest

if __name__ != "__main__":
    pytest.skip("Manual embedding diagnostic; not a pytest test module.", allow_module_level=True)
from langchain_google_genai import GoogleGenerativeAIEmbeddings

api_key = os.getenv("GEMINI_API_KEY")
print(f"API Key present: {bool(api_key)}")

models_to_try = [
    "models/text-embedding-004",
    "text-embedding-004",
    "models/gemini-embedding-001",
    "gemini-embedding-001"
]

for model in models_to_try:
    try:
        print(f"\nTrying model: {model}...")
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=api_key,
            model=model
        )
        res = embeddings.embed_query("Hello world")
        print(f"✅ Success! Dimension: {len(res)}")
    except Exception as e:
        print(f"❌ Failed: {e}")
