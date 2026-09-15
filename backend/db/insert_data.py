"""Index the checked-in seed library without re-embedding unchanged PDFs."""
from pathlib import Path

from ingestion.library import index_pdf


def insert_data(data_directory: str | Path | None = None) -> list[dict]:
    directory = Path(data_directory) if data_directory else Path(__file__).resolve().parents[2] / "Data"
    if not directory.exists():
        print(f"Data directory {directory} does not exist.")
        return []
    results = []
    for pdf in sorted(directory.glob("*.pdf")):
        result = index_pdf(pdf)
        results.append(result)
        print(f"{result['filename']}: {result['status']} ({result['chunks_inserted']} chunks)")
    return results


if __name__ == "__main__":
    insert_data()
