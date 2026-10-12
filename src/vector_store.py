import hashlib
from pathlib import Path
from typing import cast

import chromadb
from chromadb.api.types import Embeddings

from src.embeddings import embed_texts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "research_documents_minilm"


def get_collection():
    """Open or create the persistent document collection."""

    client = chromadb.PersistentClient(path=str(DB_PATH))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    return collection


def index_chunks(chunks: list[dict]) -> int:
    """Embed chunks and store their text and metadata."""

    if not chunks:
        return 0

    collection = get_collection()

    texts = [chunk["text"] for chunk in chunks]
    vectors = cast(Embeddings, embed_texts(texts))

    ids = []
    metadatas = []

    for chunk in chunks:
        identity = (
            f"{chunk['source']}|" f"{chunk['page_number']}|" f"{chunk['chunk_index']}"
        )

        chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()

        ids.append(chunk_id)

        metadatas.append(
            {
                "source": chunk["source"],
                "page_number": int(chunk["page_number"]),
                "chunk_index": int(chunk["chunk_index"]),
            }
        )

    # Upsert inserts new chunks or updates existing IDs.
    collection.upsert(
        ids=ids,
        embeddings=vectors,
        documents=texts,
        metadatas=metadatas,
    )

    return len(chunks)


def search_chunks(question: str, top_k: int = 5) -> list[dict]:
    """Return the most relevant chunks for a question."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    collection = get_collection()

    if collection.count() == 0:
        return []

    query_vector = embed_texts([question])[0]

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"]
    metadatas = results["metadatas"]
    distances = results["distances"]

    if not documents or not metadatas or not distances:
        raise RuntimeError("Vector store query did not return the requested fields.")

    matches = []

    for i, text in enumerate(documents[0]):
        matches.append(
            {
                "text": text,
                "source": metadatas[0][i]["source"],
                "page_number": metadatas[0][i]["page_number"],
                "chunk_index": metadatas[0][i]["chunk_index"],
                "distance": distances[0][i],
            }
        )

    return matches
