from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 16

_model = None


def get_embedding_model():
    global _model

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)

    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate 384-dimensional embeddings in batches."""
    if not texts:
        return []

    model = get_embedding_model()
    all_embeddings = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]

        vectors = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        all_embeddings.extend(vectors.tolist())

        completed = min(start + len(batch), len(texts))
        print(f"Embedding progress: {completed}/{len(texts)}")

    return all_embeddings
