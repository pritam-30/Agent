from utils import embedding_model


def create_embeddings(chunks):
    """
    Generate embeddings for a list of text chunks.
    """
    embeddings = embedding_model.encode(
        chunks,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return embeddings
