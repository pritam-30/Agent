import uuid
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

COLLECTION_NAME = "documents"


def get_collection():
    """
    Return the ChromaDB collection, creating it if necessary.
    """
    return client.get_or_create_collection(
        name=COLLECTION_NAME
    )


def delete_collection():
    """
    Delete the ChromaDB collection if it exists.
    """
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted collection: {COLLECTION_NAME}")
    except Exception:
        print(f"Collection '{COLLECTION_NAME}' does not exist.")


def add_document(chunks, embeddings, source):
    collection = get_collection()

    ids = [str(uuid.uuid4()) for _ in chunks]

    metadata = [
        {
            "source": source,
            "chunk_index": i
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadata
    )
