import uuid
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="documents"
)


def add_document(chunks, embeddings, source):
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
