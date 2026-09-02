from operations.parser import load_pdf
from operations.chunker import create_chunks
from operations.embedding import create_embeddings
from knowledgeBase.collection import add_document, delete_collection, get_collection


# =====================================================
# Configuration
# =====================================================

# Path to the PDF file to be processed
PDF_PATH = "XYZ_2023.pdf"

# =====================================================
# Build Vector Database
# =====================================================

print("Loading PDF...")

documents = load_pdf(PDF_PATH, start_page=21, end_page=1003)

print("Creating chunks...")

chunks = create_chunks(documents)
print(f"Total chunks after filtering: {len(chunks)}")

print("Creating embeddings...")

embeddings = create_embeddings(chunks)


# Start fresh
delete_collection()

# Recreate collection
get_collection()

print("Adding documents to ChromaDB...")

add_document(
    chunks=chunks,
    embeddings=embeddings,
    source=PDF_PATH
)

print("\n✅ Index built successfully!")
