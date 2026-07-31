from operations.parser import load_pdf
from operations.chunker import create_chunks
from operations.embedding import create_embeddings
from knowlegdeBase.collection import add_document


# =====================================================
# Configuration
# =====================================================

PDF_PATH = "ML_Bias_Variance_Interview_Guide.pdf"


# =====================================================
# Build Vector Database
# =====================================================

print("Loading PDF...")

documents = load_pdf(PDF_PATH)

print("Creating chunks...")

chunks = create_chunks(documents)

print("Creating embeddings...")

embeddings = create_embeddings(chunks)

print("Adding documents to ChromaDB...")

add_document(
    chunks=chunks,
    embeddings=embeddings,
    source=PDF_PATH
)

print("\n✅ Index built successfully!")
