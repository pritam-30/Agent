from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
