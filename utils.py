import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from time import perf_counter
from contextlib import contextmanager
load_dotenv()


tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# =========================
notes = []


@contextmanager
def timer(name):
    if os.getenv("ENABLE_TIMING", "False").lower() != "true":
        yield
        return

    start = perf_counter()
    try:
        yield
    finally:
        print(f"{name}: {perf_counter() - start:.2f} s")
