from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import tokenizer
import re

splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=512,
    chunk_overlap=64
)


def create_chunks(documents):
    """
    Convert LangChain documents into token-sized chunks.
    """

    text = "\n".join(doc.page_content for doc in documents)

    chunks = re.split(r"(?=Q\d+:)", text)

    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    final_chunks = []

    for chunk in chunks:

        num_tokens = len(tokenizer.encode(chunk))

        if num_tokens <= 512:
            final_chunks.append(chunk)

        else:
            smaller_chunks = splitter.split_text(chunk)
            final_chunks.extend(smaller_chunks)

    return final_chunks
