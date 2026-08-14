import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import tokenizer

splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=500,
    chunk_overlap=100,
    separators=[
        "\n\n",
        "\n",
        ". ",
        "? ",
        "! ",
        " ",
        ""
    ]
)

MIN_CHUNK_LENGTH = 100


def is_citation_block(text: str) -> bool:
    citation_markers = len(re.findall(
        r'\bet al\.|arXiv preprint|Proceedings of\b', text))
    numbered_lines = len(re.findall(r'^\d+[\s.]', text, re.MULTILINE))
    return citation_markers >= 2 or numbered_lines >= 2


def create_chunks(documents):

    final_chunks = []

    for doc in documents:
        for chunk in splitter.split_text(doc.page_content):
            if len(chunk.strip()) < MIN_CHUNK_LENGTH:
                continue
            if is_citation_block(chunk.strip()):
                continue
            final_chunks.append(chunk)

    return final_chunks
