from langchain_community.document_loaders import PyMuPDFLoader


def load_pdf(pdf_path, start_page: int | None = None, end_page: int | None = None):
    """
    Load a PDF and return a list of LangChain Document objects,
    optionally restricted to a page range (1-indexed, inclusive)
    to skip front matter and back-matter like the index.
    """
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    if start_page is not None or end_page is not None:
        start = (start_page - 1) if start_page else 0
        end = end_page if end_page else len(documents)
        documents = documents[start:end]

    for doc in documents:
        doc.page_content = doc.page_content.replace(
            "xyz.com or links", "").strip()

    return documents
