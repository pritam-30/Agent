from langchain_community.document_loaders import PyMuPDFLoader


def load_pdf(pdf_path):
    """
    Load a PDF and return a list of LangChain Document objects.
    """
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    return documents
