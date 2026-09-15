from langchain_community.document_loaders import PyMuPDFLoader

def load_pdf(path: str):
    loader = PyMuPDFLoader(path)
    return loader.load()