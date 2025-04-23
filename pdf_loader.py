# pdf_loader.py
"""Load a PDF and split it into overlapping chunks."""
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import settings
from typing import List


class PDFChunker:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path              # <- keep for metadata
        self.loader = PyPDFLoader(pdf_path)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

    def load_chunks(self):
        pages = self.loader.load()                    # page‑level docs
        
        page_texts: List[str] = [page.page_content for page in pages]

        docs = self.splitter.split_documents(pages)   # chunk‑level docs

        for chunk in docs:
            md = chunk.metadata
            # If PyPDFLoader gave you a page number, reuse it;
            # otherwise you may need to parse md['loc'] or similar—but typically
            # it carries over the page index as `md['page']`.
            page_idx = md.get("page", None)
            if page_idx is None:
                # fallback: if metadata looks like md['source_page'] or similar, adjust here
                page_idx = None

            md["source"] = self.pdf_path
            md["page"] = page_idx
            md["source_text"] = page_texts
        return docs