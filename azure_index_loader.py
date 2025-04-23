# azure_index_loader.py
from typing import List
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from langchain.schema import Document

from config import settings

class AzureIndexLoader:
    """
    Pull every document (chunk) out of an Azure Cognitive Search index
    and return it as a List[langchain.schema.Document].
    Expects each record in the index to have:
      - 'id'           → a unique chunk_id
      - 'content'      → the chunk text
      - optionally 'page' → page number
    """
    def __init__(self, index_name: str):
        self.client = SearchClient(
            endpoint=settings.AZURE_SEARCH_ENDPOINT,
            index_name=index_name,
            credential=AzureKeyCredential(settings.AZURE_SEARCH_KEY),
        )

    def load_chunks(self) -> List[Document]:
        docs = []
        # you can tune `search_text="*"` & `top=` as needed
        results = self.client.search(search_text="*", top=1000)
        for hit in results:
            metadata = {
                "chunk_id": hit["id"],
                "page": hit.get("page", 0),
            }
            docs.append(Document(
                page_content=hit["content"],
                metadata=metadata
            ))
        return docs