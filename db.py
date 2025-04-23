"""Very thin PGVector helper."""
from langchain_community.vectorstores.pgvector import PGVector
from langchain_openai import AzureOpenAIEmbeddings
from config import settings

class VectorDB:
    def __init__(self):
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=settings.OPENAI_ENDPOINT,
            api_key=settings.OPENAI_API_KEY,
            azure_deployment=settings.EMBEDDING_MODEL,   # often identical to model name
            api_version="2024-02-15-preview",
        )
        self.vstore = PGVector(
            connection_string=settings.PGVECTOR_URL,
            collection_name=settings.COLLECTION,
            embedding_function=self.embeddings,
            collection_metadata=settings.METADATA_COLUMNS,   # ← renamed
            use_jsonb=True,
        )

    def upsert(self, docs):
        self.vstore.add_documents(docs)

    def similarity_search(self, query: str, k: int = 10):
        return self.vstore.similarity_search(query, k=k)