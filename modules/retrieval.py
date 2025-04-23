"""
Retrieval modules
─────────────────
• BM25Retriever           - sparse lexical search
• HybridDBSFRetriever     - manual Distribution-Based Score Fusion
"""

from typing import List, Dict
from collections import defaultdict

from langchain.schema import Document
from db import VectorDB


# ───────────────────────── helpers ──────────────────────────
class _DenseRetriever:
    """Vector similarity search via PGVector."""
    def __init__(self, k: int = 10):
        self.k = k
        self.vdb = VectorDB()

    def __call__(self, query: str) -> List[Document]:
        return self.vdb.vstore.similarity_search(query, k=self.k)


class _BM25Retriever:
    """In-memory BM25 (LangChain community)."""
    def __init__(self, docs: List[Document]):
        from langchain_community.retrievers import BM25Retriever as LC_BM25
        self.bm25 = LC_BM25.from_documents(docs)

    def __call__(self, query: str, k: int = 10) -> List[Document]:
        return self.bm25.invoke(query, k=k)


# ───────────────────────── concrete classes ──────────────────────────
class BM25Retriever:
    name = "bm25"

    def __init__(self, k: int = 10):
        self.k = k
        # grab ALL docs once (simple but memory‑heavy for huge corpora)
        all_docs = VectorDB().vstore.similarity_search(" ", k=100_000)
        self.sparse = _BM25Retriever(all_docs)

    def __call__(self, query: str, k: int | None = None) -> List[Document]:
        return self.sparse(query, k or self.k)


class HybridDBSFRetriever:
    """
    Distribution-Based Score Fusion (DBSF):
        fused_score = α * 1/rank_sparse  +  (1-α) * 1/rank_dense
    where rank is 1-based.
    """
    name = "hybrid_dbsf"

    def __init__(self, alpha: float = 0.7, k: int = 10):
        self.alpha = alpha
        self.k = k

        all_docs = VectorDB().vstore.similarity_search(" ", k=100_000)
        self.sparse = _BM25Retriever(all_docs)
        self.dense  = _DenseRetriever(k=100)              # pull plenty for fusion

    def __call__(self, query: str, k: int | None = None) -> List[Document]:
        k = k or self.k

        sparse_docs = self.sparse(query, k)
        dense_docs  = self.dense(query)

        scores: Dict[str, float] = defaultdict(float)
        for r, doc in enumerate(sparse_docs, start=1):
            scores[doc.page_content] += self.alpha * (1 / r)
        for r, doc in enumerate(dense_docs, start=1):
            scores[doc.page_content] += (1 - self.alpha) * (1 / r)

        # keep first instance of each passage for metadata fidelity
        uniq_docs = {d.page_content: d for d in sparse_docs + dense_docs}

        ranked = sorted(uniq_docs.items(), key=lambda kv: scores[kv[0]], reverse=True)
        return [doc for _, doc in ranked[:k]]