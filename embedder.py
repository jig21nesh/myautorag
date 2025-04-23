from typing import List
from db import VectorDB

class Embedder:
    def __init__(self):
        self.db = VectorDB()

    def ingest(self, docs: List):
        self.db.upsert(docs)