# modules/passage_augmentation.py

from typing import List
from langchain.schema import Document

class NoAugment:
    """Return the retrieved docs unchanged."""
    name = "pass"

    def __call__(self, docs: List[Document]) -> List[Document]:
        return docs


class PrevNextAugment:
    """
    For each retrieved doc, also include its immediate neighbor(s)
    in the returned list.  Doesn't rely on any `source_text` array.
    """
    name = "prev_next"

    def __init__(self, mode: str = "both"):
        assert mode in {"prev", "next", "both"}
        self.mode = mode

    def __call__(self, docs: List[Document]) -> List[Document]:
        augmented: List[Document] = []
        n = len(docs)

        for i, doc in enumerate(docs):
            # include the doc itself
            augmented.append(doc)

            # include previous if asked and exists
            if self.mode in ("prev", "both") and i > 0:
                augmented.append(docs[i - 1])

            # include next if asked and exists
            if self.mode in ("next", "both") and i < n - 1:
                augmented.append(docs[i + 1])

        # de-duplicate while preserving order
        seen = set()
        uniq: List[Document] = []
        for d in augmented:
            # use a stable key: e.g. metadata id if present, else page_content
            key = d.metadata.get("id", d.page_content[:100])
            if key not in seen:
                seen.add(key)
                uniq.append(d)

        return uniq