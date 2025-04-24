# modules/query_expansion/base.py
from abc import ABC, abstractmethod

class QueryExpander(ABC):
    """All query‐expansion modules must subclass this."""
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def __call__(self, query: str) -> str:
        ...