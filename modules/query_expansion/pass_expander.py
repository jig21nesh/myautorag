
from .base import QueryExpander

class PassExpander(QueryExpander):
    name = "none"
    def __call__(self, query: str) -> str:
        return query