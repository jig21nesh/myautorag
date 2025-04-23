"""
Two rerankers
• PassReranker           – leave order as‑is
• FlagLLMReranker        – LLM scores relevance; keeps top_k
"""
from typing import List
from langchain_openai import AzureChatOpenAI
from langchain.schema import Document
from config import settings

class PassReranker:
    name = "pass"

    def __call__(self, query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
        return docs[:top_k]


class FlagLLMReranker:
    """
    Simple relevance‑scoring with GPT‑3.5 – mimics FlagLLM reranker idea.
    """
    name = "flag_llm"

    def __init__(self, temperature: float = 0.0):
        self.llm = AzureChatOpenAI(
            verbose=True,
            azure_endpoint=settings.OPENAI_ENDPOINT,
            api_version="2024-02-15-preview",
            azure_deployment=settings.LLM_MODEL,
            temperature=0.3,
            api_key=settings.OPENAI_API_KEY,
        )
        self.prompt_tmpl = (
            "Score (0‑10) how well this passage answers the query.\n"
            "Query: {query}\nPassage: {passage}\nScore:"
        )

    def _score(self, query: str, passage: str) -> float:
        prompt = self.prompt_tmpl.format(query=query, passage=passage[:4000])
        resp = self.llm.invoke(prompt).content.strip()
        try:
            return float(resp)
        except ValueError:
            return 0.0

    def __call__(self, query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
        scored = [(self._score(query, d.page_content), d) for d in docs]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_k]]