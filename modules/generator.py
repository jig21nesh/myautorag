"""
LLM answer generator (single option for now)
"""
from typing import Tuple, List
from langchain_openai import AzureChatOpenAI
from langchain.schema import Document
from config import settings

class GPTGenerator:
    name = "gpt_gen"

    def __init__(self):
        self.llm = AzureChatOpenAI(
            verbose=True,
            azure_endpoint=settings.OPENAI_ENDPOINT,
            api_version="2024-02-15-preview",
            azure_deployment=settings.LLM_MODEL,
            temperature=0.3,
            api_key=settings.OPENAI_API_KEY,
        )

    def __call__(self, prompt: str, docs: List[Document]) -> Tuple[str, List[Document]]:
        response = self.llm.invoke(prompt).content.strip()
        return response, docs          # return contexts too (for evaluation)