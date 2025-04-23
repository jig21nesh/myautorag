from langchain_openai import AzureChatOpenAI
from config import settings

class PassExpander:
    name = "pass"
    def __call__(self, query: str):
        return query

class HyDEExpander:
    name = "hyde"
    def __init__(self):
        self.llm = AzureChatOpenAI(
            verbose=True,
            azure_endpoint=settings.OPENAI_ENDPOINT,
            api_version="2024-02-15-preview",
            azure_deployment=settings.LLM_MODEL,
            temperature=0.3,
            api_key=settings.OPENAI_API_KEY,
        )
    def __call__(self, query: str):
        prompt = f"Write a concise answer to: {query}"
        return self.llm.invoke(prompt).content.strip()