"""
Prompt-template builders
• FStringPrompt         - classic chat prompt, passages in order
• LongContextPrompt     - duplicates best passage at end to mitigate 'lost in middle'
"""
from typing import List
from langchain.schema import Document
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from config import settings

class FStringPrompt:
    name = "f_string"

    def __call__(self, query: str, docs: List[Document]) -> str:
        context = "\n\n".join(d.page_content for d in docs)
        return (
            "You are a helpful assistant.\n"
            "Answer the user strictly using the context below.\n"
            "---\n"
            f"{context}\n"
            "---\n"
            f"Question: {query}\nAnswer:"
        )


class LongContextPrompt:
    name = "long_context_reorder"

    def __call__(self, query: str, docs: List[Document]) -> str:
        if not docs:
            return FStringPrompt()(query, docs)
        # move most relevant (first) doc to both front and end
        best = docs[0].page_content
        remaining = "\n\n".join(d.page_content for d in docs[1:])
        context = f"{best}\n\n{remaining}\n\n{best}"
        return (
            "You are a helpful assistant.\n"
            "Use the following context to answer. If the answer isn't there, say you don't know.\n"
            "---\n"
            f"{context}\n"
            "---\n"
            f"Question: {query}\nAnswer:"
        )

class DynamicPrompt:
    name = "dynamic_llm"

    def __init__(self):
        self.llm = AzureChatOpenAI(
            verbose=True,
            azure_endpoint=settings.OPENAI_ENDPOINT,
            api_version="2024-02-15-preview",
            azure_deployment=settings.LLM_MODEL,
            temperature=0.3,
            api_key=settings.OPENAI_API_KEY,
        )

    def __call__(self, query: str, docs: List[Document]) -> str:
        # 1) assemble the context passages
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        # 2) create a “meta-prompt” asking the model to author an optimized prompt
        meta_prompt = [
            SystemMessage(
                content=(
                    "You are an expert prompt engineer.  "
                    "Given the following context and a user question, "
                    "craft a single string that includes both a system "
                    "instruction and a user instruction, designed to get "
                    "the best possible answer from a helpful assistant."
                )
            ),
            HumanMessage(
                content=(
                    f"CONTEXT:\n{context}\n\n"
                    f"QUESTION: {query}\n\n"
                    "Now produce:\n"
                    "  1. A brief system message (e.g. tone, persona, rules) and\n"
                    "  2. A user message that includes the question and refers to the context.\n"
                    "Put them together exactly as ChatCompletion messages."
                )
            )
        ]
        # 3) call the LLM to generate that prompt
        response = self.llm.generate([meta_prompt])
        # 4) the LLM will return something like:
        #    System: “You are a….”\nUser: “Given the above context…” 
        #    which we just hand back as our “prompt” string
        return response.generations[0][0].text