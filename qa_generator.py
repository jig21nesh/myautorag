"""
Generate ground-truth Q-A pairs per chunk.
"""
import json
import pathlib
from typing import List, Dict

from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence

from config import settings

from tqdm.auto import tqdm

_PROMPT = """
You are an expert tutor.
For the passage below, create {k} high quality Q-A pairs answerable **only**
from the passage. Return pure JSON (list of {{'question', 'answer'}} dicts).

PASSAGE:
\"\"\"{text}\"\"\"

JSON:
"""


class QAGenerator:
    def __init__(self, qa_per_chunk: int = None):
        # allow override, otherwise use the global setting
        self.k = qa_per_chunk if qa_per_chunk is not None else settings.QA_PER_CHUNK

        self.llm = AzureChatOpenAI(
            verbose=True,
            azure_endpoint=settings.OPENAI_ENDPOINT,
            api_version="2024-02-15-preview",
            azure_deployment=settings.LLM_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.3,
        )
        self.template = PromptTemplate.from_template(_PROMPT)
        self.chain: RunnableSequence = self.template | self.llm

    def _clean_json(self, raw: str) -> str:
        """
        Strip ```json ... ``` or ``` ... ``` fences if present.
        """
        raw = raw.strip()
        if raw.startswith("```"):
            # drop first and last fence lines
            raw = "\n".join(raw.splitlines()[1:-1]).strip()
        return raw

    def qa_pairs(self, chunk_text: str, k: int) -> List[Dict]:
        resp = self.chain.invoke({"text": chunk_text, "k": k}).content
        cleaned = self._clean_json(resp)
        return json.loads(cleaned)

    def build_ground_truth(self, docs: List) -> List[Dict]:
        """
        For each Document in `docs`, generate self.k Q-A pairs,
        write out ground_truth.json, and return the list.
        """
        out_path = pathlib.Path("ground_truth.json")
        gt: List[Dict] = []

        print(f"DEBUG: QAGenerator - receiving total of {len(docs)} chunks")

        for d in tqdm(docs, desc="Chunks processed", unit="chunk"):
            for qa in self.qa_pairs(d.page_content, self.k):
                qa["chunk_id"]   = d.metadata.get("source", "") + f"_{d.metadata.get('page')}"
                qa["chunk_text"] = d.page_content
                qa["question"]   = qa["question"].strip()
                qa["answer"]     = qa["answer"].strip()
                gt.append(qa)

        with out_path.open("w") as f:
            json.dump(gt, f, indent=2)
        print(f"📝 Ground-truth saved to {out_path.resolve()}")

        return gt