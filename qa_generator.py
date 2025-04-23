"""Generate ground-truth Q-A pairs per chunk."""
import json, pathlib
from typing import List, Dict
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from config import settings

_PROMPT = """
You are an expert tutor.
For the passage below, create {k} high quality Q-A pairs answerable **only**
from the passage.  Return pure JSON (list of {{'question', 'answer'}} dicts).

PASSAGE:
\"\"\"{text}\"\"\"

JSON:
"""

class QAGenerator:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            verbose=True,
            azure_endpoint=settings.OPENAI_ENDPOINT,
            api_version="2024-02-15-preview",
            azure_deployment=settings.LLM_MODEL,
            temperature=0.3,
            api_key=settings.OPENAI_API_KEY,
        )
        self.template = PromptTemplate.from_template(_PROMPT)
        self.chain: RunnableSequence = self.template | self.llm

    def _clean_json(self, raw: str) -> str:
        """
        Strip ```json ... ``` or ``` ... ``` fences if present.
        """
        raw = raw.strip()
        if raw.startswith("```"):
            # drop first fence line and last fence line
            raw = "\n".join(raw.splitlines()[1:-1]).strip()
        return raw

    def qa_pairs(self, chunk_text: str, k: int) -> List[Dict]:
        resp = self.chain.invoke({"text": chunk_text, "k": k}).content
        cleaned = self._clean_json(resp)
        return json.loads(cleaned)

    def build_ground_truth(self, docs):
        out_path = pathlib.Path("ground_truth.json")
        gt = []
        for d in docs:
            for qa in self.qa_pairs(d.page_content, settings.QA_PER_CHUNK):
                qa["chunk_id"] = d.metadata.get("source")+f"_{d.metadata.get('page')}"
                qa["chunk_text"] = d.page_content
                qa["question"]   = qa["question"].strip()
                qa["answer"]     = qa["answer"].strip()
                gt.append(qa)
        
        with out_path.open("w") as f:
            json.dump(gt, f, indent=2)
        print(f"📝 Ground-truth saved to {out_path.resolve()}")
        
        return gt