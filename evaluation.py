# evaluation.py

"""Retrieval‑ and response‑level evaluation via RagAS."""
import copy
import json
from typing import List, Dict

from ragas.metrics import context_precision, answer_relevancy
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.dataset_schema import EvaluationDataset   # RagAS v0.2.x
from ragas.run_config import RunConfig



from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from config import settings
from llm_wrapper import LLMWrapper

import tqdm.auto
from contextlib import contextmanager


@contextmanager
def ragas_tqdm_position(pos: int):
    real_tqdm = tqdm.auto.tqdm
    def patched_tqdm(*args, **kwargs):
        kwargs.setdefault("position", pos)
        return real_tqdm(*args, **kwargs)
    tqdm.auto.tqdm = patched_tqdm
    try:
        yield
    finally:
        tqdm.auto.tqdm = real_tqdm

class Evaluator:
    """Wrap ragas.evaluate() for a list of prediction dicts."""

    def __init__(self):
        # Load the ground‑truth QA set you generated in `build`
        with open("ground_truth.json", "r") as f:
            gt = json.load(f)
        
        self.gt_answer = { item["question"]: item["answer"] for item in gt }
        self.gt_chunk  = { item["question"]: item["chunk_text"] for item in gt }

        # Build an LCEL‑compatible AzureChat LLM
        self.raw_llm = AzureChatOpenAI(
            verbose=True,
            azure_endpoint=settings.OPENAI_ENDPOINT,
            api_version="2024-02-15-preview",
            azure_deployment=settings.LLM_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.8,
        )
        self.llm = self.raw_llm

        # Embeddings for answer_relevancy
        self.emb = AzureOpenAIEmbeddings(
            azure_endpoint=settings.OPENAI_ENDPOINT,
            azure_deployment=settings.EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY,
            api_version="2024-02-15-preview",
        )

        # Configure the answer_relevancy metric
        self.answer_rel = copy.deepcopy(answer_relevancy)
        self.answer_rel.llm = LangchainLLMWrapper(self.llm)
        self.answer_rel.embeddings = LangchainEmbeddingsWrapper(self.emb)

    def score(self, predictions: List[Dict]) -> Dict[str, float]:
        """
        Expects each pred dict to have keys:
          - user_input         (the query string)
          - prediction         (the model's answer)
          - retrieved_contexts (list of raw chunk texts)
        Uses the ground truth map for reference (string).
        """
        records = []
        for p in predictions:
            q = p["user_input"]
            pred_answer = p["prediction"]

            
            contexts    = p["retrieved_contexts"]
            
            true_ans   = self.gt_answer[q]
            true_chunk = self.gt_chunk[q]
            
            records.append({
                "user_input":         q,
                "response":         pred_answer,
                "reference":          true_ans,          
                "retrieved_contexts": contexts,
                "ground_truths":       [true_chunk],  
            })

        # Build a RagAS EvaluationDataset and run metrics

        eval_dataset = EvaluationDataset.from_list(records)
        #print(f"DEBUG: Type of eval_dataset: {type(eval_dataset)}")
        #print(f"DEBUG: One record to see the values - Record: {eval_dataset}")
        return_scores = None
        try:
            with ragas_tqdm_position(2):
                eval_result = evaluate(
                    dataset=EvaluationDataset.from_list(records),
                    show_progress=True,
                    llm=LangchainLLMWrapper(self.llm),
                    embeddings=LangchainEmbeddingsWrapper(self.emb),
                    metrics=[
                        context_precision,   # retrieval metric
                        self.answer_rel,     # answer quality metric
                        
                    ],
                    raise_exceptions=True,
                    run_config=RunConfig(max_workers=1)
                )
            return_scores = eval_result.scores[0]
            print(f"DEBUG: Evaluation result : {eval_result.scores} and type of the score {type(eval_result.scores)}")
        except Exception as e:
            #print(f"Error during evaluation: {e.with_traceback()}")
            #print(f"DEBUG DEFAUL VALUES: {eval_result}")
            eval_result = {
                "context_precision": 0.0,
                "answer_relevancy": 0.0,
            }
            return_scores = eval_result
       
        print(f"DEBUG: Final evaluation result: {return_scores} and printing the context_precision {(return_scores.get('context_precision'))} and answer_relevancy {(return_scores.get('answer_relevancy'))}")
        return return_scores