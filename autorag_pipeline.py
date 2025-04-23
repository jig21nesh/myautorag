# autorag_pipeline.py


import json # <-- Add this import
import pathlib # <-- Add this import
from typing import Dict, Any
from greedy_search import GreedyAutoRAG
from pdf_loader import PDFChunker
from embedder import Embedder
from qa_generator import QAGenerator

class AutoRAGPipeline:
    """
    Orchestrates the AutoRAG flow:
      1) In __init__: chunk & embed the PDF, build ground truth QA,
         then run greedy optimisation to pick the best modules.
      2) In __call__: run the optimised pipeline on a new question,
         returning the answer, the prompt used, and the raw retrieved
         contexts (for evaluation).
    """
    def __init__(self, pdf_path: str):
        gt_path = pathlib.Path("ground_truth.json")
        gt = None # Initialize gt

        if gt_path.exists():
            print(f"Loading existing ground truth from: {gt_path}")
            try:
                with gt_path.open("r") as f:
                    gt = json.load(f)
                # Basic validation: Check if it's a list (common format)
                if not isinstance(gt, list):
                    print(f"Warning: Content in {gt_path} is not a list. Attempting generation.")
                    gt = None # Reset gt to trigger generation
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {gt_path}. Will regenerate.")
                gt = None # Reset gt to trigger generation
            except Exception as e:
                print(f"Error loading {gt_path}: {e}. Will regenerate.")
                gt = None # Reset gt to trigger generation

        # If gt is still None (file didn't exist or failed to load/validate)
        if gt is None:
            print(f"Ground truth file not found or invalid. Processing PDF and generating ground truth...")
            # Perform chunking, embedding, and GT generation only if the file doesn't exist or was invalid
            chunks = PDFChunker(pdf_path).load_chunks()
            Embedder().ingest(chunks) # Assumes embeddings should also be fresh if GT is generated
            gt = QAGenerator().build_ground_truth(chunks) # This saves the file for next time

        # Ensure gt is valid before proceeding
        if not gt or not isinstance(gt, list):
             raise RuntimeError("Failed to load or generate valid ground truth data.")

        # Proceed with optimisation using the loaded or generated ground truth
        print(f"Optimising pipeline using {len(gt)} ground truth entries...")
        self.pipeline = GreedyAutoRAG(gt).optimise()
        print("Optimisation complete.")

    def __call__(self, question: str) -> Dict[str, Any]:
        expanded_query = self.pipeline["query_expansion"](question)

        docs = self.pipeline["retrieval"](expanded_query, k=10)
        retrieved_texts = [doc.page_content for doc in docs]

        aug_docs = self.pipeline["augmentation"](docs)

        reranked_docs = self.pipeline["reranker"](question, aug_docs, top_k=5)

        prompt = self.pipeline["prompt_maker"](question, reranked_docs)

        answer, _ = self.pipeline["generator"](prompt, reranked_docs)

        return {
            "question":            question,
            "answer":              answer,
            "prompt":              prompt,
            "retrieved_contexts":  retrieved_texts,
        }