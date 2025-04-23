import json
import pathlib
from typing import Optional, Dict, Any, List

from greedy_search import GreedyAutoRAG
from pdf_loader import PDFChunker
from azure_index_loader import AzureIndexLoader
from embedder import Embedder
from qa_generator import QAGenerator
from langchain.schema import Document


class AutoRAGPipeline:
    """
    Two separate flows:
      • build_index()  → chunk or load index + QA gen only
      • ask_via_pdf() / ask_via_index() → load GT + optimize + answer
    """

    def __init__(self, pipeline):
        # private ctor: pipeline is the dict of best‐inferred modules
        self.pipeline = pipeline

    @classmethod
    def build_from_pdf(cls, pdf_path: str, qa_per_chunk: int = 2, overwrite: bool = False) -> None:
        """
        1) chunk & embed PDF
        2) generate QA ground truth and save ground_truth.json
        (no optimisation)
        """
        gt_path = pathlib.Path("ground_truth.json")
        if gt_path.exists() and not overwrite:
            print(f"✅ ground_truth.json already exists (use --overwrite to rebuild).")
            return
        print("📄 Chunking PDF…")
        chunks = PDFChunker(pdf_path).load_chunks()
        print("🔌 Embedding chunks…")
        Embedder().ingest(chunks)
        print("✍️ Generating QA ground truth…")
        QAGenerator(qa_per_chunk=qa_per_chunk).build_ground_truth(chunks)
        print("✅ build complete.")

    @classmethod
    def build_from_index(cls, index_name: str, qa_per_chunk: int = 2, overwrite: bool = False) -> None:
        """
        1) load chunks from Azure Search
        2) generate QA ground truth (same format)
        """
        gt_path = pathlib.Path("ground_truth.json")
        if gt_path.exists() and not overwrite:
            print(f"✅ ground_truth.json already exists (use --overwrite to rebuild).")
            return
        print(f"🔍 Loading chunks from index '{index_name}'…")
        chunks = AzureIndexLoader(index_name).load_chunks()
        print("✍️ Generating QA ground truth…")
        QAGenerator(qa_per_chunk=qa_per_chunk).build_ground_truth(chunks)
        print("✅ build complete.")

    @classmethod
    def ask_via_pdf(cls, pdf_path: str, qa_per_chunk: int = 2) -> "AutoRAGPipeline":
        """
        1) ensure ground_truth.json exists (or build it)
        2) optimize the pipeline
        """
        if not pathlib.Path("ground_truth.json").exists():
            cls.build_from_pdf(pdf_path, qa_per_chunk=qa_per_chunk)
        with open("ground_truth.json") as f:
            gt = json.load(f)
        print(f"🚀 Optimising pipeline on {len(gt)} GT entries…")
        best_pipeline = GreedyAutoRAG(gt).optimise()
        print("✅ optimisation done.")
        return cls(best_pipeline)

    @classmethod
    def ask_via_index(cls, index_name: str, qa_per_chunk: int = 2) -> "AutoRAGPipeline":
        """
        1) ensure ground_truth.json exists (or build it via index)
        2) optimize the pipeline
        """
        if not pathlib.Path("ground_truth.json").exists():
            cls.build_from_index(index_name, qa_per_chunk=qa_per_chunk)
        with open("ground_truth.json") as f:
            gt = json.load(f)
        print(f"🚀 Optimising pipeline on {len(gt)} GT entries…")
        best_pipeline = GreedyAutoRAG(gt).optimise()
        print("✅ optimisation done.")
        return cls(best_pipeline)

    def __call__(self, question: str) -> Dict[str, Any]:
        q2 = self.pipeline["query_expansion"](question)
        docs = self.pipeline["retrieval"](q2, k=10)
        texts = [d.page_content for d in docs]
        docs2 = self.pipeline["augmentation"](docs)
        docs3 = self.pipeline["reranker"](question, docs2, top_k=5)
        prompt = self.pipeline["prompt_maker"](question, docs3)
        answer, _ = self.pipeline["generator"](prompt, docs3)
        return {
            "question": question,
            "prompt": prompt,
            "retrieved_contexts": texts,
            "answer": answer,
        }