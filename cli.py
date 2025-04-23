# cli.py
"""
Usage
=====
# 1. Ingest PDF, embed chunks, build ground-truth QA
python cli.py build --pdf ./data/my_corpus.pdf

# 2. Optimise pipeline and answer a question
python cli.py ask  --pdf ./data/my_corpus.pdf --q "What is the warranty period?"
"""




import argparse
from pathlib import Path

# --- common helpers (no heavy retrieval logic) -----------------------------
from pdf_loader import PDFChunker
from embedder import Embedder
from qa_generator import QAGenerator

# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser(description="AutoRAG demo CLI")
ap.add_argument("command", choices=["build", "ask"],
                help="build = ingest & QA | ask = optimise & query")
ap.add_argument("--pdf", required=True, type=Path,
                help="Path to the source PDF used for both modes")
ap.add_argument("--q", help="User question (ask mode only)")
args = ap.parse_args()

if args.command == "build":
    # 1 · load & split PDF
    chunks = PDFChunker(str(args.pdf)).load_chunks()

    # 2 · embed into PGVector
    Embedder().ingest(chunks)

    # 3 · create ground‑truth question/answer pairs
    QAGenerator().build_ground_truth(chunks)

    print("Build finished: chunks embedded + GT generated")

elif args.command == "ask":
    if not args.q:
        raise SystemExit("Error: --q 'question' is required in ask mode")

    # --- lazy import to avoid retrieval modules during 'build' --------------
    from autorag_pipeline import AutoRAGPipeline

    pipeline = AutoRAGPipeline(pdf_path=str(args.pdf))   # runs greedy optimisation internally
    result = pipeline(args.q)

    print("\nAnswer:\n", result["answer"])
    print("\nPrompt used:\n", result["prompt"])