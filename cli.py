# cli.py

import argparse
import sys
from pathlib import Path

from autorag_pipeline import AutoRAGPipeline
from config import settings


def main():
    parser = argparse.ArgumentParser(prog="cli.py")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- BUILD subcommand ---
    build = sub.add_parser("build", help="Create ground truth & index")
    grp_b = build.add_mutually_exclusive_group(required=True)
    grp_b.add_argument("--pdf", type=Path,
                       help="Path to PDF file to chunk & index")
    grp_b.add_argument("--index-name", type=str,
                       help="Azure Search index name to read from")

    # --- ASK subcommand ---
    ask = sub.add_parser("ask", help="Optimize pipeline & answer")
    grp_a = ask.add_mutually_exclusive_group(required=True)
    grp_a.add_argument("--pdf", type=Path,
                       help="Path to PDF file (if reusing local index)")
    grp_a.add_argument("--index-name", type=str,
                       help="Azure Search index name to query")
    ask.add_argument("--q", "--question", dest="question", required=True,
                     help="The question to ask")

    args = parser.parse_args()

    if args.cmd == "build":
        if args.pdf:
            AutoRAGPipeline.build_from_pdf(str(args.pdf))
        else:
            # Azure‐index mode; endpoint is drawn from config.AZURE_SEARCH_ENDPOINT
            if not settings.AZURE_SEARCH_ENDPOINT:
                sys.exit("ERROR: AZURE_SEARCH_ENDPOINT not set in config")
            AutoRAGPipeline.build_from_index(
                index_name=args.index_name,
                qa_per_chunk=settings.QA_PER_CHUNK,
            )

    elif args.cmd == "ask":
        if args.pdf:
            pipeline = AutoRAGPipeline.ask_via_pdf(str(args.pdf))
        else:
            if not settings.AZURE_SEARCH_ENDPOINT:
                sys.exit("ERROR: AZURE_SEARCH_ENDPOINT not set in config")
            pipeline = AutoRAGPipeline.ask_via_index(
                index_name=args.index_name,
            )
        out = pipeline(args.question)
        print("\n➤ PROMPT\n", out["prompt"])
        print("\n➤ CONTEXTS\n", *out["retrieved_contexts"], sep="\n\n---\n\n")
        print("\n➤ ANSWER\n", out["answer"])


if __name__ == "__main__":
    main()