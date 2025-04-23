# MyAutoRAG

A command‑line toolkit for automatically optimizing Retrieval‑Augmented Generation (RAG) pipelines using RAGAS and LangChain on local PDFs or external search indexes.



## Overview
- **Build**: Chunk a PDF (or existing Azure Search index), embed it, index into PGVector or query the index.
- **Ask**: Run a question through the optimized pipeline (greedy‑search over query expansion, retrieval, augmentation, reranking, prompt construction, generation) to produce an answer.
- **Optimize**: Greedy search picks the best module at each RAG step based on aggregated context_precision (and optional answer_relevancy).


## Prerequisites
	1.	Python 3.9+
	2.	PostgreSQL with PGVector extension (if using local indexing)
	3.	Azure OpenAI resource (or compatible OpenAI API)
	4.	Environment variables (see below)



## Installation

```bash
# clone repo
git clone https://github.com/jig21nesh/myautorag.git
cd myautorag

# create & activate virtual environment
python -m venv .env
source .env/bin/activate

# install dependencies
pip install -r requirements.txt
```





## Environment Variables

| Variable          | Description                                                                                          |
|-------------------|------------------------------------------------------------------------------------------------------|
| `OPENAI_API_KEY`  | Your Azure/OpenAI API key (for chat & embeddings)                                                     |
| `PGVECTOR_URL`    | PostgreSQL URL with PGVector, e.g. `postgresql://user:pass@host:port/dbname`                           |
| `LLM_MODEL`       | (optional) Azure deployment name for the LLM<br>Default: `gpt-4o-mini`                                  |
| `EMBEDDING_MODEL` | (optional) Azure deployment for embeddings<br>Default: `text-embedding-ada-002`                        |

> **Note:** you can also point to an existing Azure Search index instead of PGVector.


## Configuration

- All hyper‑parameters and file paths live in config.py (via Pydantic Settings).
- CHUNK_SIZE, CHUNK_OVERLAP: controls PDF chunking.
- QA_PER_CHUNK: number of synthetic QA to generate per chunk.
- COLLECTION: PGVector collection name.



## Commands

### Build (generate ground truth & index)

```bash
# using a local PDF + PGVector
python cli.py build \
  --pdf ./data/YourDocument.pdf
```

- Splits PDF into chunks, generates QA ground truth, embeds and writes to PGVector.
- Outputs ground_truth.json in the repo root.

### Ask (optimize & answer)

#### ask against the same PDF/index
```bash
python cli.py ask \
  --pdf ./data/YourDocument.pdf \
  --q "What is the main takeaway?"
```

- Loads ground_truth.json, runs greedy optimization across RAG modules, then answers your question.
- Prints the final prompt, the retrieved contexts, and the generated answer.

### (Future) Ask via Azure Search Index

```bash
python cli.py ask \
  --index-endpoint https://<your-search>.search.windows.net \
  --index-name my-index \
  --q "How many chapters?"
```
- Skips PDF chunking, instead queries your existing Azure Search index.

⸻

## Internals

### GreedyAutoRAG (in `greedy_search.py`)
- Iterates over each RAG node: `query_expansion`, `retrieval`, `augmentation`, `reranker`, `prompt_maker`, `generator`.  
- For each candidate module:
  - Runs the full pipeline on every ground-truth question.  
  - Collects per-sample metrics (e.g. context_precision).  
  - Aggregates to compute a mean score.  
- Selects and locks in the module with the highest mean `context_precision` (or custom weighted metric) before moving to the next node.

### RAGAS Evaluation (in `evaluation.py`)
- Wraps `ragas.evaluate(...)` to compute:
  - `context_precision` (retrieval accuracy)  
  - `answer_relevancy` (LLM response quality)  
- Aggregates per‐sample scores across all records into a single float for the optimizer to consume.

### Module Registry (in `search_space.py`)
- Defines the list of candidate classes for each RAG node.  
- To extend:  
  1. Create your new module class under `modules/`.  
  2. Import and add it to the appropriate list in `SEARCH_SPACE`.  

⸻

### Ground Truth Format

ground_truth.json is a list of objects:

```json
[
  {
    "question": "What is X?",
    "answer":   "X is ...",
    "chunk_id": "data/doc.pdf_0",
    "chunk_text": "... original chunk content ..."
  },
  …
]
```

This file is used both for greedy optimization and (optionally) debugging.

⸻

## Extending & Contributing
- Add new modules: drop your class under modules/, then add to SEARCH_SPACE.
- Custom metrics: integrate extra RAGAS metrics by updating evaluation.py.
- Prompt engineering: edit prompt_maker.py or implement a dynamic prompt generator.

Feel free to open issues or PRs!

⸻

📜 License

MIT © Your Name