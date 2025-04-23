# greedy_search.py

from typing import List, Dict, Any
from tqdm.auto import tqdm

from evaluation import Evaluator
from search_space import SEARCH_SPACE
from functools import reduce
from operator import mul

class GreedyAutoRAG:
    """
    Greedy optimisation over each RAG node in SEARCH_SPACE:
    - For each node, try every candidate module in isolation (keeping others fixed)
      and pick the one with the highest context_precision on the ground truth.
    """
    def __init__(self, ground_truth: List[Dict[str, Any]]):
        # ground_truth is a list of dicts: {"question": str, "answer": str}
        self.gt = ground_truth
        self.evaluator = Evaluator()
        # initialize pipeline to first module of each node
        self.pipeline = { node: modules[0] for node, modules in SEARCH_SPACE.items() }

    def optimise(self) -> Dict[str, Any]:

        # --- print summary of search space ---
        counts = [len(cands) for cands in SEARCH_SPACE.values()]
        total_greedy = sum(counts)
        total_full    = reduce(mul, counts, 1)

        print(f"📊 Greedy will perform {total_greedy} trials "
            f"(one per candidate, per node)")
        print(f"🌐 Full combinatorial space is {total_full} pipelines\n")

        print(f"\n🚀 Starting greedy optimisation over {len(SEARCH_SPACE)} nodes…\n")    

        # greedy loop over each node
        for node_name, candidates in tqdm(SEARCH_SPACE.items(),
                                          desc="Optimising nodes", 
                                          position=0, 
                                          leave=True):
            
            print(f"\n🔍 Node '{node_name}' → {len(candidates)} candidate(s) to try")
            
            best_score = None
            best_mod   = self.pipeline[node_name]

            for mod in tqdm(candidates,
                        desc=f"  Candidates for {node_name}",
                        leave=False, 
                        position=1):

                mod_name = mod.__class__.__name__
                self.pipeline[node_name] = mod
                
                current_cfg = {n: m.__class__.__name__ for n, m in self.pipeline.items()}
                print(f"  ▶ Trying candidate '{mod_name}'. Pipeline now: {current_cfg}")

                
                preds = self._run_pipeline()
                score = self._score(preds)

                print(f"    ↳ Score for '{mod_name}': {score:.4f}")

                if best_score is None or score > best_score:
                    best_score = score
                    best_mod   = mod

            # lock in the best for this node
            best_name = best_mod.__class__.__name__
            print(f"✅ Best for node '{node_name}': '{best_name}' (score={best_score:.4f})\n")
            self.pipeline[node_name] = best_mod

        final_cfg = {n: m.__class__.__name__ for n, m in self.pipeline.items()}
        print("🎉 Greedy optimisation complete. Final pipeline configuration:")
        for node, mod_name in final_cfg.items():
            print(f"   • {node}: {mod_name}")
        print()
        
        return self.pipeline

    def _run_pipeline(self) -> List[Dict[str, Any]]:
        """
        Execute the current pipeline on every ground-truth question,
        collecting the fields RagAS needs:
          - user_input
          - prediction
          - reference
          - retrieved_contexts
        """
        results: List[Dict[str, Any]] = []

        for rec in self.gt:
            
            
            
            question = rec["question"]
            # 1. query expansion
            q2   = self.pipeline["query_expansion"](question)
            # 2. retrieval
            docs = self.pipeline["retrieval"](q2, k=10)
            # record raw texts
            contexts = [d.page_content for d in docs]
            # 3. augmentation
            docs2 = self.pipeline["augmentation"](docs)
            # 4. reranking
            docs3 = self.pipeline["reranker"](question, docs2, top_k=5)
            # 5. prompt
            prompt = self.pipeline["prompt_maker"](question, docs3)
            # 6. generation
            answer, _ = self.pipeline["generator"](prompt, docs3)

            

            # assemble the RagAS-compatible record
            results.append({
                "user_input":         question,
                "prediction":         answer,
                # wrap reference in list if needed
                "reference":          rec["answer"] if isinstance(rec["answer"], list)
                                       else [rec["answer"]],
                "retrieved_contexts": contexts,
            })

        return results

    def _score(self, preds: List[Dict[str, Any]]) -> float:
        """
        Evaluate using RagAS on the supplied preds, returning the
        configured metric (context_precision).
        """
        metrics = self.evaluator.score(preds)


        

        try:
            metrics = self.evaluator.score(preds)
            # Check if the key exists and the value is numeric before conversion
            score_value = metrics.get("context_precision")
            if isinstance(score_value, (int, float)):
                return float(score_value)
            elif isinstance(score_value, str) and score_value.replace('.', '', 1).isdigit():
                 return float(score_value)
            else:
                print(f"Warning: 'context_precision' missing or not a number in RAGAS results: {metrics}. Returning 0.0")
                return 0.0
        except Exception as e:
            print(f"Error during RAGAS evaluation: {e}. Returning 0.0")
            # Log the full metrics dict if needed for debugging
            # print(f"RAGAS metrics result on error: {metrics}")
            return 0.0