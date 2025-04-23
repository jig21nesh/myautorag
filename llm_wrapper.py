class LLMWrapper:
    """
    Thin proxy that delegates every attr/method to the inner LLM
    but supplies a no‑op `.set_run_config()` so RAGAS is satisfied.
    """
    def __init__(self, llm):
        self._llm = llm

    # --- ragas expects this --------------------------------------------------
    def set_run_config(self, *_, **__):
        return None

    # --- delegate everything else -------------------------------------------
    def __getattr__(self, item):
        return getattr(self._llm, item)

    # langchain calls llm.invoke(...) → forward it
    def invoke(self, *a, **kw):
        return self._llm.invoke(*a, **kw)