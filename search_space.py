from modules.retrieval import BM25Retriever, HybridDBSFRetriever
from modules.passage_augmentation import NoAugment, PrevNextAugment
from modules.reranker import PassReranker, FlagLLMReranker
from modules.prompt_maker import FStringPrompt, LongContextPrompt, DynamicPrompt
from modules.generator import GPTGenerator

import pkgutil
import importlib
from typing import List
from modules.query_expansion.base import QueryExpander

def _discover_query_expanders() -> List[QueryExpander]:
    pkg_path = "modules/query_expansion".replace("/", ".")
    expanders = []
    for finder, module_name, ispkg in pkgutil.iter_modules([ "modules/query_expansion" ]):
        module = importlib.import_module(f"{pkg_path}.{module_name}")
        for obj in vars(module).values():
            if (
                isinstance(obj, type)
                and issubclass(obj, QueryExpander)
                and obj is not QueryExpander
            ):
                expanders.append(obj())
    return expanders

def reload_search_space():
    """
    Re-import the whole search_space module so that
    any newly‐dropped files get picked up.
    """
    import search_space as _ss
    importlib.reload(_ss)
    return _ss.SEARCH_SPACE


SEARCH_SPACE = {
    "query_expansion": _discover_query_expanders(),
    "retrieval":       [BM25Retriever(), HybridDBSFRetriever()],
    "augmentation":    [NoAugment(), PrevNextAugment()],
    "reranker":        [PassReranker(), FlagLLMReranker()],
    "prompt_maker":    [FStringPrompt(), LongContextPrompt(), DynamicPrompt()],
    "generator":       [GPTGenerator()],
}