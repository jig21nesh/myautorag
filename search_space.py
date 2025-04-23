from modules.query_expansion import PassExpander, HyDEExpander
from modules.retrieval import BM25Retriever, HybridDBSFRetriever
from modules.passage_augmentation import NoAugment, PrevNextAugment
from modules.reranker import PassReranker, FlagLLMReranker
from modules.prompt_maker import FStringPrompt, LongContextPrompt, DynamicPrompt
from modules.generator import GPTGenerator

SEARCH_SPACE = {
    "query_expansion": [PassExpander(), HyDEExpander()],
    "retrieval":       [BM25Retriever(), HybridDBSFRetriever()],
    "augmentation":    [NoAugment(), PrevNextAugment()],
    "reranker":        [PassReranker(), FlagLLMReranker()],
    "prompt_maker":    [FStringPrompt(), LongContextPrompt(), DynamicPrompt()],
    "generator":       [GPTGenerator()],
}