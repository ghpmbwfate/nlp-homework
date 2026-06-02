def __getattr__(name):
    if name == "Retriever":
        from .retriever import Retriever
        return Retriever
    if name == "QueryRewriter":
        from .query_rewriting import QueryRewriter
        return QueryRewriter
    if name == "NoOpQueryRewriter":
        from .query_rewriting import NoOpQueryRewriter
        return NoOpQueryRewriter
    if name == "MultiStageReranker":
        from .reranking import MultiStageReranker
        return MultiStageReranker
    if name == "BaseRetriever":
        from .base import BaseRetriever
        return BaseRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
