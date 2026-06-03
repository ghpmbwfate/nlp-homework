"""
Dense + CrossEncoder rerank retrieval baseline.

ChromaDB dense retrieval → CrossEncoder rerank → top_k. Isolates the
contribution of the reranker on top of pure dense retrieval.
Reranker is selected via the registry (default: bge-large).
"""

import logging
from typing import List, Optional

from src.config import DEFAULT_RERANKER_ALIAS
from src.retrieval.base import BaseRetriever
from src.retrieval.retriever import load_dense_index, dense_search
from src.retrieval.reranker_backends import load_reranker

logger = logging.getLogger(__name__)


class DenseRerankRetriever(BaseRetriever):
    """Dense retrieval + CE rerank baseline."""

    def __init__(self,
                 chroma_dir: Optional[str] = None,
                 image_dir: Optional[str] = None,
                 dense_model: str = "BAAI/bge-m3",
                 reranker_alias: Optional[str] = None,
                 dense_top_k: int = 10,
                 final_top_k: int = 3):
        super().__init__(image_dir=image_dir)
        self._collection = load_dense_index(chroma_dir, dense_model)
        self._reranker = load_reranker(reranker_alias or DEFAULT_RERANKER_ALIAS)
        self.dense_top_k = dense_top_k
        self.final_top_k = final_top_k
        logger.info(f"[Dense+Rerank] Ready (reranker={reranker_alias or DEFAULT_RERANKER_ALIAS})")

    def search(self, query: str, top_k: Optional[int] = None, **kwargs) -> List[dict]:
        del kwargs
        hits = dense_search(self._collection, query, self.dense_top_k)
        if not hits:
            return []

        pairs = [(query, h["content"]) for h in hits]
        scores = self._reranker.predict(pairs)
        for h, s in zip(hits, scores):
            h["rerank_score"] = float(s)
        hits.sort(key=lambda x: x["rerank_score"], reverse=True)

        final_k = top_k if top_k is not None else self.final_top_k
        top_results = hits[:final_k]

        for h in top_results:
            filename_pdf = h.get("filename", "")
            if filename_pdf and not filename_pdf.endswith(".pdf"):
                filename_pdf += ".pdf"
                h["filename"] = filename_pdf
            h["image_path"] = self._get_page_image_path(filename_pdf, h["page"])

        return top_results

    def search_with_context(self, query: str, **kwargs) -> dict:
        return super().search_with_context(query, top_k=self.final_top_k, **kwargs)
