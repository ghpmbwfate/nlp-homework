"""
Hybrid RRF retrieval baseline (dense + BM25, no rerank).

Performs ChromaDB dense retrieval and BM25 sparse retrieval, then fuses
them via Reciprocal Rank Fusion (k=60). No CrossEncoder rerank stage.
Used to quantify the contribution of the reranker vs hybrid retrieval alone.
"""

import logging
from typing import List, Optional

from src.retrieval.base import BaseRetriever
from src.retrieval.retriever import (
    load_dense_index,
    load_bm25_index,
    dense_search,
    bm25_search,
    merge_and_deduplicate,
)

logger = logging.getLogger(__name__)


class HybridRRFRetriever(BaseRetriever):
    """Dense + BM25 with RRF fusion, no CE rerank."""

    def __init__(self,
                 chroma_dir: Optional[str] = None,
                 bm25_dir: Optional[str] = None,
                 image_dir: Optional[str] = None,
                 dense_model: str = "BAAI/bge-m3",
                 dense_top_k: int = 10,
                 bm25_top_k: int = 10,
                 final_top_k: int = 3):
        super().__init__(image_dir=image_dir)
        self._collection = load_dense_index(chroma_dir, dense_model)
        self._bm25, self._chunks = load_bm25_index(bm25_dir)
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.final_top_k = final_top_k
        logger.info("[Hybrid RRF] Indexes loaded")

    def search(self, query: str, top_k: Optional[int] = None, **kwargs) -> List[dict]:
        del kwargs  # ignore question_type
        dense_hits = dense_search(self._collection, query, self.dense_top_k)
        bm25_hits = bm25_search(self._bm25, self._chunks, query, self.bm25_top_k)

        merged = merge_and_deduplicate(dense_hits, bm25_hits)
        final_k = top_k if top_k is not None else self.final_top_k
        top_results = merged[:final_k]

        for h in top_results:
            filename_pdf = h.get("filename", "")
            if filename_pdf and not filename_pdf.endswith(".pdf"):
                filename_pdf += ".pdf"
                h["filename"] = filename_pdf
            h["image_path"] = self._get_page_image_path(filename_pdf, h["page"])

        return top_results

    def search_with_context(self, query: str, **kwargs) -> dict:
        return super().search_with_context(query, top_k=self.final_top_k, **kwargs)
