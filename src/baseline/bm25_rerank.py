"""
BM25 + CrossEncoder rerank retrieval baseline.

BM25 sparse retrieval → CrossEncoder rerank → top_k. Isolates the
contribution of the reranker on top of pure sparse retrieval.
Reranker is selected via the registry (default: bge-large).
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

import jieba
from rank_bm25 import BM25Okapi

from src.config import INDEX_BM25_DIR, DEFAULT_RERANKER_ALIAS
from src.retrieval.base import BaseRetriever
from src.retrieval.retriever import _SafeUnpickler
from src.retrieval.reranker_backends import load_reranker

logger = logging.getLogger(__name__)


class BM25RerankRetriever(BaseRetriever):
    """BM25 retrieval + CE rerank baseline."""

    def __init__(self,
                 bm25_dir: Optional[str] = None,
                 image_dir: Optional[str] = None,
                 reranker_alias: Optional[str] = None,
                 bm25_top_k: int = 10,
                 final_top_k: int = 3):
        super().__init__(image_dir=image_dir)

        bm25_path = Path(bm25_dir) if bm25_dir else INDEX_BM25_DIR
        with open(bm25_path / "bm25.pkl", "rb") as f:
            self._bm25: BM25Okapi = _SafeUnpickler(f).load()
        with open(bm25_path / "chunks.json", "r", encoding="utf-8") as f:
            self._chunks: List[dict] = json.load(f)

        self._reranker = load_reranker(reranker_alias or DEFAULT_RERANKER_ALIAS)
        self.bm25_top_k = bm25_top_k
        self.final_top_k = final_top_k
        logger.info(
            f"[BM25+Rerank] Loaded {len(self._chunks)} chunks; "
            f"reranker={reranker_alias or DEFAULT_RERANKER_ALIAS}"
        )

    def search(self, query: str, top_k: Optional[int] = None, **kwargs) -> List[dict]:
        del kwargs

        tokenized_query = list(jieba.cut(query))
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[: self.bm25_top_k]

        hits = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            c = self._chunks[idx]
            filename_pdf = c["filename"]
            if not filename_pdf.endswith(".pdf"):
                filename_pdf += ".pdf"
            hits.append({
                "chunk_id": c["chunk_id"],
                "filename": filename_pdf,
                "page": c["page"],
                "content": c["content"],
                "type": c.get("type", "text"),
                "score": score,
                "source": "bm25",
            })

        if not hits:
            return []

        pairs = [(query, h["content"]) for h in hits]
        ce_scores = self._reranker.predict(pairs)
        for h, s in zip(hits, ce_scores):
            h["rerank_score"] = float(s)
        hits.sort(key=lambda x: x["rerank_score"], reverse=True)

        final_k = top_k if top_k is not None else self.final_top_k
        top_results = hits[:final_k]

        for h in top_results:
            h["image_path"] = self._get_page_image_path(h["filename"], h["page"])

        return top_results

    def search_with_context(self, query: str, **kwargs) -> dict:
        return super().search_with_context(query, top_k=self.final_top_k, **kwargs)
