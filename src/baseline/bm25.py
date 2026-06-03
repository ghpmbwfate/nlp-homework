"""
BM25 baseline retriever.

Loads the pre-built BM25 index (bm25.pkl + chunks.json) and performs
sparse retrieval using jieba tokenization. Inherits from BaseRetriever
for compatibility with the main RAG pipeline and evaluation tools.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

import jieba
from rank_bm25 import BM25Okapi

from src.config import INDEX_BM25_DIR
from src.retrieval.base import BaseRetriever
from src.retrieval.retriever import _SafeUnpickler

logger = logging.getLogger(__name__)


class BM25Retriever(BaseRetriever):
    """BM25 retrieval baseline over pre-built chunk index."""

    def __init__(self, bm25_dir: Optional[str] = None,
                 image_dir: Optional[str] = None):
        super().__init__(image_dir=image_dir)

        bm25_path = Path(bm25_dir) if bm25_dir else INDEX_BM25_DIR

        with open(bm25_path / "bm25.pkl", "rb") as f:
            self._bm25: BM25Okapi = _SafeUnpickler(f).load()

        with open(bm25_path / "chunks.json", "r", encoding="utf-8") as f:
            self._chunks: List[dict] = json.load(f)

        logger.info(f"[BM25] Loaded index: {len(self._chunks)} chunks from {bm25_path}")

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10, **kwargs) -> List[dict]:
        """Return top-k chunks matching *query* via BM25."""
        del kwargs  # BM25 ignores extra kwargs (question_type, etc.)

        tokenized_query = list(jieba.cut(query))
        scores = self._bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

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
                "image_path": self._get_page_image_path(filename_pdf, c["page"]),
            })

        return hits


# ------------------------------------------------------------------
# Standalone evaluation
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from src.evaluation.retrieval_eval import evaluate_retrieval, print_retrieval_report

    parser = argparse.ArgumentParser(description="BM25 baseline retrieval evaluation")
    parser.add_argument("--bm25_dir", type=str, default=None)
    parser.add_argument("--ground_truth", type=str, default="questions/test_ground_truth.json")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    with open(args.ground_truth, "r", encoding="utf-8") as f:
        gold_items = json.load(f)
    questions = [
        {"question": item["question"], "filename": item["filename"], "page": item["page"]}
        for item in gold_items
    ]
    logger.info(f"Loaded {len(questions)} ground-truth questions")

    retriever = BM25Retriever(bm25_dir=args.bm25_dir)
    result = evaluate_retrieval(questions, retriever, top_ks=[1, 3, 5, 10])
    print_retrieval_report(result, output_path=args.output)
