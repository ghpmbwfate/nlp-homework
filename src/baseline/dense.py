"""
Dense vector retrieval baseline (no rerank).

Uses ChromaDB with bge-m3 embeddings directly, without BM25 fusion or
cross-encoder reranking. Inherits from BaseRetriever for compatibility
with the main RAG pipeline and evaluation tools.
"""

import json
import logging
from typing import List, Optional

from src.retrieval.base import BaseRetriever
from src.retrieval.retriever import load_dense_index, dense_search

logger = logging.getLogger(__name__)


class DenseRetriever(BaseRetriever):
    """Dense vector retrieval baseline (ChromaDB + bge-m3, no rerank)."""

    def __init__(self, chroma_dir: Optional[str] = None,
                 dense_model: str = "BAAI/bge-m3",
                 image_dir: Optional[str] = None):
        super().__init__(image_dir=image_dir)
        self._collection = load_dense_index(chroma_dir, dense_model)
        logger.info("[Dense] Index loaded")

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10, **kwargs) -> List[dict]:
        """Return top-k chunks matching *query* via dense similarity."""
        del kwargs  # Dense ignores extra kwargs (question_type, etc.)

        hits = dense_search(self._collection, query, top_k)

        for h in hits:
            filename_pdf = h.get("filename", "")
            if filename_pdf and not filename_pdf.endswith(".pdf"):
                filename_pdf += ".pdf"
                h["filename"] = filename_pdf
            h["image_path"] = self._get_page_image_path(filename_pdf, h["page"])

        return hits


# ------------------------------------------------------------------
# Standalone evaluation
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from src.evaluation.retrieval_eval import evaluate_retrieval, print_retrieval_report

    parser = argparse.ArgumentParser(description="Dense baseline retrieval evaluation")
    parser.add_argument("--chroma_dir", type=str, default=None)
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

    retriever = DenseRetriever(chroma_dir=args.chroma_dir)
    result = evaluate_retrieval(questions, retriever, top_ks=[1, 3, 5, 10])
    print_retrieval_report(result, output_path=args.output)
