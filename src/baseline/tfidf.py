"""
TF-IDF baseline retriever.

Loads page_content.json, tokenizes each page with jieba, builds a
TfidfVectorizer index. Inherits from BaseRetriever for compatibility
with the main RAG pipeline and evaluation tools.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)


class TFIDFRetriever(BaseRetriever):
    """TF-IDF retrieval baseline over page_content.json pages."""

    def __init__(self, page_content_path: str = "page_content.json",
                 image_dir: Optional[str] = None):
        super().__init__(image_dir=image_dir)

        path = Path(page_content_path)
        if not path.exists():
            raise FileNotFoundError(f"page_content.json not found: {page_content_path}")

        with open(path, "r", encoding="utf-8") as f:
            self._pages: List[dict] = json.load(f)
        logger.info(f"[TF-IDF] Loaded {len(self._pages)} pages from {page_content_path}")

        self._docs = [p.get("text", "") for p in self._pages]

        def _tokenizer(text: str) -> List[str]:
            return list(jieba.cut(text))

        self._vectorizer = TfidfVectorizer(
            tokenizer=_tokenizer,
            max_features=10000,
            lowercase=False,
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(self._docs)
        logger.info(f"[TF-IDF] Built index: {self._tfidf_matrix.shape[1]} features")

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10, **kwargs) -> List[dict]:
        """Return top-k pages matching *query*."""
        del kwargs  # TF-IDF ignores extra kwargs (question_type, etc.)

        query_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._tfidf_matrix)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]

        hits = []
        for idx in top_indices:
            score = float(sims[idx])
            if score <= 0:
                continue
            page = self._pages[idx]
            filename_pdf = page["filename"] + ".pdf"
            hits.append({
                "chunk_id": f"{page['filename']}_p{page['page']}",
                "filename": filename_pdf,
                "page": page["page"],
                "content": page.get("text", ""),
                "type": "text",
                "score": score,
                "source": "tfidf",
                "image_path": self._get_page_image_path(filename_pdf, page["page"]),
            })

        return hits


# ------------------------------------------------------------------
# Standalone evaluation
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from src.evaluation.retrieval_eval import evaluate_retrieval, print_retrieval_report

    parser = argparse.ArgumentParser(description="TF-IDF baseline retrieval evaluation")
    parser.add_argument("--page_content", type=str, default="page_content.json")
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

    retriever = TFIDFRetriever(page_content_path=args.page_content)
    result = evaluate_retrieval(questions, retriever, top_ks=[1, 3, 5, 10])
    print_retrieval_report(result, output_path=args.output)
