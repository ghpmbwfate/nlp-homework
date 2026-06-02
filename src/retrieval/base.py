"""
Abstract base class for all retrievers (VRAG and baselines).

Every retriever must implement search(query, **kwargs) -> List[dict].
search_with_context() is provided as a concrete wrapper that formats
search results into the context dict consumed by main.py.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional

from src.config import IMAGES_DIR


class BaseRetriever(ABC):
    """Abstract retriever interface.

    Subclasses override search(). Callers use search_with_context() to get
    the structured context dict that main.py's pipeline expects.
    """

    def __init__(self, image_dir: Optional[str] = None):
        self._image_dir = Path(image_dir) if image_dir else IMAGES_DIR

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def search(self, query: str, top_k: int = 10, **kwargs) -> List[dict]:
        """Return top-k results for *query*.

        Each result dict must have at least:
            filename: str   — PDF filename (with .pdf extension)
            page: int       — page number (1-based)
            content: str    — chunk / page text
            score: float    — relevance score (higher = better)

        Optional keys (used when available):
            chunk_id, type, source, rerank_score, image_path
        """
        ...

    # ------------------------------------------------------------------
    # Shared logic
    # ------------------------------------------------------------------

    def search_with_context(self, query: str, top_k: int = 3, **kwargs) -> dict:
        """Search and assemble the context dict consumed by main.py.

        Returns:
            {
                "top_filename": str,
                "top_page": int,
                "context_text": str,    # concatenated top-k text with source markers
                "image_path": str|None, # file:/// URI for top-1 page image
                "results": list,        # raw search results
            }
        """
        results = self.search(query, top_k=top_k, **kwargs)

        if not results:
            return {
                "top_filename": "",
                "top_page": 0,
                "context_text": "",
                "image_path": None,
                "results": [],
            }

        context_parts = []
        for r in results:
            context_parts.append(
                f"[来源: {r['filename']} 第{r['page']}页]\n{r.get('content', '')}"
            )
        context_text = "\n\n---\n\n".join(context_parts)

        return {
            "top_filename": results[0]["filename"],
            "top_page": results[0]["page"],
            "context_text": context_text,
            "image_path": results[0].get("image_path"),
            "results": results,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_page_image_path(filename: str, page: int,
                             image_dir: Optional[Path] = None) -> Optional[str]:
        """Return file:/// URI for a page image if it exists."""
        img_dir = image_dir or IMAGES_DIR
        # Strip .pdf if present in filename for image matching
        stem = filename.replace(".pdf", "")
        image_path = img_dir / f"{stem}_page_{page}.png"
        if image_path.exists():
            return image_path.as_uri()
        return None
