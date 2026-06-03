"""
检索模块：稠密+BM25多路召回 → cross-encoder重排序 → 返回top-k结果
"""

import io
import json
import logging
import pickle
from pathlib import Path

import jieba
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from src.config import INDEX_CHROMA_DIR, INDEX_BM25_DIR, IMAGES_DIR, _resolve_model_path, DEFAULT_RERANKER_ALIAS
from .base import BaseRetriever
from .multi_recover import title_search, keyword_search, summary_search
from .reranking import multi_stage_rerank
from .reranker_backends import load_reranker

logger = logging.getLogger(__name__)


def load_dense_index(chroma_dir: str = None,
                     model_name: str = "BAAI/bge-m3"):
    """加载ChromaDB稠密索引"""
    import torch

    chroma_dir = chroma_dir or str(INDEX_CHROMA_DIR)
    effective_model = _resolve_model_path("Xorbits/bge-m3")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"加载embedding模型: {effective_model} (device={device})")
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=effective_model,
        device=device,
    )
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_collection(
        name="financial_reports",
        embedding_function=embed_fn
    )
    return collection


class _SafeUnpickler(pickle.Unpickler):
    """Restrict pickle to only allow safe built-in types."""

    ALLOWED_CLASSES = {
        ("rank_bm25", "BM25Okapi"),
        ("collections", "Counter"),
    }

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED_CLASSES:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Blocked unsafe pickle class: {module}.{name}"
        )


def load_bm25_index(bm25_dir: str = None):
    """加载BM25索引和chunk元数据"""
    bm25_path = Path(bm25_dir) if bm25_dir else INDEX_BM25_DIR

    with open(bm25_path / "bm25.pkl", "rb") as f:
        bm25 = _SafeUnpickler(f).load()

    with open(bm25_path / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return bm25, chunks


def dense_search(collection, query: str, top_k: int = 10) -> list[dict]:
    """稠密检索"""
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    hits = []
    if results["ids"] and results["ids"][0]:
        for i, chunk_id in enumerate(results["ids"][0]):
            hits.append({
                "chunk_id": chunk_id,
                "content": results["documents"][0][i],
                "filename": results["metadatas"][0][i]["filename"],
                "page": results["metadatas"][0][i]["page"],
                "type": results["metadatas"][0][i]["type"],
                "score": 1 - results["distances"][0][i],  # 距离转相似度
                "source": "dense"
            })

    return hits


def bm25_search(bm25: BM25Okapi, chunks: list[dict],
                query: str, top_k: int = 10) -> list[dict]:
    """BM25稀疏检索"""
    tokenized_query = list(jieba.cut(query))
    scores = bm25.get_scores(tokenized_query)

    # 取top_k
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    hits = []
    for idx in top_indices:
        if scores[idx] > 0:
            c = chunks[idx]
            hits.append({
                "chunk_id": c["chunk_id"],
                "content": c["content"],
                "filename": c["filename"],
                "page": c["page"],
                "type": c["type"],
                "score": float(scores[idx]),
                "source": "bm25"
            })

    return hits


def merge_and_deduplicate(dense_hits: list[dict],
                          bm25_hits: list[dict],
                          title_hits: list[dict] | None = None,
                          keyword_hits: list[dict] | None = None,
                          summary_hits: list[dict] | None = None,
                          k: int = 60) -> list[dict]:
    """
    使用RRF (Reciprocal Rank Fusion) 合并多路检索结果。

    不直接叠加原始分数（dense score ∈ [0,1] 与 BM25 score 无界，量纲不同），
    而是基于排名进行融合：
        RRF_score = Σ 1 / (k + rank_i)
    k 取 60 为业界常用值。
    """
    scores: dict[str, float] = {}
    merged: dict[str, dict] = {}

    def _process_hits(hits, source_name):
        for rank, hit in enumerate(hits):
            cid = hit["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            if cid in merged:
                if source_name not in merged[cid]["sources"]:
                    merged[cid]["sources"].append(source_name)
            else:
                merged[cid] = {**hit}
                merged[cid]["sources"] = [source_name]

    # dense 路
    _process_hits(dense_hits, "dense")

    # bm25 路
    _process_hits(bm25_hits, "bm25")

    # title 路
    if title_hits:
        _process_hits(title_hits, "title")

    # keyword 路
    if keyword_hits:
        _process_hits(keyword_hits, "keyword")

    # summary 路
    if summary_hits:
        _process_hits(summary_hits, "summary")

    # 写入 RRF 分数并排序
    for cid in merged:
        merged[cid]["score"] = scores[cid]

    return sorted(merged.values(), key=lambda x: x["score"], reverse=True)


class Retriever(BaseRetriever):
    """检索器：封装完整的检索流程（含多路召回）"""

    def __init__(self,
                 chroma_dir: str = None,
                 bm25_dir: str = None,
                 image_dir: str = None,
                 dense_model: str = "BAAI/bge-m3",
                 reranker_model: str = "BAAI/bge-reranker-large",
                 dense_top_k: int = 10,
                 bm25_top_k: int = 10,
                 final_top_k: int = 3,
                 multi_indexes: dict | None = None,
                 enable_multi_recall: bool = False,
                 enable_multi_stage_rerank: bool = False,
                 reranker_alias: str | None = None):
        super().__init__(image_dir=image_dir)
        logger.info("初始化检索器...")
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.final_top_k = final_top_k
        self.enable_multi_recall = enable_multi_recall
        self.enable_multi_stage_rerank = enable_multi_stage_rerank
        self.multi_indexes = multi_indexes or {}

        # 加载索引
        self.collection = load_dense_index(chroma_dir, dense_model)
        self.bm25, self.chunks = load_bm25_index(bm25_dir)

        # 加载reranker (via registry)
        alias = reranker_alias or DEFAULT_RERANKER_ALIAS
        self.reranker = load_reranker(alias)
        self.reranker_alias = alias

        if enable_multi_recall and multi_indexes:
            logger.info("多路召回已启用 (title, keyword, summary)")
        if enable_multi_stage_rerank:
            logger.info("多阶段重排序已启用 (RRF→CE→MMR→TypeFilter)")
        logger.info("检索器初始化完成")

    def search(self, query: str, top_k: int | None = None,
               question_type: str | None = None, **kwargs) -> list[dict]:
        """
        完整检索流程（支持多路召回 + 多阶段重排序）
        返回: [{
            "chunk_id", "content", "filename", "page", "type",
            "rerank_score", "image_path"
        }]
        """
        # 1. 稠密检索
        dense_hits = dense_search(self.collection, query, self.dense_top_k)

        # 2. BM25检索
        bm25_hits = bm25_search(self.bm25, self.chunks, query, self.bm25_top_k)

        # 3. 多路召回（title, keyword, summary）
        title_hits = []
        keyword_hits = []
        summary_hits = []
        if self.enable_multi_recall and self.multi_indexes:
            if "title_index" in self.multi_indexes:
                title_hits = title_search(query, self.multi_indexes["title_index"], top_k=5)
            if "keyword_index" in self.multi_indexes:
                keyword_hits = keyword_search(query, self.multi_indexes["keyword_index"], top_k=5)
            if "summary_index" in self.multi_indexes:
                summary_hits = summary_search(query, self.multi_indexes["summary_index"], top_k=5)

        # 4. 合并去重
        merged = merge_and_deduplicate(
            dense_hits, bm25_hits,
            title_hits=title_hits if title_hits else None,
            keyword_hits=keyword_hits if keyword_hits else None,
            summary_hits=summary_hits if summary_hits else None,
        )

        # 5. 重排序（多阶段 或 单阶段）
        final_k = top_k if top_k is not None else self.final_top_k
        if self.enable_multi_stage_rerank:
            # 多阶段重排序: RRF→CE→MMR→TypeFilter
            top_results = multi_stage_rerank(
                query, merged,
                reranker=self.reranker,
                question_type=question_type,
                coarse_k=min(len(merged), 50),
                fine_k=min(len(merged), 20),
                final_k=final_k,
            )
        elif self.reranker is not None:
            # 单阶段 CrossEncoder 重排序（兼容旧行为）
            pairs = [(query, c["content"]) for c in merged]
            scores = self.reranker.predict(pairs)
            for i, c in enumerate(merged):
                c["rerank_score"] = float(scores[i])
            merged.sort(key=lambda x: x["rerank_score"], reverse=True)
            top_results = merged[:final_k]
        else:
            top_results = merged[:final_k]

        # 6. 添加图片路径
        for result in top_results:
            result["image_path"] = self._get_page_image_path(
                result["filename"], result["page"]
            )

        return top_results

    def search_with_context(self, query: str, **kwargs) -> dict:
        """检索并组装上下文信息（使用 final_top_k 作为默认检索数量）"""
        return super().search_with_context(query, top_k=self.final_top_k, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="检索测试工具")
    parser.add_argument("--query", type=str, required=True, help="测试查询")
    parser.add_argument("--chroma_dir", type=str, default=None)
    parser.add_argument("--bm25_dir", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=3)

    args = parser.parse_args()

    retriever = Retriever(
        chroma_dir=args.chroma_dir,
        bm25_dir=args.bm25_dir,
        image_dir=args.image_dir,
        final_top_k=args.top_k
    )

    result = retriever.search_with_context(args.query)
    logger.info(f"{'='*50}")
    logger.info(f"查询: {args.query}")
    logger.info(f"定位: {result['top_filename']} 第{result['top_page']}页")
    logger.info(f"图片: {result['image_path']}")
    logger.info(f"{'='*50}")
    logger.info(f"上下文:\n{result['context_text'][:500]}...")
