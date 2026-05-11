"""
检索模块：稠密+BM25多路召回 → cross-encoder重排序 → 返回top-k结果
"""

import json
import pickle
from pathlib import Path

import jieba
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.config import INDEX_CHROMA_DIR, INDEX_BM25_DIR, IMAGES_DIR
from .multi_recover import title_search, keyword_search, summary_search


def load_dense_index(chroma_dir: str = None,
                     model_name: str = "BAAI/bge-m3"):
    """加载ChromaDB稠密索引"""
    chroma_dir = chroma_dir or str(INDEX_CHROMA_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_collection(
        name="financial_reports",
        embedding_function=embed_fn
    )
    return collection


def load_bm25_index(bm25_dir: str = None):
    """加载BM25索引和chunk元数据"""
    bm25_path = Path(bm25_dir) if bm25_dir else INDEX_BM25_DIR

    with open(bm25_path / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)

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


def get_page_image_path(filename: str, page: int,
                        image_dir: str = None) -> str | None:
    """获取页面对应的图片路径（返回 file:/// URI）"""
    image_dir = image_dir or str(IMAGES_DIR)
    image_name = f"{filename}_page_{page}.png"
    image_path = Path(image_dir) / image_name
    if image_path.exists():
        return image_path.as_uri()
    return None


class Retriever:
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
                 enable_multi_recall: bool = False):
        print("[INFO] 初始化检索器...")
        self.image_dir = image_dir or str(IMAGES_DIR)
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.final_top_k = final_top_k
        self.enable_multi_recall = enable_multi_recall
        self.multi_indexes = multi_indexes or {}

        # 加载索引
        self.collection = load_dense_index(chroma_dir, dense_model)
        self.bm25, self.chunks = load_bm25_index(bm25_dir)

        # 加载reranker
        print(f"[INFO] 加载reranker: {reranker_model}")
        self.reranker = CrossEncoder(reranker_model)

        if enable_multi_recall and multi_indexes:
            print(f"[INFO] 多路召回已启用 (title, keyword, summary)")
        print("[INFO] 检索器初始化完成")

    def search(self, query: str) -> list[dict]:
        """
        完整检索流程（支持多路召回）
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

        # 5. 重排序
        if self.reranker is not None:
            # 用reranker预测
            pairs = [(query, c["content"]) for c in merged]
            scores = self.reranker.predict(pairs)
            for i, c in enumerate(merged):
                c["rerank_score"] = float(scores[i])
            merged.sort(key=lambda x: x["rerank_score"], reverse=True)
        top_results = merged[:self.final_top_k]

        # 6. 添加图片路径
        for result in top_results:
            result["image_path"] = get_page_image_path(
                result["filename"], result["page"], self.image_dir
            )

        return top_results

    def search_with_context(self, query: str) -> dict:
        """
        检索并组装上下文信息，供VLM使用
        返回: {
            "top_filename": str,
            "top_page": int,
            "context_text": str,  # top-k结果的拼接文本
            "image_path": str,    # top-1结果的图片路径
            "results": list       # 完整检索结果
        }
        """
        results = self.search(query)
        if not results:
            return {
                "top_filename": "",
                "top_page": 0,
                "context_text": "",
                "image_path": None,
                "results": []
            }

        # 拼接top-k的文本作为上下文
        context_parts = []
        for i, r in enumerate(results):
            context_parts.append(
                f"[来源: {r['filename']} 第{r['page']}页]\n{r['content']}"
            )
        context_text = "\n\n---\n\n".join(context_parts)

        return {
            "top_filename": results[0]["filename"],
            "top_page": results[0]["page"],
            "context_text": context_text,
            "image_path": results[0].get("image_path"),
            "results": results
        }


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
    print(f"\n{'='*50}")
    print(f"查询: {args.query}")
    print(f"定位: {result['top_filename']} 第{result['top_page']}页")
    print(f"图片: {result['image_path']}")
    print(f"{'='*50}")
    print(f"上下文:\n{result['context_text'][:500]}...")
