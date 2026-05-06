"""
分块与索引构建模块
- 按页分块 + 表格单独分块
- 稠密索引（ChromaDB + bge-m3）+ 稀疏索引（BM25 + jieba）
"""

import json
import pickle
from pathlib import Path

import jieba
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from src.config import INDEX_CHROMA_DIR, INDEX_BM25_DIR


def load_page_content(page_content_path: str = "page_content.json") -> list[dict]:
    """加载页面内容索引"""
    path = Path(page_content_path)
    if not path.exists():
        raise FileNotFoundError(f"页面内容文件不存在: {page_content_path}，请先运行pdf_parser.py")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_chunks(pages: list[dict]) -> list[dict]:
    """
    将页面内容拆分为chunk
    每页文本作为一个chunk，表格单独作为chunk
    """
    chunks = []
    chunk_id = 0

    for page in pages:
        filename = page["filename"]
        page_num = page["page"]
        text = page.get("text", "")
        tables = page.get("tables", [])

        # 文本chunk（去除表格部分后的文本）
        text_only = text
        for table in tables:
            text_only = text_only.replace(table, "")
        text_only = text_only.strip()

        if text_only and len(text_only) > 10:  # 过滤太短的文本
            chunks.append({
                "chunk_id": f"{filename}_p{page_num}_text",
                "filename": filename,
                "page": page_num,
                "type": "text",
                "content": text_only
            })
            chunk_id += 1

        # 表格chunk
        for i, table in enumerate(tables):
            if table.strip():
                chunks.append({
                    "chunk_id": f"{filename}_p{page_num}_table{i}",
                    "filename": filename,
                    "page": page_num,
                    "type": "table",
                    "content": table.strip()
                })
                chunk_id += 1

    print(f"[INFO] 共创建 {len(chunks)} 个chunk")
    return chunks


def build_dense_index(chunks: list[dict],
                      model_name: str = "BAAI/bge-m3",
                      persist_dir: str = None):
    """构建稠密向量索引（ChromaDB）"""
    persist_path = Path(persist_dir) if persist_dir else INDEX_CHROMA_DIR
    persist_path.mkdir(parents=True, exist_ok=True)

    # 使用sentence-transformers的embedding
    # bge-m3支持多语言，对中文财报效果好
    print(f"[INFO] 加载embedding模型: {model_name}")
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )

    client = chromadb.PersistentClient(path=str(persist_path))
    collection = client.get_or_create_collection(
        name="financial_reports",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )

    # 批量插入
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        ids = [c["chunk_id"] for c in batch]
        documents = [c["content"] for c in batch]
        metadatas = [
            {
                "filename": c["filename"],
                "page": c["page"],
                "type": c["type"]
            }
            for c in batch
        ]

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    print(f"[INFO] 稠密索引构建完成: {len(chunks)} 个chunk, 存储于 {persist_path}")
    return collection


def tokenize_chinese(text: str) -> list[str]:
    """中文分词"""
    return list(jieba.cut(text))


def build_bm25_index(chunks: list[dict],
                     save_dir: str = None):
    """构建BM25稀疏索引"""
    save_path = Path(save_dir) if save_dir else INDEX_BM25_DIR
    save_path.mkdir(parents=True, exist_ok=True)

    print("[INFO] 构建BM25索引...")

    # 分词
    tokenized_corpus = [tokenize_chinese(c["content"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    # 保存BM25索引和chunk元数据
    with open(save_path / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    # 保存chunk信息（用于检索时回查）
    chunk_meta = [
        {
            "chunk_id": c["chunk_id"],
            "filename": c["filename"],
            "page": c["page"],
            "type": c["type"],
            "content": c["content"]
        }
        for c in chunks
    ]
    with open(save_path / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunk_meta, f, ensure_ascii=False, indent=2)

    print(f"[INFO] BM25索引构建完成: {len(chunks)} 个chunk, 存储于 {save_path}")
    return bm25, chunk_meta


def build_all_indexes(page_content_path: str = "page_content.json",
                      dense_model: str = "BAAI/bge-m3",
                      chroma_dir: str = None,
                      bm25_dir: str = None):
    """构建所有索引"""
    print("=" * 50)
    print("Step 1: 加载页面内容")
    print("=" * 50)
    pages = load_page_content(page_content_path)
    print(f"[INFO] 加载了 {len(pages)} 页内容")

    print()
    print("=" * 50)
    print("Step 2: 创建chunk")
    print("=" * 50)
    chunks = create_chunks(pages)

    print()
    print("=" * 50)
    print("Step 3: 构建稠密索引")
    print("=" * 50)
    build_dense_index(chunks, model_name=dense_model, persist_dir=chroma_dir)

    print()
    print("=" * 50)
    print("Step 4: 构建BM25索引")
    print("=" * 50)
    build_bm25_index(chunks, save_dir=bm25_dir)

    print()
    print("[INFO] 所有索引构建完成！")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="索引构建工具")
    parser.add_argument("--page_content", type=str, default="page_content.json")
    parser.add_argument("--dense_model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--chroma_dir", type=str, default=None)
    parser.add_argument("--bm25_dir", type=str, default=None)

    args = parser.parse_args()
    build_all_indexes(
        page_content_path=args.page_content,
        dense_model=args.dense_model,
        chroma_dir=args.chroma_dir,
        bm25_dir=args.bm25_dir
    )
