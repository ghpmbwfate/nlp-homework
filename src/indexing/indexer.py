"""
分块与索引构建模块
- 直接从 outputs/parsed_data/*/chart_descriptions.json 加载 VLM 提取的结构化数据
- 按页生成文本/图表/表格 chunk
- 稠密索引（ChromaDB + bge-m3）+ 稀疏索引（BM25 + jieba）
"""

import json
import pickle
from pathlib import Path

import jieba
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from src.config import INDEX_CHROMA_DIR, INDEX_BM25_DIR, PARSED_DIR, _resolve_model_path


def load_parsed_data(parsed_dir: str = None) -> list[dict]:
    """从 outputs/parsed_data/*/chart_descriptions.json 加载所有页面数据"""
    parsed_path = Path(parsed_dir) if parsed_dir else PARSED_DIR
    if not parsed_path.exists():
        raise FileNotFoundError(
            f"解析数据目录不存在: {parsed_path}，请先运行 chart_extractor.py"
        )
    pages = []
    for cd_file in sorted(parsed_path.glob("*/chart_descriptions.json")):
        with open(cd_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        pages.extend(data)
    return pages


def create_chunks(pages: list[dict]) -> list[dict]:
    """
    将 VLM 提取的页面数据拆分为 chunk
    - 每页文本作为一个 chunk（拼接 text 列表中的 content）
    - 每个图表单独一个 chunk
    - 每个 VLM 表格单独一个 chunk
    """
    chunks = []

    for page in pages:
        filename = page.get("filename", "")
        page_num = page.get("page", 0)
        page_summary = page.get("page_summary", "")

        # 文本chunk：拼接所有 text 段落的 content
        text_parts = page.get("text", [])
        text_content = " ".join(
            t.get("content", "") for t in text_parts if t.get("content")
        ).strip()

        if text_content and len(text_content) > 10:
            # 如果有页面摘要，拼接到文本前面
            if page_summary:
                text_content = f"[页面摘要] {page_summary}\n{text_content}"
            chunks.append({
                "chunk_id": f"{filename}_p{page_num}_text",
                "filename": filename,
                "page": page_num,
                "type": "text",
                "content": text_content,
            })

        # 图表chunk
        if page.get("has_charts"):
            for ci, chart in enumerate(page.get("charts", [])):
                content = _format_chart_content(chart, page_summary)
                if content.strip():
                    chunks.append({
                        "chunk_id": f"{filename}_p{page_num}_chart{ci}",
                        "filename": filename,
                        "page": page_num,
                        "type": "chart",
                        "content": content.strip(),
                    })

        # VLM表格chunk
        if page.get("has_tables"):
            for ti, table in enumerate(page.get("tables", [])):
                content = _format_chart_table_content(table, page_summary)
                if content.strip():
                    chunks.append({
                        "chunk_id": f"{filename}_p{page_num}_charttable{ti}",
                        "filename": filename,
                        "page": page_num,
                        "type": "chart_table",
                        "content": content.strip(),
                    })

    print(f"[INFO] 共创建 {len(chunks)} 个chunk")
    return chunks


def _format_chart_content(chart: dict, page_summary: str) -> str:
    """将单个图表描述格式化为检索友好的文本chunk"""
    parts = []
    if page_summary:
        parts.append(f"[页面摘要] {page_summary}")
    chart_label = chart.get("chart_id") or "图表"
    parts.append(f"[图表] {chart_label} {chart.get('title', '')}")
    if chart.get("type"):
        parts.append(f"类型: {chart['type']}")
    if chart.get("x_axis"):
        parts.append(f"X轴: {chart['x_axis']}")
    if chart.get("y_axis"):
        parts.append(f"Y轴: {chart['y_axis']}")
    if chart.get("legend"):
        parts.append(f"图例: {', '.join(chart['legend'])}")
    if chart.get("key_data"):
        parts.append(f"关键数据: {chart['key_data']}")
    source = chart.get("source")
    parts.append(f"来源: {source or '无'}")
    return "\n".join(parts)


def _format_chart_table_content(table: dict, page_summary: str) -> str:
    """将单个VLM提取的表格格式化为检索友好的文本chunk"""
    parts = []
    if page_summary:
        parts.append(f"[页面摘要] {page_summary}")
    table_label = table.get("table_id") or "表格"
    title = table.get("title") or "未命名表格"
    parts.append(f"[图表表格] {table_label} {title}")
    if table.get("headers"):
        parts.append(f"列标题: {', '.join(table['headers'])}")
    if table.get("key_info"):
        parts.append(f"关键信息: {table['key_info']}")
    rows = table.get("rows", [])
    headers = table.get("headers", [])
    if rows:
        parts.append("数据:")
        if headers:
            parts.append("| " + " | ".join(headers) + " |")
        for row in rows:
            parts.append("| " + " | ".join(str(c) for c in row) + " |")
    source = table.get("source")
    parts.append(f"来源: {source or '无'}")
    return "\n".join(parts)


def build_dense_index(chunks: list[dict],
                      model_name: str = "BAAI/bge-m3",
                      persist_dir: str = None):
    """构建稠密向量索引（ChromaDB）"""
    import torch

    persist_path = Path(persist_dir) if persist_dir else INDEX_CHROMA_DIR
    persist_path.mkdir(parents=True, exist_ok=True)

    effective_model = _resolve_model_path("Xorbits/bge-m3")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] 加载embedding模型: {effective_model} (device={device})")
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=effective_model,
        device=device,
    )

    client = chromadb.PersistentClient(path=str(persist_path))
    collection = client.get_or_create_collection(
        name="financial_reports",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )

    # 批量插入
    batch_size = 100
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    for batch_idx, i in enumerate(range(0, len(chunks), batch_size)):
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
        print(f"\r[INFO] 稠密索引构建进度: {batch_idx + 1}/{total_batches} 批次", end="", flush=True)

    print(f"\n[INFO] 稠密索引构建完成: {len(chunks)} 个chunk, 存储于 {persist_path}")
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


def build_all_indexes(parsed_dir: str = None,
                      dense_model: str = "BAAI/bge-m3",
                      chroma_dir: str = None,
                      bm25_dir: str = None):
    """构建所有索引"""
    print("=" * 50)
    print("Step 1: 加载解析数据")
    print("=" * 50)
    pages = load_parsed_data(parsed_dir)
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
    parser.add_argument("--parsed_dir", type=str, default=None,
                        help="解析数据目录 (默认 outputs/parsed_data)")
    parser.add_argument("--dense_model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--chroma_dir", type=str, default=None)
    parser.add_argument("--bm25_dir", type=str, default=None)

    args = parser.parse_args()
    build_all_indexes(
        parsed_dir=args.parsed_dir,
        dense_model=args.dense_model,
        chroma_dir=args.chroma_dir,
        bm25_dir=args.bm25_dir
    )
