"""
语义分块模块
- 语义分块：基于文本相似度的段落边界切分，保证每个 chunk 语义内聚
- 滑动窗口：相邻 chunk 重叠 20-30%
- 保留原始页码信息用于溯源
"""

import re
import numpy as np
from typing import List, Dict, Optional


def split_sentences(text: str) -> List[str]:
    """Split Chinese text into sentences by sentence-ending punctuation."""
    # Split on Chinese sentence terminators: 。！？； followed by optional whitespace
    parts = re.split(r'(?<=[。！？；])\s*', text)
    # Also split on newlines for markdown structure
    result = []
    for part in parts:
        sub_parts = part.split('\n')
        for sp in sub_parts:
            sp = sp.strip()
            if sp and len(sp) > 2:
                result.append(sp)
    return result


def jieba_similarity(s1: str, s2: str) -> float:
    """Compute similarity using jieba token overlap (Jaccard)."""
    import jieba
    tokens1 = set(jieba.cut(s1))
    tokens2 = set(jieba.cut(s2))
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(intersection) / len(union) if union else 0.0


def compute_similarity(s1: str, s2: str, model=None) -> float:
    """Compute semantic similarity between two sentences.

    If model is provided (SentenceTransformer), use cosine similarity of embeddings.
    Otherwise falls back to jieba Jaccard token overlap.
    """
    if model is not None:
        emb = model.encode([s1, s2])
        dot = float(np.dot(emb[0], emb[1]))
        norm = float(np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]))
        if norm == 0:
            return 0.0
        return dot / norm
    else:
        return jieba_similarity(s1, s2)


def semantic_chunking(pages: List[Dict],
                      chunk_size: int = 500,
                      overlap: int = 100,
                      model_name: Optional[str] = "BAAI/bge-m3") -> List[Dict]:
    """
    Semantic chunking with sliding window overlap.

    策略：
    1. 将每页文本拆分为句子
    2. 计算相邻句子的语义相似度
    3. 在相似度显著下降处切分（semantic breakpoints）
    4. 将句子分组为 ~chunk_size 大小的 chunk
    5. 相邻 chunk 重叠 overlap 个字符
    6. 表格单独作为 chunk

    Args:
        pages: List of page dicts with "filename", "page", "text", "tables"
        chunk_size: Target characters per chunk (default 500)
        overlap: Overlap characters between adjacent chunks (default 100)
        model_name: SentenceTransformer model name. If None, uses jieba fallback.

    Returns:
        List of chunk dicts with keys: chunk_id, filename, page, type, content
    """
    # Load sentence-transformers model if specified
    model = None
    if model_name:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
        except Exception:
            pass  # Fall back to jieba similarity

    chunks = []
    chunk_idx = 0

    for page in pages:
        filename = page.get("filename", "")
        page_num = page.get("page", 0)
        text = page.get("text", "")
        tables = page.get("tables", [])

        if not text or len(text.strip()) < 10:
            # Still add tables even if text is short
            for ti, table in enumerate(tables):
                if table and table.strip():
                    chunks.append({
                        "chunk_id": f"{filename}_p{page_num}_table{ti}",
                        "filename": filename,
                        "page": page_num,
                        "type": "table",
                        "content": table.strip(),
                    })
                    chunk_idx += 1
            continue

        # Step 1: Split into sentences
        sentences = split_sentences(text)
        if not sentences:
            continue

        # Step 2: Compute consecutive pairwise similarities
        sims = []
        for i in range(len(sentences) - 1):
            sim = compute_similarity(sentences[i], sentences[i + 1], model)
            sims.append(sim)

        # Step 3: Find breakpoints (where similarity drops significantly)
        breakpoints = set()
        if sims:
            mean_sim = np.mean(sims)
            std_sim = np.std(sims) if len(sims) > 1 else 0.1
            threshold = max(mean_sim - 1.0 * std_sim, 0.15)

            for i, sim in enumerate(sims):
                if sim < threshold:
                    breakpoints.add(i + 1)  # Break AFTER sentence i+1

        # Also add breakpoints at natural boundaries (very long sentences, section markers)
        for i, sent in enumerate(sentences):
            if re.match(r'^#{1,4}\s+', sent):  # Markdown heading
                breakpoints.add(i)

        # Step 4: Group sentences between breakpoints into chunks
        current_group = []
        current_len = 0
        last_chunk_text = ""  # For overlap reference

        def finalize_chunk(group, prev_chunk_overlap=""):
            nonlocal chunk_idx, last_chunk_text
            if not group:
                return

            chunk_text = "".join(group)

            # Apply overlap from previous chunk
            if prev_chunk_overlap and not chunk_text.startswith(prev_chunk_overlap):
                chunk_text = prev_chunk_overlap + chunk_text

            if len(chunk_text.strip()) > 10:
                chunks.append({
                    "chunk_id": f"{filename}_p{page_num}_chunk{chunk_idx}",
                    "filename": filename,
                    "page": page_num,
                    "type": "text",
                    "content": chunk_text.strip(),
                })
                chunk_idx += 1
                last_chunk_text = chunk_text

        overlap_text = ""
        for i, sent in enumerate(sentences):
            current_group.append(sent)
            current_len += len(sent)

            should_break = (i in breakpoints and current_len >= 50)
            if not should_break and current_len >= chunk_size:
                # Check if next sentence is semantically distant
                if i + 1 < len(sentences) and sims and i < len(sims):
                    if sims[i] < 0.25:
                        should_break = True
                # Also break if we're significantly over chunk_size
                if current_len >= chunk_size * 1.5:
                    should_break = True

            if should_break:
                finalize_chunk(current_group, overlap_text)

                # Prepare overlap for next chunk
                chunk_text = "".join(current_group)
                if overlap > 0 and len(chunk_text) > overlap:
                    overlap_text = chunk_text[-overlap:]
                else:
                    overlap_text = ""

                current_group = []
                current_len = 0

        # Step 5: Finalize remaining sentences
        if current_group:
            finalize_chunk(current_group, overlap_text)

        # Step 6: Add tables as separate chunks
        for ti, table in enumerate(tables):
            if table and table.strip():
                chunks.append({
                    "chunk_id": f"{filename}_p{page_num}_table{ti}",
                    "filename": filename,
                    "page": page_num,
                    "type": "table",
                    "content": table.strip(),
                })
                chunk_idx += 1

    return chunks
