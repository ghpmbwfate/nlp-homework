"""Semantic chunking with sliding window overlap.

Splits pages into semantically coherent chunks using text similarity,
with configurable overlap to avoid boundary information loss.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


_DEFAULT_CHUNK_SIZE = 500
_DEFAULT_OVERLAP = 100
_DEFAULT_SIMILARITY_THRESHOLD = 0.6


def _split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs by blank lines."""
    paragraphs = []
    for para in text.split("\n\n"):
        para = para.strip()
        if para:
            paragraphs.append(para)
    if not paragraphs:
        # Fallback: split by single newlines
        for line in text.split("\n"):
            line = line.strip()
            if line:
                paragraphs.append(line)
    return paragraphs


def _merge_short_paragraphs(paragraphs: list[str], min_length: int = 30) -> list[str]:
    """Merge very short paragraphs with neighbors."""
    if not paragraphs:
        return []
    merged = [paragraphs[0]]
    for para in paragraphs[1:]:
        if len(merged[-1]) < min_length:
            merged[-1] = merged[-1] + "\n" + para
        elif len(para) < min_length:
            merged[-1] = merged[-1] + "\n" + para
        else:
            merged.append(para)
    return merged


def _compute_similarities(
    paragraphs: list[str], model: SentenceTransformer
) -> list[float]:
    """Compute cosine similarity between adjacent paragraphs."""
    if len(paragraphs) <= 1:
        return []
    embeddings = model.encode(paragraphs, show_progress_bar=False)
    sims = []
    for i in range(len(embeddings) - 1):
        sim = float(
            cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1),
            )[0, 0]
        )
        sims.append(sim)
    return sims


def _chunk_by_similarity(
    paragraphs: list[str],
    similarities: list[float],
    chunk_size: int,
    threshold: float,
) -> list[list[str]]:
    """Group paragraphs into chunks based on similarity breakpoints."""
    if not paragraphs:
        return []
    if len(paragraphs) == 1:
        return [paragraphs]

    chunks: list[list[str]] = []
    current: list[str] = [paragraphs[0]]
    current_len = len(paragraphs[0])

    for i in range(1, len(paragraphs)):
        para = paragraphs[i]
        sim = similarities[i - 1] if i - 1 < len(similarities) else 0.0
        para_len = len(para)

        # Break if similarity is low AND current chunk has enough content
        if sim < threshold and current_len >= chunk_size * 0.5:
            chunks.append(current)
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += para_len

        # Hard break if chunk exceeds size
        if current_len >= chunk_size:
            chunks.append(current)
            current = []
            current_len = 0

    if current:
        chunks.append(current)

    return chunks


def _apply_sliding_window(
    chunks: list[list[str]], overlap: int
) -> list[list[str]]:
    """Apply sliding window overlap between adjacent chunks."""
    if not chunks or overlap <= 0:
        return chunks

    result: list[list[str]] = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            result.append(chunk)
            continue

        # Take last paragraphs from previous chunk as overlap
        prev_chunk = chunks[i - 1]
        overlap_paras: list[str] = []
        overlap_len = 0
        for para in reversed(prev_chunk):
            if overlap_len + len(para) > overlap:
                break
            overlap_paras.insert(0, para)
            overlap_len += len(para)

        new_chunk = overlap_paras + chunk
        result.append(new_chunk)

    return result


def semantic_chunking(
    pages: list[dict],
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    overlap: int = _DEFAULT_OVERLAP,
    similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
    model_name: str = "BAAI/bge-m3",
) -> list[dict]:
    """Split pages into semantic chunks with sliding window overlap.

    Args:
        pages: List of page dicts with keys: filename, page, text
        chunk_size: Target chunk size in characters
        overlap: Overlap size in characters between adjacent chunks
        similarity_threshold: Cosine similarity threshold for breaking chunks
        model_name: Sentence transformer model for similarity computation

    Returns:
        List of chunk dicts with keys: chunk_id, filename, page, type, content
    """
    print(f"[INFO] 加载语义分块模型: {model_name}")
    model = SentenceTransformer(model_name)

    all_chunks: list[dict] = []
    chunk_counter = 0

    for page in pages:
        filename = page.get("filename", "")
        page_num = page.get("page", 0)
        text = page.get("text", "")

        if not text.strip():
            continue

        paragraphs = _split_into_paragraphs(text)
        paragraphs = _merge_short_paragraphs(paragraphs)

        if not paragraphs:
            continue

        if len(paragraphs) == 1:
            raw_chunks = [paragraphs]
        else:
            similarities = _compute_similarities(paragraphs, model)
            raw_chunks = _chunk_by_similarity(
                paragraphs, similarities, chunk_size, similarity_threshold
            )

        windowed_chunks = _apply_sliding_window(raw_chunks, overlap)

        for i, chunk_paras in enumerate(windowed_chunks):
            content = "\n\n".join(chunk_paras).strip()
            if not content or len(content) < 20:
                continue
            all_chunks.append({
                "chunk_id": f"{filename}_p{page_num}_c{i}",
                "filename": filename,
                "page": page_num,
                "type": "text",
                "content": content,
            })
            chunk_counter += 1

    print(f"[INFO] 语义分块完成: {chunk_counter} 个chunk")
    return all_chunks
