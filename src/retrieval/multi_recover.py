"""
多路召回增强模块：
- 标题召回: 提取并匹配文档标题
- 关键词召回: jieba + TF-IDF 关键词匹配
- 摘要召回: 每页摘要检索
"""

import json
import re
from pathlib import Path
from typing import List, Dict

import jieba
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Lines/patterns to skip when detecting plain-text headings
_TITLE_SKIP_PREFIXES = ("免责声明", "分析师声明", "正文目录", "插图目录", "表格目录")
_COMPANY_PATTERN = re.compile(r'^[\u4e00-\u9fff\u00b7]{2,10}\s*\(\d{4,6}\s*CH\)')


def _is_boilerplate(line: str) -> bool:
    """Check if a line is boilerplate (disclaimer, company name, etc.)."""
    if any(line.startswith(p) for p in _TITLE_SKIP_PREFIXES):
        return True
    if _COMPANY_PATTERN.match(line):
        return True
    return False


def _looks_like_title(line: str) -> bool:
    """
    Heuristic: line looks like a section heading in plain text.
    Matches TOC entries (trailing ......), section headings with Chinese colon,
    and short Chinese phrases that are clearly not full sentences.
    """
    # Basic length constraints
    if len(line) < 4 or len(line) > 100:
        return False
    if _is_boilerplate(line):
        return False
    # Must contain Chinese characters
    if not any('\u4e00' <= c <= '\u9fff' for c in line):
        return False
    # Skip lines that are full sentences (have sentence-ending punctuation or conjunctions)
    if re.search(r'[。！？]', line):
        return False
    # Skip lines with discourse markers common in running text
    if re.search(r'^(我们认为|我们认[为是]|首先|其次|再次|最后|一方面|另一方面|因此|所以|但是|然而|不过)', line):
        return False
    # Skip lines that contain clause connectors indicating a complex sentence
    if re.search(r'[，；][但并因所于而在从被把让使为与和或者]', line):
        return False
    # Skip table headers or data lines (short with numbers/dates)
    if re.match(r'^[\u4e00-\u9fff]{1,6}\s*[\d%]', line):
        return False
    # TOC entries: end with many trailing dots
    if re.search(r'\.{6,}\s*$', line):
        return True
    # Section headings with Chinese colon in first half
    if '：' in line:
        colon_pos = line.index('：')
        if colon_pos <= len(line) * 0.5 and colon_pos >= 2:
            return True
    # Short standalone phrases (4-40 chars, no decimal digits at end)
    if len(line) <= 40 and not re.search(r'\d', line) and len(line) >= 4:
        return True
    return False


def extract_titles(pages: List[Dict]) -> List[Dict]:
    """
    Extract heading lines from page text.
    Supports both markdown-style (#, ##, ###, ####) and plain-text headings
    (TOC entries with trailing dots, lines containing Chinese colons, etc.).
    """
    titles = []
    for page in pages:
        text = page.get("text", "")
        filename = page.get("filename", "")
        page_num = page.get("page", 0)
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            title_text = None
            # Markdown headings
            m = re.match(r'^#{1,4}\s+(.+)', line)
            if m:
                title_text = m.group(1).strip()
            # Plain-text headings: look for lines with  Chinese colon or trailing dots
            elif _looks_like_title(line):
                # Strip trailing dots (TOC entries)
                cleaned = re.sub(r'\s*\.{3,}\s*$', '', line).strip()
                if cleaned and len(cleaned) > 2:
                    title_text = cleaned
            if title_text and len(title_text) > 2:
                titles.append({
                    "title": title_text,
                    "filename": filename,
                    "page": page_num,
                    "content": title_text,
                })
    return titles


def build_title_index(pages: List[Dict]) -> Dict:
    """Build title BM25 index from pages."""
    titles = extract_titles(pages)
    if not titles:
        return {"titles": [], "bm25": None}
    tokenized = [list(jieba.cut(t["title"])) for t in titles]
    bm25 = BM25Okapi(tokenized)
    return {"titles": titles, "bm25": bm25}


def title_search(query: str, title_index: Dict, top_k: int = 5) -> List[Dict]:
    """Search titles using BM25, with scores normalized to [0, 1]."""
    if not title_index.get("bm25") or not title_index.get("titles"):
        return []
    tokenized_query = list(jieba.cut(query))
    scores = title_index["bm25"].get_scores(tokenized_query)
    if scores is None or len(scores) == 0:
        return []
    max_score = float(scores.max())
    if max_score <= 0:
        return []
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    hits = []
    for idx in top_indices:
        if scores[idx] > 0:
            t = title_index["titles"][idx]
            normalized_score = float(scores[idx]) / max_score
            hits.append({
                "chunk_id": f"{t['filename']}_p{t['page']}_title",
                "content": t["content"],
                "filename": t["filename"],
                "page": t["page"],
                "type": "title",
                "score": normalized_score,
                "source": "title"
            })
    return hits


def build_keyword_index(pages: List[Dict]) -> Dict:
    """Build TF-IDF keyword index from page texts."""
    texts = []
    metas = []
    for page in pages:
        text = page.get("text", "")
        if text.strip():
            tokenized = " ".join(jieba.cut(text))
            texts.append(tokenized)
            metas.append({
                "filename": page.get("filename", ""),
                "page": page.get("page", 0),
                "content": text
            })
    if not texts:
        return {"vectorizer": None, "tfidf_matrix": None, "metas": metas}
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    return {"vectorizer": vectorizer, "tfidf_matrix": tfidf_matrix, "metas": metas}


def keyword_search(query: str, keyword_index: Dict, top_k: int = 5) -> List[Dict]:
    """Search using TF-IDF keyword matching."""
    if keyword_index.get("vectorizer") is None or keyword_index.get("tfidf_matrix") is None:
        return []
    query_tokens = " ".join(jieba.cut(query))
    query_vec = keyword_index["vectorizer"].transform([query_tokens])
    similarities = cosine_similarity(query_vec, keyword_index["tfidf_matrix"])[0]
    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]
    hits = []
    for idx in top_indices:
        if similarities[idx] > 0:
            meta = keyword_index["metas"][idx]
            hits.append({
                "chunk_id": f"{meta['filename']}_p{meta['page']}_keyword",
                "content": meta["content"],
                "filename": meta["filename"],
                "page": meta["page"],
                "type": "keyword",
                "score": float(similarities[idx]),
                "source": "keyword"
            })
    return hits


def extract_summary(text: str) -> str:
    """Extract a meaningful summary sentence from text, skipping boilerplate."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines:
        if re.match(r'^#{1,4}\s+', line):
            continue
        if len(line) < 10:
            continue
        if re.match(r'^[\d\s|｜\-+]+$', line):
            continue
        # Skip boilerplate
        if _is_boilerplate(line):
            continue
        # Skip pure table-of-contents lines (trailing dots)
        if re.search(r'\.{6,}\s*$', line):
            continue
        first_sent = re.split(r'[。；，]', line)[0]
        if len(first_sent) > 5:
            return first_sent[:200]
    return ""


def build_summary_index(pages: List[Dict]) -> Dict:
    """Build summary index using extractive first meaningful sentence per page."""
    summaries = []
    for page in pages:
        text = page.get("text", "")
        filename = page.get("filename", "")
        page_num = page.get("page", 0)
        summary = extract_summary(text)
        if summary:
            summaries.append({
                "summary": summary,
                "filename": filename,
                "page": page_num,
                "content": text,
                "type": "summary"
            })
    if not summaries:
        return {"summaries": [], "bm25": None}
    tokenized = [list(jieba.cut(s["summary"])) for s in summaries]
    bm25 = BM25Okapi(tokenized)
    return {"summaries": summaries, "bm25": bm25}


def summary_search(query: str, summary_index: Dict, top_k: int = 5) -> List[Dict]:
    """Search summaries using BM25, with scores normalized to [0, 1]."""
    if not summary_index.get("bm25") or not summary_index.get("summaries"):
        return []
    tokenized_query = list(jieba.cut(query))
    scores = summary_index["bm25"].get_scores(tokenized_query)
    if scores is None or len(scores) == 0:
        return []
    max_score = float(scores.max())
    if max_score <= 0:
        return []
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    hits = []
    for idx in top_indices:
        if scores[idx] > 0:
            s = summary_index["summaries"][idx]
            normalized_score = float(scores[idx]) / max_score
            hits.append({
                "chunk_id": f"{s['filename']}_p{s['page']}_summary",
                "content": s["content"],
                "filename": s["filename"],
                "page": s["page"],
                "type": s["type"],
                "score": normalized_score,
                "source": "summary"
            })
    return hits


def build_all_multi_indexes(pages: List[Dict]) -> Dict:
    """Build all multi-recovery indexes (title, keyword, summary)."""
    return {
        "title_index": build_title_index(pages),
        "keyword_index": build_keyword_index(pages),
        "summary_index": build_summary_index(pages),
    }
