"""Citation extraction and prompt augmentation for answer source tracing."""

import re
from typing import List, Dict, Optional


CITATION_INSTRUCTION = """
请在答案末尾标注信息来源，格式为：[来源: 文档名 第X页]
如果答案参考了多页内容，请列出所有来源。
"""


def add_citation_instruction(prompt_template: str) -> str:
    """Add citation instruction to a prompt template."""
    return prompt_template + "\n" + CITATION_INSTRUCTION


def extract_citations(answer: str) -> List[Dict]:
    """Extract citation markers from answer text.

    Handles patterns like:
    - [来源: 千味央厨 第5页]
    - [来源: 广联达深度报告 第12页]
    - [来源: 千味央厨-千寻百味乘势而上-221227 第3页]

    Returns list of {"filename": str, "page": int}
    """
    pattern = r'\[来源:\s*(.+?)\s*第(\d+)页\]'

    citations = []
    seen = set()
    for match in re.finditer(pattern, answer):
        filename = match.group(1).strip()
        if not filename:
            continue
        page = int(match.group(2))
        key = (filename, page)
        if key not in seen:
            seen.add(key)
            citations.append({
                "filename": filename,
                "page": page,
            })
    return citations


def format_citation(filename: str, page: int) -> str:
    """Format a standard citation string."""
    return f"[来源: {filename} 第{page}页]"


def clean_answer_no_citations(answer: str) -> str:
    """Remove citation markers from answer (for evaluation)."""
    return re.sub(r'\[来源:\s*.+?\s*第\d+页\]', '', answer).strip()


def has_citations(answer: str) -> bool:
    """Check if answer contains any citation markers."""
    return bool(re.search(r'\[来源:\s*.+?\s*第\d+页\]', answer))
