"""
Audit answer quality across all submit files.

Classifies each answer into one of:
  - ok               : non-empty, valid ending
  - empty            : answer.strip() == ""
  - error_fallback   : starts with "处理失败" or "[模型返回空答案"
  - likely_truncated : length >= 1900 AND ends mid-Markdown-row / dangling [来源:

Output:
  - stdout: per-run summary table
  - outputs/answer_quality_audit.json: structured per-run report incl. broken indices
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).parent.parent
OUTPUTS = ROOT / "outputs"

RUNS: list[tuple[str, str]] = [
    ("TF-IDF",          "submit_tfidf_v3.json"),
    ("BM25",            "submit_bm25.json"),
    ("Dense",           "submit_dense.json"),
    ("A: Hybrid RRF",   "submit_hybrid_rrf.json"),
    ("B: Dense+CE",     "submit_dense_rerank.json"),
    ("C: BM25+CE",      "submit_bm25_rerank.json"),
    ("D: Hybrid+CE",    "submit_hybrid_ce.json"),
    ("D: bge-v2-m3",    "submit_hybrid_ce_bge_v2m3.json"),
    ("D: qwen3-0.6b",   "submit_hybrid_ce_qwen3_06b.json"),
]

# Char set considered a "dangling" ending (mid-Markdown-row / mid-list-item)
DANGLING_LAST_CHARS = set("|,，、（(*")

Verdict = Literal["ok", "empty", "error_fallback", "likely_truncated"]


def classify(answer: str) -> Verdict:
    stripped = answer.strip() if answer else ""
    if not stripped:
        return "empty"
    if stripped.startswith("处理失败") or stripped.startswith("[模型返回空答案"):
        return "error_fallback"
    if len(stripped) >= 1900:
        # Check last line first for dangling citation
        last_line = stripped.rsplit("\n", 1)[-1].strip()
        if last_line.startswith("[来源:") and "]" not in last_line:
            return "likely_truncated"
        # Check last non-whitespace char
        last_char = stripped[-1]
        if last_char in DANGLING_LAST_CHARS:
            return "likely_truncated"
        # Mid-Markdown table row check: ends with no closing | but line has many |
        if last_line.count("|") >= 2 and not last_line.rstrip().endswith("|"):
            return "likely_truncated"
    return "ok"


def main() -> None:
    report: dict[str, dict] = {}
    print(f"{'Run':<20} {'total':>5} {'ok':>5} {'empty':>5} {'errfb':>5} {'trunc':>5}")
    print("-" * 50)

    for name, filename in RUNS:
        path = OUTPUTS / filename
        if not path.exists():
            print(f"{name:<20}  -- file not found: {filename}")
            continue
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        counts: dict[Verdict, int] = {
            "ok": 0, "empty": 0, "error_fallback": 0, "likely_truncated": 0
        }
        broken_indices: dict[Verdict, list[int]] = {
            "empty": [], "error_fallback": [], "likely_truncated": []
        }
        for i, item in enumerate(data):
            v = classify(item.get("answer", ""))
            counts[v] += 1
            if v != "ok":
                broken_indices[v].append(i)

        report[name] = {
            "file": filename,
            "total": len(data),
            "counts": counts,
            "broken_indices": broken_indices,
        }

        print(
            f"{name:<20} {len(data):>5} {counts['ok']:>5} "
            f"{counts['empty']:>5} {counts['error_fallback']:>5} "
            f"{counts['likely_truncated']:>5}"
        )

    out_path = OUTPUTS / "answer_quality_audit.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
