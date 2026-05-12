"""Error analysis tools for end-to-end evaluation.

Provides:
- Per-question-type error rate statistics
- Per-document error rate statistics
- Bad-case extraction (question + retrieval + prediction + gold)
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict


def analyze_errors(
    pred_data: List[dict],
    gold_data: List[dict],
    question_types: List[str] = None,
) -> Dict:
    """Analyze errors by question type and document.

    Args:
        pred_data: Predictions with keys: question, answer, question_type
        gold_data: Ground truth with keys: question, answer
        question_types: Optional list of known question types

    Returns:
        Dict with error statistics and bad cases.
    """
    gold_map = {item["question"].strip(): item for item in gold_data}

    by_type: dict[str, dict] = defaultdict(
        lambda: {"total": 0, "errors": 0, "details": []}
    )
    by_doc: dict[str, dict] = defaultdict(
        lambda: {"total": 0, "errors": 0, "details": []}
    )
    bad_cases: List[dict] = []

    for pred in pred_data:
        q = pred.get("question", "").strip()
        if q not in gold_map:
            continue

        gold = gold_map[q]
        pred_ans = pred.get("answer", "").strip()
        gold_ans = gold.get("answer", "").strip()
        qtype = pred.get("question_type", "unknown")
        filename = pred.get("filename", "unknown")

        # Simple exact-match error detection
        is_error = pred_ans != gold_ans

        by_type[qtype]["total"] += 1
        by_doc[filename]["total"] += 1
        if is_error:
            by_type[qtype]["errors"] += 1
            by_doc[filename]["errors"] += 1
            bad_cases.append({
                "question": q,
                "question_type": qtype,
                "filename": filename,
                "predicted": pred_ans,
                "gold": gold_ans,
                "citations": pred.get("citations", []),
            })

    # Compute rates
    type_stats = {}
    for qtype, stats in by_type.items():
        total = stats["total"]
        type_stats[qtype] = {
            "total": total,
            "errors": stats["errors"],
            "error_rate": stats["errors"] / total if total > 0 else 0.0,
        }

    doc_stats = {}
    for doc, stats in by_doc.items():
        total = stats["total"]
        doc_stats[doc] = {
            "total": total,
            "errors": stats["errors"],
            "error_rate": stats["errors"] / total if total > 0 else 0.0,
        }

    return {
        "by_question_type": type_stats,
        "by_document": doc_stats,
        "bad_cases": bad_cases,
        "total_questions": sum(s["total"] for s in by_type.values()),
        "total_errors": sum(s["errors"] for s in by_type.values()),
    }


def print_error_report(analysis: Dict, output_path: str = None):
    """Print error analysis report."""
    print("\n" + "=" * 60)
    print("错误分析报告")
    print("=" * 60)
    total = analysis["total_questions"]
    errors = analysis["total_errors"]
    print(f"总问题数: {total}")
    print(f"错误数:   {errors}")
    print(f"错误率:   {errors / total:.2%}" if total > 0 else "错误率:   N/A")

    print("\n【按问题类型统计】")
    for qtype, stats in sorted(
        analysis["by_question_type"].items(),
        key=lambda x: x[1]["error_rate"],
        reverse=True,
    ):
        print(
            f"  {qtype:20s} 错误率: {stats['error_rate']:.2%} "
            f"({stats['errors']}/{stats['total']})"
        )

    print("\n【按文档统计】")
    for doc, stats in sorted(
        analysis["by_document"].items(),
        key=lambda x: x[1]["error_rate"],
        reverse=True,
    ):
        print(
            f"  {doc:40s} 错误率: {stats['error_rate']:.2%} "
            f"({stats['errors']}/{stats['total']})"
        )

    print(f"\n【Bad Case 数量】{len(analysis['bad_cases'])}")
    print("=" * 60)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 详细分析已保存至 {output_path}")
