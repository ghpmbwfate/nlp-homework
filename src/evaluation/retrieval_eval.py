"""Retrieval evaluation metrics.

Metrics:
- Recall@k: correct page in top-k results
- MRR: Mean Reciprocal Rank of correct page
- Hit Rate@k: whether correct page is retrieved at all in top-k
"""

import json
from pathlib import Path
from typing import List, Dict


def evaluate_retrieval(
    questions: List[dict],
    retriever,
    top_ks: List[int] = None,
) -> Dict:
    """Evaluate retrieval quality against ground-truth page annotations.

    Args:
        questions: List of dicts with keys: question, filename, page
        retriever: Retriever instance
        top_ks: List of k values to evaluate (default [1, 3, 5, 10])

    Returns:
        Dict with metrics per k and per-question details.
    """
    if top_ks is None:
        top_ks = [1, 3, 5, 10]

    max_k = max(top_ks)
    results = []
    recalls = {k: 0 for k in top_ks}
    hits = {k: 0 for k in top_ks}
    reciprocal_ranks: List[float] = []

    for item in questions:
        question = item.get("question", "")
        gold_filename = item.get("filename", "")
        gold_page = item.get("page", -1)

        if not question or gold_page < 0:
            continue

        search_results = retriever.search(question)
        top_results = search_results[:max_k]

        # Find rank of correct page (1-indexed)
        rank = -1
        for i, r in enumerate(top_results):
            if (
                r["filename"] == gold_filename
                and r["page"] == gold_page
            ):
                rank = i + 1
                break

        rr = 1.0 / rank if rank > 0 else 0.0
        reciprocal_ranks.append(rr)

        for k in top_ks:
            in_top_k = rank > 0 and rank <= k
            if in_top_k:
                recalls[k] += 1
                hits[k] += 1

        results.append({
            "question": question,
            "gold_filename": gold_filename,
            "gold_page": gold_page,
            "rank": rank,
            "reciprocal_rank": rr,
            "retrieved_pages": [
                {"filename": r["filename"], "page": r["page"]}
                for r in top_results
            ],
        })

    n = len(results)
    if n == 0:
        return {"count": 0, "metrics": {}, "details": []}

    metrics = {
        "count": n,
        "mrr": sum(reciprocal_ranks) / n,
    }
    for k in top_ks:
        metrics[f"recall@{k}"] = recalls[k] / n
        metrics[f"hit_rate@{k}"] = hits[k] / n

    return {"count": n, "metrics": metrics, "details": results}


def print_retrieval_report(eval_result: Dict, output_path: str = None):
    """Print retrieval evaluation report."""
    metrics = eval_result["metrics"]
    n = eval_result["count"]

    print("\n" + "=" * 60)
    print("检索评估报告")
    print("=" * 60)
    print(f"样本数量: {n}")
    print("-" * 60)
    print(f"MRR:           {metrics.get('mrr', 0):.4f}")
    for k in [1, 3, 5, 10]:
        rk = metrics.get(f"recall@{k}", 0)
        hk = metrics.get(f"hit_rate@{k}", 0)
        print(f"Recall@{k}:      {rk:.4f}")
        print(f"Hit Rate@{k}:    {hk:.4f}")
    print("=" * 60)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(eval_result, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] 详细结果已保存至 {output_path}")
