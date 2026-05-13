"""
多阶段重排序优化模块

- Stage 1: RRF 粗排（在 retriever.py 中完成，取 top N）
- Stage 2: CrossEncoder 精排（取 top 20）
- Stage 3: MMR 多样性重排（避免内容重复）
- Stage 4: 按问题类型过滤（优先表格chunk用于数字类问题）
"""

import re
import numpy as np
from typing import List, Dict, Optional


def mmr_rerank(query: str,
               candidates: List[Dict],
               lambda_param: float = 0.7,
               top_k: int = 5,
               reranker=None) -> List[Dict]:
    """
    MMR (Maximal Marginal Relevance) 多样性重排序。

    平衡相关性（与 query 的相似度）与多样性（与已选结果的差异度）。

    MMR = argmax [λ * relevance(d_i, Q) - (1-λ) * max similarity(d_i, d_j)]
    其中 d_j 是已选中的结果。

    Args:
        query: 搜索查询
        candidates: 候选项列表，每项至少有 "content", "rerank_score" (或 "score")
        lambda_param: 相关性权重 (0~1)，越高越偏好相关性
        top_k: 返回结果数
        reranker: CrossEncoder 模型（可选，用于精确相关性/相似度计算）

    Returns:
        多样性重排后的结果列表
    """
    if len(candidates) <= top_k:
        return list(candidates)

    n = len(candidates)

    # 获取相关性分数
    if reranker is not None:
        pairs = [(query, c["content"]) for c in candidates]
        relevance = np.array(reranker.predict(pairs))
    else:
        relevance = np.array([
            c.get("rerank_score", c.get("score", 0.0))
            for c in candidates
        ])

    # 计算候选之间的成对相似度（用于多样性惩罚）
    similarity_matrix = np.zeros((n, n))

    if reranker is not None:
        # 使用 reranker 预测候选间的相似度
        for i in range(n):
            for j in range(i + 1, n):
                pair_score = float(reranker.predict(
                    [(candidates[i]["content"], candidates[j]["content"])]
                )[0])
                similarity_matrix[i][j] = pair_score
                similarity_matrix[j][i] = pair_score
    else:
        # Fallback: 使用 jieba Jaccard 相似度
        import jieba
        token_sets = [set(jieba.cut(c["content"])) for c in candidates]
        for i in range(n):
            for j in range(i + 1, n):
                if token_sets[i] and token_sets[j]:
                    jaccard = len(token_sets[i] & token_sets[j]) / len(token_sets[i] | token_sets[j])
                else:
                    jaccard = 0.0
                similarity_matrix[i][j] = jaccard
                similarity_matrix[j][i] = jaccard

    # MMR 贪心选择
    selected_indices = []
    remaining = list(range(n))

    for _ in range(min(top_k, n)):
        mmr_scores = []
        for idx in remaining:
            # 相关性得分
            rel_score = lambda_param * relevance[idx]

            # 多样性惩罚（与已选结果的最大相似度）
            diversity_penalty = 0.0
            if selected_indices:
                max_sim = max(similarity_matrix[idx][s] for s in selected_indices)
                diversity_penalty = (1.0 - lambda_param) * max_sim

            mmr_scores.append(rel_score - diversity_penalty)

        best_local = remaining[np.argmax(mmr_scores)]
        selected_indices.append(best_local)
        remaining.remove(best_local)

    return [candidates[i] for i in selected_indices]


def filter_by_question_type(candidates: List[Dict],
                            question_type: str) -> List[Dict]:
    """
    根据问题类型对候选结果重新排序。

    - fact_extraction: 提升表格 chunk 优先级（数值类问题）
    - analysis_summary: 保持原序
    - chart_understanding: 提升包含图表引用的 chunk
    - comparison: 保持原序
    """
    if not candidates:
        return candidates

    if question_type == "fact_extraction":
        # 表格 chunk 优先（数值更精确）
        tables = [c for c in candidates if c.get("type") == "table"]
        texts = [c for c in candidates if c.get("type") != "table"]
        return tables + texts

    elif question_type == "chart_understanding":
        # 包含图表引用的 chunk 优先
        def has_chart_ref(c):
            content = c.get("content", "")
            return bool(re.search(r'图\d+|表\d+|如图|见图|图表|上图|下图', content))

        chart_chunks = [c for c in candidates if has_chart_ref(c)]
        other_chunks = [c for c in candidates if not has_chart_ref(c)]
        return chart_chunks + other_chunks

    else:
        # analysis_summary, comparison: 保持原序
        return candidates


def multi_stage_rerank(query: str,
                       merged_results: List[Dict],
                       reranker=None,
                       question_type: Optional[str] = None,
                       coarse_k: int = 50,
                       fine_k: int = 20,
                       final_k: int = 5,
                       mmr_lambda: float = 0.7) -> List[Dict]:
    """
    完整的多阶段重排序流程。

    Stage 1: RRF 粗排 → 取 top coarse_k（在 retriever 中已完成）
    Stage 2: CrossEncoder 精排 → 取 top fine_k
    Stage 3: MMR 多样性重排 → 取 top final_k
    Stage 4: 问题类型过滤 → 最终排序

    Args:
        query: 搜索查询
        merged_results: RRF 融合后的结果列表（已按 RRF score 排序）
        reranker: CrossEncoder 模型
        question_type: 问题类型（fact_extraction/analysis_summary/...）
        coarse_k: Stage 1 保留数量
        fine_k: Stage 2 保留数量
        final_k: 最终返回数量
        mmr_lambda: MMR 相关性权重

    Returns:
        多阶段重排后的 top final_k 结果
    """
    results = list(merged_results)

    if not results:
        return results

    # Stage 1: RRF 粗排（取 top coarse_k）
    stage1 = results[:coarse_k]

    # Stage 2: CrossEncoder 精排
    if reranker is not None and len(stage1) > 1:
        pairs = [(query, c["content"]) for c in stage1]
        ce_scores = reranker.predict(pairs)
        for i, c in enumerate(stage1):
            c["rerank_score"] = float(ce_scores[i])
        stage1.sort(key=lambda x: x["rerank_score"], reverse=True)
    stage2 = stage1[:fine_k]

    # Stage 3: MMR 多样性重排
    stage3 = mmr_rerank(
        query, stage2,
        lambda_param=mmr_lambda,
        top_k=final_k,
        reranker=reranker
    )

    # Stage 4: 问题类型过滤
    if question_type:
        stage4 = filter_by_question_type(stage3, question_type)
    else:
        stage4 = stage3

    return stage4[:final_k]
