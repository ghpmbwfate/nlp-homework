"""
Self-RAG 自洽性检查模块

- 答案支持度判断：检查生成答案是否被检索内容支持
- 自洽性投票：对多次生成的答案进行一致性投票
- 检索反馈：根据判断结果给出检索调整建议
"""

import re
from typing import List, Dict, Tuple, Literal, Optional

Verdict = Literal["Support", "Contradict", "Neutral"]


def extract_numbers(text: str) -> List[str]:
    """Extract numeric values from text (integers, decimals, percentages)."""
    pattern = r'\d+(?:\.\d+)?%?'
    return re.findall(pattern, text)


def tokenize(text: str) -> set:
    """Tokenize Chinese text using jieba, excluding stop words."""
    import jieba

    STOP_WORDS = {
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
        '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
        '没有', '看', '好', '自己', '这', '那', '他', '她', '它', '们',
        '与', '及', '而', '或', '但', '被', '把', '让', '从', '对', '向',
        '将', '以', '为', '所', '其', '之', '等', '中', '已', '还', '又',
        '可以', '需要', '应该', '能够', '可能', '已经', '以及', '并且',
    }

    tokens = set(jieba.cut(text))
    return tokens - STOP_WORDS


def check_support(answer: str, context: str) -> Verdict:
    """
    检查答案是否被检索上下文支持。

    规则判断（无需 LLM，可直接测试）：
    1. 提取答案和上下文中的数字
    2. 检查数字重叠率
    3. 检查关键词（jieba 分词）重叠率
    4. 综合判断

    Args:
        answer: VLM 生成的答案文本
        context: 检索到的上下文文本

    Returns:
        "Support" - 答案被上下文支持
        "Contradict" - 答案与上下文矛盾（可能幻觉）
        "Neutral" - 无法明确判断
    """
    if not context or not answer:
        return "Neutral"

    # --- 特殊模式：答案声称无信息（优先判断，避免误判） ---
    if re.search(r'(无法回答|未找到|没有相关|信息不足|无法确定)', answer):
        return "Neutral"

    # --- 数字检查 ---
    answer_nums = set(extract_numbers(answer))
    context_nums = set(extract_numbers(context))

    if answer_nums:
        num_overlap = answer_nums & context_nums
        num_ratio = len(num_overlap) / len(answer_nums)

        if num_ratio >= 0.8:
            # 80%+ 答案数字在上下文中 → Support
            return "Support"

        # 多数答案数字不在上下文中且数字较多 → Contradict（即使关键词重叠高）
        missing_ratio = len(answer_nums - context_nums) / len(answer_nums)
        if missing_ratio >= 0.75 and len(answer_nums) >= 3:
            return "Contradict"

    # --- 关键词检查 ---
    answer_tokens = tokenize(answer)
    context_tokens = tokenize(context)

    if answer_tokens and context_tokens:
        overlap = answer_tokens & context_tokens
        overlap_ratio = len(overlap) / len(answer_tokens)

        if overlap_ratio >= 0.5:
            return "Support"
        elif overlap_ratio <= 0.15:
            return "Contradict"

    return "Neutral"


def self_consistency_vote(answers: List[str]) -> Tuple[str, int]:
    """
    对多次生成的答案进行一致性投票，返回最一致的答案。

    使用 Jaccard 相似度投票：每个答案与其他答案的相似度之和最高的胜出。

    Args:
        answers: 多次生成的答案列表（至少 1 个）

    Returns:
        (best_answer, vote_count): 最佳答案和相似答案的数量
    """
    if not answers:
        return "", 0
    if len(answers) == 1:
        return answers[0], 1

    n = len(answers)
    tokenized = [tokenize(a) for a in answers]

    # 计算成对 Jaccard 相似度
    scores = [0.0] * n
    for i in range(n):
        for j in range(n):
            if i != j and tokenized[i] and tokenized[j]:
                jaccard = len(tokenized[i] & tokenized[j]) / len(tokenized[i] | tokenized[j])
                scores[i] += jaccard

    best_idx = 0
    best_score = -1.0
    for i, s in enumerate(scores):
        if s > best_score:
            best_score = s
            best_idx = i

    # 计算"同意"该答案的数量（相似度 >= 最佳分 * 0.8）
    threshold = best_score * 0.8 if best_score > 0 else 0
    vote_count = sum(1 for s in scores if s >= threshold)

    return answers[best_idx], vote_count


def get_retrieval_feedback(verdict: Verdict) -> Dict:
    """
    根据支持度判断生成检索反馈建议。

    Args:
        verdict: "Support", "Contradict", 或 "Neutral"

    Returns:
        {
            "action": "keep" | "expand_retrieval" | "review",
            "reason": str,
            "suggestion": str
        }
    """
    if verdict == "Contradict":
        return {
            "action": "expand_retrieval",
            "reason": "生成答案可能包含未被检索上下文支持的幻觉信息",
            "suggestion": "扩大检索范围（增加 top-k、使用查询改写、尝试多路召回）"
        }
    elif verdict == "Support":
        return {
            "action": "keep",
            "reason": "答案内容被检索上下文充分支持",
            "suggestion": "可直接采纳该答案"
        }
    else:
        return {
            "action": "review",
            "reason": "无法明确判断答案是否充分被上下文支持",
            "suggestion": "建议人工审核或尝试不同角度重新检索"
        }


def run_self_check(answer: str, context: str) -> Dict:
    """
    执行完整的 Self-RAG 自洽性检查，返回结构化结果。

    Args:
        answer: 生成的答案
        context: 检索上下文

    Returns:
        {
            "verdict": str,       # Support / Contradict / Neutral
            "feedback": dict,     # 检索反馈
            "num_support_ratio": float,    # 数字支持率
            "keyword_overlap_ratio": float # 关键词重叠率
        }
    """
    verdict = check_support(answer, context)
    feedback = get_retrieval_feedback(verdict)

    # 计算详细指标
    answer_nums = set(extract_numbers(answer))
    context_nums = set(extract_numbers(context))
    num_ratio = len(answer_nums & context_nums) / len(answer_nums) if answer_nums else 1.0

    answer_tokens = tokenize(answer)
    context_tokens = tokenize(context)
    kw_ratio = len(answer_tokens & context_tokens) / len(answer_tokens) if answer_tokens else 0.0

    return {
        "verdict": verdict,
        "feedback": feedback,
        "num_support_ratio": round(num_ratio, 4),
        "keyword_overlap_ratio": round(kw_ratio, 4),
        "answer_numbers_found": list(answer_nums),
        "context_numbers_found": list(context_nums),
    }
