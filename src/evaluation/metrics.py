"""
评估指标模块：对比预测结果与标准答案，计算多项指标

支持指标：
- Accuracy (准确率，等同于 Exact Match)
- Exact Match (精确匹配率)
- Char F1 (字符级 F1)
- Word F1 (词级 F1，基于 jieba 分词)
- ROUGE-1 / ROUGE-2 (词级 n-gram 召回)
- ROUGE-L (基于最长公共子序列)
- BLEU-4 (n-gram 精度 + 短句惩罚)
- Number Accuracy (数字提取准确率，针对财报场景)
"""

import json
import logging
import math
import re
import string
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple

import jieba

logger = logging.getLogger(__name__)

# 需要去除的中英文标点
_PUNCTUATION = string.punctuation + "，。？！；：\"\"''（）【】《》、"
_TRANS_TABLE = str.maketrans("", "", _PUNCTUATION)


def normalize_text(text: str) -> str:
    """标准化文本：去除空格、换行、常见标点，转为小写"""
    text = text.lower().strip()
    text = text.translate(_TRANS_TABLE)
    return text


def extract_numbers(text: str) -> List[str]:
    """提取文本中的数字（包括整数、小数、百分比，支持千分位逗号）"""
    # 匹配数字模式：整数、小数、百分比、负数和千分位格式
    pattern = r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?|-?\d+(?:\.\d+)?%?"
    return re.findall(pattern, text)


def longest_common_subsequence(x: List[str], y: List[str]) -> int:
    """计算最长公共子序列长度"""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0

    # 使用滚动数组优化空间
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev

    return prev[n]


def char_f1(pred: str, gold: str) -> Tuple[float, float, float]:
    """字符级 F1（基于 Counter，保留频率信息）"""
    pred_chars = Counter(pred)
    gold_chars = Counter(gold)

    if not pred_chars and not gold_chars:
        return 1.0, 1.0, 1.0
    if not pred_chars or not gold_chars:
        return 0.0, 0.0, 0.0

    common = sum((pred_chars & gold_chars).values())
    precision = common / sum(pred_chars.values())
    recall = common / sum(gold_chars.values())
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def word_f1(pred: str, gold: str) -> Tuple[float, float, float]:
    """词级 F1（基于 jieba 分词 + Counter，保留频率信息）"""
    pred_words = Counter(
        w for w in jieba.lcut(pred)
        if w.strip() and not re.match(r"^\W+$", w)
    )
    gold_words = Counter(
        w for w in jieba.lcut(gold)
        if w.strip() and not re.match(r"^\W+$", w)
    )

    if not pred_words and not gold_words:
        return 1.0, 1.0, 1.0
    if not pred_words or not gold_words:
        return 0.0, 0.0, 0.0

    common = sum((pred_words & gold_words).values())
    precision = common / sum(pred_words.values())
    recall = common / sum(gold_words.values())
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def rouge_l(pred: str, gold: str) -> Tuple[float, float, float]:
    """ROUGE-L：基于最长公共子序列"""
    pred_chars = list(pred)
    gold_chars = list(gold)

    if not pred_chars and not gold_chars:
        return 1.0, 1.0, 1.0
    if not pred_chars or not gold_chars:
        return 0.0, 0.0, 0.0

    lcs_len = longest_common_subsequence(pred_chars, gold_chars)

    precision = lcs_len / len(pred_chars)
    recall = lcs_len / len(gold_chars)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def _tokenize(text: str) -> List[str]:
    """jieba 分词，过滤纯标点 token"""
    return [w for w in jieba.lcut(text) if w.strip() and not re.match(r"^\W+$", w)]


def _get_ngrams(tokens: List[str], n: int) -> Counter:
    """提取 n-gram 并计数"""
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def rouge_n(pred: str, gold: str, n: int = 1) -> Tuple[float, float, float]:
    """ROUGE-N：基于 n-gram 重叠"""
    pred_tokens = _tokenize(pred)
    gold_tokens = _tokenize(gold)

    pred_ngrams = _get_ngrams(pred_tokens, n)
    gold_ngrams = _get_ngrams(gold_tokens, n)

    if not pred_ngrams and not gold_ngrams:
        return 1.0, 1.0, 1.0
    if not pred_ngrams or not gold_ngrams:
        return 0.0, 0.0, 0.0

    common = sum((pred_ngrams & gold_ngrams).values())
    precision = common / sum(pred_ngrams.values())
    recall = common / sum(gold_ngrams.values())
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def rouge_1(pred: str, gold: str) -> Tuple[float, float, float]:
    """ROUGE-1：unigram 重叠"""
    return rouge_n(pred, gold, n=1)


def rouge_2(pred: str, gold: str) -> Tuple[float, float, float]:
    """ROUGE-2：bigram 重叠"""
    return rouge_n(pred, gold, n=2)


def bleu(pred: str, gold: str, max_n: int = 4) -> float:
    """BLEU：n-gram 精度几何平均 + 短句惩罚"""
    pred_tokens = _tokenize(pred)
    gold_tokens = _tokenize(gold)

    if not pred_tokens or not gold_tokens:
        return 0.0

    # Brevity penalty
    bp = math.exp(1 - len(gold_tokens) / len(pred_tokens)) if len(pred_tokens) < len(gold_tokens) else 1.0

    # 各阶 n-gram 精度
    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = _get_ngrams(pred_tokens, n)
        gold_ngrams = _get_ngrams(gold_tokens, n)
        if not pred_ngrams:
            precisions.append(0.0)
            continue
        clipped = sum((pred_ngrams & gold_ngrams).values())
        total = sum(pred_ngrams.values())
        precisions.append(clipped / total if total > 0 else 0.0)

    # 若任一阶精度为 0，整体为 0
    if any(p == 0.0 for p in precisions):
        return 0.0

    log_avg = sum(math.log(p) for p in precisions) / max_n
    return bp * math.exp(log_avg)


def number_accuracy(pred: str, gold: str) -> float:
    """
    数字提取准确率。
    提取预测答案和参考答案中的所有数字，计算集合的 F1。
    对财报问答场景特别重要。
    """
    pred_nums = set(extract_numbers(pred))
    gold_nums = set(extract_numbers(gold))

    if not pred_nums and not gold_nums:
        return 1.0
    if not pred_nums or not gold_nums:
        return 0.0

    common = pred_nums & gold_nums
    precision = len(common) / len(pred_nums)
    recall = len(common) / len(gold_nums)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


def evaluate(pred_data: List[dict], gold_data: List[dict]) -> Dict:
    """
    评估主函数

    匹配策略：
    - 优先按 question 字段匹配
    - 如果 question 不匹配或不存在，则按顺序配对
    """
    # 建立 gold 的 question -> answer 映射
    gold_map = {}
    for item in gold_data:
        q = item.get("question", "").strip()
        if q:
            gold_map[q] = item

    results = []
    total = {"exact_match": 0, "char_f1": 0.0, "word_f1": 0.0,
             "rouge_1_f1": 0.0, "rouge_2_f1": 0.0, "rouge_l_f1": 0.0,
             "bleu": 0.0, "number_f1": 0.0}

    for pred_item in pred_data:
        pred_q = pred_item.get("question", "").strip()
        pred_ans = pred_item.get("answer", "")

        # 必须按 question 精确匹配
        if not pred_q or pred_q not in gold_map:
            logger.warning(f"未匹配到问题，已跳过: {pred_q[:60]}...")
            continue

        gold_item = gold_map[pred_q]

        gold_ans = gold_item.get("answer", "")

        # 标准化
        pred_norm = normalize_text(pred_ans)
        gold_norm = normalize_text(gold_ans)

        # 计算指标
        em = 1.0 if pred_norm == gold_norm else 0.0
        _, _, c_f1 = char_f1(pred_norm, gold_norm)
        _, _, w_f1 = word_f1(pred_norm, gold_norm)
        _, _, r1_f1 = rouge_1(pred_norm, gold_norm)
        _, _, r2_f1 = rouge_2(pred_norm, gold_norm)
        _, _, rl_f1 = rouge_l(pred_norm, gold_norm)
        b_score = bleu(pred_norm, gold_norm)
        n_f1 = number_accuracy(pred_ans, gold_ans)

        total["exact_match"] += em
        total["char_f1"] += c_f1
        total["word_f1"] += w_f1
        total["rouge_1_f1"] += r1_f1
        total["rouge_2_f1"] += r2_f1
        total["rouge_l_f1"] += rl_f1
        total["bleu"] += b_score
        total["number_f1"] += n_f1

        results.append({
            "question": pred_q,
            "predicted": pred_ans,
            "gold": gold_ans,
            "exact_match": em,
            "char_f1": c_f1,
            "word_f1": w_f1,
            "rouge_1_f1": r1_f1,
            "rouge_2_f1": r2_f1,
            "rouge_l_f1": rl_f1,
            "bleu": b_score,
            "number_f1": n_f1
        })

    n = len(results)
    if n == 0:
        logger.error("没有匹配到任何样本，请检查输入文件格式")
        return {}

    metrics = {
        "count": n,
        "accuracy": total["exact_match"] / n,
        "exact_match": total["exact_match"] / n,
        "char_f1": total["char_f1"] / n,
        "word_f1": total["word_f1"] / n,
        "rouge_1_f1": total["rouge_1_f1"] / n,
        "rouge_2_f1": total["rouge_2_f1"] / n,
        "rouge_l_f1": total["rouge_l_f1"] / n,
        "bleu": total["bleu"] / n,
        "number_f1": total["number_f1"] / n,
        "details": results
    }

    return metrics


def print_report(metrics: Dict, output_path: str = None):
    """打印评估报告"""
    logger.info("\n" + "=" * 60)
    logger.info("评估报告")
    logger.info("=" * 60)
    logger.info(f"样本数量: {metrics['count']}")
    logger.info("-" * 60)
    logger.info(f"准确率 (Accuracy):            {metrics['accuracy']:.4f}")
    logger.info(f"精确匹配率 (Exact Match):     {metrics['exact_match']:.4f}")
    logger.info(f"字符级 F1 (Char F1):          {metrics['char_f1']:.4f}")
    logger.info(f"词级 F1 (Word F1):            {metrics['word_f1']:.4f}")
    logger.info(f"ROUGE-1 F1:                   {metrics['rouge_1_f1']:.4f}")
    logger.info(f"ROUGE-2 F1:                   {metrics['rouge_2_f1']:.4f}")
    logger.info(f"ROUGE-L F1:                   {metrics['rouge_l_f1']:.4f}")
    logger.info(f"BLEU-4:                       {metrics['bleu']:.4f}")
    logger.info(f"数字提取 F1 (Number F1):      {metrics['number_f1']:.4f}")
    logger.info("=" * 60)

    # 找出表现最好和最差的样本
    details = metrics["details"]
    best = max(details, key=lambda x: x["rouge_l_f1"])
    worst = min(details, key=lambda x: x["rouge_l_f1"])

    logger.info("\n【ROUGE-L 最高样本】")
    logger.info(f"问题: {best['question'][:80]}...")
    logger.info(f"预测: {best['predicted'][:100]}...")
    logger.info(f"参考: {best['gold'][:100]}...")
    logger.info(f"分数: EM={best['exact_match']:.2f}, ROUGE-L={best['rouge_l_f1']:.4f}")

    logger.info("\n【ROUGE-L 最低样本】")
    logger.info(f"问题: {worst['question'][:80]}...")
    logger.info(f"预测: {worst['predicted'][:100]}...")
    logger.info(f"参考: {worst['gold'][:100]}...")
    logger.info(f"分数: EM={worst['exact_match']:.2f}, ROUGE-L={worst['rouge_l_f1']:.4f}")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"详细结果已保存至 {output_path}")
