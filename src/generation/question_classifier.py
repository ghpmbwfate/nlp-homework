"""Question type classifier using keyword/pattern matching."""

import re
from enum import Enum
from typing import Tuple


class QuestionType(Enum):
    FACT_EXTRACTION = "fact_extraction"
    ANALYSIS_SUMMARY = "analysis_summary"
    CHART_UNDERSTANDING = "chart_understanding"
    COMPARISON = "comparison"


# Keyword patterns for each question type
FACT_PATTERNS = [
    r'是多少', r'多少', r'几个', r'几年', r'何时', r'什么时候', r'哪年', r'哪一年', r'多少钱', r'营收', r'利润', r'收入', r'增长率', r'同比增长', r'同比下降', r'占比', r'比例', r'达到', r'规模', r'金额', r'数值', r'数据', r'什么.*业务', r'哪些', r'哪家', r'哪个', r'主要.*产品', r'核心.*业务',
]

ANALYSIS_PATTERNS = [
    r'分析', r'评估', r'总结', r'概括', r'简述', r'概述', r'说明', r'描述', r'介绍', r'阐述', r'解释', r'原因', r'因素', r'影响', r'驱动', r'逻辑', r'优势', r'劣势', r'风险', r'机遇', r'挑战', r'战略', r'布局', r'发展.*方向', r'趋势', r'如何.*看待', r'怎么.*看', r'观点', r'前景', r'展望', r'未来',
]

CHART_PATTERNS = [
    r'图表', r'图\d+', r'表\d+', r'如图', r'见表', r'图中', r'表中', r'数据图', r'折线图', r'柱状图', r'饼图', r'趋势图', r'根据图', r'从图', r'上图', r'下图',
]

COMPARISON_PATTERNS = [
    r'对比', r'比较', r'相比', r'相对于', r'不同于', r'区别于', r'差异', r'异同', r'孰优', r'哪个.*更', r'高于', r'低于', r'优于', r'不如', r'与.*相比', r'和.*比', r'同.*比较', r'行业.*对比', r'同业.*比较', r'竞争.*对比',
]


def _count_matches(text: str, patterns: list) -> int:
    """Count how many patterns match the text."""
    count = 0
    for pattern in patterns:
        if re.search(pattern, text):
            count += 1
    return count


def classify_question(question: str) -> QuestionType:
    """Classify question type based on keyword patterns."""
    scores = {
        QuestionType.FACT_EXTRACTION: _count_matches(question, FACT_PATTERNS),
        QuestionType.ANALYSIS_SUMMARY: _count_matches(question, ANALYSIS_PATTERNS),
        QuestionType.CHART_UNDERSTANDING: _count_matches(question, CHART_PATTERNS),
        QuestionType.COMPARISON: _count_matches(question, COMPARISON_PATTERNS),
    }
    best_type = max(scores, key=scores.get)
    if scores[best_type] == 0:
        return QuestionType.FACT_EXTRACTION
    return best_type


def classify_question_with_score(question: str) -> Tuple[QuestionType, dict]:
    """Classify question and return scores for all types."""
    scores = {
        "fact_extraction": _count_matches(question, FACT_PATTERNS),
        "analysis_summary": _count_matches(question, ANALYSIS_PATTERNS),
        "chart_understanding": _count_matches(question, CHART_PATTERNS),
        "comparison": _count_matches(question, COMPARISON_PATTERNS),
    }
    max_type = max(scores, key=scores.get)
    if scores[max_type] == 0:
        return QuestionType.FACT_EXTRACTION, scores
    return QuestionType(max_type), scores
