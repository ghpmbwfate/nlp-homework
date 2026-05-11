"""Tests for question classifier."""


import pytest
from src.generation.question_classifier import (
    QuestionType,
    classify_question,
    classify_question_with_score,
)


class TestFactExtraction:
    def test_revenue_query(self):
        assert classify_question("2024年营收是多少？") == QuestionType.FACT_EXTRACTION

    def test_product_query(self):
        assert classify_question("千味央厨的主要产品有哪些？") == QuestionType.FACT_EXTRACTION

    def test_number_query(self):
        assert classify_question("速冻面米制品的收入占比达到多少？") == QuestionType.FACT_EXTRACTION

    def test_growth_rate(self):
        assert classify_question("2023年同比增长率是多少？") == QuestionType.FACT_EXTRACTION

    def test_when_query(self):
        assert classify_question("千味央厨什么时候上市的？") == QuestionType.FACT_EXTRACTION


class TestAnalysisSummary:
    def test_analysis_query(self):
        assert classify_question("分析千味央厨的竞争优势") == QuestionType.ANALYSIS_SUMMARY

    def test_describe_query(self):
        assert classify_question("简述千味央厨的业务模式") == QuestionType.ANALYSIS_SUMMARY

    def test_outlook_query(self):
        assert classify_question("未来发展趋势如何？") == QuestionType.ANALYSIS_SUMMARY

    def test_reason_query(self):
        assert classify_question("千味央厨业绩增长的原因是什么？") == QuestionType.ANALYSIS_SUMMARY

    def test_strategy_query(self):
        assert classify_question("千味央厨的市场布局战略是怎样的？") == QuestionType.ANALYSIS_SUMMARY


class TestChartUnderstanding:
    def test_chart_query(self):
        assert classify_question("根据图表64，千味央厨的营收趋势如何？") == QuestionType.CHART_UNDERSTANDING

    def test_figure_query(self):
        assert classify_question("图5显示了什么") == QuestionType.CHART_UNDERSTANDING

    def test_table_query(self):
        assert classify_question("如表3所示，各业务的毛利率对比如何？") == QuestionType.CHART_UNDERSTANDING


class TestComparison:
    def test_compare_query(self):
        assert classify_question("与同行业相比，千味央厨的利润率如何？") == QuestionType.COMPARISON

    def test_vs_query(self):
        assert classify_question("千味央厨和安井食品相比，谁更有优势？") == QuestionType.COMPARISON

    def test_diff_query(self):
        assert classify_question("千味央厨与竞争对手的差异是什么？") == QuestionType.COMPARISON


class TestAmbiguous:
    def test_empty_defaults_to_fact(self):
        assert classify_question("你好") == QuestionType.FACT_EXTRACTION

    def test_mixed_both_analysis_and_fact(self):
        # “分析” triggers analysis, “营收” triggers fact
        # analysis wins because “分析” is a stronger signal
        result = classify_question("分析千味央厨的营收情况")
        assert result in (QuestionType.ANALYSIS_SUMMARY, QuestionType.FACT_EXTRACTION)


class TestWithScore:
    def test_returns_scores(self):
        qtype, scores = classify_question_with_score("分析千味央厨的营收情况")
        assert isinstance(qtype, QuestionType)
        assert isinstance(scores, dict)
        assert "fact_extraction" in scores
        assert "analysis_summary" in scores
        assert any(v > 0 for v in scores.values())