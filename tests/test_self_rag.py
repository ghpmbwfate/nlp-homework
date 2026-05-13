"""Tests for Self-RAG self-consistency check module."""

import pytest
from src.generation.self_rag import (
    extract_numbers,
    tokenize,
    check_support,
    self_consistency_vote,
    get_retrieval_feedback,
    run_self_check,
)


class TestExtractNumbers:
    def test_integers(self):
        assert "123" in extract_numbers("营收123亿元")

    def test_decimals(self):
        nums = extract_numbers("增长12.5个百分点")
        assert "12.5" in nums

    def test_percentages(self):
        nums = extract_numbers("同比增长25%")
        assert "25%" in nums

    def test_mixed(self):
        nums = extract_numbers("营收50亿元，增长12.5%，利润8亿")
        assert len(nums) >= 3

    def test_empty(self):
        assert extract_numbers("没有数字") == []


class TestTokenize:
    def test_chinese_tokenize(self):
        tokens = tokenize("速冻面米制品市场营收增长")
        assert len(tokens) > 0
        assert "速冻" in tokens or "面米" in tokens or "制品" in tokens

    def test_stop_words_removed(self):
        tokens = tokenize("这是一个测试的句子")
        assert "的" not in tokens
        assert "是" not in tokens


class TestCheckSupport:
    def test_well_supported(self):
        context = "公司2024年营收为50亿元，同比增长20%。主要产品包括速冻面米制品。"
        answer = "2024年公司营收为50亿元，增长了20%。"
        assert check_support(answer, context) == "Support"

    def test_contradict_hallucinated_numbers(self):
        context = "公司2024年营收为50亿元。"
        answer = "公司2024年营收为120亿元，利润为30亿元，市场份额为15%。"
        result = check_support(answer, context)
        assert result in ("Contradict", "Neutral")

    def test_empty_context(self):
        assert check_support("营收50亿", "") == "Neutral"

    def test_empty_answer(self):
        assert check_support("", "营收50亿") == "Neutral"

    def test_unable_to_answer(self):
        context = "公司主要生产汽车零部件。"
        answer = "根据提供的信息无法回答关于营收的问题。"
        result = check_support(answer, context)
        assert result == "Neutral"

    def test_similar_keywords(self):
        context = "速冻面米制品市场规模持续扩大，公司在该领域具有竞争优势。"
        answer = "速冻面米制品市场在扩大，公司有竞争优势。"
        assert check_support(answer, context) == "Support"


class TestSelfConsistencyVote:
    def test_single_answer(self):
        answer, count = self_consistency_vote(["答案A"])
        assert answer == "答案A"
        assert count == 1

    def test_identical_answers(self):
        answers = ["速冻面米制品营收增长", "速冻面米制品营收增长", "速冻面米制品营收增长"]
        answer, count = self_consistency_vote(answers)
        assert "速冻" in answer
        assert count >= 2

    def test_different_answers(self):
        answers = [
            "速冻面米制品营收增长20%",
            "速冻面米制品营收增长大约20%",
            "汽车零部件市场下滑",
        ]
        answer, count = self_consistency_vote(answers)
        # The two similar answers should win
        assert "20" in answer or "速冻" in answer
        assert count >= 2

    def test_empty_list(self):
        answer, count = self_consistency_vote([])
        assert answer == ""
        assert count == 0


class TestGetRetrievalFeedback:
    def test_contradict_feedback(self):
        fb = get_retrieval_feedback("Contradict")
        assert fb["action"] == "expand_retrieval"
        assert "幻觉" in fb["reason"]

    def test_support_feedback(self):
        fb = get_retrieval_feedback("Support")
        assert fb["action"] == "keep"

    def test_neutral_feedback(self):
        fb = get_retrieval_feedback("Neutral")
        assert fb["action"] == "review"


class TestRunSelfCheck:
    def test_supported_answer(self):
        context = "公司2024年营收为50亿元，利润为8亿元。"
        answer = "2024年营收50亿元，利润8亿元。"
        result = run_self_check(answer, context)
        assert result["verdict"] == "Support"
        assert "feedback" in result
        assert "num_support_ratio" in result
        assert result["num_support_ratio"] >= 0.8

    def test_contradict_answer(self):
        context = "公司营收约10亿元。"
        answer = "公司营收120亿元，利润30亿元，市场份额15%。"
        result = run_self_check(answer, context)
        assert result["verdict"] in ("Contradict", "Neutral")
        assert result["num_support_ratio"] <= 0.5

    def test_includes_details(self):
        result = run_self_check("营收50亿", "营收50亿元")
        assert "answer_numbers_found" in result
        assert "context_numbers_found" in result
        assert isinstance(result["num_support_ratio"], float)
        assert isinstance(result["keyword_overlap_ratio"], float)
