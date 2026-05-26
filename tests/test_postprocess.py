"""Tests for postprocess module."""

import pytest

from src.generation.postprocess import (
    extract_numbers,
    verify_numbers,
    normalize_format,
    handle_empty,
    control_length,
    PostProcessor,
)


class TestExtractNumbers:
    def test_integer(self):
        assert extract_numbers("营收100亿元") == ["100"]

    def test_decimal(self):
        assert extract_numbers("利润率12.5%") == ["12.5%"]

    def test_percentage(self):
        assert extract_numbers("增长50%") == ["50%"]

    def test_negative(self):
        assert extract_numbers("下降-5%") == ["-5%"]

    def test_thousand_separator(self):
        assert extract_numbers("营收1,234万元") == ["1,234"]

    def test_no_numbers(self):
        assert extract_numbers("没有数字") == []

    def test_multiple_numbers(self):
        result = extract_numbers("2023年营收100亿元，利润20亿元")
        assert "100" in result
        assert "20" in result


class TestVerifyNumbers:
    def test_all_numbers_in_context(self):
        answer = "营收100亿元"
        context = "公司2023年营收100亿元"
        assert verify_numbers(answer, context) == []

    def test_hallucinated_numbers(self):
        answer = "营收999亿元"
        context = "公司2023年营收100亿元"
        assert verify_numbers(answer, context) == ["999"]

    def test_no_numbers(self):
        assert verify_numbers("没有数字", "也没有") == []


class TestNormalizeFormat:
    def test_percentage_space(self):
        assert normalize_format("增长 50 %") == "增长 50%"

    def test_currency_space(self):
        assert normalize_format("100 亿元") == "100亿元"
        assert normalize_format("50 万元") == "50万元"
        assert normalize_format("10 元") == "10元"

    def test_no_change_needed(self):
        text = "营收100亿元，增长50%"
        assert normalize_format(text) == text


class TestHandleEmpty:
    def test_empty_string(self):
        assert handle_empty("") == "根据提供的信息无法回答。"

    def test_whitespace_only(self):
        assert handle_empty("   ") == "根据提供的信息无法回答。"

    def test_unknown_phrases(self):
        assert handle_empty("不知道") == "根据提供的信息无法回答。"
        assert handle_empty("无法回答") == "根据提供的信息无法回答。"

    def test_normal_answer(self):
        ans = "营收100亿元"
        assert handle_empty(ans) == ans


class TestControlLength:
    def test_within_bounds(self):
        text = "这是一个正常长度的答案。"
        assert control_length(text, min_len=5, max_len=100) == text

    def test_truncate_long(self):
        text = "a" * 600
        result = control_length(text, max_len=500)
        assert len(result) <= 503  # allow "..."

    def test_truncate_at_sentence_boundary(self):
        text = "第一句。" + "b" * 600
        result = control_length(text, max_len=500)
        assert result.endswith("。") or result.endswith("...")


class TestPostProcessor:
    def test_process_empty(self):
        pp = PostProcessor()
        result = pp.process("")
        assert result["was_empty"] is True
        assert "无法回答" in result["answer"]

    def test_process_hallucination(self):
        pp = PostProcessor()
        result = pp.process(
            "营收999亿元", context_text="营收100亿元"
        )
        assert "999" in result["hallucinated_numbers"]

    def test_process_truncation(self):
        pp = PostProcessor(max_length=20)
        result = pp.process("a" * 100)
        assert result["was_truncated"] is True
        assert len(result["answer"]) <= 23

    def test_process_returns_dict(self):
        pp = PostProcessor()
        result = pp.process("正常答案", context_text="上下文")
        assert isinstance(result, dict)
        assert "answer" in result
        assert "hallucinated_numbers" in result
        assert "was_truncated" in result
        assert "was_empty" in result
