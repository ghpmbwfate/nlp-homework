"""Tests for citation module."""

import pytest
from src.generation.citation import (
    CITATION_INSTRUCTION,
    add_citation_instruction,
    extract_citations,
    format_citation,
    clean_answer_no_citations,
    has_citations,
)


class TestAddCitationInstruction:
    def test_appends_instruction(self):
        prompt = "请回答问题：{question}"
        result = add_citation_instruction(prompt)
        assert prompt in result
        assert "来源" in result
        assert "第X页" in result

    def test_instruction_non_empty(self):
        assert len(CITATION_INSTRUCTION.strip()) > 0


class TestExtractCitations:
    def test_single_citation(self):
        answer = "公司营收为XX亿元 [来源: 千味央厨 第5页]"
        citations = extract_citations(answer)
        assert len(citations) == 1
        assert citations[0]["filename"] == "千味央厨"
        assert citations[0]["page"] == 5

    def test_multiple_citations(self):
        answer = (
            "公司营收为XX亿元 [来源: 千味央厨 第5页]，"
            "利润为YY亿元 [来源: 千味央厨 第12页]"
        )
        citations = extract_citations(answer)
        assert len(citations) == 2
        assert citations[0]["page"] == 5
        assert citations[1]["page"] == 12

    def test_duplicate_citations_deduplicated(self):
        answer = "数据参考 [来源: 千味央厨 第3页] 和 [来源: 千味央厨 第3页]"
        citations = extract_citations(answer)
        assert len(citations) == 1

    def test_filename_with_hyphens(self):
        answer = "数据显示 [来源: 千味央厨-千寻百味乘势而上-221227 第3页]"
        citations = extract_citations(answer)
        assert len(citations) == 1
        assert "千味央厨" in citations[0]["filename"]
        assert citations[0]["page"] == 3

    def test_filename_with_spaces(self):
        answer = "根据报告 [来源: 伊利股份 深度报告 第8页]"
        citations = extract_citations(answer)
        assert len(citations) == 1
        assert "伊利股份" in citations[0]["filename"]

    def test_no_citations(self):
        answer = "公司营收约为50亿元，同比增长10%。"
        citations = extract_citations(answer)
        assert len(citations) == 0

    def test_malformed_citation_ignored(self):
        answer = "[来源: 第5页] 数据不完整"
        citations = extract_citations(answer)
        assert len(citations) == 0

    def test_non_numeric_page_ignored(self):
        answer = "[来源: 千味央厨 第X页]"
        citations = extract_citations(answer)
        assert len(citations) == 0


class TestFormatCitation:
    def test_format(self):
        result = format_citation("千味央厨", 5)
        assert result == "[来源: 千味央厨 第5页]"


class TestCleanAnswerNoCitations:
    def test_removes_citations(self):
        answer = "营收为XX亿元 [来源: 千味央厨 第5页] 这是结论。"
        cleaned = clean_answer_no_citations(answer)
        assert "[来源:" not in cleaned
        assert "营收为XX亿元" in cleaned

    def test_no_citations_unchanged(self):
        answer = "营收为XX亿元。"
        cleaned = clean_answer_no_citations(answer)
        assert cleaned == answer


class TestHasCitations:
    def test_has_citations(self):
        assert has_citations("答案 [来源: 千味央厨 第5页]") is True

    def test_no_citations(self):
        assert has_citations("普通答案无引用") is False
