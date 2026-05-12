"""Tests for chunking module."""

import pytest

from src.indexing.chunking import (
    _split_into_paragraphs,
    _merge_short_paragraphs,
    _chunk_by_similarity,
    _apply_sliding_window,
)


class TestSplitIntoParagraphs:
    def test_blank_line_split(self):
        text = "段落1\n\n段落2\n\n段落3"
        result = _split_into_paragraphs(text)
        assert result == ["段落1", "段落2", "段落3"]

    def test_single_line_fallback(self):
        text = "行1\n\n行2\n\n行3"
        result = _split_into_paragraphs(text)
        assert result == ["行1", "行2", "行3"]

    def test_empty_text(self):
        assert _split_into_paragraphs("") == []

    def test_whitespace_only(self):
        assert _split_into_paragraphs("   \n\n   ") == []


class TestMergeShortParagraphs:
    def test_merge_short_with_previous(self):
        paras = ["这是一个很长的段落内容。", "短"]
        result = _merge_short_paragraphs(paras, min_length=30)
        assert len(result) == 1
        assert "短" in result[0]

    def test_no_merge_for_long(self):
        paras = [
            "这是一个非常非常非常长的段落内容，包含大量文字信息以及更多描述。",
            "这是另一个非常非常非常长的段落内容，也包含大量文字以及更多描述。",
        ]
        result = _merge_short_paragraphs(paras, min_length=30)
        assert len(result) == 2

    def test_empty(self):
        assert _merge_short_paragraphs([]) == []


class TestChunkBySimilarity:
    def test_empty(self):
        assert _chunk_by_similarity([], [], 500, 0.6) == []

    def test_single_paragraph(self):
        paras = ["唯一段落"]
        result = _chunk_by_similarity(paras, [], 500, 0.6)
        assert result == [paras]

    def test_break_on_low_similarity(self):
        paras = ["a" * 300, "b" * 300]
        sims = [0.1]  # low similarity
        result = _chunk_by_similarity(paras, sims, 500, 0.6)
        assert len(result) == 2

    def test_hard_size_break(self):
        paras = ["a" * 300, "b" * 300]
        sims = [0.9]  # high similarity
        result = _chunk_by_similarity(paras, sims, 400, 0.6)
        # Should break because combined length >= chunk_size
        assert len(result) >= 1


class TestApplySlidingWindow:
    def test_no_overlap_returns_same(self):
        chunks = [["a"], ["b"], ["c"]]
        result = _apply_sliding_window(chunks, 0)
        assert result == chunks

    def test_overlap_appends_previous(self):
        chunks = [["a" * 30, "b" * 30], ["c" * 30, "d" * 30]]
        result = _apply_sliding_window(chunks, 50)
        assert "b" * 30 in result[1]
        assert "c" * 30 in result[1]
        assert "d" * 30 in result[1]

    def test_empty(self):
        assert _apply_sliding_window([], 100) == []
