"""Tests for multi_recover module using 千味央厨 data."""

import pytest
from src.retrieval.multi_recover import (
    extract_titles,
    build_title_index,
    title_search,
    build_keyword_index,
    keyword_search,
    extract_summary,
    build_summary_index,
    summary_search,
    build_all_multi_indexes,
)


class TestTitleIndex:
    def test_extract_titles_returns_non_empty(self, qianwei_pages):
        titles = extract_titles(qianwei_pages)
        assert len(titles) > 0, "Should find titles in 千味央厨 report"
        for t in titles:
            assert "title" in t
            assert "filename" in t
            assert "page" in t

    def test_build_title_index(self, qianwei_pages):
        index = build_title_index(qianwei_pages)
        assert index["titles"], "Title index should have titles"
        assert index["bm25"] is not None, "BM25 index should be built"

    def test_title_search_relevant(self, qianwei_pages):
        index = build_title_index(qianwei_pages)
        results = title_search("千味央厨", index, top_k=3)
        assert len(results) > 0, "Should find matching titles"
        for r in results:
            assert r["source"] == "title"
            assert "score" in r
            assert r["score"] > 0

    def test_title_search_returns_empty_for_gibberish(self, qianwei_pages):
        index = build_title_index(qianwei_pages)
        results = title_search("xyzabc123不存在", index, top_k=5)
        # Should return empty or very low scores
        assert len(results) == 0 or all(r["score"] <= 1 for r in results)


class TestKeywordIndex:
    def test_build_keyword_index(self, qianwei_pages):
        index = build_keyword_index(qianwei_pages)
        assert index["vectorizer"] is not None
        assert index["tfidf_matrix"] is not None
        assert len(index["metas"]) > 0

    def test_keyword_search_relevant(self, qianwei_pages):
        index = build_keyword_index(qianwei_pages)
        results = keyword_search("餐饮 供应链", index, top_k=3)
        assert len(results) > 0, "Should find keyword matches"
        for r in results:
            assert r["source"] == "keyword"
            assert r["score"] > 0


class TestSummaryIndex:
    def test_extract_summary_returns_text(self):
        text = "公司主营业务为速冻面米制品的研发、生产和销售。公司产品包括..."
        summary = extract_summary(text)
        assert len(summary) > 5

    def test_extract_summary_empty_short_text(self):
        assert extract_summary("") == ""
        assert extract_summary("短") == ""

    def test_build_summary_index(self, qianwei_pages):
        index = build_summary_index(qianwei_pages)
        assert len(index["summaries"]) > 0, "Should create summaries for 千味央厨 pages"
        assert index["bm25"] is not None

    def test_summary_search(self, qianwei_pages):
        index = build_summary_index(qianwei_pages)
        results = summary_search("速冻面米制品", index, top_k=3)
        assert len(results) > 0, "Should find summary matches"
        for r in results:
            assert r["source"] == "summary"


class TestBuildAll:
    def test_build_all_multi_indexes(self, qianwei_pages):
        indexes = build_all_multi_indexes(qianwei_pages)
        assert "title_index" in indexes
        assert "keyword_index" in indexes
        assert "summary_index" in indexes
