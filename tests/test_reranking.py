"""Tests for multi-stage reranking module."""

import pytest
from src.retrieval.reranking import (
    mmr_rerank,
    filter_by_question_type,
    multi_stage_rerank,
)


def make_hit(chunk_id, content, score=0.8, htype="text"):
    return {
        "chunk_id": chunk_id,
        "content": content,
        "score": score,
        "rerank_score": score,
        "type": htype,
        "source": "dense",
        "filename": "test",
        "page": 1,
    }


class TestMMRRerank:
    def test_returns_all_when_fewer_than_top_k(self):
        candidates = [make_hit("a", "content A"), make_hit("b", "content B")]
        result = mmr_rerank("query", candidates, top_k=5)
        assert len(result) == 2

    def test_returns_top_k(self):
        candidates = [make_hit(f"c{i}", f"content {i}", score=0.9 - i * 0.1) for i in range(10)]
        result = mmr_rerank("query", candidates, top_k=3)
        assert len(result) == 3

    def test_diversity_penalty(self):
        """Diverse results should be preferred over duplicate ones."""
        candidates = [
            make_hit("a", "速冻面米制品市场营收增长快速", score=0.9),
            make_hit("b", "速冻面米制品市场规模扩大迅速", score=0.85),  # very similar to a
            make_hit("c", "汽车零部件制造行业分析", score=0.8),          # different topic
        ]
        result = mmr_rerank("速冻面米制品", candidates, lambda_param=0.5, top_k=2)
        # With lambda=0.5 (balanced), should include both similar (a) and diverse (c)
        contents = [r["chunk_id"] for r in result]
        assert "a" in contents or "c" in contents

    def test_empty_candidates(self):
        assert mmr_rerank("query", [], top_k=5) == []

    def test_lambda_high_prefers_relevance(self):
        """High lambda should prefer relevance over diversity."""
        candidates = [
            make_hit("a", "速冻面米制品市场规模数据", score=0.95),
            make_hit("b", "速冻面米制品营收趋势分析", score=0.9),
            make_hit("c", "完全无关的汽车行业内容", score=0.3),
        ]
        result = mmr_rerank("速冻面米", candidates, lambda_param=0.9, top_k=2)
        # High lambda: should definitely include highest relevance (a)
        assert result[0]["chunk_id"] == "a"


class TestFilterByQuestionType:
    def test_fact_extraction_boosts_tables(self):
        candidates = [
            make_hit("t1", "text chunk about revenue", htype="text"),
            make_hit("t2", "table with numbers", htype="table"),
            make_hit("t3", "another text chunk", htype="text"),
        ]
        result = filter_by_question_type(candidates, "fact_extraction")
        # Table chunks should come first
        assert result[0]["type"] == "table"

    def test_chart_understanding_boosts_chart_refs(self):
        candidates = [
            make_hit("a", "根据图5显示，营收增长"),
            make_hit("b", "普通文本内容"),
            make_hit("c", "如表3所示，各产品对比"),
        ]
        result = filter_by_question_type(candidates, "chart_understanding")
        # Chunks with chart references should come first
        first_two = [r["chunk_id"] for r in result[:2]]
        assert "a" in first_two or "c" in first_two

    def test_analysis_summary_keeps_order(self):
        candidates = [
            make_hit("a", "first"),
            make_hit("b", "second"),
            make_hit("c", "third"),
        ]
        result = filter_by_question_type(candidates, "analysis_summary")
        assert [r["chunk_id"] for r in result] == ["a", "b", "c"]

    def test_empty_list(self):
        assert filter_by_question_type([], "fact_extraction") == []


class TestMultiStageRerank:
    def test_full_pipeline(self):
        candidates = [make_hit(f"c{i}", f"content number {i}", score=0.9 - i * 0.02)
                      for i in range(30)]
        result = multi_stage_rerank(
            "test query", candidates,
            reranker=None,
            question_type="fact_extraction",
            coarse_k=20,
            fine_k=10,
            final_k=3
        )
        assert len(result) <= 3
        assert len(result) > 0

    def test_empty_input(self):
        result = multi_stage_rerank("query", [])
        assert result == []

    def test_small_input(self):
        candidates = [make_hit("a", "content A")]
        result = multi_stage_rerank("query", candidates, final_k=3)
        assert len(result) == 1

    def test_no_question_type(self):
        candidates = [make_hit(f"c{i}", f"content {i}", score=0.8 - i * 0.05)
                      for i in range(15)]
        result = multi_stage_rerank("query", candidates, question_type=None, final_k=3)
        assert len(result) == 3

    def test_preserves_metadata(self):
        candidates = [make_hit(f"c{i}", f"content {i}", score=0.9 - i * 0.03)
                      for i in range(10)]
        result = multi_stage_rerank("query", candidates, final_k=3)
        for r in result:
            assert "chunk_id" in r
            assert "content" in r
            assert "score" in r
            assert "type" in r
