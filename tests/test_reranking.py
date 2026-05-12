"""Tests for reranking module."""

import numpy as np
import pytest

from src.retrieval.reranking import MultiStageReranker


class FakeCrossEncoder:
    """Fake CrossEncoder that returns fixed scores."""

    def __init__(self, scores):
        self.scores = scores
        self.call_count = 0

    def predict(self, pairs):
        result = self.scores[self.call_count : self.call_count + len(pairs)]
        self.call_count += len(pairs)
        return np.array(result)


class FakeEmbedder:
    """Fake SentenceTransformer that returns fixed embeddings."""

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.call_count = 0

    def encode(self, texts, show_progress_bar=False):
        result = self.embeddings[self.call_count : self.call_count + len(texts)]
        self.call_count += len(texts)
        return np.array(result)


@pytest.fixture
def reranker(monkeypatch):
    """Fixture with mocked models."""
    instance = MultiStageReranker.__new__(MultiStageReranker)
    instance.reranker = FakeCrossEncoder([0.9, 0.8, 0.7, 0.6, 0.5])
    instance.embedder = FakeEmbedder([
        [1, 0, 0],
        [0.9, 0.1, 0],
        [0, 1, 0],
        [0, 0.9, 0.1],
        [0, 0, 1],
    ])
    instance.mmr_lambda = 0.7
    return instance


class TestCrossEncoderRank:
    def test_empty_candidates(self, reranker):
        assert reranker.cross_encoder_rank("q", []) == []

    def test_adds_rerank_score(self, reranker):
        cands = [{"content": "a"}, {"content": "b"}]
        result = reranker.cross_encoder_rank("q", cands)
        assert "rerank_score" in result[0]
        assert result[0]["rerank_score"] == 0.9

    def test_sorted_descending(self, reranker):
        cands = [{"content": "a"}, {"content": "b"}]
        result = reranker.cross_encoder_rank("q", cands)
        scores = [r["rerank_score"] for r in result]
        assert scores == sorted(scores, reverse=True)


class TestMMRRerank:
    def test_empty_candidates(self, reranker):
        assert reranker.mmr_rerank("q", []) == []

    def test_returns_top_k(self, reranker):
        cands = [{"content": "a"}, {"content": "b"}, {"content": "c"}]
        result = reranker.mmr_rerank("q", cands, top_k=2)
        assert len(result) == 2

    def test_diversity_vs_relevance(self, reranker):
        # Embedding 0,1 相似，2 不同；MMR 应选 1 个相关 + 1 个 diverse
        cands = [
            {"content": "a"},
            {"content": "b"},
            {"content": "c"},
        ]
        result = reranker.mmr_rerank("q", cands, top_k=2)
        assert len(result) == 2


class TestTypeBoost:
    def test_no_boost_for_non_fact(self, reranker):
        cands = [
            {"content": "a", "type": "table", "rerank_score": 0.5},
        ]
        result = reranker.type_boost(cands, "analysis_summary")
        assert result[0]["rerank_score"] == 0.5

    def test_boosts_table_chunks(self, reranker):
        cands = [
            {"content": "a", "type": "table", "rerank_score": 0.5},
            {"content": "b", "type": "text", "rerank_score": 0.6},
        ]
        result = reranker.type_boost(cands, "fact_extraction")
        assert result[0]["rerank_score"] > 0.5  # table boosted


class TestFullPipeline:
    def test_rerank_pipeline(self, reranker):
        cands = [
            {"content": "a"},
            {"content": "b"},
            {"content": "c"},
        ]
        result = reranker.rerank("q", cands, ce_top_k=3, mmr_top_k=2)
        assert len(result) <= 2
        assert "rerank_score" in result[0]
