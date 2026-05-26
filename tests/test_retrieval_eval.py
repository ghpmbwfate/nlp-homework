"""Tests for retrieval_eval module."""

import pytest

from src.evaluation.retrieval_eval import evaluate_retrieval, print_retrieval_report


class MockRetriever:
    """Mock retriever that returns fixed results."""

    def __init__(self, results):
        self._results = results

    def search(self, query):
        return self._results


class TestEvaluateRetrieval:
    def test_empty_questions(self):
        retriever = MockRetriever([])
        result = evaluate_retrieval([], retriever)
        assert result["count"] == 0

    def test_perfect_retrieval(self):
        retriever = MockRetriever([
            {"filename": "doc.pdf", "page": 5},
        ])
        questions = [
            {"question": "q", "filename": "doc.pdf", "page": 5},
        ]
        result = evaluate_retrieval(questions, retriever, top_ks=[1, 3])
        assert result["metrics"]["recall@1"] == 1.0
        assert result["metrics"]["mrr"] == 1.0

    def test_miss_retrieval(self):
        retriever = MockRetriever([
            {"filename": "other.pdf", "page": 1},
        ])
        questions = [
            {"question": "q", "filename": "doc.pdf", "page": 5},
        ]
        result = evaluate_retrieval(questions, retriever, top_ks=[1])
        assert result["metrics"]["recall@1"] == 0.0
        assert result["metrics"]["mrr"] == 0.0

    def test_mrr_computation(self):
        retriever = MockRetriever([
            {"filename": "other.pdf", "page": 1},
            {"filename": "doc.pdf", "page": 5},
        ])
        questions = [
            {"question": "q", "filename": "doc.pdf", "page": 5},
        ]
        result = evaluate_retrieval(questions, retriever, top_ks=[1, 2])
        assert result["metrics"]["recall@1"] == 0.0
        assert result["metrics"]["recall@2"] == 1.0
        assert result["metrics"]["mrr"] == 0.5

    def test_hit_rate(self):
        retriever = MockRetriever([
            {"filename": "doc.pdf", "page": 5},
        ])
        questions = [
            {"question": "q", "filename": "doc.pdf", "page": 5},
        ]
        result = evaluate_retrieval(questions, retriever, top_ks=[1])
        assert result["metrics"]["hit_rate@1"] == 1.0


class TestPrintRetrievalReport:
    def test_prints_without_error(self, capsys):
        eval_result = {
            "count": 1,
            "metrics": {
                "mrr": 1.0,
                "recall@1": 1.0,
                "hit_rate@1": 1.0,
            },
            "details": [],
        }
        print_retrieval_report(eval_result)
        captured = capsys.readouterr()
        assert "MRR" in captured.out
