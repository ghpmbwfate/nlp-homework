"""Tests for error_analysis module."""

import pytest

from src.evaluation.error_analysis import analyze_errors, print_error_report


class TestAnalyzeErrors:
    def test_empty_data(self):
        result = analyze_errors([], [])
        assert result["total_questions"] == 0
        assert result["total_errors"] == 0
        assert result["bad_cases"] == []

    def test_no_errors(self):
        pred = [
            {
                "question": "q1",
                "answer": "a",
                "question_type": "fact",
                "filename": "doc.pdf",
            }
        ]
        gold = [{"question": "q1", "answer": "a"}]
        result = analyze_errors(pred, gold)
        assert result["total_errors"] == 0
        assert result["by_question_type"]["fact"]["error_rate"] == 0.0

    def test_all_errors(self):
        pred = [
            {
                "question": "q1",
                "answer": "wrong",
                "question_type": "fact",
                "filename": "doc.pdf",
            }
        ]
        gold = [{"question": "q1", "answer": "right"}]
        result = analyze_errors(pred, gold)
        assert result["total_errors"] == 1
        assert result["by_question_type"]["fact"]["error_rate"] == 1.0

    def test_by_question_type(self):
        pred = [
            {"question": "q1", "answer": "a", "question_type": "fact", "filename": "f"},
            {"question": "q2", "answer": "b", "question_type": "analysis", "filename": "f"},
        ]
        gold = [
            {"question": "q1", "answer": "wrong"},
            {"question": "q2", "answer": "b"},
        ]
        result = analyze_errors(pred, gold)
        assert result["by_question_type"]["fact"]["error_rate"] == 1.0
        assert result["by_question_type"]["analysis"]["error_rate"] == 0.0

    def test_by_document(self):
        pred = [
            {"question": "q1", "answer": "a", "question_type": "t", "filename": "a.pdf"},
            {"question": "q2", "answer": "b", "question_type": "t", "filename": "b.pdf"},
        ]
        gold = [
            {"question": "q1", "answer": "wrong"},
            {"question": "q2", "answer": "b"},
        ]
        result = analyze_errors(pred, gold)
        assert result["by_document"]["a.pdf"]["error_rate"] == 1.0
        assert result["by_document"]["b.pdf"]["error_rate"] == 0.0

    def test_bad_cases(self):
        pred = [
            {
                "question": "q1",
                "answer": "wrong",
                "question_type": "fact",
                "filename": "doc.pdf",
                "citations": [],
            }
        ]
        gold = [{"question": "q1", "answer": "right"}]
        result = analyze_errors(pred, gold)
        assert len(result["bad_cases"]) == 1
        assert result["bad_cases"][0]["question"] == "q1"

    def test_missing_question_skipped(self):
        pred = [{"question": "q1", "answer": "a", "question_type": "t", "filename": "f"}]
        gold = [{"question": "q2", "answer": "a"}]
        result = analyze_errors(pred, gold)
        assert result["total_questions"] == 0


class TestPrintErrorReport:
    def test_prints_without_error(self, capsys):
        analysis = {
            "total_questions": 2,
            "total_errors": 1,
            "by_question_type": {
                "fact": {"total": 1, "errors": 1, "error_rate": 1.0},
            },
            "by_document": {
                "doc.pdf": {"total": 1, "errors": 1, "error_rate": 1.0},
            },
            "bad_cases": [{"question": "q1"}],
        }
        print_error_report(analysis)
        captured = capsys.readouterr()
        assert "错误率" in captured.out
