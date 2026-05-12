"""Tests for self_rag module."""

import pytest

from src.generation.self_rag import SelfRAG


class TestSelfRAGInit:
    def test_init_with_api_key(self):
        sr = SelfRAG(api_key="fake-key")
        assert sr.model == "qwen-turbo"
        assert sr.client is not None

    def test_init_without_api_key_raises(self, monkeypatch):
        monkeypatch.setattr(
            "src.generation.self_rag.DASHSCOPE_API_KEY", ""
        )
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key"):
            SelfRAG()

    def test_init_reads_env_var(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")
        sr = SelfRAG()
        assert sr.client is not None


class TestVerify:
    def test_support_verdict(self, monkeypatch):
        sr = SelfRAG(api_key="fake")
        monkeypatch.setattr(sr, "_call_llm", lambda s, u: "support")
        assert sr.verify("答案", "上下文") == "support"

    def test_contradict_verdict(self, monkeypatch):
        sr = SelfRAG(api_key="fake")
        monkeypatch.setattr(sr, "_call_llm", lambda s, u: "contradict")
        assert sr.verify("答案", "上下文") == "contradict"

    def test_neutral_verdict(self, monkeypatch):
        sr = SelfRAG(api_key="fake")
        monkeypatch.setattr(sr, "_call_llm", lambda s, u: "neutral")
        assert sr.verify("答案", "上下文") == "neutral"

    def test_fallback_to_neutral(self, monkeypatch):
        sr = SelfRAG(api_key="fake")
        monkeypatch.setattr(sr, "_call_llm", lambda s, u: "unknown")
        assert sr.verify("答案", "上下文") == "neutral"


class TestVote:
    def test_empty(self):
        sr = SelfRAG(api_key="fake")
        assert sr.vote([]) == ""

    def test_single(self):
        sr = SelfRAG(api_key="fake")
        assert sr.vote(["唯一答案"]) == "唯一答案"

    def test_most_consistent(self):
        sr = SelfRAG(api_key="fake")
        # 答案1 和 答案2 更相似，答案3 不同
        answers = ["营收100亿元", "营收约100亿元", "利润50亿元"]
        result = sr.vote(answers)
        assert result in ["营收100亿元", "营收约100亿元"]


class MockGenerator:
    """Mock generator for testing check_and_refine."""

    def __init__(self, answers=None):
        self.answers = answers or []
        self.call_count = 0

    def generate(self, **kwargs):
        ans = self.answers[self.call_count % len(self.answers)]
        self.call_count += 1
        return {"answer": ans}


class TestCheckAndRefine:
    def test_support_no_regen(self, monkeypatch):
        sr = SelfRAG(api_key="fake")
        monkeypatch.setattr(sr, "_call_llm", lambda s, u: "support")
        gen = MockGenerator()
        result = sr.check_and_refine("答案", "上下文", gen, "问题")
        assert result["verification"] == "support"
        assert result["attempts"] == 1

    def test_contradict_regenerates(self, monkeypatch):
        sr = SelfRAG(api_key="fake")
        monkeypatch.setattr(sr, "_call_llm", lambda s, u: "contradict")
        gen = MockGenerator(["新答案"])
        result = sr.check_and_refine("答案", "上下文", gen, "问题")
        assert result["verification"] == "contradict"
        assert result["attempts"] == 2
        assert result["answer"] == "新答案"

    def test_max_attempts(self, monkeypatch):
        sr = SelfRAG(api_key="fake")
        monkeypatch.setattr(sr, "_call_llm", lambda s, u: "contradict")
        gen = MockGenerator(["a", "b", "c"])
        result = sr.check_and_refine(
            "答案", "上下文", gen, "问题", max_attempts=3
        )
        assert result["attempts"] == 3
