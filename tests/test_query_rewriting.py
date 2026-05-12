"""Tests for query_rewriting module."""

import pytest

from src.retrieval.query_rewriting import QueryRewriter, NoOpQueryRewriter


class TestNoOpQueryRewriter:
    def test_rewrite_returns_original(self):
        rewriter = NoOpQueryRewriter()
        assert rewriter.rewrite("测试问题") == ["测试问题"]

    def test_decompose_returns_original(self):
        rewriter = NoOpQueryRewriter()
        assert rewriter.decompose("测试问题") == ["测试问题"]

    def test_hyde_returns_original(self):
        rewriter = NoOpQueryRewriter()
        assert rewriter.hyde_generate("测试问题") == "测试问题"

    def test_rewrite_ignores_num_variants(self):
        rewriter = NoOpQueryRewriter()
        assert rewriter.rewrite("测试", num_variants=10) == ["测试"]


class TestQueryRewriterInit:
    def test_init_with_api_key(self):
        rewriter = QueryRewriter(api_key="fake-key")
        assert rewriter.model == "qwen-turbo"
        assert rewriter.client is not None

    def test_init_without_api_key_raises(self, monkeypatch):
        monkeypatch.setattr(
            "src.retrieval.query_rewriting.DASHSCOPE_API_KEY", ""
        )
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key"):
            QueryRewriter()

    def test_init_reads_env_var(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")
        rewriter = QueryRewriter()
        assert rewriter.client is not None

    def test_init_custom_model(self):
        rewriter = QueryRewriter(api_key="fake", model="qwen-max")
        assert rewriter.model == "qwen-max"

    def test_init_custom_base_url(self):
        url = "https://custom.example.com/v1"
        rewriter = QueryRewriter(api_key="fake", base_url=url)
        assert rewriter.client is not None


class TestQueryRewriterRewrite:
    def test_rewrite_returns_list(self, monkeypatch):
        rewriter = QueryRewriter(api_key="fake")
        monkeypatch.setattr(
            rewriter, "_call_llm", lambda s, u: "改写1\n改写2"
        )
        result = rewriter.rewrite("原始问题")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_rewrite_includes_original(self, monkeypatch):
        rewriter = QueryRewriter(api_key="fake")
        monkeypatch.setattr(
            rewriter, "_call_llm", lambda s, u: "改写1\n改写2"
        )
        result = rewriter.rewrite("原始问题")
        assert "原始问题" in result

    def test_rewrite_deduplicates(self, monkeypatch):
        rewriter = QueryRewriter(api_key="fake")
        monkeypatch.setattr(
            rewriter, "_call_llm", lambda s, u: "原始问题\n原始问题"
        )
        result = rewriter.rewrite("原始问题")
        assert result.count("原始问题") == 1

    def test_rewrite_empty_response(self, monkeypatch):
        rewriter = QueryRewriter(api_key="fake")
        monkeypatch.setattr(rewriter, "_call_llm", lambda s, u: "")
        result = rewriter.rewrite("原始问题")
        assert result == ["原始问题"]


class TestQueryRewriterDecompose:
    def test_decompose_returns_list(self, monkeypatch):
        rewriter = QueryRewriter(api_key="fake")
        monkeypatch.setattr(
            rewriter, "_call_llm", lambda s, u: "子问题1\n子问题2"
        )
        result = rewriter.decompose("复合问题")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_decompose_fallback(self, monkeypatch):
        rewriter = QueryRewriter(api_key="fake")
        monkeypatch.setattr(rewriter, "_call_llm", lambda s, u: "")
        result = rewriter.decompose("单一问题")
        assert result == ["单一问题"]


class TestQueryRewriterHyDE:
    def test_hyde_returns_string(self, monkeypatch):
        rewriter = QueryRewriter(api_key="fake")
        monkeypatch.setattr(
            rewriter, "_call_llm", lambda s, u: "假设文档内容"
        )
        result = rewriter.hyde_generate("问题")
        assert isinstance(result, str)
        assert result == "假设文档内容"

    def test_hyde_not_empty(self, monkeypatch):
        rewriter = QueryRewriter(api_key="fake")
        monkeypatch.setattr(
            rewriter, "_call_llm", lambda s, u: "文档"
        )
        result = rewriter.hyde_generate("问题")
        assert len(result) > 0
