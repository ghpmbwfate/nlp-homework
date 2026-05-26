"""Tests for LLMGenerator (cloud LLM answer generation)."""

import pytest

from src.generation.generator import LLMGenerator


class TestLLMGeneratorInit:
    def test_init_with_api_key(self):
        gen = LLMGenerator(api_key="fake-key")
        assert gen.client is not None
        assert gen.model  # 从 OPENAI_MODEL 或默认值读取

    def test_init_without_api_key_raises(self, monkeypatch):
        monkeypatch.setattr(
            "src.generation.generator.OPENAI_API_KEY", ""
        )
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key"):
            LLMGenerator()

    def test_init_reads_env_var(self, monkeypatch):
        monkeypatch.setattr(
            "src.generation.generator.OPENAI_API_KEY", ""
        )
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        gen = LLMGenerator()
        assert gen.client is not None

    def test_init_custom_model(self):
        gen = LLMGenerator(api_key="fake", model="custom-model")
        assert gen.model == "custom-model"

    def test_init_custom_base_url(self):
        gen = LLMGenerator(
            api_key="fake", base_url="http://example.com/v1"
        )
        assert gen.client is not None

    def test_init_default_temperature_is_zero(self):
        gen = LLMGenerator(api_key="fake")
        assert gen.temperature == 0.0

    def test_init_custom_temperature(self):
        gen = LLMGenerator(api_key="fake", temperature=0.7)
        assert gen.temperature == 0.7


class TestLLMGeneratorGenerate:
    def test_generate_returns_dict_with_required_keys(self, monkeypatch):
        gen = LLMGenerator(api_key="fake")
        monkeypatch.setattr(
            gen, "_call_llm", lambda prompt, max_tokens: "营收为100亿。"
        )
        result = gen.generate(
            question="2022年营收是多少？",
            context_text="营收100亿",
        )
        assert set(result.keys()) == {"answer", "question_type", "citations"}
        assert result["answer"] == "营收为100亿。"

    def test_generate_uses_classifier_when_type_none(self, monkeypatch):
        gen = LLMGenerator(api_key="fake")
        monkeypatch.setattr(
            gen, "_call_llm", lambda prompt, max_tokens: "ans"
        )
        # "是多少" → fact_extraction
        result = gen.generate(
            question="2022年营收是多少？",
            context_text="ctx",
        )
        assert result["question_type"] == "fact_extraction"

    def test_generate_respects_question_type_override(self, monkeypatch):
        gen = LLMGenerator(api_key="fake")
        monkeypatch.setattr(
            gen, "_call_llm", lambda prompt, max_tokens: "ans"
        )
        # 即使问题中包含 "是多少"，显式 override 也应胜出
        result = gen.generate(
            question="2022年营收是多少？",
            context_text="ctx",
            question_type="analysis_summary",
        )
        assert result["question_type"] == "analysis_summary"

    def test_generate_invalid_question_type_raises(self):
        gen = LLMGenerator(api_key="fake")
        with pytest.raises(ValueError):
            gen.generate(
                question="问题",
                context_text="ctx",
                question_type="bogus_type",
            )

    def test_generate_passes_max_new_tokens_to_call(self, monkeypatch):
        gen = LLMGenerator(api_key="fake")
        captured = {}

        def fake_call(prompt, max_tokens):
            captured["max_tokens"] = max_tokens
            return "ans"

        monkeypatch.setattr(gen, "_call_llm", fake_call)
        gen.generate(
            question="问题",
            context_text="ctx",
            max_new_tokens=256,
        )
        assert captured["max_tokens"] == 256

    def test_generate_prompt_contains_question_and_context(
        self, monkeypatch
    ):
        gen = LLMGenerator(api_key="fake")
        captured = {}

        def fake_call(prompt, max_tokens):
            captured["prompt"] = prompt
            return "ans"

        monkeypatch.setattr(gen, "_call_llm", fake_call)
        gen.generate(
            question="问题XYZ",
            context_text="上下文ABC",
        )
        assert "问题XYZ" in captured["prompt"]
        assert "上下文ABC" in captured["prompt"]


class TestLLMGeneratorImageIgnored:
    def test_image_path_kwarg_accepted_but_ignored(self, monkeypatch):
        gen = LLMGenerator(api_key="fake")
        captured = {}

        def fake_call(prompt, max_tokens):
            captured["prompt"] = prompt
            return "ans"

        monkeypatch.setattr(gen, "_call_llm", fake_call)
        # 传入一个明显的伪路径，不应抛错，也不应进入 prompt
        result = gen.generate(
            question="问题",
            context_text="ctx",
            image_path="/totally/fake/path/image.png",
        )
        assert result["answer"] == "ans"
        assert "/totally/fake/path/image.png" not in captured["prompt"]

    def test_image_path_none_default_works(self, monkeypatch):
        gen = LLMGenerator(api_key="fake")
        monkeypatch.setattr(
            gen, "_call_llm", lambda prompt, max_tokens: "ans"
        )
        result = gen.generate(question="问题", context_text="ctx")
        assert result["answer"] == "ans"


class TestLLMGeneratorCitationIntegration:
    def test_citations_extracted_from_answer(self, monkeypatch):
        gen = LLMGenerator(api_key="fake")
        monkeypatch.setattr(
            gen, "_call_llm",
            lambda prompt, max_tokens: "营收100亿。[来源: 千味央厨 第3页]"
        )
        result = gen.generate(question="营收?", context_text="ctx")
        assert result["citations"] == [
            {"filename": "千味央厨", "page": 3}
        ]

    def test_no_citations_returns_empty_list(self, monkeypatch):
        gen = LLMGenerator(api_key="fake")
        monkeypatch.setattr(
            gen, "_call_llm", lambda prompt, max_tokens: "纯文本无引用"
        )
        result = gen.generate(question="问题", context_text="ctx")
        assert result["citations"] == []

    def test_multiple_citations_deduplicated(self, monkeypatch):
        gen = LLMGenerator(api_key="fake")
        ans = (
            "见 [来源: 千味央厨 第3页] 与 [来源: 千味央厨 第5页]"
            " 以及 [来源: 千味央厨 第3页]"  # 第3页重复
        )
        monkeypatch.setattr(
            gen, "_call_llm", lambda prompt, max_tokens: ans
        )
        result = gen.generate(question="问题", context_text="ctx")
        assert len(result["citations"]) == 2


class TestLLMGeneratorCallLLM:
    def test_call_llm_wraps_openai_errors(self, monkeypatch):
        gen = LLMGenerator(api_key="fake")

        def raising_create(**kwargs):
            raise RuntimeError("network down")

        monkeypatch.setattr(
            gen.client.chat.completions, "create", raising_create
        )
        with pytest.raises(RuntimeError, match="LLM API call failed"):
            gen._call_llm("p", max_tokens=10)

    def test_call_llm_handles_empty_content(self, monkeypatch):
        gen = LLMGenerator(api_key="fake")

        class _Msg:
            content = None

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        monkeypatch.setattr(
            gen.client.chat.completions, "create",
            lambda **kw: _Resp()
        )
        assert gen._call_llm("p", max_tokens=10) == ""

    def test_call_llm_returns_stripped_content(self, monkeypatch):
        gen = LLMGenerator(api_key="fake")

        class _Msg:
            content = "  hello  "

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        monkeypatch.setattr(
            gen.client.chat.completions, "create",
            lambda **kw: _Resp()
        )
        assert gen._call_llm("p", max_tokens=10) == "hello"


class TestLLMGeneratorBatch:
    def test_batch_generate_processes_all(self, monkeypatch):
        gen = LLMGenerator(api_key="fake")
        monkeypatch.setattr(
            gen, "_call_llm", lambda prompt, max_tokens: "ans"
        )
        items = [
            {"question": "q1", "context_text": "c1"},
            {"question": "q2", "context_text": "c2"},
            {"question": "q3", "context_text": "c3"},
        ]
        results = gen.batch_generate(items)
        assert len(results) == 3
        assert all(r["answer"] == "ans" for r in results)

    def test_batch_generate_propagates_question_type(self, monkeypatch):
        gen = LLMGenerator(api_key="fake")
        monkeypatch.setattr(
            gen, "_call_llm", lambda prompt, max_tokens: "ans"
        )
        items = [
            {
                "question": "q",
                "context_text": "c",
                "question_type": "comparison",
            }
        ]
        results = gen.batch_generate(items)
        assert results[0]["question_type"] == "comparison"

    def test_batch_generate_ignores_image_path(self, monkeypatch):
        gen = LLMGenerator(api_key="fake")
        captured_prompts = []

        def fake_call(prompt, max_tokens):
            captured_prompts.append(prompt)
            return "ans"

        monkeypatch.setattr(gen, "_call_llm", fake_call)
        items = [
            {
                "question": "q",
                "context_text": "c",
                "image_path": "/fake/p.png",
            }
        ]
        gen.batch_generate(items)
        assert "/fake/p.png" not in captured_prompts[0]
