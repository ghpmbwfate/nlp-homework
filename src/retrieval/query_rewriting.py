"""Query rewriting and expansion for retrieval enhancement.

Supports:
- Keyword expansion via LLM
- HyDE (Hypothetical Document Embeddings)
- Query decomposition for multi-condition questions
"""

import os
from typing import List, Optional

from openai import OpenAI

from src.config import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    DASHSCOPE_MODEL,
)


class QueryRewriter:
    """Rewrite and expand queries to improve retrieval recall."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = None,
    ):
        api_key = api_key or DASHSCOPE_API_KEY or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "DashScope API key is required. "
                "Set DASHSCOPE_API_KEY in .env or pass api_key."
            )
        base_url = base_url or DASHSCOPE_BASE_URL or os.environ.get(
            "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model or DASHSCOPE_MODEL or "qwen-turbo"
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM with system and user prompts."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=512,
            )
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}") from e

    def rewrite(self, query: str, num_variants: int = 3) -> List[str]:
        """Expand query into keyword-rich retrieval variants.

        Returns the original query plus LLM-generated variants.
        """
        system_prompt = (
            "你是一个财务报告检索助手。"
            "请将用户问题改写为适合信息检索的查询语句。"
            "要求：保留核心关键词、添加同义词、去除口语化表达。"
            "每行输出一个改写版本，不要添加编号或解释。"
        )
        user_prompt = (
            f"请将以下问题改写为 {num_variants} 个检索查询（每行一个）：\n"
            f"问题：{query}"
        )

        raw = self._call_llm(system_prompt, user_prompt)
        variants = [line.strip() for line in raw.split("\n") if line.strip()]

        # Deduplicate while preserving order
        seen = set()
        result = []
        for v in [query] + variants:
            if v not in seen:
                seen.add(v)
                result.append(v)
        return result

    def decompose(self, query: str) -> List[str]:
        """Decompose a multi-condition query into sub-questions.

        If the query contains multiple conditions (e.g., A and B),
        split it into separate sub-queries.
        """
        system_prompt = (
            "你是一个财务报告问题分解助手。"
            "如果用户问题包含多个条件，请将其拆分为独立的子问题。"
            "如果问题本身已经很聚焦，直接返回原问题。"
            "每行输出一个子问题，不要添加编号或解释。"
        )
        user_prompt = f"请将以下问题拆分为子问题（每行一个）：\n问题：{query}"

        raw = self._call_llm(system_prompt, user_prompt)
        parts = [line.strip() for line in raw.split("\n") if line.strip()]

        if not parts:
            return [query]
        return parts

    def hyde_generate(self, query: str) -> str:
        """Generate a hypothetical answer document for dense retrieval.

        Uses HyDE technique: generate a plausible answer, then use it
        as the dense retrieval query.
        """
        system_prompt = (
            "你是一个财务报告分析助手。"
            "请根据用户问题，生成一段假设性的答案文档。"
            "这段文档应该包含问题可能涉及的关键事实和数据。"
            "文档长度控制在 200 字以内，只输出文档内容。"
        )
        user_prompt = f"请生成关于以下问题的假设性答案文档：\n问题：{query}"

        return self._call_llm(system_prompt, user_prompt)


class NoOpQueryRewriter:
    """No-op rewriter for when API is unavailable. Returns query unchanged."""

    def rewrite(self, query: str, num_variants: int = 3) -> List[str]:
        return [query]

    def decompose(self, query: str) -> List[str]:
        return [query]

    def hyde_generate(self, query: str) -> str:
        return query
