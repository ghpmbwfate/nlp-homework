"""Self-RAG: factual consistency checking and answer voting.

Uses an LLM to verify whether the generated answer is supported by
the retrieved context, and supports multi-generation voting.
"""

import os
from typing import Literal, Optional

from openai import OpenAI

from src.config import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    DASHSCOPE_MODEL,
)

VerificationResult = Literal["support", "contradict", "neutral"]


class SelfRAG:
    """Self-RAG verifier using LLM-based factuality checking."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = None,
    ):
        api_key = api_key or DASHSCOPE_API_KEY or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "DashScope API key required. "
                "Set DASHSCOPE_API_KEY in .env or pass api_key."
            )
        base_url = base_url or DASHSCOPE_BASE_URL or os.environ.get(
            "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model or DASHSCOPE_MODEL or "qwen-turbo"
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM and return content."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=128,
            )
            content = response.choices[0].message.content
            return content.strip().lower() if content else ""
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}") from e

    def verify(
        self, answer: str, context_text: str
    ) -> VerificationResult:
        """Check if answer is supported by context.

        Returns:
            "support": answer is fully supported
            "contradict": answer contradicts context
            "neutral": cannot determine
        """
        system_prompt = (
            "你是一个事实核查助手。请判断给定答案是否被参考文本支持。"
            "只输出以下三种结果之一，不要解释："
            "- support（答案被参考文本支持）"
            "- contradict（答案与参考文本矛盾）"
            "- neutral（无法判断）"
        )
        user_prompt = (
            f"参考文本：\n{context_text[:2000]}\n\n"
            f"答案：{answer}\n\n"
            f"请判断："
        )

        raw = self._call_llm(system_prompt, user_prompt)
        if "contradict" in raw:
            return "contradict"
        if "support" in raw:
            return "support"
        return "neutral"

    def vote(self, answers: list[str]) -> str:
        """Vote among multiple answers and return the most consistent one.

        Uses a simple heuristic: prefer longer answers with more overlap
        to other answers. Falls back to the first answer if no clear winner.
        """
        if not answers:
            return ""
        if len(answers) == 1:
            return answers[0]

        # Compute pairwise similarity (character overlap)
        def _sim(a: str, b: str) -> float:
            set_a = set(a)
            set_b = set(b)
            if not set_a or not set_b:
                return 0.0
            inter = len(set_a & set_b)
            return inter / max(len(set_a), len(set_b))

        scores = []
        for i, a in enumerate(answers):
            total = sum(_sim(a, answers[j]) for j in range(len(answers)) if j != i)
            scores.append(total)

        best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        return answers[best_idx]

    def check_and_refine(
        self,
        answer: str,
        context_text: str,
        generator,
        question: str,
        max_attempts: int = 2,
    ) -> dict:
        """Verify answer and regenerate if contradicted.

        Args:
            answer: generated answer
            context_text: retrieved context
            generator: VLMGenerator instance for re-generation
            question: original question
            max_attempts: max regeneration attempts

        Returns:
            dict with answer, verification, attempts
        """
        result = {
            "answer": answer,
            "verification": "support",
            "attempts": 1,
        }

        for attempt in range(max_attempts):
            verdict = self.verify(result["answer"], context_text)
            result["verification"] = verdict
            if verdict != "contradict":
                break
            if attempt < max_attempts - 1:
                print(
                    f"[INFO] Self-RAG: answer contradicted, "
                    f"regenerating (attempt {attempt + 2})"
                )
                gen_result = generator.generate(
                    question=question,
                    context_text=context_text,
                    max_new_tokens=512,
                )
                result["answer"] = gen_result["answer"]
                result["attempts"] = attempt + 2

        return result
