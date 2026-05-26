"""
云端 LLM 答案生成模块：通过 OpenAI 兼容 API 调用 GPT-OSS-20B-BF16
文本-only 生成；支持分问题类型 Prompt 与引用溯源
"""

import os
from typing import Optional

from openai import OpenAI

from src.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL

from .citation import add_citation_instruction, extract_citations
from .prompts import load_prompt_template
from .question_classifier import QuestionType, classify_question


class LLMGenerator:
    """云端 LLM 答案生成器（OpenAI 兼容 API，文本-only）"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        api_key = api_key or OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Set OPENAI_API_KEY in .env or pass api_key."
            )
        base_url = base_url or OPENAI_BASE_URL or os.environ.get(
            "OPENAI_BASE_URL", "http://api.bnuzh.top:8080/v1"
        )
        self.model = model or OPENAI_MODEL or "GPT-OSS-20B-BF16"
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        print(f"[INFO] LLM 生成器就绪: model={self.model}, base_url={base_url}")

    def generate(
        self,
        question: str,
        context_text: str,
        image_path: Optional[str] = None,  # 接受但忽略：文本-only 模式
        max_new_tokens: int = 512,
        question_type: Optional[str] = None,
    ) -> dict:
        """
        生成答案（支持分问题类型 Prompt + 引用溯源）

        Args:
            question: 用户问题
            context_text: 检索到的上下文文本
            image_path: 兼容旧接口，当前实现忽略此参数
            max_new_tokens: 最大生成 token 数（映射到 OpenAI 的 max_tokens）
            question_type: 问题类型，None 时自动分类

        Returns:
            {"answer": str, "question_type": str, "citations": list}
        """
        del image_path  # 显式标注：未使用

        # 分类问题类型
        if question_type is None:
            qtype = classify_question(question)
        else:
            qtype = QuestionType(question_type)
        qtype_str = qtype.value

        # 加载模板 + 追加引用溯源指令
        prompt_template = load_prompt_template(qtype_str)
        prompt_with_citation = add_citation_instruction(prompt_template)
        prompt = prompt_with_citation.format(
            question=question, context=context_text
        )

        # 调用云端 LLM
        answer = self._call_llm(prompt, max_tokens=max_new_tokens)

        # 提取引用
        citations = extract_citations(answer)

        return {
            "answer": answer,
            "question_type": qtype_str,
            "citations": citations,
        }

    def _call_llm(self, prompt: str, max_tokens: int) -> str:
        """调用 OpenAI 兼容 chat completion；异常包装为 RuntimeError"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}") from e

    def batch_generate(
        self, questions: list[dict], max_new_tokens: int = 512
    ) -> list[dict]:
        """
        批量生成答案

        Args:
            questions: [{"question": str, "context_text": str,
                         "image_path": str (ignored), "question_type": str (opt)}]

        Returns:
            [{"answer": str, "question_type": str, "citations": list}]
        """
        results = []
        for i, q in enumerate(questions):
            print(
                f"[INFO] 生成答案 {i + 1}/{len(questions)}: "
                f"{q['question'][:30]}..."
            )
            result = self.generate(
                question=q["question"],
                context_text=q["context_text"],
                image_path=q.get("image_path"),
                max_new_tokens=max_new_tokens,
                question_type=q.get("question_type"),
            )
            results.append(result)
        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM 生成测试工具")
    parser.add_argument("--model", type=str, default=None,
                        help="模型名（默认读取 OPENAI_MODEL）")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--context", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    args = parser.parse_args()

    generator = LLMGenerator(model=args.model)
    result = generator.generate(
        question=args.question,
        context_text=args.context,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"\n{'=' * 50}")
    print(f"问题: {args.question}")
    print(f"类型: {result['question_type']}")
    print(f"答案: {result['answer']}")
    print(f"引用: {result['citations']}")
    print(f"{'=' * 50}")
