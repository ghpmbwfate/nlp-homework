"""
云端 LLM 答案生成模块：通过 OpenAI 兼容 API 调用 GPT-OSS-20B-BF16
文本-only 生成；支持分问题类型 Prompt 与引用溯源
"""

import logging
import os
from typing import Optional

from openai import OpenAI

from src.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL

from .citation import add_citation_instruction, extract_citations
from .prompts import load_prompt_template
from .question_classifier import QuestionType, classify_question

logger = logging.getLogger(__name__)


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
        base_url = base_url or OPENAI_BASE_URL or os.environ.get("OPENAI_BASE_URL")
        if not base_url:
            raise ValueError(
                "OpenAI base URL is required. "
                "Set OPENAI_BASE_URL in .env or pass base_url."
            )
        self.model = model or OPENAI_MODEL or os.environ.get("OPENAI_MODEL")
        if not self.model:
            raise ValueError(
                "OpenAI model name is required. "
                "Set OPENAI_MODEL in .env or pass model."
            )
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info(f"LLM 生成器就绪: model={self.model}, base_url={base_url}")

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
            choice = response.choices[0]
            content = choice.message.content
            finish_reason = getattr(choice, "finish_reason", None)

            # 兼容 reasoning 模型：尝试读取 reasoning_content 字段
            reasoning_content = getattr(choice.message, "reasoning_content", None)

            if content and content.strip():
                if finish_reason == "length":
                    logger.warning(
                        f"答案被截断 (finish_reason=length, max_tokens={max_tokens})"
                    )
                return content.strip()

            # content 为空时记录诊断信息
            r_len = len(reasoning_content) if reasoning_content else 0
            logger.warning(
                f"LLM 返回空 content: finish_reason={finish_reason}, "
                f"reasoning_content 长度={r_len}"
            )

            # Fallback: 用 reasoning_content（截取最后 1500 字作为答案）
            if reasoning_content and reasoning_content.strip():
                return reasoning_content.strip()[-1500:]

            return f"[模型返回空答案，finish_reason={finish_reason}]"
        except (ConnectionError, TimeoutError) as e:
            raise RuntimeError(f"LLM API network error: {e}") from e
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
            logger.info(
                f"生成答案 {i + 1}/{len(questions)}: "
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

    logger.info(f"{'=' * 50}")
    logger.info(f"问题: {args.question}")
    logger.info(f"类型: {result['question_type']}")
    logger.info(f"答案: {result['answer']}")
    logger.info(f"引用: {result['citations']}")
    logger.info(f"{'=' * 50}")
