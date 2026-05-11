"""
VLM答案生成模块：加载Qwen2-VL，结合检索文本+页面图片生成答案
支持分问题类型Prompt和引用溯源
"""

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from .question_classifier import classify_question, QuestionType
from .prompts import load_prompt_template
from .citation import add_citation_instruction, extract_citations


PROMPT_TEMPLATE = """你是一个专业的财报分析助手。请根据以下参考信息准确回答问题。

问题：{question}

参考文本：
{context}

要求：
1. 答案要准确、简洁，直接回答问题
2. 如果信息来自图表或表格，请结合图表内容回答
3. 不要添加参考信息中没有的内容
4. 用中文回答

答案："""


PROMPT_TEMPLATE_NO_IMAGE = """你是一个专业的财报分析助手。请根据以下参考文本准确回答问题。

问题：{question}

参考文本：
{context}

要求：
1. 答案要准确、简洁，直接回答问题
2. 不要添加参考文本中没有的内容
3. 用中文回答

答案："""


class VLMGenerator:
    """VLM答案生成器"""

    def __init__(self,
                 model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
                 device: str = "cuda",
                 load_in_4bit: bool = True):
        print(f"[INFO] 加载VLM模型: {model_name}")
        self.device = device

        # 4bit量化加载，节省显存
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )

        self.processor = AutoProcessor.from_pretrained(model_name)
        print("[INFO] VLM模型加载完成")

    def generate(self, question: str, context_text: str,
                 image_path: str | None = None, max_new_tokens: int = 512,
                 question_type: str | None = None) -> dict:
        """
        生成答案（支持分问题类型Prompt + 引用溯源）

        Args:
            question: 用户问题
            context_text: 检索到的上下文文本
            image_path: 页面图片路径（可选）
            max_new_tokens: 最大生成token数
            question_type: 问题类型，None时自动分类

        Returns:
            {"answer": str, "question_type": str, "citations": list}
        """
        # 分类问题类型
        if question_type is None:
            qtype = classify_question(question)
        else:
            qtype = QuestionType(question_type)
        qtype_str = qtype.value

        # 加载对应类型的 Prompt 模板
        prompt_template = load_prompt_template(qtype_str)

        # 追加引用溯源指令
        prompt_with_citation = add_citation_instruction(prompt_template)

        # 生成答案
        if image_path:
            answer = self._generate_with_prompt(
                question, context_text, image_path,
                prompt_with_citation, max_new_tokens
            )
        else:
            answer = self._generate_text_with_prompt(
                question, context_text,
                prompt_with_citation, max_new_tokens
            )

        # 提取引用溯源
        citations = extract_citations(answer)

        return {
            "answer": answer,
            "question_type": qtype_str,
            "citations": citations,
        }

    def _generate_with_prompt(self, question: str, context_text: str,
                               image_path: str, prompt_template: str,
                               max_new_tokens: int) -> str:
        """带图片 + 自定义Prompt的答案生成"""
        prompt = prompt_template.format(question=question, context=context_text)

        from pathlib import Path
        image_uri = Path(image_path).as_uri()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_uri},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        return self._run_inference(messages, max_new_tokens)

    def _generate_text_with_prompt(self, question: str, context_text: str,
                                    prompt_template: str,
                                    max_new_tokens: int) -> str:
        """纯文本 + 自定义Prompt的答案生成"""
        prompt = prompt_template.format(question=question, context=context_text)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        return self._run_inference(messages, max_new_tokens)

    def _run_inference(self, messages: list, max_new_tokens: int) -> str:
        """执行模型推理"""
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False  # 贪心解码，保证事实性问答的确定性输出
            )

        # 只取新生成的部分
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return output_text[0].strip() if output_text else ""

    def batch_generate(self, questions: list[dict], max_new_tokens: int = 512) -> list[dict]:
        """
        批量生成答案

        Args:
            questions: [{"question": str, "context_text": str, "image_path": str}]

        Returns:
            [{"answer": str, "question_type": str, "citations": list}]
        """
        results = []
        for i, q in enumerate(questions):
            print(f"[INFO] 生成答案 {i+1}/{len(questions)}: {q['question'][:30]}...")
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

    parser = argparse.ArgumentParser(description="VLM生成测试工具")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--context", type=str, required=True)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--no_4bit", action="store_true")

    args = parser.parse_args()

    generator = VLMGenerator(
        model_name=args.model,
        load_in_4bit=not args.no_4bit
    )

    answer = generator.generate(
        question=args.question,
        context_text=args.context,
        image_path=args.image
    )

    print(f"\n{'='*50}")
    print(f"问题: {args.question}")
    print(f"答案: {answer}")
    print(f"{'='*50}")
