"""
主流程：读取test.json → 检索 → VLM生成 → 输出submit.json
"""

import os
import json
import argparse
from pathlib import Path

from retriever import Retriever
from generator import VLMGenerator


def load_test_data(test_path: str = "test.json") -> list[dict]:
    """加载测试集"""
    with open(test_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 兼容不同格式：可能是列表或带key的字典
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "questions" in data:
        return data["questions"]
    elif isinstance(data, dict):
        return [data]
    return data


def run_pipeline(test_path: str = "test.json",
                 output_path: str = "submit.json",
                 chroma_dir: str = "index_data/chroma",
                 bm25_dir: str = "index_data/bm25",
                 image_dir: str = "page_images",
                 dense_model: str = "BAAI/bge-m3",
                 reranker_model: str = "BAAI/bge-reranker-large",
                 vlm_model: str = "Qwen/Qwen2-VL-7B-Instruct",
                 load_in_4bit: bool = True,
                 dense_top_k: int = 10,
                 bm25_top_k: int = 10,
                 final_top_k: int = 3,
                 max_new_tokens: int = 512):
    """运行完整pipeline"""

    # 1. 加载测试数据
    print("=" * 50)
    print("Step 1: 加载测试数据")
    print("=" * 50)
    questions = load_test_data(test_path)
    print(f"[INFO] 加载了 {len(questions)} 个问题")

    # 2. 初始化检索器
    print()
    print("=" * 50)
    print("Step 2: 初始化检索器")
    print("=" * 50)
    retriever = Retriever(
        chroma_dir=chroma_dir,
        bm25_dir=bm25_dir,
        image_dir=image_dir,
        dense_model=dense_model,
        reranker_model=reranker_model,
        dense_top_k=dense_top_k,
        bm25_top_k=bm25_top_k,
        final_top_k=final_top_k
    )

    # 3. 初始化VLM生成器
    print()
    print("=" * 50)
    print("Step 3: 初始化VLM生成器")
    print("=" * 50)
    generator = VLMGenerator(
        model_name=vlm_model,
        load_in_4bit=load_in_4bit
    )

    # 4. 逐题检索+生成
    print()
    print("=" * 50)
    print("Step 4: 逐题处理")
    print("=" * 50)

    results = []
    for i, item in enumerate(questions):
        question = item.get("question", item.get("query", ""))

        print(f"\n[{i+1}/{len(questions)}] 问题: {question[:50]}...")

        # 检索
        context = retriever.search_with_context(question)
        top_filename = context["top_filename"]
        top_page = context["top_page"]
        context_text = context["context_text"]
        image_path = context["image_path"]

        print(f"  定位: {top_filename} 第{top_page}页")

        # 生成答案
        answer = generator.generate(
            question=question,
            context_text=context_text,
            image_path=image_path,
            max_new_tokens=max_new_tokens
        )

        print(f"  答案: {answer[:80]}...")

        results.append({
            "question": question,
            "filename": top_filename,
            "page": top_page,
            "answer": answer
        })

    # 5. 保存结果
    print()
    print("=" * 50)
    print("Step 5: 保存结果")
    print("=" * 50)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 结果已保存至 {output_path}")
    print(f"[INFO] 共处理 {len(results)} 个问题")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="财报RAG问答系统 - 主流程")
    parser.add_argument("--test", type=str, default="test.json", help="测试集路径")
    parser.add_argument("--output", type=str, default="submit.json", help="输出路径")
    parser.add_argument("--chroma_dir", type=str, default="index_data/chroma")
    parser.add_argument("--bm25_dir", type=str, default="index_data/bm25")
    parser.add_argument("--image_dir", type=str, default="page_images")
    parser.add_argument("--dense_model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--reranker_model", type=str, default="BAAI/bge-reranker-large")
    parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--no_4bit", action="store_true", help="不使用4bit量化")
    parser.add_argument("--dense_top_k", type=int, default=10)
    parser.add_argument("--bm25_top_k", type=int, default=10)
    parser.add_argument("--final_top_k", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    args = parser.parse_args()

    run_pipeline(
        test_path=args.test,
        output_path=args.output,
        chroma_dir=args.chroma_dir,
        bm25_dir=args.bm25_dir,
        image_dir=args.image_dir,
        dense_model=args.dense_model,
        reranker_model=args.reranker_model,
        vlm_model=args.vlm_model,
        load_in_4bit=not args.no_4bit,
        dense_top_k=args.dense_top_k,
        bm25_top_k=args.bm25_top_k,
        final_top_k=args.final_top_k,
        max_new_tokens=args.max_new_tokens
    )
