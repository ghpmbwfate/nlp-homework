"""
主流程：读取test.json → 检索 → LLM生成 → 输出submit.json
"""

import json
import argparse
from pathlib import Path

from src.retrieval import Retriever
from src.generation import LLMGenerator
from src.generation.citation import clean_answer_no_citations, has_citations
from src.generation.question_classifier import classify_question
from src.generation.self_rag import run_self_check
from src.config import (
    DEFAULT_TEST_PATH,
    DEFAULT_OUTPUT_PATH,
    INDEX_CHROMA_DIR,
    INDEX_BM25_DIR,
    IMAGES_DIR,
    DENSE_MODEL,
    RERANKER_MODEL,
    LLM_MODEL,
    DENSE_TOP_K,
    BM25_TOP_K,
    FINAL_TOP_K,
    MAX_NEW_TOKENS,
)


def load_test_data(test_path: str) -> list[dict]:
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


def run_pipeline(test_path: str = None,
                 output_path: str = None,
                 chroma_dir: str = None,
                 bm25_dir: str = None,
                 image_dir: str = None,
                 dense_model: str = None,
                 reranker_model: str = None,
                 llm_model: str = None,
                 dense_top_k: int = None,
                 bm25_top_k: int = None,
                 final_top_k: int = None,
                 max_new_tokens: int = None):
    """运行完整pipeline"""

    # 使用默认值
    test_path = test_path or str(DEFAULT_TEST_PATH)
    output_path = output_path or str(DEFAULT_OUTPUT_PATH)
    chroma_dir = chroma_dir or str(INDEX_CHROMA_DIR)
    bm25_dir = bm25_dir or str(INDEX_BM25_DIR)
    image_dir = image_dir or str(IMAGES_DIR)
    dense_model = dense_model or DENSE_MODEL
    reranker_model = reranker_model or RERANKER_MODEL
    llm_model = llm_model or LLM_MODEL
    dense_top_k = dense_top_k if dense_top_k is not None else DENSE_TOP_K
    bm25_top_k = bm25_top_k if bm25_top_k is not None else BM25_TOP_K
    final_top_k = final_top_k if final_top_k is not None else FINAL_TOP_K
    max_new_tokens = max_new_tokens if max_new_tokens is not None else MAX_NEW_TOKENS

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

    # 3. 初始化LLM生成器
    print()
    print("=" * 50)
    print("Step 3: 初始化LLM生成器")
    print("=" * 50)
    generator = LLMGenerator(model=llm_model)

    # 4. 逐题检索+生成
    print()
    print("=" * 50)
    print("Step 4: 逐题处理")
    print("=" * 50)

    results = []
    for i, item in enumerate(questions):
        question = item.get("question", item.get("query", ""))

        print(f"\n[{i+1}/{len(questions)}] 问题: {question[:50]}...")

        # 预分类问题类型（供检索和生成共用）
        qtype_enum = classify_question(question)
        qtype_str = qtype_enum.value
        print(f"  类型: {qtype_str}")

        # 检索（传入问题类型以启用多阶段重排序的类型过滤）
        context = retriever.search_with_context(question, question_type=qtype_str)
        top_filename = context["top_filename"]
        top_page = context["top_page"]
        context_text = context["context_text"]
        image_path = context["image_path"]

        print(f"  定位: {top_filename} 第{top_page}页")

        # 生成答案（image_path 在文本-only LLM 模式下被忽略，保留兼容）
        gen_result = generator.generate(
            question=question,
            context_text=context_text,
            image_path=image_path,
            max_new_tokens=max_new_tokens
        )
        answer = gen_result["answer"]
        qtype = gen_result["question_type"]
        citations = gen_result["citations"]

        # Self-RAG: 自洽性检查
        self_check = run_self_check(answer, context_text)
        verdict = self_check["verdict"]
        print(f"  答案: {answer[:80]}...")
        print(f"  自洽性: {verdict} (数字支持率={self_check['num_support_ratio']:.2f})")

        results.append({
            "question": question,
            "filename": top_filename,
            "page": top_page,
            "answer": answer,
            "question_type": qtype,
            "citations": citations,
            "has_citations": has_citations(answer),
            "self_check": {
                "verdict": verdict,
                "num_support_ratio": self_check["num_support_ratio"],
                "keyword_overlap_ratio": self_check["keyword_overlap_ratio"],
                "feedback": self_check["feedback"],
            },
        })

    # 5. 保存结果
    print()
    print("=" * 50)
    print("Step 5: 保存结果")
    print("=" * 50)

    # 确保输出目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 结果已保存至 {output_path}")
    print(f"[INFO] 共处理 {len(results)} 个问题")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="财报RAG问答系统 - 主流程")
    parser.add_argument("--test", type=str, default=None, help="测试集路径")
    parser.add_argument("--output", type=str, default=None, help="输出路径")
    parser.add_argument("--chroma_dir", type=str, default=None)
    parser.add_argument("--bm25_dir", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--dense_model", type=str, default=None)
    parser.add_argument("--reranker_model", type=str, default=None)
    parser.add_argument("--llm_model", type=str, default=None)
    parser.add_argument("--dense_top_k", type=int, default=None)
    parser.add_argument("--bm25_top_k", type=int, default=None)
    parser.add_argument("--final_top_k", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)

    args = parser.parse_args()

    run_pipeline(
        test_path=args.test,
        output_path=args.output,
        chroma_dir=args.chroma_dir,
        bm25_dir=args.bm25_dir,
        image_dir=args.image_dir,
        dense_model=args.dense_model,
        reranker_model=args.reranker_model,
        llm_model=args.llm_model,
        dense_top_k=args.dense_top_k,
        bm25_top_k=args.bm25_top_k,
        final_top_k=args.final_top_k,
        max_new_tokens=args.max_new_tokens
    )
