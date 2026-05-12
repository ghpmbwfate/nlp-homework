"""
主流程：读取test.json → 检索 → VLM生成 → 输出submit.json
"""

import json
import argparse
from pathlib import Path

from src.retrieval import Retriever, QueryRewriter
from src.retrieval.multi_recover import build_all_multi_indexes
from src.generation import VLMGenerator, PostProcessor, SelfRAG
from src.generation.citation import clean_answer_no_citations, has_citations
from src.config import (
    DEFAULT_TEST_PATH,
    DEFAULT_OUTPUT_PATH,
    INDEX_CHROMA_DIR,
    INDEX_BM25_DIR,
    IMAGES_DIR,
    DENSE_MODEL,
    RERANKER_MODEL,
    VLM_MODEL,
    DENSE_TOP_K,
    BM25_TOP_K,
    FINAL_TOP_K,
    MAX_NEW_TOKENS,
    LOAD_IN_4BIT,
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
                 vlm_model: str = None,
                 load_in_4bit: bool = None,
                 dense_top_k: int = None,
                 bm25_top_k: int = None,
                 final_top_k: int = None,
                 max_new_tokens: int = None,
                 enable_query_rewrite: bool = False,
                 enable_multi_recall: bool = False,
                 enable_postprocess: bool = False,
                 enable_self_rag: bool = False):
    """Run full pipeline with optional enhancements."""

    test_path = test_path or str(DEFAULT_TEST_PATH)
    output_path = output_path or str(DEFAULT_OUTPUT_PATH)
    chroma_dir = chroma_dir or str(INDEX_CHROMA_DIR)
    bm25_dir = bm25_dir or str(INDEX_BM25_DIR)
    image_dir = image_dir or str(IMAGES_DIR)
    dense_model = dense_model or DENSE_MODEL
    reranker_model = reranker_model or RERANKER_MODEL
    vlm_model = vlm_model or VLM_MODEL
    load_in_4bit = LOAD_IN_4BIT if load_in_4bit is None else load_in_4bit
    dense_top_k = dense_top_k if dense_top_k is not None else DENSE_TOP_K
    bm25_top_k = bm25_top_k if bm25_top_k is not None else BM25_TOP_K
    final_top_k = final_top_k if final_top_k is not None else FINAL_TOP_K
    max_new_tokens = max_new_tokens if max_new_tokens is not None else MAX_NEW_TOKENS

    # Step 1: Load test data
    print("=" * 50)
    print("Step 1: 加载测试数据")
    print("=" * 50)
    questions = load_test_data(test_path)
    print(f"[INFO] 加载了 {len(questions)} 个问题")

    # Step 2: Build multi-recall indexes if enabled
    multi_indexes = {}
    if enable_multi_recall:
        print()
        print("=" * 50)
        print("Step 2a: 构建多路召回索引")
        print("=" * 50)
        from src.indexing.indexer import load_page_content
        pages = load_page_content("page_content.json")
        multi_indexes = build_all_multi_indexes(pages)
        print("[INFO] 多路召回索引构建完成")

    # Step 3: Init query rewriter if enabled
    query_rewriter = None
    if enable_query_rewrite:
        print()
        print("=" * 50)
        print("Step 2b: 初始化Query改写器")
        print("=" * 50)
        try:
            query_rewriter = QueryRewriter()
            print("[INFO] Query改写器初始化完成")
        except ValueError as e:
            print(f"[WARN] Query改写器初始化失败: {e}")

    # Step 4: Init retriever
    print()
    print("=" * 50)
    print("Step 3: 初始化检索器")
    print("=" * 50)
    retriever = Retriever(
        chroma_dir=chroma_dir,
        bm25_dir=bm25_dir,
        image_dir=image_dir,
        dense_model=dense_model,
        reranker_model=reranker_model,
        dense_top_k=dense_top_k,
        bm25_top_k=bm25_top_k,
        final_top_k=final_top_k,
        multi_indexes=multi_indexes,
        enable_multi_recall=enable_multi_recall,
        query_rewriter=query_rewriter,
        enable_query_rewrite=enable_query_rewrite,
    )

    # Step 5: Init VLM generator
    print()
    print("=" * 50)
    print("Step 4: 初始化VLM生成器")
    print("=" * 50)
    generator = VLMGenerator(
        model_name=vlm_model,
        load_in_4bit=load_in_4bit
    )

    # Step 6: Init optional modules
    postprocessor = PostProcessor() if enable_postprocess else None
    self_rag = None
    if enable_self_rag:
        try:
            self_rag = SelfRAG()
            print("[INFO] Self-RAG初始化完成")
        except ValueError as e:
            print(f"[WARN] Self-RAG初始化失败: {e}")

    # Step 7: Process questions
    print()
    print("=" * 50)
    print("Step 5: 逐题处理")
    print("=" * 50)

    results = []
    for i, item in enumerate(questions):
        question = item.get("question", item.get("query", ""))

        print(f"\n[{i+1}/{len(questions)}] 问题: {question[:50]}...")

        # Retrieve
        context = retriever.search_with_context(question)
        top_filename = context["top_filename"]
        top_page = context["top_page"]
        context_text = context["context_text"]
        image_path = context["image_path"]

        print(f"  定位: {top_filename} 第{top_page}页")

        # Generate answer
        gen_result = generator.generate(
            question=question,
            context_text=context_text,
            image_path=image_path,
            max_new_tokens=max_new_tokens
        )
        answer = gen_result["answer"]
        qtype = gen_result["question_type"]
        citations = gen_result["citations"]

        # Post-process
        if postprocessor is not None:
            pp_result = postprocessor.process(answer, context_text, qtype)
            answer = pp_result["answer"]
            print(f"  后处理: 幻觉数字={len(pp_result['hallucinated_numbers'])}, "
                  f"截断={pp_result['was_truncated']}, 空答案={pp_result['was_empty']}")

        # Self-RAG
        if self_rag is not None:
            sr_result = self_rag.check_and_refine(
                answer, context_text, generator, question
            )
            answer = sr_result["answer"]
            print(f"  Self-RAG: 验证={sr_result['verification']}, "
                  f"尝试次数={sr_result['attempts']}")

        print(f"  类型: {qtype}")
        print(f"  答案: {answer[:80]}...")

        results.append({
            "question": question,
            "filename": top_filename,
            "page": top_page,
            "answer": answer,
            "question_type": qtype,
            "citations": citations,
            "has_citations": has_citations(answer),
        })

    # Step 8: Save results
    print()
    print("=" * 50)
    print("Step 6: 保存结果")
    print("=" * 50)

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
    parser.add_argument("--vlm_model", type=str, default=None)
    parser.add_argument("--no_4bit", action="store_true", help="不使用4bit量化")
    parser.add_argument("--dense_top_k", type=int, default=None)
    parser.add_argument("--bm25_top_k", type=int, default=None)
    parser.add_argument("--final_top_k", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--enable_query_rewrite", action="store_true",
                        help="启用Query改写")
    parser.add_argument("--enable_multi_recall", action="store_true",
                        help="启用多路召回")
    parser.add_argument("--enable_postprocess", action="store_true",
                        help="启用答案后处理")
    parser.add_argument("--enable_self_rag", action="store_true",
                        help="启用Self-RAG校验")

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
        max_new_tokens=args.max_new_tokens,
        enable_query_rewrite=args.enable_query_rewrite,
        enable_multi_recall=args.enable_multi_recall,
        enable_postprocess=args.enable_postprocess,
        enable_self_rag=args.enable_self_rag,
    )
