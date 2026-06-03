"""
主流程：读取ground_truth → 检索 → LLM生成 → 输出submit.json

默认从 test_ground_truth.json 提取问题（仅 question 字段用于检索/生成），
输出格式与 ground_truth 一致：{filename, page, question, answer}

支持 --retriever 切换检索后端:
    python main.py --retriever vrag   # VRAG (dense + BM25 + rerank)
    python main.py --retriever tfidf  # TF-IDF baseline
"""

import json
import logging
import argparse
from pathlib import Path

from src.generation import LLMGenerator
from src.generation.question_classifier import classify_question
from src.generation.self_rag import run_self_check
from src.config import (
    DEFAULT_TEST_PATH,
    DEFAULT_GROUND_TRUTH_PATH,
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
    PROMPT_VERSION,
)

logger = logging.getLogger(__name__)


def _build_retriever(retriever_type: str, **kwargs):
    """Factory: instantiate the selected retriever backend."""
    if retriever_type == "vrag":
        from src.retrieval import Retriever
        return Retriever(
            chroma_dir=kwargs.get("chroma_dir"),
            bm25_dir=kwargs.get("bm25_dir"),
            image_dir=kwargs.get("image_dir"),
            dense_model=kwargs.get("dense_model", DENSE_MODEL),
            reranker_model=kwargs.get("reranker_model", RERANKER_MODEL),
            dense_top_k=kwargs.get("dense_top_k", DENSE_TOP_K),
            bm25_top_k=kwargs.get("bm25_top_k", BM25_TOP_K),
            final_top_k=kwargs.get("final_top_k", FINAL_TOP_K),
            reranker_alias=kwargs.get("reranker_alias"),
        )
    elif retriever_type == "hybrid_ce":
        # Hybrid + CE only (no multi-recall, no multi-stage rerank) — Baseline D
        from src.retrieval import Retriever
        return Retriever(
            chroma_dir=kwargs.get("chroma_dir"),
            bm25_dir=kwargs.get("bm25_dir"),
            image_dir=kwargs.get("image_dir"),
            dense_model=kwargs.get("dense_model", DENSE_MODEL),
            dense_top_k=kwargs.get("dense_top_k", DENSE_TOP_K),
            bm25_top_k=kwargs.get("bm25_top_k", BM25_TOP_K),
            final_top_k=kwargs.get("final_top_k", FINAL_TOP_K),
            enable_multi_recall=False,
            enable_multi_stage_rerank=False,
            reranker_alias=kwargs.get("reranker_alias"),
        )
    elif retriever_type == "tfidf":
        from src.baseline.tfidf import TFIDFRetriever
        return TFIDFRetriever(
            page_content_path=kwargs.get("page_content", "page_content.json"),
            image_dir=kwargs.get("image_dir"),
        )
    elif retriever_type == "bm25":
        from src.baseline.bm25 import BM25Retriever
        return BM25Retriever(
            bm25_dir=kwargs.get("bm25_dir"),
            image_dir=kwargs.get("image_dir"),
        )
    elif retriever_type == "dense":
        from src.baseline.dense import DenseRetriever
        return DenseRetriever(
            chroma_dir=kwargs.get("chroma_dir"),
            dense_model=kwargs.get("dense_model", DENSE_MODEL),
            image_dir=kwargs.get("image_dir"),
        )
    elif retriever_type == "hybrid_rrf":
        # Baseline A: dense + BM25 RRF, no rerank
        from src.baseline.hybrid_rrf import HybridRRFRetriever
        return HybridRRFRetriever(
            chroma_dir=kwargs.get("chroma_dir"),
            bm25_dir=kwargs.get("bm25_dir"),
            image_dir=kwargs.get("image_dir"),
            dense_model=kwargs.get("dense_model", DENSE_MODEL),
            dense_top_k=kwargs.get("dense_top_k", DENSE_TOP_K),
            bm25_top_k=kwargs.get("bm25_top_k", BM25_TOP_K),
            final_top_k=kwargs.get("final_top_k", FINAL_TOP_K),
        )
    elif retriever_type == "dense_rerank":
        # Baseline B: dense + CE rerank
        from src.baseline.dense_rerank import DenseRerankRetriever
        return DenseRerankRetriever(
            chroma_dir=kwargs.get("chroma_dir"),
            image_dir=kwargs.get("image_dir"),
            dense_model=kwargs.get("dense_model", DENSE_MODEL),
            reranker_alias=kwargs.get("reranker_alias"),
            dense_top_k=kwargs.get("dense_top_k", DENSE_TOP_K),
            final_top_k=kwargs.get("final_top_k", FINAL_TOP_K),
        )
    elif retriever_type == "bm25_rerank":
        # Baseline C: BM25 + CE rerank
        from src.baseline.bm25_rerank import BM25RerankRetriever
        return BM25RerankRetriever(
            bm25_dir=kwargs.get("bm25_dir"),
            image_dir=kwargs.get("image_dir"),
            reranker_alias=kwargs.get("reranker_alias"),
            bm25_top_k=kwargs.get("bm25_top_k", BM25_TOP_K),
            final_top_k=kwargs.get("final_top_k", FINAL_TOP_K),
        )
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")


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
                 retriever_type: str = "vrag",
                 page_content: str = "page_content.json",
                 chroma_dir: str = None,
                 bm25_dir: str = None,
                 image_dir: str = None,
                 dense_model: str = None,
                 reranker_model: str = None,
                 reranker_alias: str = None,
                 llm_model: str = None,
                 dense_top_k: int = None,
                 bm25_top_k: int = None,
                 final_top_k: int = None,
                 max_new_tokens: int = None,
                 prompt_version: str = None):
    """运行完整pipeline"""

    # 默认从 ground_truth 提取问题
    test_path = test_path or str(DEFAULT_GROUND_TRUTH_PATH)
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
    prompt_version = prompt_version or PROMPT_VERSION

    # 1. 加载测试数据
    logger.info("=" * 50)
    logger.info("Step 1: 加载测试数据")
    logger.info("=" * 50)
    questions = load_test_data(test_path)
    logger.info(f"加载了 {len(questions)} 个问题")

    # 2. 初始化检索器
    logger.info("=" * 50)
    logger.info(f"Step 2: 初始化检索器 ({retriever_type})")
    logger.info("=" * 50)
    retriever = _build_retriever(
        retriever_type,
        chroma_dir=chroma_dir,
        bm25_dir=bm25_dir,
        image_dir=image_dir,
        page_content=page_content,
        dense_model=dense_model,
        reranker_model=reranker_model,
        reranker_alias=reranker_alias,
        dense_top_k=dense_top_k,
        bm25_top_k=bm25_top_k,
        final_top_k=final_top_k,
    )

    # 3. 初始化LLM生成器
    logger.info("=" * 50)
    logger.info("Step 3: 初始化LLM生成器")
    logger.info("=" * 50)
    generator = LLMGenerator(model=llm_model, prompt_version=prompt_version)

    # 4. 逐题检索+生成
    logger.info("=" * 50)
    logger.info("Step 4: 逐题处理")
    logger.info("=" * 50)

    results = []
    for i, item in enumerate(questions):
        question = item.get("question", item.get("query", ""))

        logger.info(f"[{i+1}/{len(questions)}] 问题: {question[:50]}...")

        try:
            # 预分类问题类型（供检索和生成共用）
            qtype_enum = classify_question(question)
            qtype_str = qtype_enum.value
            logger.info(f"类型: {qtype_str}")

            # 检索（传入问题类型以启用多阶段重排序的类型过滤）
            context = retriever.search_with_context(question, question_type=qtype_str)
            top_filename = context["top_filename"]
            top_page = context["top_page"]
            context_text = context["context_text"]
            image_path = context["image_path"]

            logger.info(f"定位: {top_filename} 第{top_page}页")

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

            if not answer.strip():
                logger.warning(f"[{i+1}] 答案为空: question={question[:50]}")

            # Self-RAG: 自洽性检查
            self_check = run_self_check(answer, context_text)
            verdict = self_check["verdict"]
            logger.info(f"答案: {answer[:80]}...")
            logger.info(f"自洽性: {verdict} (数字支持率={self_check['num_support_ratio']:.2f})")

            results.append({
                "filename": top_filename,
                "page": top_page,
                "question": question,
                "answer": answer,
            })
        except Exception as e:
            logger.error(f"处理失败: {e}")
            results.append({
                "filename": "",
                "page": -1,
                "question": question,
                "answer": f"处理失败: {e}",
            })

    # 5. 保存结果
    logger.info("=" * 50)
    logger.info("Step 5: 保存结果")
    logger.info("=" * 50)

    # 确保输出目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"结果已保存至 {output_path}")
    logger.info(f"共处理 {len(results)} 个问题")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="财报RAG问答系统 - 主流程")
    parser.add_argument("--test", type=str, default=None, help="测试集路径（默认: test_ground_truth.json）")
    parser.add_argument("--output", type=str, default=None, help="输出路径")
    parser.add_argument("--retriever", type=str, default="vrag",
                        choices=["vrag", "tfidf", "bm25", "dense",
                                 "hybrid_rrf", "dense_rerank", "bm25_rerank", "hybrid_ce"],
                        help="检索后端 (default: vrag)")
    parser.add_argument("--page_content", type=str, default="page_content.json",
                        help="page_content.json 路径 (TF-IDF baseline 使用)")
    parser.add_argument("--chroma_dir", type=str, default=None)
    parser.add_argument("--bm25_dir", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--dense_model", type=str, default=None)
    parser.add_argument("--reranker_model", type=str, default=None)
    parser.add_argument("--reranker_alias", type=str, default=None,
                        help="Reranker 别名: bge-large | bge-v2-m3 | qwen3-0.6b")
    parser.add_argument("--llm_model", type=str, default=None)
    parser.add_argument("--dense_top_k", type=int, default=None)
    parser.add_argument("--bm25_top_k", type=int, default=None)
    parser.add_argument("--final_top_k", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--prompt_version", type=str, default=None,
                        choices=["v1", "v2"],
                        help="Prompt 版本 (default: 读取 config.PROMPT_VERSION = v1)")

    args = parser.parse_args()

    run_pipeline(
        test_path=args.test,
        output_path=args.output,
        retriever_type=args.retriever,
        page_content=args.page_content,
        chroma_dir=args.chroma_dir,
        bm25_dir=args.bm25_dir,
        image_dir=args.image_dir,
        dense_model=args.dense_model,
        reranker_model=args.reranker_model,
        reranker_alias=args.reranker_alias,
        llm_model=args.llm_model,
        dense_top_k=args.dense_top_k,
        bm25_top_k=args.bm25_top_k,
        final_top_k=args.final_top_k,
        max_new_tokens=args.max_new_tokens,
        prompt_version=args.prompt_version,
    )
