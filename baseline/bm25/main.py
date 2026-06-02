"""
BM25 Baseline — 纯 BM25 关键词检索 + LLM 生成

对比项（被去除的高级组件）：
- ✗ 稠密检索 (dense/chroma)
- ✗ Cross-Encoder 重排序
- ✗ 多路融合 (RRF)
- ✗ 标题/关键词辅助召回
- ✗ Self-RAG 自洽性检查
- ✓ 仅保留: BM25 检索 → LLM 生成
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.retrieval.retriever import load_bm25_index, bm25_search
from src.generation.generator import LLMGenerator
from src.config import (
    DEFAULT_TEST_PATH,
    INDEX_BM25_DIR,
    FINAL_TOP_K,
    MAX_NEW_TOKENS,
    LLM_MODEL,
)

# 基线专用输出路径
DEFAULT_BASELINE_OUTPUT = Path("outputs") / "baseline_bm25.json"


def load_test_data(test_path: str = None) -> list[dict]:
    test_path = test_path or str(DEFAULT_TEST_PATH)
    test_path_obj = Path(test_path)

    if not test_path_obj.exists():
        raise FileNotFoundError(f"测试文件不存在: {test_path}")

    with open(test_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        if "questions" in data:
            return data["questions"]
        else:
            return [data]
    else:
        raise ValueError(f"测试数据格式错误: 期望 list 或 dict，得到 {type(data)}")


def main():
    print("=" * 50)
    print("  BM25 Baseline — 纯关键词检索 + LLM")
    print("=" * 50)

    # 1. 加载测试问题
    print("\n[1/3] 加载测试数据...")
    questions = load_test_data()
    print(f"  -> 加载了 {len(questions)} 个问题")

    # 2. 加载 BM25 索引
    print("\n[2/3] 初始化 BM25 检索器...")
    bm25, chunks = load_bm25_index(str(INDEX_BM25_DIR))
    print(f"  -> BM25 索引加载完成, 文档数: {len(chunks)}")

    # 3. 初始化 LLM 生成器
    print("\n[3/3] 初始化 LLM 生成器...")
    generator = LLMGenerator(model=LLM_MODEL)

    # 4. 逐题 BM25 检索 → LLM 生成
    print(f"\n{'=' * 50}")
    print(f"  开始处理 {len(questions)} 个问题...")
    print(f"{'=' * 50}\n")

    results = []
    total_start = time.time()

    for i, q in enumerate(questions):
        q_start = time.time()
        qid = q.get("id", i + 1)
        question_text = q["question"]

        print(f"[{i + 1}/{len(questions)}] Q{qid}: {question_text[:60]}...")

        # BM25 检索
        hits = bm25_search(bm25, chunks, question_text, top_k=FINAL_TOP_K)
        if hits:
            top1 = hits[0]
            context_text = "\n\n---\n\n".join([
                f"[来源: {h['filename']} 第{h['page']}页]\n{h['content']}"
                for h in hits
            ])
            print(f"  -> BM25 Top-1: {top1['filename']} 第{top1['page']}页 (score={top1['score']:.4f})")
        else:
            context_text = "暂无相关文档"
            print("  -> 未检索到相关文档")

        # LLM 生成答案
        gen_result = generator.generate(
            question=question_text,
            context_text=context_text,
            max_new_tokens=MAX_NEW_TOKENS,
        )

        answer = gen_result["answer"]
        top1 = hits[0] if hits else {}
        result = {
            "filename": top1.get("filename", ""),
            "page": top1.get("page", 0),
            "question": question_text,
            "answer": answer,
        }

        results.append(result)

        q_elapsed = time.time() - q_start
        print(f"  -> 耗时: {q_elapsed:.1f}s, 答案长度: {len(answer)}字")
        print(f"  -> 答案预览: {answer[:80]}...")
        print()

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 50}")
    print(f"  全部完成!")
    print(f"  总耗时: {total_elapsed:.1f}s")
    print(f"  平均: {total_elapsed / len(questions):.1f}s/题")
    print(f"{'=' * 50}\n")

    # 5. 保存结果
    output_path = DEFAULT_BASELINE_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[OK] 结果已保存至: {output_path}")


if __name__ == "__main__":
    main()