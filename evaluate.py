"""
评估入口：对比 submit.json 与 test_ground_truth.json，计算多项指标
"""

import json
import argparse

from src.evaluation import evaluate, print_report
from src.config import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_GROUND_TRUTH_PATH,
    DEFAULT_EVAL_OUTPUT_PATH,
)


def main():
    parser = argparse.ArgumentParser(description="RAG 系统评估工具")
    parser.add_argument("--pred", type=str, default=str(DEFAULT_OUTPUT_PATH), help="预测结果路径")
    parser.add_argument("--gold", type=str, default=str(DEFAULT_GROUND_TRUTH_PATH), help="标准答案路径")
    parser.add_argument("--output", type=str, default=str(DEFAULT_EVAL_OUTPUT_PATH), help="详细结果输出路径")

    args = parser.parse_args()

    # 加载数据
    with open(args.pred, "r", encoding="utf-8") as f:
        pred_data = json.load(f)
    with open(args.gold, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    print(f"[INFO] 加载预测结果: {len(pred_data)} 条")
    print(f"[INFO] 加载标准答案: {len(gold_data)} 条")

    metrics = evaluate(pred_data, gold_data)
    if metrics:
        print_report(metrics, args.output)


if __name__ == "__main__":
    main()
