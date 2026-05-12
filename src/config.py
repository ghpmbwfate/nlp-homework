"""
统一配置：管理所有默认路径和模型参数
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# 数据目录
DATA_DIR = PROJECT_ROOT / "data" / "reports"
QUESTIONS_DIR = PROJECT_ROOT / "questions"

# 输出目录
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PARSED_DIR = OUTPUTS_DIR / "parsed_data"
IMAGES_DIR = OUTPUTS_DIR / "page_images"
INDEX_CHROMA_DIR = OUTPUTS_DIR / "index_data" / "chroma"
INDEX_BM25_DIR = OUTPUTS_DIR / "index_data" / "bm25"

# 默认文件路径
DEFAULT_TEST_PATH = QUESTIONS_DIR / "test.json"
DEFAULT_OUTPUT_PATH = OUTPUTS_DIR / "submit.json"
DEFAULT_GROUND_TRUTH_PATH = QUESTIONS_DIR / "test_ground_truth.json"
DEFAULT_EVAL_OUTPUT_PATH = OUTPUTS_DIR / "evaluation_result.json"

# 模型配置
DENSE_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-large"
VLM_MODEL = "Qwen/Qwen2-VL-7B-Instruct"

# 检索参数
DENSE_TOP_K = 10
BM25_TOP_K = 10
FINAL_TOP_K = 3

# 生成参数
MAX_NEW_TOKENS = 512
LOAD_IN_4BIT = True

# LLM API 配置
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
DASHSCOPE_MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen-turbo")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
