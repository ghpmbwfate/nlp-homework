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

# LLM API 配置（OPENAI_* 用于云端答案生成，DASHSCOPE_* 用于 query rewriter + VLM 图表提取）
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
DASHSCOPE_MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen-turbo")
DASHSCOPE_VL_MODEL = os.environ.get("DASHSCOPE_VL_MODEL", "qwen-vl-max")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "")

# 模型缓存（使用 ModelScope 下载和管理）
MODEL_CACHE_DIR = str(PROJECT_ROOT / ".cache" / "modelscope")
DENSE_MODEL_MS_ID = "Xorbits/bge-m3"  # ModelScope 模型 ID
RERANKER_MODEL_MS_ID = "Xorbits/bge-reranker-large"  # ModelScope 模型 ID


def _resolve_model_path(ms_model_id: str) -> str:
    """通过 ModelScope snapshot_download 解析本地模型路径，未下载则自动下载"""
    from modelscope import snapshot_download
    return snapshot_download(ms_model_id, cache_dir=MODEL_CACHE_DIR)


# 模型配置
DENSE_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-large"
LLM_MODEL = OPENAI_MODEL  # 答案生成 LLM（OpenAI 兼容 API 模型名）

# 检索参数
DENSE_TOP_K = 10
BM25_TOP_K = 10
FINAL_TOP_K = 3

# 生成参数
MAX_NEW_TOKENS = 2048
