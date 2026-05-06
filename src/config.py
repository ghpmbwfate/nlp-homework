"""
统一配置：管理所有默认路径和模型参数
"""

from pathlib import Path

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
