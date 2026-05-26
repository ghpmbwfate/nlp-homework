# 环境搭建与测试记录

## 时间
2026-05-06

## 目标
在 conda 环境 `nlp` 中测试 RAG 检索流程的基础通路。

---

## 一、缺失包安装

以下包在 `nlp` 环境中缺失，已通过 `pip install` 补装：

| 包名 | 版本 | 用途 |
|------|------|------|
| `jieba` | 0.42.1 | 中文分词（BM25检索） |
| `chromadb` | 1.5.9 | 稠密向量索引与检索 |
| `rank_bm25` | 0.2.2 | BM25稀疏检索 |
| `qwen-vl-utils` | 0.0.14 | Qwen2-VL视觉语言模型工具 |
| `bitsandbytes` | 0.41.3 | 4bit量化（兼容torch 2.1） |
| `pdf2image` | 1.17.0 | PDF转PNG页面图片 |
| `av` | 17.0.1 | qwen-vl-utils视频依赖 |

chromadb 的传递依赖（pydantic-settings, pybase64, onnxruntime, opentelemetry-*, grpcio, kubernetes, mmh3, orjson, httpx, build 等）一并安装。

---

## 二、PDF解析方案变更

### 原方案：MinerU (magic-pdf)
- 安装了 `magic-pdf[full]` 及其全部依赖（含 ultralytics, opencv, torch 等）
- 从 HuggingFace (`opendatalab/PDF-Extract-Kit`) 下载了 LayoutLMv3、MFD YOLO、MFR UniMERNet 模型
- 修复了 MFR 模型文件名问题（pytorch_model.pth → pytorch_model.bin）
- **最终阻塞**：LayoutLMv3 依赖 `detectron2`，该包在 Windows CUDA 12.6 + torch 2.11 环境下无预编译 wheel，无法安装

### 替代方案：PyMuPDF (fitz)
- 使用已安装的 `PyMuPDF` 直接提取 PDF 文本
- 生成了 `page_content.json`（307页，6份PDF），格式与 indexer 期望的完全一致
- **不涉及任何已有代码修改**──仅通过单独脚本生成数据文件

---

## 三、索引构建

执行 `build_all_indexes()` 成功：

| 项目 | 结果 |
|------|------|
| 总 chunk 数 | 306 |
| ChromaDB 索引 | `outputs/index_data/chroma/` |
| BM25 索引 | `outputs/index_data/bm25/` |
| Embedding 模型 | BAAI/bge-m3 |
| BM25 分词 | jieba |

---

## 四、main.py 全流程测试

| 步骤 | 状态 | 说明 |
|------|------|------|
| Step 1: 加载测试数据 | ✅ 通过 | 加载 104 个问题 |
| Step 2: 初始化检索器 | ✅ 通过 | ChromaDB + BM25 + bge-reranker-large 全部就绪 |
| Step 3: 初始化VLM生成器 | ❌ 失败 | Qwen2-VL-7B-Instruct 下载阶段磁盘空间不足 |
| Step 4: 逐题处理 | ⏭ 未执行 | |
| Step 5: 保存结果 | ⏭ 未执行 | |

### 检索链路验证结论
**dense(BAAI/bge-m3) + BM25(jieba) → RRF融合 → CrossEncoder(bge-reranker-large) 重排序** 整条检索通路加载和初始化完全通过。失败点仅在于 VLM 模型权重下载（需约 15GB 可用磁盘空间）。

---

## 五、环境副作用

以下不相关包的版本被升级，可能影响 `nlp` 环境中其他项目：

| 包 | 旧版本 | 新版本 |
|----|--------|--------|
| torch | 2.1.2+cu118 | 2.11.0+cu126 |
| transformers | 4.35.0 | 4.57.6 |
| torchvision | 0.16.2 | 0.26.0 |
| tokenizers | 0.14.1 | 0.22.2 |
| huggingface-hub | 0.17.3 | 0.36.2 |
| pydantic | 2.4.2 | 2.10.6 |
| safetensors | 0.4.1 | 0.7.0 |

base 环境的 torch DLL 被破坏（magic-pdf 安装时污染），但 nlp 环境的 `D:\IT\anaconda_envs\nlp\python.exe` 可正常工作。

---

## 六、待解决问题

1. **磁盘空间不足**：清理磁盘或更换模型存储路径后，VLM 生成阶段可继续
2. **conda activate 不生效**：PowerShell 子进程中需手动设置 `$env:PATH` 指向 nlp 环境的 Scripts 目录
3. **MinerU 不可用**：Windows + CUDA 12.6 环境下 detectron2 无预编译包，MinerU PDF 解析无法使用
