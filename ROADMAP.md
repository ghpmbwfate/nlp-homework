# 财报 RAG 系统改进路线图

> 本文档供团队分工参考，按优先级 P0→P1→P2→P3 排列。
> 每个改进项标注了：负责人建议、预估工作量、依赖项、预期收益。

---

## 一、当前系统现状

```
PDF → MinerU解析 → markdown → 按页拆分 → 分块 → ChromaDB + BM25索引
                                              ↓
问题 → dense检索 + BM25检索 → RRF融合 → CrossEncoder重排序 → top-k上下文
                                              ↓
                              top-k文本 + top-1页面图片 → VLM生成 → 答案
```

### 已具备的能力
- PDF 文本/表格提取（MinerU）
- 稠密 + 稀疏双路召回
- RRF 融合 + CrossEncoder 重排序
- VLM（Qwen2-VL）图文联合生成
- 5 项评估指标（EM / CharF1 / WordF1 / ROUGE-L / NumberF1）

### 当前主要瓶颈
1. **检索精度不足**：问题直接检索，没有改写/分解；召回只有两路；RRF 参数未调优
2. **生成质量不稳**：固定 Prompt，无引用溯源，无答案校验
3. **分块粒度粗**：按页分块，语义边界不清晰
4. **缺少中间评估**：只有端到端指标，无法定位是检索问题还是生成问题
5. **工程化不足**：串行处理慢、无日志、无缓存

---

## 二、改进项总览

| 优先级 | 模块 | 改进项 | 预估工时 | 依赖 |
|--------|------|--------|----------|------|
| **P0** | 检索 | Query 改写与扩展 | 4h | 无 |
| **P0** | 检索 | 多路召回增强 | 6h | 无 |
| **P0** | 检索 | RRF 参数调优 | 3h | 多路召回 |
| **P0** | 生成 | Prompt 工程（分问题类型） | 4h | 无 |
| **P0** | 生成 | 引用溯源（答案标注来源） | 3h | 无 |
| **P0** | 工程 | 并行处理加速 | 3h | 无 |
| **P1** | 分块 | 语义分块 + 滑动窗口 | 5h | 无 |
| **P1** | 检索 | 重排序多阶段优化 | 4h | 多路召回 |
| **P1** | 生成 | 答案后处理（数字校验等） | 4h | 无 |
| **P1** | 生成 | Self-RAG / 自洽性检查 | 6h | Prompt工程 |
| **P1** | 评估 | 检索评估指标 + 错误分析 | 5h | 无 |
| **P2** | 解析 | 图表信息提取（OCR/VLM） | 8h | 无 |
| **P2** | 解析 | 结构化元数据（章节、标题） | 4h | MinerU输出 |
| **P2** | 检索 | 文档结构过滤 | 3h | 结构化元数据 |
| **P2** | 工程 | 日志系统 + 缓存 + 配置管理 | 4h | 无 |
| **P3** | 全栈 | Embedding/Reranker 微调 | 12h | 训练数据 |
| **P3** | 全栈 | 向量索引参数调优（HNSW） | 3h | 无 |

---

## 三、P0 优先项（立即开始）

### 3.1 检索：Query 改写与扩展

**问题**：用户问题往往口语化、模糊，直接检索召回率低。例如：
> 问题："广联达施工业务怎么样？"
> 文档中对应："广联达数字施工业务2020年合同额增长XX%"

**方案**：
- **关键词扩展**：用 LLM 将问题扩展为 3-5 个关键词丰富的检索 query
- **HyDE (Hypothetical Document Embeddings)**：用 LLM 生成假想的答案文档，用该文档做 dense 检索
- **Query 分解**：将多条件问题拆分为子问题分别检索后合并

**文件位置**：`src/retrieval/query_rewriting.py`
**接口设计**：
```python
class QueryRewriter:
    def rewrite(self, query: str) -> list[str]:  # 返回多个改写版本
    def decompose(self, query: str) -> list[str]:  # 拆分为子问题
    def hyde_generate(self, query: str) -> str:  # 生成假想文档
```

**预期收益**：检索 Recall@10 提升 10-20%

---

### 3.2 检索：多路召回增强

**问题**：当前只有 dense + BM25 两路，召回覆盖不足。

**方案**：增加以下召回通道：
1. **标题召回**：提取文档各级标题建索引，问题先匹配标题（适合"某章节讲了什么"类问题）
2. **关键词召回**：用 jieba + TF-IDF 做关键词匹配（适合数字、专有名词类问题）
3. **摘要召回**：每页生成一句话摘要建索引（适合概括类问题）

**文件位置**：`src/retrieval/multi_recover.py`
**接口设计**：
```python
def title_search(query: str, title_index, top_k: int = 5) -> list[dict]
def keyword_search(query: str, keyword_index, top_k: int = 5) -> list[dict]
def summary_search(query: str, summary_index, top_k: int = 5) -> list[dict]
```

**预期收益**：检索 Recall@10 提升 15-25%

---

### 3.3 检索：RRF 参数调优

**问题**：RRF k=60 是固定经验值，未针对财报场景调优。

**方案**：
- 在开发集上 Grid Search k ∈ [20, 40, 60, 80, 100]
- 为不同召回路分配权重（dense 路权重可略高于 BM25）
- 考虑 weighted RRF：`score = Σ w_i / (k + rank_i)`

**文件位置**：`src/retrieval/fusion.py`
**接口设计**：
```python
def weighted_rrf(dense_hits, bm25_hits, title_hits, keyword_hits,
                 weights: dict = None, k: int = 60) -> list[dict]
```

**预期收益**：检索排序质量提升 5-10%

---

### 3.4 生成：Prompt 工程（分问题类型）

**问题**：当前一个固定 Prompt 处理所有问题。财报问题其实可以分类：
- **事实提取型**："2024年营收是多少？" → 需要精确数字
- **分析总结型**："分析竞争优势" → 需要多角度概括
- **图表理解型**："根据图表64..." → 需要强调图表信息
- **比较型**："与同行业相比..." → 需要对比框架

**方案**：
- 在 `src/generation/prompts/` 下按问题类型定义不同 Prompt 模板
- 用轻量级分类器（关键词匹配或 LLM）判断问题类型，选择对应 Prompt

**文件位置**：
- `src/generation/prompts/fact_extraction.md`
- `src/generation/prompts/analysis_summary.md`
- `src/generation/prompts/chart_understanding.md`
- `src/generation/prompts/comparison.md`
- `src/generation/question_classifier.py`

**预期收益**：不同类型问题准确率提升 10-20%

---

### 3.5 生成：引用溯源（答案标注来源）

**问题**：答案没有标注来源，无法验证，也无法定位错误。

**方案**：
- 修改 Prompt，要求 VLM 在答案后标注 `[来源: 文件名 第X页]`
- 生成后解析引用标记，写入 submit.json 的 `citation` 字段
- 评估时检查引用是否准确（答案内容是否确实出现在引用页）

**文件位置**：`src/generation/citation.py`
**输出格式**：
```json
{
  "question": "...",
  "answer": "广联达2024年营收为XX亿元 [来源: 广联达深度报告 第12页]",
  "citations": [
    {"filename": "广联达深度报告", "page": 12, "chunk_id": "..."}
  ]
}
```

**预期收益**：可解释性大幅提升，便于错误分析

---

### 3.6 工程：并行处理加速

**问题**：当前 120+ 个问题串行处理，每题加载 VLM 推理很慢。

**方案**：
- 检索阶段并行（多线程，每个问题独立检索）
- VLM 生成阶段：若显存允许，batch inference；否则多进程并行
- 使用 `concurrent.futures.ThreadPoolExecutor`

**文件位置**：`src/utils/parallel.py` + 修改 `main.py`

**预期收益**：处理时间缩短 3-5 倍

---

## 四、P1 重要项（P0 完成后）

### 4.1 分块：语义分块 + 滑动窗口

**问题**：按页分块粒度太粗，一页可能包含多个主题；没有重叠导致边界信息丢失。

**方案**：
- **语义分块**：使用文本相似度（如 sentence-transformers）在段落边界切分，保证每个 chunk 语义内聚
- **滑动窗口**：相邻 chunk 重叠 20-30%，避免答案在边界被截断
- 保留原始页码信息用于溯源

**文件位置**：`src/indexing/chunking.py`
**接口设计**：
```python
def semantic_chunking(pages: list[dict],
                      chunk_size: int = 500,
                      overlap: int = 100,
                      model_name: str = "BAAI/bge-m3") -> list[dict]
```

---

### 4.2 检索：重排序多阶段优化

**问题**：当前只做了一轮 CrossEncoder 重排序，没有利用问题的结构化信息。

**方案**：
- **第一阶段**：RRF 粗排（召回 50 个）
- **第二阶段**：CrossEncoder 精排（取 top 20）
- **第三阶段**：多样性重排（MMR，避免重复内容，保证不同角度）
- **第四阶段**：按问题类型过滤（如数字类问题优先表格 chunk）

**文件位置**：`src/retrieval/reranking.py`

---

### 4.3 生成：答案后处理（数字校验等）

**问题**：VLM 可能生成幻觉数字，或格式不统一。

**方案**：
- **数字校验**：提取答案中的数字，检查是否在检索到的上下文中出现过
- **格式标准化**：统一百分比、金额等格式
- **空答案处理**：若检索结果为空，返回 "根据提供的信息无法回答"
- **答案长度控制**：过短/过长时重新生成或截断

**文件位置**：`src/generation/postprocess.py`

---

### 4.4 生成：Self-RAG / 自洽性检查

**问题**：VLM 可能生成与检索内容矛盾的答案。

**方案**：
- **Self-RAG**：生成答案后，用 LLM 判断答案是否被检索内容支持（Support / Contradict / Neutral）
- **自洽性**：对同一问题生成 3 次答案，取最一致的（vote）
- **检索反馈**：若判断为 Contradict，重新检索或扩大检索范围

**文件位置**：`src/generation/self_rag.py`

---

### 4.5 评估：检索评估指标 + 错误分析

**问题**：只有端到端评估，无法定位是检索召回不足还是生成质量差。

**方案**：
- **检索评估**：
  - Recall@k：正确答案所在的页是否在 top-k 检索结果中
  - MRR (Mean Reciprocal Rank)：正确答案页的平均排名倒数
  - Hit Rate@k
- **错误分析工具**：
  - 按问题类型统计错误率
  - 按文档统计错误率
  - 可视化 bad case（问题 + 检索结果 + 预测答案 + 标准答案）

**文件位置**：`src/evaluation/retrieval_eval.py`, `src/evaluation/error_analysis.py`

---

## 五、P2 增强项（有精力时做）

### 5.1 解析：图表信息提取（OCR/VLM）

**问题**：当前只提取了文本和 markdown 表格，PDF 中的图片图表（折线图、柱状图、饼图）信息丢失。

**方案**：
- 对 page_images 中的每张图片，用 VLM（如 Qwen2-VL 或更轻的模型）生成图表描述
- 将图表描述作为 chunk 加入索引
- 针对 "根据图表X" 类问题，优先匹配图表描述 chunk

**文件位置**：`src/parsing/chart_extractor.py`

---

### 5.2 解析：结构化元数据（章节、标题）

**问题**：没有利用文档的章节结构信息。

**方案**：
- 从 MinerU 输出中提取标题层级（H1, H2, H3）
- 为每个 chunk 附加章节路径（如 "3.2 竞争优势 > 市场份额"）
- 检索时可按章节过滤

**文件位置**：`src/parsing/structure_parser.py`

---

### 5.3 检索：文档结构过滤

**问题**：检索时没有利用文档结构信息。

**方案**：
- 若问题包含章节名（如 "第三章..."），优先检索该章节下的 chunk
- 若问题包含图表编号（如 "图表64"），优先检索对应页的 chunk
- 用正则表达式从问题中提取结构线索

**文件位置**：`src/retrieval/structure_filter.py`

---

### 5.4 工程：日志系统 + 缓存 + 配置管理

**方案**：
- **日志**：用 `logging` 模块替代 `print`，支持 DEBUG/INFO/ERROR 级别
- **缓存**：用 `diskcache` 缓存检索结果和 VLM 生成结果（相同问题直接返回）
- **配置**：用 `pydantic-settings` 或 `python-dotenv` 支持环境变量和 YAML 配置

**文件位置**：`src/utils/logger.py`, `src/utils/cache.py`, `config.yaml`

---

## 六、P3 长期项（竞赛后期或后续优化）

### 6.1 Embedding / Reranker 微调

**问题**：通用模型（bge-m3, bge-reranker）对财报领域术语理解可能不够精准。

**方案**：
- 收集领域正样本（问题-相关 chunk 对），微调 Embedding 模型
- 收集领域重排序样本，微调 Reranker
- 可用 `sentence-transformers` 的 `MultipleNegativesRankingLoss`

**依赖**：需要标注一定量的领域训练数据

---

### 6.2 向量索引参数调优（HNSW）

**方案**：
- 调整 ChromaDB 的 HNSW 参数：`M`（邻居数）、`efConstruction`（构建时搜索范围）、`ef`（查询时搜索范围）
- 在速度和召回率之间找到平衡点

---

## 七、团队分工建议

### 建议分组

| 小组 | 负责模块 | 建议人数 | 技能要求 |
|------|----------|----------|----------|
| **检索组** | Query改写、多路召回、RRF调优、重排序 | 2人 | NLP、信息检索 |
| **生成组** | Prompt工程、引用溯源、后处理、Self-RAG | 2人 | LLM Prompt Engineering |
| **数据组** | 语义分块、图表提取、结构化解析 | 1-2人 | 数据处理、CV |
| **工程组** | 并行处理、日志、缓存、配置、评估工具 | 1人 | Python工程 |

### 执行顺序

```
Week 1（P0 攻坚）:
  Day 1-2: 检索组完成 Query改写 + 多路召回框架
  Day 2-3: 生成组完成 Prompt分类 + 引用溯源
  Day 3-4: 工程组完成 并行处理
  Day 4-5: 全组联调，跑通完整 pipeline

Week 2（P0 收尾 + P1 开始）:
  Day 1-2: RRF参数调优 + 检索评估指标搭建
  Day 3-4: 语义分块实验 + 重排序优化
  Day 5: 答案后处理 + 错误分析

Week 3（P1 深化 + P2 启动）:
  Day 1-3: Self-RAG + 图表提取实验
  Day 4-5: 工程化完善（日志、缓存、配置）

Week 4（P2 + P3）:
  根据剩余时间和算力，选择性地做 图表提取、模型微调
```

---

## 八、快速启动检查清单

开始改进前，确保：
- [ ] 当前 baseline 能跑通（`python main.py` 生成 submit.json）
- [ ] 当前 baseline 评估分数已记录（用于对比）
- [ ] 每个改进分支都从 `main` 切出，改完提 PR
- [ ] 每次改进后都跑完整评估，记录分数变化
- [ ] 建立共享文档记录 bad case（问题 + 检索结果 + 预测 + 标准答案）
