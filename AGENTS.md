# AGENTS.md — nlp-homework

Financial report RAG Q&A pipeline. Conda env `nlp`, Windows only (PowerShell 5.1).

## Quick commands

```powershell
# Full pipeline — VRAG (dense + BM25 + rerank)
python main.py --test questions/test.json --output outputs/submit.json --retriever vrag

# Full pipeline — TF-IDF baseline
python main.py --test questions/test.json --output outputs/submit.json --retriever tfidf

# Evaluation (generation quality)
python evaluate.py --pred outputs/submit.json --gold questions/test_ground_truth.json --output outputs/evaluation_result.json

# Retrieval-only evaluation (TF-IDF baseline)
python -m src.baseline.tfidf --page_content page_content.json --ground_truth questions/test_ground_truth.json --output outputs/tfidf_eval.json

# Build indexes (run after page_content.json changes)
python -m src.indexing.indexer --page_content page_content.json

# PDF parsing → page_content.json (PyMuPDF text) + page images
python -m src.parsing.pdf_parser

# VLM chart extraction → outputs/parsed_data/{pdf_name}/chart_descriptions.json
python -m src.parsing.chart_extractor --per_pdf 1

# Run all tests
pytest tests/ -v
```

## Architecture

```
PDF reports
  ├── PyMuPDF (fitz) ──────────► page_content.json  (plain text per page)
  └── VLM (qwen-vl-max) ───────► outputs/parsed_data/{pdf_name}/chart_descriptions.json
                                    (text blocks + chart descriptions + table data per page)
                                              ↓
               chunking (per-page text + tables separately)
                                    ↓
               ChromaDB (dense, bge-m3) + BM25 (sparse, jieba)
                                    ↓
question → classify (4 types) → dense + BM25 → RRF fusion → CE rerank → top-k
                                    ↓
              API LLM (GPT-OSS-20B-BF16, OpenAI-compat) → answer + citations
                                    ↓
              Self-RAG check (number/keyword overlap vs retrieved context)
```

### Data sources

- **`outputs/parsed_data/{pdf_name}/chart_descriptions.json`** — VLM-extracted full content per page: typed text blocks (title/paragraph/caption), chart metadata (type, axes, key_data), structured table data. This is the **richest data source**.
- **`page_content.json`** — PyMuPDF (fitz) fallback: plain text per page. Used when VLM is unavailable or for quick baselines. All indexes derive from this file.
- **`outputs/page_images/`** — PNG screenshots of each PDF page (200 DPI), consumed by VLM chart extraction.

### Retriever architecture

All retrievers inherit from `BaseRetriever` (`src/retrieval/base.py`):

```
BaseRetriever (ABC)
  ├── search(query, top_k, **kwargs) → List[dict]     # abstract — subclass implements
  └── search_with_context(query, **kwargs) → dict      # concrete — wraps search()
```

**VRAG** (`src/retrieval/retriever.py`): dense (ChromaDB + bge-m3) + BM25 (jieba) → RRF fusion → CrossEncoder rerank → optional MMR + type filter.

**Baselines** (`src/baseline/`): each in its own file, inherits `BaseRetriever`:
- `tfidf.py` — TF-IDF over page_content.json (jieba tokenizer, sklearn TfidfVectorizer)

To add a new baseline, inherit `BaseRetriever`, implement `search()`, and register in `main.py`'s `_build_retriever()`.

### Module map

| Module | Purpose |
|--------|---------|
| `src/parsing/` | PDF→text via MinerU/PyMuPDF fallback; page→images; VLM chart extraction → `chart_descriptions.json` |
| `src/indexing/` | Chunking (per-page text + table); ChromaDB dense + BM25 sparse indexes |
| `src/retrieval/` | `BaseRetriever` ABC; `Retriever` (VRAG); dense/BM25/RRF/rerank/MMR; query rewriting; multi-recovery |
| `src/baseline/` | Baseline retrievers (TF-IDF, etc.) — one file per method, all inherit `BaseRetriever` |
| `src/generation/` | OpenAI-compat LLM generation; question classifier (keyword patterns → 4 types); per-type prompt templates; citation extraction; self-RAG consistency check |
| `src/evaluation/` | EM, Char F1, Word F1, ROUGE-L, Number F1; retrieval Recall@k/MRR; per-question error analysis |

### Question types (drive prompt selection + retrieval filtering)

1. `fact_extraction` — numerical facts → prioritizes table chunks
2. `analysis_summary` — "分析/评估/总结" → balanced
3. `chart_understanding` — "图表/如图" → prioritizes chart-reference chunks
4. `comparison` — "对比/相比" → balanced

Classification is keyword-pattern based (`src/generation/question_classifier.py`). Fallback: `fact_extraction`.

## Environment constraints

- **Conda env `nlp`**. `conda activate` doesn't work in PowerShell subprocesses — set `$env:PATH` manually if spawning conda commands.
- **Windows + CUDA 12.6**: MinerU (`magic-pdf`) does NOT work (`detectron2` has no prebuilt wheel). PyMuPDF (`fitz`) is the fallback. Never attempt to install `detectron2` or use MinerU.
- **API keys**: Copy `.env.example` to `.env`, fill in `OPENAI_API_KEY` and `DASHSCOPE_API_KEY`.
- **Local VLM is optional**: `GPT-OSS-20B-BF16` runs via OpenAI-compat HTTP endpoint. The text-only path avoids local GPU entirely.

## Key patterns

- **Lazy imports via `__getattr__`**: `src/retrieval/__init__.py` and `src/generation/__init__.py` use `__getattr__` for lazy imports. This is intentional — do not replace with regular imports.
- **Config lives in `src/config.py`**: Always import paths, model names, and defaults from there. Never hardcode.
- **`BaseRetriever.search_with_context()`** returns `{"top_filename", "top_page", "context_text", "image_path", "results"}` — this is the main retrieval interface consumed by `main.py`. All retrievers inherit it.
- **`BaseRetriever.search()`** is abstract. Each retriever implements its own logic; results must have `{filename, page, content, score}`.
- **`LLMGenerator.generate()`** returns `{"answer", "question_type", "citations"}`. The `image_path` param is accepted for compatibility but **ignored** (text-only path).
- **`page_content.json`** is the parsed PDF database from PyMuPDF. All indexes derive from it. If missing, rebuild with `python -m src.indexing.indexer`.
- **Retrieval features are opt-in**: Multi-recovery (`enable_multi_recall`) and multi-stage reranking (`enable_multi_stage_rerank`) default to `False` in `Retriever.__init__`.

## Testing

- pytest with fixtures in `tests/conftest.py`. Test data files (`test_pages_qianwei.json`, `test_questions_qianwei.json`, `test_ground_truth_qianwei.json`) use real content from 千味央厨 report.
- Tests do NOT require live API keys or model downloads — they use mocked/fixture data.

## DO NOT

- Install `detectron2` or attempt to use MinerU on Windows.
- Hardcode paths or API keys — use `src/config.py` and `.env`.
- Replace `_call_llm` timeout logic with any synchronous blocking without retry — the API endpoint may be slow.
- Remove or refactor `__getattr__` lazy imports in `__init__.py` files.
