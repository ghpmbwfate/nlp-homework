# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See also AGENTS.md for broader project conventions.

## Quick commands

```powershell
# Full QA pipeline — VRAG (dense + BM25 + rerank)
python main.py --retriever vrag --test questions/test.json --output outputs/submit.json

# Full QA pipeline — TF-IDF baseline
python main.py --retriever tfidf --test questions/test.json --output outputs/submit.json

# Generation quality evaluation
python evaluate.py --pred outputs/submit.json --gold questions/test_ground_truth.json --output outputs/evaluation_result.json

# Retrieval-only evaluation (TF-IDF baseline)
python -m src.baseline.tfidf --page_content page_content.json --ground_truth questions/test_ground_truth.json --output outputs/tfidf_eval.json

# Build indexes (run after page_content.json changes)
python -m src.indexing.indexer --page_content page_content.json

# PDF parsing → page_content.json + page images
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
                                    (rich: text blocks + chart metadata + table data per page)
                                              ↓
               chunking → ChromaDB (bge-m3) + BM25 (jieba) indexes
                                    ↓
question → classify (4 types) → dense + BM25 → RRF fusion → CE rerank → top-k
                                    ↓
              API LLM (GPT-OSS-20B-BF16, OpenAI-compat) → answer + citations
```

### Data: two layers

- **`chart_descriptions.json`** (VLM) — the richer source. Per page: typed text blocks, chart titles/axes/key_data, structured table rows.
- **`page_content.json`** (PyMuPDF) — plain text fallback. Used for quick baselines and index building.

### Retriever hierarchy

```
BaseRetriever (src/retrieval/base.py)  ← ABC
  ├── search(query, top_k, **kwargs) → List[dict]    # abstract
  └── search_with_context(query, **kwargs) → dict     # concrete (wraps search)
```

**VRAG** (`src/retrieval/retriever.py`): `Retriever(BaseRetriever)` — ChromaDB dense + BM25 sparse → RRF → CrossEncoder rerank.

**Baselines** (`src/baseline/`): each file has one class inheriting `BaseRetriever`.
- `tfidf.py` — `TFIDFRetriever`: jieba + sklearn TfidfVectorizer over page_content.json

To add a baseline: inherit `BaseRetriever`, implement `search()`, register in `main.py:_build_retriever()`.

## Environment constraints

- **Conda env `nlp`**. `conda activate` doesn't work in PowerShell subprocesses — set `$env:PATH` manually.
- **Windows + CUDA 12.6**: MinerU (`magic-pdf`) does NOT work. PyMuPDF is the fallback. Never install `detectron2`.
- **API keys**: Copy `.env.example` to `.env`, fill `OPENAI_API_KEY` and `DASHSCOPE_API_KEY`.
- **Config**: `src/config.py` — all paths, model names, defaults. Never hardcode.

## Key patterns

- **Lazy `__getattr__` imports** in `src/retrieval/__init__.py` and `src/generation/__init__.py` — do not replace.
- **`BaseRetriever.search_with_context()`** → `{top_filename, top_page, context_text, image_path, results}` — the interface `main.py` consumes.
- **`BaseRetriever.search()`** is abstract. Results must have `{filename, page, content, score}`.
- **`LLMGenerator.generate()`** → `{answer, question_type, citations}`. `image_path` param accepted but ignored.
- **Retrieval features are opt-in**: `enable_multi_recall` and `enable_multi_stage_rerank` default `False`.

## Testing

- pytest + fixtures in `tests/conftest.py`. Test data uses 千味央厨 report pages.
- Tests do not require API keys or model downloads.
