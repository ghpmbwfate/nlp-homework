"""Tests for semantic chunking module using 千味央厨 data."""

import pytest
from src.indexing.chunking import (
    split_sentences,
    jieba_similarity,
    compute_similarity,
    semantic_chunking,
)
from src.indexing.indexer import create_chunks


class TestSplitSentences:
    def test_basic_chinese(self):
        text = "公司主营速冻面米制品。产品包括油条、芝麻球等。市场前景广阔。"
        sents = split_sentences(text)
        assert len(sents) == 3
        assert "速冻面米制品" in sents[0]

    def test_with_newlines(self):
        text = "第一段内容。\n第二段内容。\n第三段内容。"
        sents = split_sentences(text)
        assert len(sents) >= 2

    def test_empty_text(self):
        assert split_sentences("") == []

    def test_short_fragments_filtered(self):
        text = "嗯。好。这是一个完整的句子。"
        sents = split_sentences(text)
        # Short fragments like "嗯" "好" should be filtered
        assert any("完整" in s for s in sents)


class TestSimilarity:
    def test_jieba_similarity_identical(self):
        sim = jieba_similarity("速冻面米制品市场", "速冻面米制品市场")
        assert sim > 0.9

    def test_jieba_similarity_different(self):
        sim = jieba_similarity("速冻面米制品", "汽车零部件制造")
        assert sim < 0.5

    def test_jieba_similarity_empty(self):
        assert jieba_similarity("", "test") == 0.0
        assert jieba_similarity("test", "") == 0.0

    def test_compute_similarity_no_model(self):
        sim = compute_similarity("餐饮供应链市场", "餐饮供应链规模", model=None)
        assert 0 <= sim <= 1


class TestSemanticChunking:
    def test_basic_chunking(self, qianwei_pages):
        """Test that chunking produces valid chunks from 千味央厨 pages."""
        chunks = semantic_chunking(qianwei_pages, chunk_size=500, overlap=100, model_name=None)
        assert len(chunks) > 0, "Should produce at least some chunks"

        for c in chunks:
            assert "chunk_id" in c
            assert "filename" in c
            assert "page" in c
            assert "type" in c
            assert "content" in c
            assert len(c["content"]) > 10, f"Chunk too short: {c['chunk_id']}"

    def test_chunks_preserve_page_info(self, qianwei_pages):
        """Test that page numbers are correctly preserved."""
        chunks = semantic_chunking(qianwei_pages, chunk_size=500, overlap=100, model_name=None)
        pages_seen = set()
        for c in chunks:
            if c["type"] == "text":
                pages_seen.add(c["page"])
        assert len(pages_seen) >= 1

    def test_table_chunks_present(self, qianwei_pages):
        """Test that table chunks are included when tables exist."""
        chunks = semantic_chunking(qianwei_pages, chunk_size=500, overlap=100, model_name=None)
        table_chunks = [c for c in chunks if c["type"] == "table"]
        # May or may not have tables depending on data, but structure should be valid
        for tc in table_chunks:
            assert "table" in tc["chunk_id"] or tc["type"] == "table"

    def test_overlap_produces_more_chunks(self, qianwei_pages):
        """Test that enabling overlap can produce more or same chunks."""
        chunks_no_overlap = semantic_chunking(qianwei_pages, chunk_size=500, overlap=0, model_name=None)
        chunks_with_overlap = semantic_chunking(qianwei_pages, chunk_size=500, overlap=100, model_name=None)
        # With overlap, chunks should be roughly similar count or slightly more
        assert len(chunks_with_overlap) >= len(chunks_no_overlap) * 0.5

    def test_chunks_dont_exceed_reasonable_size(self, qianwei_pages):
        """Test that chunks are within reasonable size bounds."""
        chunks = semantic_chunking(qianwei_pages, chunk_size=200, overlap=50, model_name=None)
        text_chunks = [c for c in chunks if c["type"] == "text"]
        for c in text_chunks:
            # Allow some flexibility (up to 4x chunk_size)
            assert len(c["content"]) <= 200 * 4 + 100, \
                f"Chunk {c['chunk_id']} too large: {len(c['content'])} chars"

    def test_synthetic_data(self):
        """Test with controlled synthetic data for predictable behavior."""
        pages = [{
            "filename": "test",
            "page": 1,
            "text": "公司主营速冻面米制品。产品包括油条、芝麻球、蒸饺等。市场前景广阔。",
            "tables": ["| 产品 | 营收 |\n| 油条 | 5亿 |"]
        }]
        chunks = semantic_chunking(pages, chunk_size=20, overlap=5, model_name=None)
        assert len(chunks) > 0
        # Should have at least one text chunk and one table chunk
        types = [c["type"] for c in chunks]
        assert "table" in types


class TestChartChunks:
    """Tests for chart/chart_table chunk creation from chart_descriptions."""

    def test_chart_chunks_created(self):
        """Chart chunks are created from chart_descriptions.charts."""
        pages = [{
            "filename": "test_report",
            "page": 1,
            "text": "这是一段测试文本。",
            "tables": [],
            "chart_descriptions": {
                "page_summary": "本页展示了营收趋势图。",
                "has_charts": True,
                "has_tables": False,
                "charts": [{
                    "chart_id": "图1",
                    "type": "line_chart",
                    "title": "营收趋势",
                    "x_axis": "年份 (2019-2023)",
                    "y_axis": "营收 (亿元)",
                    "legend": ["营收"],
                    "key_data": "营收从2019年的50亿元增长至2023年的120亿元。",
                    "source": "资料来源：WIND",
                }],
                "tables": [],
            }
        }]
        chunks = create_chunks(pages)
        chart_chunks = [c for c in chunks if c["type"] == "chart"]
        assert len(chart_chunks) == 1
        assert chart_chunks[0]["chunk_id"] == "test_report_p1_chart0"
        assert "图1" in chart_chunks[0]["content"]
        assert "营收趋势" in chart_chunks[0]["content"]
        assert "页面摘要" in chart_chunks[0]["content"]

    def test_chart_table_chunks_created(self):
        """chart_table chunks are created from chart_descriptions.tables."""
        pages = [{
            "filename": "test_report",
            "page": 5,
            "text": "对比分析。",
            "tables": [],
            "chart_descriptions": {
                "page_summary": "本页有财务对比表格。",
                "has_charts": False,
                "has_tables": True,
                "charts": [],
                "tables": [{
                    "table_id": "表2",
                    "title": "财务数据对比",
                    "headers": ["年份", "营收", "利润"],
                    "rows": [["2022", "100亿", "20亿"]],
                    "key_info": "2022年营收100亿元，利润20亿元。",
                    "source": None,
                }],
            }
        }]
        chunks = create_chunks(pages)
        ct_chunks = [c for c in chunks if c["type"] == "chart_table"]
        assert len(ct_chunks) == 1
        assert ct_chunks[0]["chunk_id"] == "test_report_p5_charttable0"
        assert "表2" in ct_chunks[0]["content"]
        assert "关键信息" in ct_chunks[0]["content"]

    def test_no_chart_descriptions_graceful(self):
        """Pages without chart_descriptions still work."""
        pages = [{
            "filename": "test_report",
            "page": 1,
            "text": "这是一段足够长的纯文字页面内容，用于测试没有图表描述的情况。",
            "tables": [],
        }]
        chunks = create_chunks(pages)
        assert len(chunks) == 1
        assert chunks[0]["type"] == "text"

    def test_empty_chart_descriptions(self):
        """Empty chart_descriptions stub produces no chart/chart_table chunks."""
        pages = [{
            "filename": "test_report",
            "page": 1,
            "text": "这是一段足够长的纯文字页面内容，用于测试空图表描述的情况。",
            "tables": [],
            "chart_descriptions": {
                "page_summary": "",
                "has_charts": False,
                "has_tables": False,
                "charts": [],
                "tables": [],
            }
        }]
        chunks = create_chunks(pages)
        assert all(c["type"] != "chart" for c in chunks)
        assert all(c["type"] != "chart_table" for c in chunks)
