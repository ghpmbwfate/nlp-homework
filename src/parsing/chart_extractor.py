"""
VLM 图表提取模块：使用 DashScope 多模态模型分析 PDF 页面图片，
提取其中的图表（折线图、柱状图、饼图等）和表格信息。
"""

import base64
import json
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

from src.config import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    DASHSCOPE_VL_MODEL,
    IMAGES_DIR,
    PARSED_DIR,
)

# ── 输出路径 ──────────────────────────────────────────────────
# 每个 PDF 的结果保存在 PARSED_DIR/{pdf_name}/chart_descriptions.json
# 例如: outputs/parsed_data/伊利股份-.../chart_descriptions.json

# ── 提示词 ────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是一个专业的财报图表分析专家。你的任务是从 PDF 页面截图中提取所有文字、图表和表格的信息。

请仔细观察图片中的所有视觉元素，并输出一个 JSON 对象，格式如下：

{
  "page_summary": "本页主要内容的一句话概括（中文）",
  "has_text": true/false,
  "has_charts": true/false,
  "has_tables": true/false,
  "text": [
    {
      "text_id": "文本编号（如无编号则填 null）",
      "type": "paragraph / title / caption / other",
      "content": "文本内容（中文）"
    }
  ],
  "charts": [
    {
      "chart_id": "图1 或 图表编号（如无编号则填 null）",
      "type": "line_chart / bar_chart / pie_chart / scatter_plot / area_chart / combo_chart / other",
      "title": "图表标题",
      "x_axis": "X 轴含义和刻度说明",
      "y_axis": "Y 轴含义和刻度说明",
      "legend": ["图例项1", "图例项2"],
      "key_data": "图表中最重要的数据点、趋势或结论（中文描述）",
      "source": "数据来源标注（如 '资料来源：万得，信达证券研发中心'，无则填 null）"
    }
  ],
  "tables": [
    {
      "table_id": "表1 或 表格编号（如无编号则填 null）",
      "title": "表格标题",
      "headers": ["列标题1", "列标题2"],
      "rows": [
        ["行1列1", "行1列2"],
        ["行2列1", "行2列2"]
      ],
      "key_info": "表格中最重要的信息概括（中文）",
      "source": "数据来源标注（无则填 null）"
    }
  ]
}

规则：
0. 文本内容指页面上除图表和表格以外的所有文本，包括标题、段落、注释等。标题仅包括页面上的主标题。段落包括一切正文段落（若次级标题后跟随正文，也属于该段落）。注释指页面底部或图表附近的说明文字。
1. 文本/图表/表格的内容、编号、标题、图例、轴标签必须从图片中原文提取，不要编造
2. 表格的行列数据尽量完整提取；如果表格太大，至少提取前 5 行和最关键的汇总行
3. key_data / key_info 用中文概括最重要的事实和趋势
4. 如果页面是纯文字，has_charts 和 has_tables 都设为 false，charts 和 tables 为空数组
5. 封皮页、目录页、纯文字页也正常分析——page_summary 要准确概括
6. 只输出 JSON，不要有任何额外文字或 markdown 标记"""

USER_PROMPT_TEMPLATE = """请分析这张 PDF 页面截图，提取其中所有文本、图表和表格的信息。
文件名: {filename}
页码: 第 {page_num} 页"""

# ── 工具函数 ──────────────────────────────────────────────────


def _image_to_base64(image_path: str) -> str:
    """将图片文件编码为 base64 data URL"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _build_data_url(image_path: str) -> str:
    """构建 base64 data URL"""
    b64 = _image_to_base64(image_path)
    return f"data:image/png;base64,{b64}"


def _parse_json_response(raw_text: str) -> Optional[dict]:
    """从 VLM 响应中解析 JSON，处理常见的格式问题"""
    text = raw_text.strip()
    # 去除可能的 markdown 代码块标记
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试在文本中查找 JSON 块
        import re
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None


# ── 主类 ──────────────────────────────────────────────────────


class ChartExtractor:
    """VLM 图表提取器：逐页分析 PDF 截图，提取图表和表格信息"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        page_image_dir: Optional[str] = None,
    ):
        api_key = api_key or DASHSCOPE_API_KEY
        if not api_key:
            raise ValueError(
                "DashScope API key is required. "
                "Set DASHSCOPE_API_KEY in .env or pass api_key."
            )
        base_url = base_url or DASHSCOPE_BASE_URL
        self.model = model or DASHSCOPE_VL_MODEL
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.page_image_dir = Path(page_image_dir) if page_image_dir else IMAGES_DIR
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        print(f"[INFO] 图表提取器就绪: model={self.model}, base_url={base_url}")

    def _call_vlm(self, image_path: str, filename: str, page_num: int) -> dict:
        """调用 DashScope VLM API 分析单页图片"""
        data_url = _build_data_url(image_path)
        user_text = USER_PROMPT_TEMPLATE.format(
            filename=filename, page_num=page_num
        )

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": data_url},
                                },
                                {"type": "text", "text": user_text},
                            ],
                        },
                    ],
                    temperature=0.0,
                    # max_tokens=4096,
                )
                raw_text = response.choices[0].message.content
                if not raw_text:
                    raise RuntimeError("VLM 返回空响应")

                parsed = _parse_json_response(raw_text)
                if parsed is not None:
                    return parsed

                # JSON 解析失败——当作 raw 结果返回
                print(
                    f"  [WARNING] JSON 解析失败，保留原始响应"
                )
                return {
                    "page_summary": "JSONDecodeError: 无法解析 VLM 响应",
                    "has_charts": False,
                    "has_tables": False,
                    "charts": [],
                    "tables": [],
                    "_raw_response": raw_text,
                    "_parse_error": True,
                }

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait = self.retry_delay * (2 ** attempt)
                    print(
                        f"  [WARNING] API 调用失败 (尝试 {attempt + 1}/{self.max_retries}), "
                        f"{wait:.0f}s 后重试: {e}"
                    )
                    time.sleep(wait)
                else:
                    print(f"  [ERROR] API 调用最终失败: {e}")

        raise RuntimeError(
            f"VLM API call failed after {self.max_retries} attempts: {last_error}"
        )

    def extract_page(
        self, filename: str, page_num: int
    ) -> Optional[dict]:
        """分析单页图片，返回结构化图表信息"""
        image_name = f"{filename}_page_{page_num}.png"
        image_path = self.page_image_dir / image_name
        if not image_path.exists():
            print(f"  [SKIP] 图片不存在: {image_name}")
            return None

        print(f"  [INFO] 分析: {filename} 第{page_num}页")
        result = self._call_vlm(str(image_path), filename, page_num)
        result["filename"] = filename
        result["page"] = page_num
        return result

    def _output_path_for(self, filename: str) -> Path:
        """推导某个 PDF 的输出文件路径"""
        return PARSED_DIR / filename / "chart_descriptions.json"

    def _pre_scan(
        self, tasks: list[tuple[str, int]]
    ) -> tuple[dict[str, list[dict]], set[tuple[str, int]]]:
        """
        预处理：扫描所有 per-PDF 输出文件，汇总已处理页面。

        Returns:
            (all_results, existing_keys)
            - all_results: {pdf_name: [page_dict, ...]}  已有的完整结果
            - existing_keys: {(filename, page), ...}     已处理的页面集合
        """
        print("=" * 50)
        print("预扫描：检查已处理页面")
        print("=" * 50)

        all_results: dict[str, list[dict]] = {}
        existing_keys: set[tuple[str, int]] = set()
        pdf_names = sorted(set(t[0] for t in tasks))

        total_pages_by_pdf: dict[str, int] = {}
        for fn, _ in tasks:
            total_pages_by_pdf[fn] = total_pages_by_pdf.get(fn, 0) + 1

        done_total = 0
        remain_total = 0

        for fn in pdf_names:
            existing = self._load_per_pdf(fn)
            all_results[fn] = existing
            total_pdf = total_pages_by_pdf.get(fn, 0)
            done_pdf = len(existing)
            remain_pdf = max(0, total_pdf - done_pdf)
            done_total += done_pdf
            remain_total += remain_pdf

            for r in existing:
                existing_keys.add((r["filename"], r["page"]))

            status = "[DONE]" if remain_pdf == 0 else f"还需 {remain_pdf} 页"
            print(
                f"  [{done_pdf:>3}/{total_pdf:>3}] {fn[:45]:45s}  {status}"
            )

        total_all = done_total + remain_total
        print("-" * 50)
        print(
            f"  合计: 已处理 {done_total} 页 / 共 {total_all} 页"
            f" ({remain_total} 页待处理)"
        )
        if done_total == total_all and total_all > 0:
            print("  所有页面已完成，无需处理。")
        print()

        return all_results, existing_keys

    def extract_all(
        self,
        image_index_path: Optional[str] = None,
        start_page: int = 1,
        max_pages: Optional[int] = None,
        per_pdf: Optional[int] = None,
        no_skip: bool = False,
        delay_between_calls: float = 0.5,
    ) -> dict[str, list[dict]]:
        """
        批量分析所有页面图片，每个 PDF 的结果保存在各自的目录下。
        自动跳过已处理页面；使用 --no_skip 强制重新处理。

        Args:
            image_index_path: image_index.json 路径，None 则自动扫描图片目录
            start_page: 起始页码（1-based），用于断点续传
            max_pages: 最大处理总页数，None 表示全量
            per_pdf: 每个 PDF 最多处理前 N 页，None 表示全量（用于测试时均匀采样）
            no_skip: 不跳过已处理页面，强制重新分析所有页面
            delay_between_calls: 两次 API 调用之间的延迟（秒），避免触发限流

        Returns:
            {pdf_name: [page_results]} 字典
        """
        tasks = self._collect_tasks(image_index_path)
        total = len(tasks)
        print(f"[INFO] 共发现 {total} 张页面图片")

        # 按 PDF 分组，应用 per_pdf 限制
        if per_pdf is not None:
            pdf_counters: dict[str, int] = {}
            filtered_tasks = []
            for filename, page_num in tasks:
                cnt = pdf_counters.get(filename, 0)
                if cnt < per_pdf:
                    filtered_tasks.append((filename, page_num))
                    pdf_counters[filename] = cnt + 1
            tasks = filtered_tasks
            print(f"[INFO] 每 PDF 最多 {per_pdf} 页，共 {len(tasks)} 个任务")

        # ── 预扫描：加载已有结果 ──
        if no_skip:
            print("[INFO] --no_skip 已启用，将忽略已有结果重新处理所有页面")
            all_results: dict[str, list[dict]] = {}
            existing_keys: set[tuple[str, int]] = set()
        else:
            all_results, existing_keys = self._pre_scan(tasks)
            if not existing_keys:
                print("[INFO] 无已有结果，将处理所有页面\n")
            else:
                skipped_in_tasks = sum(
                    1 for t in tasks if t in existing_keys
                )
                if skipped_in_tasks == len(tasks):
                    print("[INFO] 所有待处理页面均已完成，无需处理。\n")
                    return all_results
                print(
                    f"[INFO] 在本次任务中，已有 {skipped_in_tasks} 页将被跳过。"
                    f" 开始处理...\n"
                )

        processed = 0
        skipped = 0
        failed = 0
        dirty_pdfs: set[str] = set()  # 有变更的 PDF

        for i, (filename, page_num) in enumerate(tasks):
            if page_num < start_page:
                continue
            if (filename, page_num) in existing_keys:
                skipped += 1
                continue
            if max_pages is not None and processed >= max_pages:
                print(f"[INFO] 已达到最大处理数 {max_pages}，停止")
                break

            try:
                result = self.extract_page(filename, page_num)
                if result is not None:
                    all_results.setdefault(filename, []).append(result)
                    processed += 1
                    dirty_pdfs.add(filename)
            except Exception as e:
                print(f"  [ERROR] 第{page_num}页处理失败: {e}")
                failed += 1
                all_results.setdefault(filename, []).append({
                    "filename": filename,
                    "page": page_num,
                    "page_summary": f"提取失败: {e}",
                    "has_charts": False,
                    "has_tables": False,
                    "charts": [],
                    "tables": [],
                    "_error": str(e),
                })
                dirty_pdfs.add(filename)

            print(
                f"  [PROGRESS] {i + 1}/{len(tasks)} "
                f"(已处理 {processed}, 跳过 {skipped}, 失败 {failed})"
            )
            # 每处理 10 页保存一次有变更的 PDF
            if processed > 0 and processed % 10 == 0:
                for fn in dirty_pdfs:
                    self._save_per_pdf(fn, all_results[fn])
                dirty_pdfs.clear()

            if delay_between_calls > 0:
                time.sleep(delay_between_calls)

        # 最终保存有变更的 PDF
        for fn in dirty_pdfs:
            self._save_per_pdf(fn, all_results[fn])
        dirty_pdfs.clear()

        total_processed = sum(len(v) for v in all_results.values())
        print(
            f"\n[INFO] 处理完成! 共 {total} 页, "
            f"成功 {processed}, 跳过 {skipped}, 失败 {failed}"
        )
        print(f"[INFO] 结果已保存至 {PARSED_DIR}")
        for fn in sorted(all_results):
            print(f"  {self._output_path_for(fn)} ({len(all_results[fn])} 页)")

        return all_results

    def _collect_tasks(
        self, image_index_path: Optional[str]
    ) -> list[tuple[str, int]]:
        """收集所有待处理的 (filename, page_num) 任务"""
        if image_index_path is not None:
            index_path = Path(image_index_path)
            if index_path.exists():
                with open(index_path, "r", encoding="utf-8") as f:
                    index = json.load(f)
                tasks = []
                for key in sorted(index.keys()):
                    # key 格式: "{filename}_page_{page_num}"
                    parts = key.rsplit("_page_", 1)
                    if len(parts) == 2:
                        filename = parts[0]
                        page_num = int(parts[1])
                        tasks.append((filename, page_num))
                return tasks

        # 回退：扫描图片目录
        tasks = []
        for img_path in sorted(self.page_image_dir.glob("*.png")):
            stem = img_path.stem
            # 文件名格式: "{filename}_page_{page_num}"
            parts = stem.rsplit("_page_", 1)
            if len(parts) == 2:
                filename = parts[0]
                try:
                    page_num = int(parts[1])
                except ValueError:
                    continue
                tasks.append((filename, page_num))
        return tasks

    def _load_per_pdf(self, filename: str) -> list[dict]:
        """加载某个 PDF 的已有结果（用于增量处理）"""
        output_path = self._output_path_for(filename)
        if output_path.exists():
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data
            except (json.JSONDecodeError, IOError):
                pass
        return []

    def _save_per_pdf(self, filename: str, results: list[dict]):
        """保存某个 PDF 的结果到 JSON 文件"""
        output_path = self._output_path_for(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


# ── 便捷函数 ──────────────────────────────────────────────────


def merge_with_page_content(
    chart_results: dict[str, list[dict]] | list[dict],
    page_content_path: str = "page_content.json",
    output_path: Optional[str] = None,
) -> list[dict]:
    """
    将图表提取结果与 page_content.json 合并，为每页补充 chart_descriptions 字段。
    兼容旧版单文件列表和新版 per-PDF 字典。

    Args:
        chart_results: ChartExtractor.extract_all() 返回的 {pdf_name: [pages]} 或旧版列表
        page_content_path: page_content.json 路径
        output_path: 合并后输出路径，None 则覆盖原文件

    Returns:
        合并后的 page_content 列表
    """
    output_path = Path(output_path) if output_path else Path(page_content_path)

    with open(page_content_path, "r", encoding="utf-8") as f:
        pages = json.load(f)

    # 归一化：统一转为列表
    if isinstance(chart_results, dict):
        flat_results = []
        for results in chart_results.values():
            flat_results.extend(results)
    else:
        flat_results = chart_results

    chart_map = {}
    for cr in flat_results:
        key = (cr["filename"], cr["page"])
        chart_map[key] = {
            "page_summary": cr.get("page_summary", ""),
            "has_charts": cr.get("has_charts", False),
            "has_tables": cr.get("has_tables", False),
            "charts": cr.get("charts", []),
            "tables": cr.get("tables", []),
        }

    for page in pages:
        key = (page["filename"], page["page"])
        if key in chart_map:
            page["chart_descriptions"] = chart_map[key]
        else:
            page["chart_descriptions"] = {
                "page_summary": "",
                "has_charts": False,
                "has_tables": False,
                "charts": [],
                "tables": [],
            }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 合并完成，已写入 {output_path}")
    return pages


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="VLM 图表提取工具 —— 使用 DashScope 多模态模型分析 PDF 页面图片"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=f"VLM 模型名（默认: {DASHSCOPE_VL_MODEL}）"
    )
    parser.add_argument(
        "--image_dir", type=str, default=None,
        help="页面图片目录（默认 outputs/page_images/）"
    )
    parser.add_argument(
        "--start_page", type=int, default=1,
        help="起始页码（用于断点续传）"
    )
    parser.add_argument(
        "--max_pages", type=int, default=None,
        help="最大处理总页数（用于测试）"
    )
    parser.add_argument(
        "--per_pdf", type=int, default=None,
        help="每个 PDF 最多取前 N 页（用于均匀采样测试，如 --per_pdf 1）"
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="API 调用间隔秒数（默认 0.5）"
    )
    parser.add_argument(
        "--no_skip", action="store_true",
        help="不跳过已处理页面，强制重新分析"
    )
    parser.add_argument(
        "--page_content", type=str, default=None,
        help="与 page_content.json 合并的路径（可选）"
    )

    args = parser.parse_args()

    extractor = ChartExtractor(
        model=args.model,
        page_image_dir=args.image_dir,
    )

    results = extractor.extract_all(
        image_index_path=None,
        start_page=args.start_page,
        max_pages=args.max_pages,
        per_pdf=args.per_pdf,
        no_skip=args.no_skip,
        delay_between_calls=args.delay,
    )

    # 可选：合并到 page_content.json
    if args.page_content:
        merge_with_page_content(
            results,
            page_content_path=args.page_content,
        )
