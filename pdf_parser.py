"""
PDF解析模块：MinerU提取文本/表格 + 每页转图片
输出：parsed_data/（markdown）和 page_images/（PNG图片）
"""

import os
import json
import subprocess
from pathlib import Path
from pdf2image import convert_from_path


def parse_pdfs_with_mineru(pdf_dir: str, output_dir: str = "parsed_data"):
    """使用MinerU解析所有PDF，输出markdown"""
    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"[WARNING] 未找到PDF文件: {pdf_dir}")
        return {}

    # MinerU的magic-pdf命令行解析
    # 也可以用API方式调用，这里用命令行更简单
    results = {}
    for pdf_path in pdf_files:
        pdf_name = pdf_path.stem
        print(f"[INFO] 正在解析: {pdf_path.name}")

        # 调用magic-pdf命令行
        cmd = [
            "magic-pdf",
            "-p", str(pdf_path),
            "-o", str(output_dir / pdf_name),
            "-m", "auto"  # 自动选择解析模式
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
            print(f"[INFO] 解析完成: {pdf_path.name}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] 解析失败 {pdf_path.name}: {e.stderr}")
            continue
        except subprocess.TimeoutExpired:
            print(f"[ERROR] 解析超时 {pdf_path.name}")
            continue

        # 读取解析出的markdown
        md_dir = output_dir / pdf_name / "auto"
        md_file = None
        for f in md_dir.glob("*.md"):
            md_file = f
            break

        if md_file and md_file.exists():
            content = md_file.read_text(encoding="utf-8")
            results[pdf_name] = {
                "source_pdf": str(pdf_path),
                "markdown_path": str(md_file),
                "content": content
            }

    # 保存解析结果索引
    index_path = output_dir / "parse_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        # 不存content本身，只存路径映射
        index_data = {
            name: {
                "source_pdf": info["source_pdf"],
                "markdown_path": info["markdown_path"]
            }
            for name, info in results.items()
        }
        json.dump(index_data, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 解析完成，共处理 {len(results)} 个PDF")
    return results


def convert_pdfs_to_images(pdf_dir: str, output_dir: str = "page_images", dpi: int = 200):
    """将每个PDF的每一页转为PNG图片"""
    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"[WARNING] 未找到PDF文件: {pdf_dir}")
        return {}

    image_index = {}  # {filename_page: image_path}

    for pdf_path in pdf_files:
        pdf_name = pdf_path.stem
        print(f"[INFO] 正在转换图片: {pdf_path.name}")

        try:
            images = convert_from_path(str(pdf_path), dpi=dpi)
        except Exception as e:
            print(f"[ERROR] 转换失败 {pdf_path.name}: {e}")
            continue

        for page_num, image in enumerate(images, start=1):
            image_name = f"{pdf_name}_page_{page_num}.png"
            image_path = output_dir / image_name
            image.save(str(image_path), "PNG")
            image_index[f"{pdf_name}_page_{page_num}"] = str(image_path)

        print(f"[INFO] 转换完成: {pdf_path.name}, 共 {len(images)} 页")

    # 保存图片索引
    index_path = output_dir / "image_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(image_index, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 图片转换完成，共 {len(image_index)} 张")
    return image_index


def split_markdown_by_page(markdown_content: str, pdf_name: str) -> list[dict]:
    """
    将MinerU输出的markdown按页拆分
    MinerU通常用 <!-- Page XX --> 或类似标记分隔页面
    返回: [{"page": int, "text": str, "tables": [str]}]
    """
    # MinerU的页面分隔标记
    import re
    page_split_pattern = r'(?:<!--\s*Page\s+(\d+)\s*-->|\n---\n|\f)'

    parts = re.split(page_split_pattern, markdown_content)

    pages = []
    current_page = 1
    current_text = ""

    i = 0
    while i < len(parts):
        part = parts[i]
        # 检查是否是页码标记
        page_match = re.match(r'^\d+$', part.strip()) if part else False
        if page_match and i + 1 < len(parts):
            # 保存上一页
            if current_text.strip():
                tables = extract_tables(current_text)
                pages.append({
                    "filename": pdf_name,
                    "page": current_page,
                    "text": current_text.strip(),
                    "tables": tables
                })
            current_page = int(part.strip())
            current_text = parts[i + 1] if i + 1 < len(parts) else ""
            i += 2
        else:
            current_text += part
            i += 1

    # 最后一页
    if current_text.strip():
        tables = extract_tables(current_text)
        pages.append({
            "filename": pdf_name,
            "page": current_page,
            "text": current_text.strip(),
            "tables": tables
        })

    # 如果没检测到分页标记，整篇作为一页
    if not pages:
        pages.append({
            "filename": pdf_name,
            "page": 1,
            "text": markdown_content.strip(),
            "tables": extract_tables(markdown_content)
        })

    return pages


def extract_tables(text: str) -> list[str]:
    """从markdown文本中提取表格"""
    import re
    # 匹配markdown表格
    table_pattern = r'(\|[^\n]+\|\n(?:\|[-:| ]+\|\n)?(?:\|[^\n]+\|\n)*)'
    tables = re.findall(table_pattern, text)
    return [t.strip() for t in tables if t.strip()]


def build_page_content_index(parsed_data_dir: str = "parsed_data",
                              output_path: str = "page_content.json"):
    """
    构建页面内容索引：将所有PDF的markdown按页拆分并保存
    输出格式: [{"filename": str, "page": int, "text": str, "tables": [str]}]
    """
    parsed_dir = Path(parsed_data_dir)
    index_file = parsed_dir / "parse_index.json"

    if not index_file.exists():
        print("[ERROR] 未找到parse_index.json，请先运行parse_pdfs_with_mineru")
        return []

    with open(index_file, "r", encoding="utf-8") as f:
        parse_index = json.load(f)

    all_pages = []
    for pdf_name, info in parse_index.items():
        md_path = Path(info["markdown_path"])
        if not md_path.exists():
            print(f"[WARNING] markdown文件不存在: {md_path}")
            continue

        content = md_path.read_text(encoding="utf-8")
        pages = split_markdown_by_page(content, pdf_name)
        all_pages.extend(pages)
        print(f"[INFO] {pdf_name}: 拆分为 {len(pages)} 页")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_pages, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 页面内容索引已保存: {output_path}, 共 {len(all_pages)} 页")
    return all_pages


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PDF解析工具")
    parser.add_argument("--pdf_dir", type=str, default="财报数据库", help="PDF文件目录")
    parser.add_argument("--parsed_dir", type=str, default="parsed_data", help="MinerU解析输出目录")
    parser.add_argument("--image_dir", type=str, default="page_images", help="页面图片输出目录")
    parser.add_argument("--dpi", type=int, default=200, help="图片DPI")
    parser.add_argument("--skip_parse", action="store_true", help="跳过MinerU解析，只转图片")
    parser.add_argument("--skip_images", action="store_true", help="跳过图片转换")

    args = parser.parse_args()

    if not args.skip_parse:
        print("=" * 50)
        print("Step 1: MinerU解析PDF")
        print("=" * 50)
        parse_pdfs_with_mineru(args.pdf_dir, args.parsed_dir)
        print()

        print("=" * 50)
        print("Step 2: 构建页面内容索引")
        print("=" * 50)
        build_page_content_index(args.parsed_dir)
        print()

    if not args.skip_images:
        print("=" * 50)
        print("Step 3: 转换PDF为页面图片")
        print("=" * 50)
        convert_pdfs_to_images(args.pdf_dir, args.image_dir, args.dpi)
