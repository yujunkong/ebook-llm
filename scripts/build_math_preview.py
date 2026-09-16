#!/usr/bin/env python3
"""Markdown 강의를 KaTeX로 렌더되는 HTML 미리보기로 변환한다.

사용 예:
  python3 scripts/build_math_preview.py books/01_python_tensor_math_pytorch/09강_Scalar_Vector_Matrix_Tensor.md
  python3 scripts/build_math_preview.py --book 1
"""

from __future__ import annotations

import argparse
import html
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "epub" / "preview"

BOOK_DIRS = {
    1: ROOT / "books" / "01_python_tensor_math_pytorch",
    2: ROOT / "books" / "02_tokenizer_transformer",
    3: ROOT / "books" / "03_gpt_pretraining_sft",
    4: ROOT / "books" / "04_rlhf_ppo_grpo",
    5: ROOT / "books" / "05_vllm_glm_dgx",
}

# KaTeX CDN — 브라우저에서 $$ / $ / \\( \\) 수식을 렌더한다.
KATEX_HEAD = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {
    delimiters: [
      {left: '$$', right: '$$', display: true},
      {left: '\\\\(', right: '\\\\)', display: false},
      {left: '\\\\[', right: '\\\\]', display: true},
      {left: '$', right: '$', display: false}
    ],
    throwOnError: false
  });"></script>
<style>
  :root { color-scheme: light dark; }
  body {
    font-family: "Pretendard", "Noto Sans KR", system-ui, sans-serif;
    line-height: 1.7;
    max-width: 52rem;
    margin: 2rem auto;
    padding: 0 1.25rem;
  }
  pre, code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  pre { overflow-x: auto; padding: 0.9rem 1rem; border-radius: 8px;
        background: color-mix(in srgb, Canvas 92%, CanvasText 8%); }
  .katex-display { margin: 1.25rem 0; overflow-x: auto; overflow-y: hidden; }
  h1,h2,h3 { line-height: 1.35; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
  th, td { border: 1px solid color-mix(in srgb, CanvasText 25%, transparent);
           padding: 0.4rem 0.6rem; }
</style>
"""


def md_to_html_body(md_path: Path) -> str:
    """pandoc으로 Markdown→HTML 본문 변환 (수식 구분자는 그대로 남겨 KaTeX가 처리)."""
    proc = subprocess.run(
        [
            "pandoc",
            str(md_path),
            "-f",
            "markdown",
            "-t",
            "html5",
            "--no-highlight",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout


def wrap_page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
{KATEX_HEAD}
</head>
<body>
{body}
</body>
</html>
"""


def build_one(md_path: Path, out_path: Path) -> None:
    title = md_path.stem
    body = md_to_html_body(md_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(wrap_page(title, body), encoding="utf-8")
    print(f"[OK] {md_path.relative_to(ROOT)} → {out_path.relative_to(ROOT)}")


def build_book_index(book_num: int, lecture_files: list[Path], out_dir: Path) -> None:
    links = []
    for p in lecture_files:
        html_name = p.stem + ".html"
        links.append(f'<li><a href="{html.escape(html_name)}">{html.escape(p.stem)}</a></li>')
    index = wrap_page(
        f"밑바닥부터 LLM {book_num}권 미리보기",
        f"<h1>《밑바닥부터 LLM》 {book_num}권 수식 미리보기</h1>"
        f"<p>브라우저에서 열면 KaTeX가 <code>$$…$$</code> 수식을 렌더합니다.</p>"
        f"<ol>\n{chr(10).join(links)}\n</ol>",
    )
    (out_dir / "index.html").write_text(index, encoding="utf-8")
    print(f"[OK] index → {out_dir.relative_to(ROOT)}/index.html")


def main() -> int:
    parser = argparse.ArgumentParser(description="KaTeX HTML 미리보기 생성")
    parser.add_argument("markdown", nargs="?", help="단일 Markdown 파일 경로")
    parser.add_argument("--book", type=int, choices=[1, 2, 3, 4, 5], help="권 단위로 전체 변환")
    args = parser.parse_args()

    if not args.markdown and not args.book:
        parser.error("markdown 파일 또는 --book N 이 필요합니다.")

    if args.markdown:
        md = Path(args.markdown)
        if not md.is_absolute():
            md = (ROOT / md).resolve()
        out = OUT_DIR / md.stem / f"{md.stem}.html"
        # flatter: epub/preview/<stem>.html
        out = OUT_DIR / f"{md.stem}.html"
        build_one(md, out)
        print(f"\n브라우저에서 열기:\n  file://{out}")
        return 0

    book_dir = BOOK_DIRS[args.book]
    lectures = sorted(book_dir.glob("*강*.md"), key=lambda p: int(re.match(r"(\d+)", p.name).group(1)))
    out_dir = OUT_DIR / f"book{args.book}"
    for md in lectures:
        build_one(md, out_dir / f"{md.stem}.html")
    build_book_index(args.book, lectures, out_dir)
    print(f"\n브라우저에서 열기:\n  file://{out_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
