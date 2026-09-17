#!/usr/bin/env python3
"""권 단위 Markdown을 EPUB3(MathML 수식)로 변환한다.

사용 예:
  python3 scripts/build_epub.py --book 1
  python3 scripts/build_epub.py --book all

주의:
- 원고의 `---` 구분선은 Pandoc이 YAML로 오인할 수 있어
  빌드 직전에 `* * *` thematic break로 치환한다.
- Apple Books는 Pandoc이 넣는 `.svgz`(실제로는 비압축 SVG)를
  깨뜨리는 경우가 많아, 가능하면 PNG를 쓰고 EPUB 사후처리로
  `.svgz` → `.svg` 이름도 교정한다.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "epub"

BOOKS = {
    1: ("01_밑바닥부터_LLM_1권.epub", ROOT / "books" / "01_python_tensor_math_pytorch", "1권. Python · Tensor · 수학 · PyTorch"),
    2: ("02_밑바닥부터_LLM_2권.epub", ROOT / "books" / "02_tokenizer_transformer", "2권. Tokenizer와 Transformer"),
    3: ("03_밑바닥부터_LLM_3권.epub", ROOT / "books" / "03_gpt_pretraining_sft", "3권. GPT Pretraining과 SFT"),
    4: ("04_밑바닥부터_LLM_4권.epub", ROOT / "books" / "04_rlhf_ppo_grpo", "4권. RLHF · PPO · GRPO"),
    5: ("05_밑바닥부터_LLM_5권.epub", ROOT / "books" / "05_vllm_glm_dgx", "5권. vLLM · GLM · DGX Spark"),
}


def lecture_sort_key(path: Path) -> int:
    m = re.match(r"(\d+)", path.name)
    return int(m.group(1)) if m else 0


def preprocess_markdown(text: str, book_dir: Path) -> str:
    """Pandoc/Apple Books 호환을 위한 전처리."""
    fences: list[str] = []

    def save_fence(m: re.Match[str]) -> str:
        fences.append(m.group(0))
        return f"@@@FENCE{len(fences)-1}@@@"

    text = re.sub(r"```[\s\S]*?```", save_fence, text)

    # YAML로 오인되는 --- 구분선 → thematic break
    text = re.sub(r"(?m)^---\s*$", "* * *", text)

    # Apple Books: SVG보다 PNG가 안정적. 동일 stem PNG가 있으면 교체
    def prefer_png(m: re.Match[str]) -> str:
        rel = m.group(1)
        png = book_dir / Path(rel).with_suffix(".png")
        if png.is_file():
            return f"]({Path(rel).with_suffix('.png').as_posix()})"
        return m.group(0)

    text = re.sub(r"\]\((images/[^)]+\.svg)\)", prefer_png, text)

    for i, fence in enumerate(fences):
        text = text.replace(f"@@@FENCE{i}@@@", fence)
    return text


def fix_epub_media(epub_path: Path) -> None:
    """Pandoc `.svgz` 확장자를 Apple Books가 읽는 `.svg`로 고친다.

    Pandoc 3.x는 비압축 SVG를 `.svgz` 이름으로 넣는 경우가 있다.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(epub_path, "r") as zin:
            zin.extractall(tmp_path)

        renamed: dict[str, str] = {}
        media_dir = tmp_path / "EPUB" / "media"
        if media_dir.is_dir():
            for path in list(media_dir.iterdir()):
                if path.suffix.lower() == ".svgz":
                    new_path = path.with_suffix(".svg")
                    # 이미 비압축 SVG인 경우 그대로 이름만 변경
                    path.rename(new_path)
                    renamed[path.name] = new_path.name

        if renamed:
            for path in tmp_path.rglob("*"):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in {".xhtml", ".html", ".opf", ".ncx", ".css", ".xml"}:
                    continue
                raw = path.read_bytes()
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                original = text
                for old, new in renamed.items():
                    text = text.replace(old, new)
                if text != original:
                    path.write_text(text, encoding="utf-8")

        # EPUB은 mimetype이 첫 엔트리·무압축이어야 함
        out_tmp = tmp_path / "out.epub"
        with zipfile.ZipFile(out_tmp, "w") as zout:
            mimetype = tmp_path / "mimetype"
            if mimetype.is_file():
                zout.write(mimetype, "mimetype", compress_type=zipfile.ZIP_STORED)
            for path in sorted(tmp_path.rglob("*")):
                if not path.is_file() or path.name == "out.epub":
                    continue
                arc = path.relative_to(tmp_path).as_posix()
                if arc == "mimetype":
                    continue
                zout.write(path, arc, compress_type=zipfile.ZIP_DEFLATED)
        out_tmp.replace(epub_path)


def build_book(book_num: int) -> Path:
    out_name, book_dir, title = BOOKS[book_num]
    lectures = sorted(book_dir.glob("*강*.md"), key=lecture_sort_key)
    if not lectures:
        raise SystemExit(f"강의 파일이 없습니다: {book_dir}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / out_name

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        combined = tmp_path / "combined.md"
        parts: list[str] = []
        for lec in lectures:
            body = preprocess_markdown(lec.read_text(encoding="utf-8"), book_dir)
            parts.append(body.rstrip() + "\n\n")
        combined.write_text("".join(parts), encoding="utf-8")

        cmd = [
            "pandoc",
            str(combined),
            "-f",
            "markdown-yaml_metadata_block",
            "-t",
            "epub3",
            "--mathml",
            "--toc",
            "--toc-depth=2",
            "--split-level=1",
            f"--resource-path={book_dir}",
            "-o",
            str(out_path),
            "--metadata",
            f"title={title}",
            "--metadata",
            "author=밑바닥부터 LLM",
            "--metadata",
            "lang=ko",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            raise SystemExit(f"pandoc failed for book {book_num} (exit {proc.returncode})")
        if proc.stderr.strip():
            for line in proc.stderr.splitlines():
                if line.strip():
                    print(f"  [warn] {line}")

    fix_epub_media(out_path)

    size_kb = out_path.stat().st_size / 1024
    print(f"[OK] {out_path.relative_to(ROOT)}  ({len(lectures)} lectures, {size_kb:.0f} KB)")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", required=True, help="1~5 또는 all")
    args = parser.parse_args()

    if args.book == "all":
        for n in range(1, 6):
            build_book(n)
    else:
        build_book(int(args.book))
    return 0


if __name__ == "__main__":
    sys.exit(main())
