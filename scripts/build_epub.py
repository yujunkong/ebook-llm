#!/usr/bin/env python3
"""권 단위 Markdown을 EPUB3(MathML 수식)로 변환한다.

사용 예:
  python3 scripts/build_epub.py --book 1
  python3 scripts/build_epub.py --book all

pandoc이 Markdown의 $$…$$ / \\(…\\) 수식을 MathML로 넣어
Sigil·여러 EPUB 리더에서 수식으로 보이게 한다.
(리더마다 MathML 지원 차이는 있음 — 미리보기는 build_math_preview.py 권장)
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
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


def build_book(book_num: int) -> Path:
    out_name, book_dir, title = BOOKS[book_num]
    lectures = sorted(book_dir.glob("*강*.md"), key=lecture_sort_key)
    if not lectures:
        raise SystemExit(f"강의 파일이 없습니다: {book_dir}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / out_name

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # 권 제목 + 강의들을 하나의 pandoc 입력으로 합친다.
        combined = tmp_path / "combined.md"
        parts = [f"% {title}\n% 《밑바닥부터 LLM》\n\n"]
        for lec in lectures:
            parts.append(lec.read_text(encoding="utf-8"))
            parts.append("\n\n")
        combined.write_text("".join(parts), encoding="utf-8")

        # MathML로 수식 변환 (EPUB3)
        cmd = [
            "pandoc",
            str(combined),
            "-f",
            "markdown",
            "-t",
            "epub3",
            "--mathml",
            "--toc",
            "--toc-depth=2",
            "-o",
            str(out_path),
            "--metadata",
            f"title={title}",
            "--metadata",
            "lang=ko",
        ]
        subprocess.run(cmd, check=True)

    print(f"[OK] {out_path.relative_to(ROOT)}  ({len(lectures)} lectures)")
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
