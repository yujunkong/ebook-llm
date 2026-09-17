#!/usr/bin/env python3
"""권 단위 Markdown을 EPUB3(MathML 수식)로 변환한다.

사용 예:
  python3 scripts/build_epub.py --book 1
  python3 scripts/build_epub.py --book all

주의: 원고의 `---` 구분선은 Pandoc이 YAML로 오인할 수 있어
빌드 직전에 `* * *` thematic break로 치환한다.
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


def preprocess_markdown(text: str) -> str:
    """Pandoc 호환을 위한 전처리."""
    # 코드 펜스 보호
    fences: list[str] = []

    def save_fence(m: re.Match[str]) -> str:
        fences.append(m.group(0))
        return f"@@@FENCE{len(fences)-1}@@@"

    text = re.sub(r"```[\s\S]*?```", save_fence, text)

    # YAML로 오인되는 --- 구분선 → thematic break
    text = re.sub(r"(?m)^---\s*$", "* * *", text)

    # HTML 주석은 유지(네비 마커). Pandoc이 그대로 둘 수 있음.

    for i, fence in enumerate(fences):
        text = text.replace(f"@@@FENCE{i}@@@", fence)
    return text


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
            body = preprocess_markdown(lec.read_text(encoding="utf-8"))
            parts.append(body.rstrip() + "\n\n")
        combined.write_text("".join(parts), encoding="utf-8")

        # yaml_metadata_block 비활성: 본문 --- 잔여 오인 방지
        # smart typography는 수식/코드에 영향 줄 수 있어 기본만 사용
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
            f"--resource-path={book_dir}",  # images/ 상대경로 해석
            "-o",
            str(out_path),
            "--metadata",
            f"title={title}",
            "--metadata",
            "author=밑바닥부터 LLM",
            "--metadata",
            "lang=ko",  # ko-KR은 pandoc translations/ko.yaml 파싱 경고 유발
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            raise SystemExit(f"pandoc failed for book {book_num} (exit {proc.returncode})")
        if proc.stderr.strip():
            # warnings only
            for line in proc.stderr.splitlines():
                if line.strip():
                    print(f"  [warn] {line}")

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
