#!/usr/bin/env python3
"""120강 Markdown을 Apple Books 가이드 구조로 일괄 정규화한다.

적용 항목:
- H1: `# 제N강.` → `# N강.`
- 학습 목표 인용 블록 → `## 이번 강에서 배우는 내용`
- 표준 절 제목 통일 (왜 중요한가 / 선수 개념 / LLM 연결 / 핵심 요약 / 용어 사전 / 연습문제)
- 표준 H2의 앞번호(`## 12. …`) 제거
- LLM 연결 절이 없으면 핵심 요약 앞에 최소 절 삽입
- 문체·본문 내용은 건드리지 않음 (10·11강 내용 보강은 별도)
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOKS = ROOT / "books"

H1_RE = re.compile(r"^# 제(\d+)강\.\s*(.+)\s*$", re.M)

GOAL_BLOCK_RE = re.compile(
    r"^> \*\*학습 목표\*\*\s*\n(?P<body>(?:^>.*\n)+)",
    re.M,
)

# (패턴, 새 제목) — 번호 유무 모두 매칭
SECTION_RENAMES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^## (?:\d+\.\s*)?왜 이것을 배우는가\s*$", re.M), "## 왜 중요한가?"),
    (re.compile(r"^## (?:\d+\.\s*)?왜 총정리가 필요한가\s*$", re.M), "## 왜 중요한가?"),
    (re.compile(r"^## (?:\d+\.\s*)?먼저 알아야 할 개념(?:\s*\(.*\))?\s*$", re.M), "## 선수 개념"),
    (
        re.compile(
            r"^## (?:\d+\.\s*)?(?:실제 LLM에서는 어떻게 사용하는가|LLM Serving 연결|LLM 연결)\s*$",
            re.M,
        ),
        "## LLM에서는 어디에 사용될까?",
    ),
    (re.compile(r"^## (?:\d+\.\s*)?핵심 정리\s*$", re.M), "## 핵심 요약"),
    (re.compile(r"^## (?:\d+\.\s*)?핵심 용어\s*$", re.M), "## 용어 사전"),
    (re.compile(r"^## (?:\d+\.\s*)?연습 문제\s*$", re.M), "## 연습문제"),
    (re.compile(r"^## (?:\d+\.\s*)?다음 강의와 연결\s*$", re.M), "## 다음 강의와 연결"),
    (re.compile(r"^## (?:\d+\.\s*)?자주 하는 실수\s*$", re.M), "## 자주 하는 실수"),
    (re.compile(r"^## (?:\d+\.\s*)?실습\s*$", re.M), "## 실습"),
    (re.compile(r"^## (?:\d+\.\s*)?정답 및 해설\s*$", re.M), "## 정답 및 해설"),
    (re.compile(r"^## (?:\d+\.\s*)?핵심 개념 설명\s*$", re.M), "## 핵심 개념"),
    (re.compile(r"^## (?:\d+\.\s*)?직관적으로 이해하기\s*$", re.M), "## 직관적으로 이해하기"),
    (re.compile(r"^## (?:\d+\.\s*)?수학적으로 이해하기\s*$", re.M), "## 수학적으로 이해하기"),
    (re.compile(r"^## (?:\d+\.\s*)?작은 숫자로 직접 계산하기(?:（.*）|\(.*\))?\s*$", re.M), "## 작은 숫자로 직접 계산하기"),
    (re.compile(r"^## (?:\d+\.\s*)?코드로 구현하기(?:\s*\(.*\))?\s*$", re.M), "## 코드로 구현하기"),
    (re.compile(r"^## (?:\d+\.\s*)?PyTorch로 구현하기\s*$", re.M), "## PyTorch로 구현하기"),
]

STANDARD_UNNUMBER = {
    "왜 중요한가?",
    "선수 개념",
    "핵심 개념",
    "직관적으로 이해하기",
    "수학적으로 이해하기",
    "작은 숫자로 직접 계산하기",
    "코드로 구현하기",
    "PyTorch로 구현하기",
    "실습",
    "자주 하는 실수",
    "LLM에서는 어디에 사용될까?",
    "핵심 요약",
    "용어 사전",
    "연습문제",
    "정답 및 해설",
    "다음 강의와 연결",
    "이번 강에서 배우는 내용",
}


def convert_goals(text: str) -> str:
    m = GOAL_BLOCK_RE.search(text)
    if not m:
        return text
    lines = []
    for raw in m.group("body").splitlines():
        line = raw.lstrip(">").strip()
        if not line:
            continue
        if line.startswith("-"):
            lines.append(line)
        else:
            lines.append(f"- {line}")
    block = "## 이번 강에서 배우는 내용\n\n" + "\n".join(lines) + "\n"
    text = text[: m.start()] + block + text[m.end() :]
    # 학습 목표 직후 --- 제거
    text = re.sub(r"(## 이번 강에서 배우는 내용\n(?:.*\n)*?)\n---\s*\n", r"\1\n", text, count=1)
    return text


def rename_sections(text: str) -> str:
    for pat, repl in SECTION_RENAMES:
        text = pat.sub(repl, text)
    return text


def strip_standard_h2_numbers(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        title = m.group(1).strip()
        if title in STANDARD_UNNUMBER or title.startswith("LLM에서는"):
            return f"## {title}"
        # 그 외 번호 붙은 H2도 Apple Books에서는 서술형 제목 권장 → 번호만 제거
        return f"## {title}"

    return re.sub(r"^## \d+\.\s*(.+?)\s*$", repl, text, flags=re.M)


def ensure_llm_section(text: str, lecture_num: int) -> str:
    if "LLM에서는 어디에 사용될까?" in text:
        return text
    # 핵심 요약 앞에 삽입
    stub = (
        "## LLM에서는 어디에 사용될까?\n\n"
        f"이번 {lecture_num}강에서 배운 개념은 이후 Transformer · GPT · 서빙 강의에서 "
        "반복해서 등장합니다. 각 수식·코드 블록을 “실제 모델의 어느 단계인가”와 연결해 다시 읽어 보세요.\n\n"
    )
    if "## 핵심 요약" in text:
        return text.replace("## 핵심 요약", stub + "## 핵심 요약", 1)
    if "## 용어 사전" in text:
        return text.replace("## 용어 사전", stub + "## 용어 사전", 1)
    return text


def convert_h1(text: str) -> str:
    return H1_RE.sub(r"# \1강. \2", text)


def process_file(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    m = re.match(r"(\d+)강_", path.name)
    num = int(m.group(1)) if m else 0

    text = original
    text = convert_h1(text)
    text = convert_goals(text)
    text = rename_sections(text)
    text = strip_standard_h2_numbers(text)
    text = ensure_llm_section(text, num)
    # 연속 빈 줄 정리
    text = re.sub(r"\n{3,}", "\n\n", text)

    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> int:
    files = sorted(BOOKS.rglob("*강*.md"))
    updated = 0
    for path in files:
        if process_file(path):
            updated += 1
            print(f"[OK] {path.relative_to(ROOT)}")
    print(f"done: updated {updated} / {len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
