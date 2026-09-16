#!/usr/bin/env python3
"""강의 Markdown 파일의 기본 구조를 점검하는 초안 스크립트."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOKS = ROOT / "books"

REQUIRED_HINTS = [
    r"이번 강의에서 배울 것",
    r"핵심 정리",
    r"복습 문제",
    r"다음 강의와 연결",
]


def check_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    issues: list[str] = []
    if not re.search(r"^## 제\d+강", text, re.M):
        issues.append("강의 H2 제목(제N강) 없음")
    for hint in REQUIRED_HINTS:
        if not re.search(hint, text):
            issues.append(f"권장 섹션 누락: {hint}")
    if "좋아!" in text or "알겠어!" in text:
        issues.append("대화체 표현 감지")
    return issues


def main() -> int:
    files = sorted(BOOKS.rglob("*.md"))
    lecture_files = [p for p in files if re.search(r"\d+강_", p.name)]
    if not lecture_files:
        print("검사할 강의 파일이 없습니다.")
        return 0

    failed = 0
    for path in lecture_files:
        issues = check_file(path)
        rel = path.relative_to(ROOT)
        if issues:
            failed += 1
            print(f"[FAIL] {rel}")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"[OK] {rel}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
