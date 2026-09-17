#!/usr/bin/env python3
"""강의 Markdown의 기본 구조를 점검한다 (Apple Books 가이드 기준)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOKS = ROOT / "books"

# 신규 템플릿 + 기존 원고 호환
REQUIRED_HINTS = [
    r"이번 강에서 배우는 내용|학습 목표",
    r"핵심 요약|핵심 정리",
    r"연습문제|연습 문제|복습 문제",
    r"LECTURE_NAV|다음 강의와 연결|강의 이동",
]

H1_RE = re.compile(r"^# (?:제)?(\d+)강\.\s+", re.M)


def check_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    issues: list[str] = []
    if not H1_RE.search(text):
        issues.append("강의 H1 제목(# N강. / # 제N강.) 없음")
    first_h1 = re.search(r"^# (.+)$", text, re.M)
    if first_h1 and not re.match(r"(?:제)?\d+강\.", first_h1.group(1)):
        issues.append(f"H1이 강의 제목이 아님: {first_h1.group(1)[:40]}")
    for hint in REQUIRED_HINTS:
        if not re.search(hint, text):
            issues.append(f"권장 섹션 누락: {hint}")
    if "좋아!" in text or "알겠어!" in text:
        issues.append("대화체 표현 감지")
    body = re.sub(r"```[\s\S]*?```", "", text)
    if re.search(r"\\\(|\\\)", body):
        issues.append("인라인 수식 \\( \\) 잔존 ( $...$ 로 변환 필요)")
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
        if issues:
            failed += 1
            rel = path.relative_to(ROOT)
            print(f"[FAIL] {rel}")
            for issue in issues:
                print(f"  - {issue}")
    print(f"검사 {len(lecture_files)}개, 문제 {failed}개")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
