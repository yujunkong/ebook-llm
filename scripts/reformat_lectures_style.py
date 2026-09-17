#!/usr/bin/env python3
"""120강 Markdown을 Sigil EPUB 공식 스타일로 정규화한다.

주요 변환:
1. H1을 `# 제N강. 제목`으로 통일 (권 제목 H1 제거)
2. 기존 ### → ##, #### → ### (최대 4단)
3. \\( \\) 인라인 수식 → $...$
4. `이번 강의에서 배울 것` → `> **학습 목표**` 블록
5. `복습 문제` → `연습 문제` 제목 통일
6. LECTURE_NAV 블록은 유지
7. 학습 목표 다음에 `---` 구분선 보장

사용:
  python3 scripts/reformat_lectures_style.py
  python3 scripts/reformat_lectures_style.py --book 1
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOKS = {
    1: ROOT / "books" / "01_python_tensor_math_pytorch",
    2: ROOT / "books" / "02_tokenizer_transformer",
    3: ROOT / "books" / "03_gpt_pretraining_sft",
    4: ROOT / "books" / "04_rlhf_ppo_grpo",
    5: ROOT / "books" / "05_vllm_glm_dgx",
}

NAV_START = "<!-- LECTURE_NAV -->"
NAV_END = "<!-- /LECTURE_NAV -->"
TITLE_RE = re.compile(r"^##\s*제(\d+)강\.\s*(.+?)\s*$", re.M)
TITLE_H1_RE = re.compile(r"^#\s*제(\d+)강\.\s*(.+?)\s*$", re.M)
NUM_RE = re.compile(r"^(\d+)강_")


def split_nav(text: str) -> tuple[str, str]:
    if NAV_START in text and NAV_END in text:
        before, rest = text.split(NAV_START, 1)
        nav_body, after = rest.split(NAV_END, 1)
        nav = f"{NAV_START}{nav_body}{NAV_END}\n"
        return before.rstrip() + "\n", nav + after.lstrip("\n")
    return text.rstrip() + "\n", ""


def protect_fences(text: str) -> tuple[str, list[str]]:
    chunks: list[str] = []

    def repl(m: re.Match[str]) -> str:
        chunks.append(m.group(0))
        return f"@@@FENCE{len(chunks)-1}@@@"

    return re.sub(r"```[\s\S]*?```", repl, text), chunks


def restore_fences(text: str, chunks: list[str]) -> str:
    for i, chunk in enumerate(chunks):
        text = text.replace(f"@@@FENCE{i}@@@", chunk)
    return text


def convert_inline_math(text: str) -> str:
    """\\( ... \\) → $...$ (코드 펜스 외부만)."""
    text, fences = protect_fences(text)

    def repl(m: re.Match[str]) -> str:
        inner = m.group(1)
        # trim only outer spaces that were stylistic
        return f"${inner}$"

    text = re.sub(r"\\\((.+?)\\\)", repl, text, flags=re.S)
    return restore_fences(text, fences)


def shift_headings(text: str) -> str:
    """강의가 H1이 된 뒤, 기존 ###/####를 ##/###로 한 단계 올린다.

    이미 `# 제N강`만 있는 파일은 ###가 본문일 수 있으므로
    '제N강' H1 이후 구간에만 적용한다.
    """
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    past_title = False
    in_fence = False
    for line in lines:
        if line.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if not in_fence and TITLE_H1_RE.match(line):
            past_title = True
            out.append(line)
            continue
        if past_title and not in_fence:
            if line.startswith("###### "):
                line = "#### " + line[7:]
            elif line.startswith("##### "):
                line = "#### " + line[6:]
            elif line.startswith("#### "):
                line = "### " + line[5:]
            elif line.startswith("### "):
                line = "## " + line[4:]
            # avoid promoting leftover book-style ## that isn't 제N강
            elif line.startswith("## ") and not line.startswith("## 제"):
                pass
        out.append(line)
    return "".join(out)


def extract_title(text: str, path: Path) -> tuple[int, str]:
    m = TITLE_H1_RE.search(text) or TITLE_RE.search(text)
    if m:
        return int(m.group(1)), m.group(2).strip()
    num = int(NUM_RE.match(path.name).group(1))
    title = re.sub(r"^\d+강_", "", path.stem).replace("_", " ")
    return num, title


def extract_goals(text: str) -> list[str]:
    """'배울 것' 섹션의 불릿을 학습 목표로 추출."""
    # match ## or ### section
    m = re.search(
        r"^#{2,3}\s*\d*\.?\s*이번 강의에서 배울 것\s*$([\s\S]*?)(?=^#{2,3}\s|\Z)",
        text,
        re.M,
    )
    if not m:
        m = re.search(
            r"^#{2,3}\s*.*배울 것.*\s*$([\s\S]*?)(?=^#{2,3}\s|\Z)",
            text,
            re.M,
        )
    if not m:
        return []
    body = m.group(1)
    bullets = re.findall(r"^\s*[-*]\s+(.+)$", body, re.M)
    # keep short meaningful goals
    goals = []
    for b in bullets:
        b = re.sub(r"\*\*(.+?)\*\*", r"\1", b)
        b = re.sub(r"\\\((.+?)\\\)", r"$\1$", b)
        goals.append(b.strip())
        if len(goals) >= 6:
            break
    return goals


def strip_old_title_blocks(text: str) -> str:
    """권 H1 / 제N강 H2를 제거하고 본문만 남긴다."""
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    # skip leading blank
    while i < len(lines) and not lines[i].strip():
        i += 1
    # skip book H1 if present
    if i < len(lines) and lines[i].startswith("# ") and not TITLE_H1_RE.match(lines[i]):
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
    # skip lecture H2 제N강
    if i < len(lines) and TITLE_RE.match(lines[i]):
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
    # if already H1 제N강, skip it (will rewrite)
    if i < len(lines) and TITLE_H1_RE.match(lines[i]):
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
    out.extend(lines[i:])
    return "".join(out)


def remove_goals_section(text: str) -> str:
    """본문에서 '이번 강의에서 배울 것' 섹션을 제거(학습 목표로 승격했으므로)."""
    return re.sub(
        r"^#{2,3}\s*\d*\.?\s*이번 강의에서 배울 것\s*\n[\s\S]*?(?=^#{2,3}\s|\Z)",
        "",
        text,
        count=1,
        flags=re.M,
    )


def rename_sections(text: str) -> str:
    text = re.sub(
        r"^(#{2,3}\s*\d*\.?\s*)복습 문제\s*$",
        r"\1연습 문제",
        text,
        flags=re.M,
    )
    text = re.sub(
        r"^(#{2,3}\s*)핵심 용어\s*$",
        r"\1용어 정리",
        text,
        flags=re.M,
    )
    return text


def ensure_blank_around_display_math(text: str) -> str:
    text, fences = protect_fences(text)
    # ensure newline before/after $$ blocks
    text = re.sub(r"([^\n])\n\$\$", r"\1\n\n$$", text)
    text = re.sub(r"\$\$\n([^\n])", r"$$\n\n\1", text)
    return restore_fences(text, fences)


def renumber_top_sections(text: str) -> str:
    """본문 최상위 ## N. 절을 1부터 연속 번호로 맞춘다."""
    if NAV_START in text:
        body, nav = text.split(NAV_START, 1)
        nav = NAV_START + nav
    else:
        body, nav = text, ""

    h2 = re.compile(r"^(##\s+)(\d+)(\.\s+)(.*)$")
    lines = body.splitlines(keepends=True)
    out: list[str] = []
    in_fence = False
    n = 0
    for line in lines:
        if line.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        m = h2.match(line.rstrip("\n")) if not in_fence else None
        if m:
            n += 1
            new = f"{m.group(1)}{n}{m.group(3)}{m.group(4)}"
            out.append(new + ("\n" if line.endswith("\n") else ""))
        else:
            out.append(line)
    body = "".join(out)
    return body.rstrip() + ("\n\n" + nav if nav else "\n")


def renumber_subsections(text: str) -> str:
    """### a.b 형태를 현재 상위 ## a 번호에 맞게 재부여한다."""
    if NAV_START in text:
        body, nav = text.split(NAV_START, 1)
        nav = NAV_START + nav
    else:
        body, nav = text, ""

    h2 = re.compile(r"^##\s+(\d+)\.\s+")
    h3 = re.compile(r"^(###\s+)(\d+)(\.)(\d+)(\s+)(.*)$")
    lines = body.splitlines(keepends=True)
    out: list[str] = []
    in_fence = False
    cur = None
    idx = 0
    for line in lines:
        if line.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if not in_fence:
            m2 = h2.match(line)
            if m2:
                cur = int(m2.group(1))
                idx = 0
                out.append(line)
                continue
            m3 = h3.match(line.rstrip("\n"))
            if m3 and cur is not None:
                idx += 1
                new = f"{m3.group(1)}{cur}.{idx}{m3.group(5)}{m3.group(6)}"
                out.append(new + ("\n" if line.endswith("\n") else ""))
                continue
        out.append(line)
    body = "".join(out)
    return body.rstrip() + ("\n\n" + nav if nav else "\n")


def build_header(num: int, title: str, goals: list[str]) -> str:
    lines = [f"# 제{num}강. {title}", ""]
    lines.append("> **학습 목표**")
    if goals:
        for g in goals:
            lines.append(f"> - {g}")
    else:
        lines.append("> - 이번 강의의 핵심 개념을 이해한다")
        lines.append("> - 관련 수식·코드를 따라 계산·실행한다")
        lines.append("> - 실제 LLM에서의 쓰임과 연결한다")
    lines.extend(["", "---", ""])
    return "\n".join(lines)


def reformat_file(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    body, nav = split_nav(original)
    num, title = extract_title(body, path)
    goals = extract_goals(body)

    body = strip_old_title_blocks(body)
    body = remove_goals_section(body)
    body = convert_inline_math(body)
    body = rename_sections(body)

    # 임시로 본문에 H1 제목을 심고 heading shift
    tmp = f"# 제{num}강. {title}\n\n{body.lstrip()}"
    tmp = shift_headings(tmp)
    # shift 후 첫 H1 제거(헤더에서 다시 씀)
    tmp = TITLE_H1_RE.sub("", tmp, count=1).lstrip()

    # 학습 목표 블록이 본문 앞에 이미 있으면 제거
    tmp = re.sub(
        r"^>\s*\*\*학습 목표\*\*[\s\S]*?(?=\n---\n|\n##\s|\Z)",
        "",
        tmp,
        count=1,
    ).lstrip()

    header = build_header(num, title, goals)
    new_body = header + tmp
    new_body = ensure_blank_around_display_math(new_body)
    new_body = re.sub(r"\n{3,}", "\n\n", new_body)

    if nav:
        if not new_body.endswith("\n"):
            new_body += "\n"
        new_text = new_body.rstrip() + "\n\n" + nav.lstrip()
        if not new_text.endswith("\n"):
            new_text += "\n"
    else:
        new_text = new_body if new_body.endswith("\n") else new_body + "\n"

    new_text = renumber_top_sections(new_text)
    new_text = renumber_subsections(new_text)

    if new_text != original:
        path.write_text(new_text, encoding="utf-8")
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", type=int, choices=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    dirs = [BOOKS[args.book]] if args.book else list(BOOKS.values())
    changed = 0
    total = 0
    for d in dirs:
        for path in sorted(d.glob("*강*.md"), key=lambda p: int(NUM_RE.match(p.name).group(1))):
            total += 1
            if reformat_file(path):
                changed += 1
                print(f"[OK] {path.relative_to(ROOT)}")
            else:
                print(f"[SKIP] {path.relative_to(ROOT)}")
    print(f"done: changed {changed}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
