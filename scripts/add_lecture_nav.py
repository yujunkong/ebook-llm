#!/usr/bin/env python3
"""모든 강의 Markdown 맨 아래에 이전 강 / 다음 강 링크를 넣거나 갱신한다.

멱등: <!-- LECTURE_NAV --> ... <!-- /LECTURE_NAV --> 블록을 교체한다.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOKS = [
    ROOT / "books" / "01_python_tensor_math_pytorch",
    ROOT / "books" / "02_tokenizer_transformer",
    ROOT / "books" / "03_gpt_pretraining_sft",
    ROOT / "books" / "04_rlhf_ppo_grpo",
    ROOT / "books" / "05_vllm_glm_dgx",
]

NAV_START = "<!-- LECTURE_NAV -->"
NAV_END = "<!-- /LECTURE_NAV -->"
TITLE_RE = re.compile(r"^#{1,2}\s*제(\d+)강\.\s*(.+?)\s*$", re.M)
NUM_RE = re.compile(r"^(\d+)강_")


def lecture_num(path: Path) -> int:
    m = NUM_RE.match(path.name)
    if not m:
        raise ValueError(path.name)
    return int(m.group(1))


def lecture_title(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    m = TITLE_RE.search(text)
    if m:
        return m.group(2).strip()
    # fallback: filename
    stem = path.stem
    return re.sub(r"^\d+강_", "", stem).replace("_", " ")


def rel_link(from_path: Path, to_path: Path) -> str:
    """같은 권이면 파일명만, 다른 권이면 상대경로."""
    if from_path.parent == to_path.parent:
        return to_path.name
    return Path(os_path_rel(from_path.parent, to_path)).as_posix()


def os_path_rel(start: Path, target: Path) -> Path:
    return Path(os_relpath(str(target), str(start)))


def os_relpath(target: str, start: str) -> str:
    import os

    return os.path.relpath(target, start)


def build_nav(prev: Path | None, curr: Path, nxt: Path | None) -> str:
    lines = [
        "",
        NAV_START,
        "",
        "---",
        "",
        "### 강의 이동",
        "",
    ]
    if prev is None:
        lines.append("- **이전 강:** 없음 (시리즈 시작)")
    else:
        title = lecture_title(prev)
        num = lecture_num(prev)
        href = rel_link(curr, prev)
        lines.append(f"- **이전 강:** [제{num}강. {title}]({href})")

    if nxt is None:
        lines.append("- **다음 강:** 없음 (시리즈 끝)")
    else:
        title = lecture_title(nxt)
        num = lecture_num(nxt)
        href = rel_link(curr, nxt)
        lines.append(f"- **다음 강:** [제{num}강. {title}]({href})")

    lines.extend(["", NAV_END, ""])
    return "\n".join(lines)


def strip_existing_nav(text: str) -> str:
    pattern = re.compile(
        re.escape(NAV_START) + r".*?" + re.escape(NAV_END) + r"\n?",
        re.S,
    )
    text = pattern.sub("", text)
    return text.rstrip() + "\n"


def main() -> None:
    lectures: list[Path] = []
    for book in BOOKS:
        lectures.extend(sorted(book.glob("*강*.md"), key=lecture_num))
    lectures.sort(key=lecture_num)

    if len(lectures) != 120:
        print(f"[WARN] expected 120 lectures, found {len(lectures)}")

    updated = 0
    for i, path in enumerate(lectures):
        prev = lectures[i - 1] if i > 0 else None
        nxt = lectures[i + 1] if i + 1 < len(lectures) else None
        nav = build_nav(prev, path, nxt)
        original = path.read_text(encoding="utf-8")
        base = strip_existing_nav(original)
        new_text = base.rstrip() + "\n" + nav
        if new_text != original:
            path.write_text(new_text, encoding="utf-8")
            updated += 1
            print(f"[OK] {path.relative_to(ROOT)}")

    print(f"done: updated {updated} / {len(lectures)}")


if __name__ == "__main__":
    main()
