# 《밑바닥부터 LLM》 Markdown 작성 규칙

이 문서는 120강 전체 전자책의 **공식 Markdown 스타일 가이드**이다.

**최종 목표 매체는 Apple Books(EPUB3)** 이다. 상세 제작 원칙은 [APPLE_BOOKS_GUIDE.md](APPLE_BOOKS_GUIDE.md)를 따른다.

목표:

1. Apple Books(iPhone · iPad · Mac)에서 읽기 좋다.
2. GitHub에서도 읽기 좋다.
3. MathJax/KaTeX · Pandoc MathML 수식이 정상 렌더링된다.
4. Markdown 원본만으로 EPUB를 재현할 수 있다 (Sigil/HTML 의존 최소화).

저장소 경로:

```text
books/
  01_python_tensor_math_pytorch/
    NN강_제목.md
    images/figNN-01.svg
  02_tokenizer_transformer/
  03_gpt_pretraining_sft/
  04_rlhf_ppo_grpo/
  05_vllm_glm_dgx/
```

---

## 1. 문서 기본 구조 (모든 강의)

```markdown
# 10강. 선형대수 기초 — 내적과 행렬곱

## 이번 강에서 배우는 내용

- …

## 왜 중요한가?

…

## 개념 …

## NumPy 실습 / PyTorch 실습

## LLM에서는 어디에 사용될까?

## 핵심 요약

## 용어 사전

## 연습문제

<!-- LECTURE_NAV -->
```

| 순서 | 내용 |
|---:|---|
| 1 | `# N강. 강의 제목` (파일당 H1 하나) |
| 2 | 이번 강에서 배우는 내용 |
| 3 | 왜 중요한가? |
| 4 | 개념 · 수식 · 그림 · 실습 |
| 5 | LLM 연결 |
| 6 | 핵심 요약 · 용어 사전 · 연습문제 |
| 7 | 강의 이동 링크 |

권 제목을 강의 H1로 쓰지 않는다.

---

## 2. 제목 규칙

| Heading | 용도 |
|---|---|
| `#` | 강의 제목 (`N강. …` / 호환 `제N강. …`) |
| `##` | 큰 챕터 |
| `###` | 소주제 · 실습 제목 |
| `####` | 참고 / 주의 / 심화 |

**최대 4단계**까지.

---

## 3. 문단 · 강조

의미 단위로 빈 줄을 둔다. 2~4문장마다 줄바꿈.

| 용도 | 표현 |
|---|---|
| 핵심 용어 | **Attention** |
| 주의 | `> ⚠️ **주의**` |
| 팁 | `> 💡 **팁**` |
| 핵심 | `> **핵심**` |
| 심화 | `> 📘 **심화**` |

---

## 4. 수식

- 인라인: `$L=f(x)$` (`\(...\)` 금지)
- 블록: `$$` 앞뒤 빈 줄
- 행렬 행 구분: `\\`
- 식 번호 생략. 문장으로 참조

상세: [MATH_RENDERING.md](MATH_RENDERING.md)

---

## 5. 코드

언어 지정 필수. 예제당 약 10~25줄. 코드 아래 짧은 설명.

---

## 6. 표 · 이미지

- 표: Markdown만, **4열 이하** 권장
- 이미지: 권별 `images/`, 캡션은 **위**에 `**그림 N-K. …**`
- 수학 도식은 **SVG 우선** (`fig10-01.svg`)

```markdown
**그림 10-1. 벡터의 방향과 내적**

![그림 10-1](images/fig10-01.svg)
```

---

## 7. 실습 · 요약 · 용어 · 문제

- 실습: `### 실습 10-1. …` (NumPy / PyTorch 최소 2개)
- 핵심 요약: 체크리스트 불릿
- 용어 사전: 표 10~20개
- 연습문제: 기초 · 계산 · 코드 · 응용

---

## 8. 강의 이동 링크

`scripts/add_lecture_nav.py`가 관리하는 `<!-- LECTURE_NAV -->` 블록.

---

## 9. EPUB / Pandoc

| 항목 | 규칙 |
|---|---|
| 수식 | `$…$`, `$$…$$` |
| HTML/CSS | 최소화 (Sigil 전용 금지) |
| 이미지 | 상대경로 `images/…` |
| 코드 | 언어 지정 |
| 표 | Markdown |

```bash
python3 scripts/build_epub.py --book all
python3 scripts/build_math_preview.py --book 1
```

---

## 10. 파일명

`01강_제목.md` … `120강_….md` (두 자리 번호 + `강_` + 제목, 공백 없음)

---

## 11. 권 구성 (고정)

| 권 | 강의 | 주제 |
|---|---|---|
| 1권 | 1~26 | Python · Tensor · 수학 · PyTorch |
| 2권 | 27~54 | Tokenizer와 Transformer |
| 3권 | 55~78 | GPT Pretraining · SFT |
| 4권 | 79~98 | RLHF · DPO · GRPO · RL |
| 5권 | 99~120 | vLLM · KV Cache · DGX Spark 등 |

120강 · 5권 구조는 임의 변경하지 않는다.
