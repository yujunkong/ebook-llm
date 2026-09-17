# Apple Books 전자책 제작 가이드라인 (ebook-llm 최종본)

대상: ebook-llm 프로젝트 (120강 LLM 교재)

목표: Apple Books(iPhone, iPad, Mac)에서 가장 읽기 좋은 EPUB 제작.

원칙: **Markdown 원본 하나로** Apple Books EPUB 생성. Sigil 전용 문법·HTML 의존 최소화.

관련 문서: [MARKDOWN_STYLE_GUIDE.md](MARKDOWN_STYLE_GUIDE.md) · [MATH_RENDERING.md](MATH_RENDERING.md) · [PROJECT_GUIDELINES.md](PROJECT_GUIDELINES.md)

---

## 1. 전자책 전체 원칙

### 책 구성

| 단위 | 규칙 |
|---|---|
| 권(Book) | 1권, 2권, 3권, 4권, 5권 |
| 장(Lesson) | 1강 = 하나의 Markdown 파일 |
| 챕터 | `##`, `###` Heading 사용 |
| 이미지 | PNG 또는 SVG (`images/`) |
| 수식 | LaTeX 블록(`$$`) · 인라인(`$`) |
| 코드 | Markdown fenced code block |

### 파일 구조

```text
books/
 ├── 01_python_tensor_math_pytorch/
 │    ├── 01강_....md
 │    ├── 10강_선형대수_기초_내적과_행렬곱.md
 │    ├── images/
 │    │     fig10-01.svg
 │    │     fig10-02.svg
 │    └── README.md
 ├── 02_tokenizer_transformer/
 ...
```

규칙:

- 이미지 전용 `images/` 폴더 (권 단위).
- 한 강에서 쓰는 이미지는 `fig10-01.png`처럼 **강번호-순번**.
- 파일명은 UTF-8 한글 유지 가능.

---

## 2. Markdown 작성 규칙

### 제목 계층

```markdown
# 10강. 선형대수 기초 — 내적과 행렬곱

## 이번 강에서 배우는 내용

### 벡터란 무엇인가?

#### 참고
```

| Heading | 용도 |
|---|---|
| `#` | 강 제목 (파일당 1개). 형식: `N강. 제목` |
| `##` | 큰 챕터 |
| `###` | 세부 개념 |
| `####` | 참고 / 주의 / 심화 |

Apple Books 목차는 Heading에서 자동 생성된다.

> 호환: 기존 원고의 `# 제N강.` 도 허용한다. **신규·보강 강의는 `# N강.` 형식을 쓴다.**

### 문단 규칙

좋은 예:

```markdown
벡터는 크기와 방향을 가진 수학적 객체입니다.

LLM에서는 토큰 하나가 하나의 벡터로 표현됩니다.

벡터끼리 계산하여 의미를 비교합니다.
```

권장:

- 2~4문장마다 빈 줄.
- 한 문단에 여러 개념을 섞지 않는다.
- 모바일 폭 기준 가독성 유지.

### 강조 규칙

| 용도 | 문법 |
|---|---|
| 중요 개념 | **Gradient** |
| 용어 | **Embedding** |
| 주의 | `> ⚠️ **주의**` |
| 팁 | `> 💡 **팁**` |
| 핵심 | `> **핵심**` |
| 심화 | `> 📘 **심화**` |

---

## 3. 수식 작성 규칙

### 인라인

```markdown
손실은 $L=(y-\hat{y})^2$ 로 씁니다.
```

### 블록

```markdown
$$
a \cdot b
=
\sum_{i=1}^{n} a_i b_i
$$
```

### 여러 줄

블록 안에서 `\\`로 줄을 나눈다.

```markdown
$$
y = Wx + b \\
z = \mathrm{ReLU}(y) \\
\hat{y} = \mathrm{Softmax}(z)
$$
```

### 번호

전자책에서는 **식 번호를 붙이지 않는다.** 문장으로 참조한다.

---

## 4. 코드 작성 규칙

- 언어를 반드시 지정한다 (`python`, `text`, `bash` …).
- 예제 하나당 약 10~25줄. 긴 코드는 블록을 나눈다.
- 코드 아래에 한두 문장 설명을 둔다.

````markdown
```python
x = torch.tensor([1.0, 2.0, 3.0])
```

`torch.tensor()`는 PyTorch Tensor 객체를 생성합니다.
````

---

## 5. 이미지 규칙 (Apple Books 최적화)

캡션은 **이미지 위**에 둔다.

```markdown
**그림 10-1. 벡터의 방향과 내적**

![그림 10-1](images/fig10-01.svg)
```

| 종류 | 권장 |
|---|---|
| 도식 | 폭 약 1200px 상당 / SVG 우선 |
| 그래프 | 폭 약 1600px / SVG 또는 PNG |
| 수학 도식 | **SVG 우선** (확대 선명 · 용량↓) |

이미지는 개념 설명 **직후**에 배치한다.

---

## 6. 표 작성 규칙

단순 Markdown 표만 사용한다. **4열 이하** 권장. 5열 이상은 표를 나눈다.

```markdown
| Shape | 의미 |
|-------|------|
| (3,) | 벡터 |
| (3, 4) | 행렬 |
```

---

## 7. 콜아웃(Callout) 규칙

```markdown
> **핵심**
>
> Attention은 Query와 Key의 내적입니다.

> 💡 **팁**
>
> 벡터 정규화는 코사인 유사도 계산 전에 자주 수행합니다.

> ⚠️ **주의**
>
> 행렬곱은 가운데 차원이 맞아야 합니다.

> 📘 **심화**
>
> 이 내용은 37강 Dot-Product Attention에서 다시 사용됩니다.
```

---

## 8. LLM 교재 연결 규칙

각 강에 **반드시** LLM 연결 절을 둔다.

```markdown
## LLM에서는 어디에 사용될까?

벡터는 토큰을 표현합니다.

Embedding Matrix에서 토큰은 고차원 벡터로 변환됩니다.

Attention은 벡터의 내적을 사용합니다.
```

| 강 | 연결 예시 |
|---|---|
| 10강 | Embedding, Attention ($QK^\top$) |
| 11강 | Gradient, Autograd |
| 20강 | Autograd |
| 31강 | Embedding |
| 37~40강 | Self-Attention |
| 55강 | GPT Decoder |

---

## 9. 실습 작성 규칙

### 실습 N-1. NumPy / ### 실습 N-2. PyTorch

구조: 목표 → 코드 → 결과 설명 → 배운 점.

```markdown
### 실습 10-1. 벡터 내적 계산

### 실습 10-2. 코사인 유사도 계산
```

NumPy + PyTorch를 **최소 2개** 이상 권장한다.

---

## 10. 용어 사전 · 핵심 요약 · 연습문제

### 용어 사전

강 말미. 표로 10~20개.

### 핵심 요약

체크리스트 불릿.

```markdown
## 핵심 요약

- 벡터는 방향과 크기를 가진다.
- 내적은 두 벡터의 유사도를 계산한다.
```

### 연습문제 (4단계)

1. 기초 (객관식/단답)
2. 계산
3. NumPy/코드
4. 응용 (LLM 연결)

---

## 11. 페이지 구성 템플릿 (모든 강)

```text
강 제목 (# N강. …)
│
├─ 이번 강에서 배우는 내용
├─ 왜 중요한가?
├─ 개념 1 …
├─ 개념 2 …
├─ 수학 공식 · 작은 숫자 예
├─ 그림
├─ NumPy 실습
├─ PyTorch 실습
├─ LLM에서는 어디에 사용될까?
├─ 핵심 요약
├─ 용어 사전
├─ 연습문제 (+ 정답)
└─ 강의 이동 (LECTURE_NAV)
```

---

## 12. Apple Books 최적화 체크리스트

### 반드시

- [ ] `#` / `##` / `###` Heading
- [ ] LaTeX `$` / `$$` 수식
- [ ] PNG/SVG 이미지 (있으면)
- [ ] 코드블록 언어 지정
- [ ] 짧은 문단
- [ ] 요약 · 용어사전 · 연습문제
- [ ] LLM 연결 절

### 사용하지 않음

- ❌ Sigil 전용 HTML/CSS
- ❌ `<div>` 스타일 · 인라인 CSS
- ❌ 5열 이상 복잡한 표
- ❌ 페이지 번호 · EPUB 전용 태그 남용

---

## 13. 공통 품질 기준

| 항목 | 기준 |
|---|---|
| 독자 수준 | Python 초급 → LLM 연구 입문 |
| 설명 방식 | 직관 → 수학 → 코드 → LLM 연결 |
| 분량 | 강당 약 20~35 Apple Books 페이지 |
| 실습 | NumPy + PyTorch 최소 2개 |
| 수식 | LaTeX 표준 |
| LLM 연결 | Transformer/GPT 예시 포함 |
| 용어 | 강마다 10~20개 |
| 연습문제 | 기초 · 계산 · 코드 · 응용 |

### 빌드

```bash
# 구조 정규화(필요 시)
python3 scripts/apply_apple_books_structure.py

python3 scripts/build_epub.py --book all
python3 scripts/check_lecture.py
```

산출물: `epub/01_…` ~ `epub/05_….epub` (EPUB3 + MathML)

Markdown만 유지하면 Pandoc 등 다른 EPUB 도구로도 동일 결과를 목표로 한다.
