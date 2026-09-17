# 《밑바닥부터 LLM》 Markdown 작성 규칙 (Sigil EPUB 표준)

이 문서는 120강 전체 전자책의 **공식 Markdown 스타일 가이드**이다.

목표:

1. Sigil EPUB에서 깨지지 않는다.
2. GitHub에서도 읽기 좋다.
3. MathJax/KaTeX 수식이 정상 렌더링된다.
4. Pandoc → EPUB/PDF 변환과 호환된다.

저장소 경로 규칙(프로젝트 실제 구조):

```text
books/
  01_python_tensor_math_pytorch/
  02_tokenizer_transformer/
  03_gpt_pretraining_sft/
  04_rlhf_ppo_grpo/
  05_vllm_glm_dgx/
```

권 폴더명은 위 구조를 유지한다. 강의 파일명 규칙은 아래 §22를 따른다.

---

## 1. 문서 기본 구조 (모든 강의 동일)

```markdown
# 제11강. 미분과 편미분

> **학습 목표**
> - 미분의 의미 이해
> - 편미분의 의미 이해
> - PyTorch Autograd와 연결

---

## 1. 미분이란?

설명...

## 2. 편미분이란?

설명...

## 3. 실습

```python
...
```

## 4. 핵심 정리

...

## 5. 연습 문제

...
```

| 순서 | 내용 |
|---:|---|
| 1 | `# 제N강. 강의 제목` |
| 2 | 학습 목표 |
| 3 | 본문 |
| 4 | 실습 코드 |
| 5 | 핵심 요약 |
| 6 | 연습 문제 |
| 7 | (권장) 용어 정리 · 강의 이동 링크 |

권 제목(`1권. …`)을 강의 H1로 쓰지 않는다. 권 정보는 README·EPUB 메타데이터·목차에서 다룬다.

---

## 2. 제목 규칙

| Heading | 용도 |
|---|---|
| `#` | 강의 제목 (`제N강. …`) |
| `##` | 큰 챕터 |
| `###` | 소주제 |
| `####` | 정의 / 참고 |

**최대 4단계**까지만 사용한다.

---

## 3. 문단 스타일

의미 단위로 **빈 줄**을 둔다.

```markdown
Transformer는 Attention을 이용하여 문맥 정보를 계산한다.

Attention은 Query, Key, Value 세 가지 벡터를 사용한다.

다음 절에서 각각을 자세히 살펴본다.
```

---

## 4. 강조 규칙

| 용도 | 표현 |
|---|---|
| 핵심 용어 | **Attention** |
| 새로운 개념 | **Gradient** |
| 주의 | `> ⚠️ **주의**` |
| 팁 | `> 💡 **Tip**` |
| 정의 | `> **정의**` |

```markdown
> 💡 **Tip**
> Gradient는 기울기 벡터이다.
```

---

## 5. 수식 작성 규칙

### 인라인

`$L=f(x)$` 형식만 사용한다. (`\(...\)` 사용 금지)

```markdown
Loss는 $L=f(x)$ 로 표현한다.
```

### 블록

앞뒤로 **빈 줄**을 둔다. 등호는 수식 안에 넣는다.

```markdown
$$
f'(x)=2x
$$
```

### 권장 표기

| 개념 | 표기 |
|---|---|
| 벡터 | `\mathbf{x}` |
| 행렬 | `\mathbf{W}` |
| 텐서 | `\mathcal{X}` |
| Loss | `L` |
| Gradient | `\nabla L` |
| Expectation | `\mathbb{E}` |
| Query / Key / Value | `Q` / `K` / `V` |
| Attention Score | `QK^{T}` |
| Dimension | `d_k` |

행렬 행 구분은 `\\`를 쓴다.

```markdown
$$
A=
\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix}
$$
```

**금지:** `[svg](...)` 수식 이미지, HTML `<math>` 수동 삽입(변환기가 넣는 경우는 제외).

---

## 6. 코드 블록

언어를 반드시 지정한다: `python`, `bash`, `json`, `yaml`, `text` 등.

출력은 코드와 분리한다.

````markdown
```python
print(loss)
```

출력 결과:

```text
tensor(0.2384)
```
````

---

## 7. 표

Markdown 표만 사용한다. 복잡한 병합 셀은 쓰지 않는다. 열은 대략 6~8개 이하.

---

## 8. 이미지

```markdown
![Transformer 구조](images/transformer_architecture.png)

*그림 11-2. Transformer 구조*
```

- 권별 `images/` 폴더에만 둔다. 권 간 공유하지 않는다.
- PNG 권장, 파일명은 영문 소문자+언더바, 공백 없음.

---

## 9. 예제 · 실습 · 정리 · 문제

### 예제

문제 → 풀이 → 결과.

### 실습

목표 → 코드 → 실행 결과 → 설명 (또는 Step 1…N).

### 핵심 정리

불릿 **5개 내외**.

### 연습 문제

`## 연습 문제` 아래 `### 문제 1` …

---

## 10. 파일명

```text
01강_제목.md
11강_미분과_편미분.md
27강_텍스트가_숫자가_되는_과정.md
```

- 강의 번호 + `강_` + 제목
- 공백 없음, 언더바 사용
- 1~9강은 `01강`…`09강`처럼 두 자리

---

## 11. 강의 이동 링크

모든 강의 말미에 네비게이션 블록을 둔다. (`scripts/add_lecture_nav.py`)

```markdown
<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** …
- **다음 강:** …

<!-- /LECTURE_NAV -->
```

---

## 12. EPUB / Pandoc

| 항목 | 규칙 |
|---|---|
| 수식 | `$…$`, `$$…$$`만 사용 |
| SVG 수식 링크 | 금지 |
| HTML | 최소화 |
| 이미지 | 상대경로 |
| 코드 | 언어 지정 |
| 표 | Markdown 표 |

권장:

```bash
python3 scripts/build_math_preview.py --book 1   # KaTeX HTML
python3 scripts/build_epub.py --book 1           # EPUB3 MathML
```

상세: [MATH_RENDERING.md](MATH_RENDERING.md)

---

## 13. 공통 템플릿

```markdown
# 제N강. 강의 제목

> **학습 목표**
> - 목표 1
> - 목표 2
> - 목표 3

---

## 1. 개념 소개

설명

### 정의

정의 설명

### 예제

수식 또는 예제

---

## 2. 동작 원리

설명

---

## 3. 코드 / PyTorch 실습

### 목표

설명

### 코드

```python
...
```

### 실행 결과

```text
...
```

### 코드 해설

설명

---

## 4. LLM에서 어떻게 사용되는가?

연결 설명

---

## 핵심 정리

- …
- …

---

## 용어 정리

| 용어 | 설명 |
|------|------|
| … | … |

---

## 연습 문제

### 문제 1

…

### 문제 2

…

### 문제 3 (실습)

…
```

강의 성격에 따라 절 제목·개수는 조절할 수 있으나, **H1·학습 목표·핵심 정리·연습 문제·강의 이동**은 유지한다.

---

## 14. 권 구성 (고정)

| 권 | 강의 | 주제 |
|---|---|---|
| 1권 | 1~26 | Python · Tensor · 수학 · PyTorch |
| 2권 | 27~54 | Tokenizer와 Transformer |
| 3권 | 55~78 | GPT Pretraining · SFT |
| 4권 | 79~98 | RLHF · DPO · GRPO · RL |
| 5권 | 99~120 | vLLM · KV Cache · Speculative Decoding · DGX Spark 등 |

120강 번호와 5권 구조는 임의로 바꾸지 않는다.
