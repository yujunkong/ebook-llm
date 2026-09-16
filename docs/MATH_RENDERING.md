# 수식이 스크린샷처럼 보이게 하려면

원고 Markdown에는 `$$…$$`, `\(…\)` 형태의 LaTeX를 **그대로** 둔다.
문제는 원고가 아니라 **뷰어가 수식 엔진을 안 켜서** `$$` 원문이 그대로 보이는 것이다.

스크린샷처럼 보이게 하는 방법:

## 1) 브라우저 미리보기 (권장, 즉시 확인)

```bash
python3 scripts/build_math_preview.py books/01_python_tensor_math_pytorch/09강_Scalar_Vector_Matrix_Tensor.md
```

생성된 HTML을 브라우저로 연다. KaTeX가 수식을 렌더한다.

한 권 전체:

```bash
python3 scripts/build_math_preview.py --book 1
```

`epub/preview/book1/index.html` 을 연다.

## 2) EPUB3 (MathML)

```bash
python3 scripts/build_epub.py --book 1
```

산출물: `epub/01_밑바닥부터_LLM_1권.epub`  
Sigil·Apple Books 등 MathML을 지원하는 리더에서 수식으로 보인다.

## 3) Cursor / 일반 Markdown 미리보기

에디터 미리보기가 LaTeX를 지원해야 한다.
지원하지 않으면 `$$`가 그대로 보인다. 이 경우 1) HTML 미리보기를 사용한다.

## 원고 작성 규칙

올바른 예 (행렬 행 구분 `\\`):

```markdown
$$
\mathbf{x} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}
$$
```

잘못된 예 (`\ ` 한 칸만 쓰면 행이 안 나뉨):

```markdown
$$ \mathbf{x} = \begin{bmatrix} 1 \ 2 \ 3 \end{bmatrix} $$
```

인라인은 `\(...\)` 를 기본으로 한다.

```markdown
Loss \(L\) 와 Perplexity \(PPL = e^{L}\)
```
