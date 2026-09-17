# 8강. NumPy로 배열 다루기
## 이번 강에서 배우는 내용

- NumPy가 무엇인지, 왜 LLM/딥러닝에서 먼저 배우는지 설명한다
- ndarray를 만들고 shape, dtype을 읽는다
- 인덱싱·슬라이싱으로 부분 배열을 꺼낸다
- 원소별 연산과 벡터화(vectorization)의 의미를 안다
- Broadcasting의 기본 규칙을 작은 예로 확인한다
- NumPy 배열이 이후 Tensor로 어떻게 이어지는지 연결한다

## 왜 중요한가?
순수 Python 리스트로도 숫자는 다룰 수 있다. 다만 LLM이 요구하는 규모에서는 부족하다.

예를 들어 배치 하나에도 이런 양이 생긴다.

```text
batch_size = 32
seq_len = 512
hidden = 768

원소 수 ≈ 32 × 512 × 768 ≈ 1.2천만 개
```

이를 파이썬 `for` 루프로 하나식 곱하면 너무 느리다.  
**NumPy(Numerical Python)**는 이런 다차원 숫자 배열을 **빠르고 간결하게** 다루기 위한 라이브러리이다.

또한 PyTorch의 Tensor는 NumPy ndarray와 개념적으로 매우 닮았다.

```text
NumPy ndarray  --개념적으로 확장→  PyTorch Tensor (GPU·자동미분)
```

8강은 “텐서로 가기 전의 마지막 징검다리”이다.

## 선수 개념
### 2.1 Array와 ndarray

**Array(배열)**는 같은 종류의 값을 격자처럼 늘어놓은 자료구조이다.  
NumPy의 핵심 객체는 **ndarray(N-dimensional array)**이다.

- 1차원: 벡터처럼 보이는 수열
- 2차원: 행렬
- 3차원 이상: 배치·채널·시퀀스 등을 함께 담는 구조

### 2.2 Vectorization

**Vectorization(벡터화)**는 파이썬 루프 대신, 배열 전체에 대한 연산을 한 번에 표현·실행하는 방식이다.

```python
# 느린 패턴 (개념)
out = []
for x in xs:
    out.append(x * 2)

# 벡터화
out = xs * 2
```

가독성과 속도가 동시에 좋아지는 경우가 많다.

### 2.3 설치와 import

가상환경에서:

```bash
python -m pip install numpy
```

코드:

```python
import numpy as np
print(np.__version__)
```

관례적으로 `numpy`를 `np`로 줄여 쓴다.

## 핵심 개념
### 3.1 배열 만들기

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([[1.0, 2.0], [3.0, 4.0]])

print(a)
print(b)
```

주요 생성 함수:

```python
zeros = np.zeros((2, 3))       # 0으로 채움
ones = np.ones((2, 3))         # 1로 채움
full = np.full((2, 3), 7)      # 7로 채움
arange = np.arange(0, 5)       # 0,1,2,3,4
linspace = np.linspace(0, 1, 5)  # 0~1을 5등분
eye = np.eye(3)                # 단위행렬
```

난수:

```python
rng = np.random.default_rng(42)  # 재현용 시드
x = rng.normal(size=(2, 3))      # 정규분포 샘플
```

LLM 실험에서 **시드 고정**은 재현성의 기본이다.

### 3.2 shape, dtype, ndim, size

```python
x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
print(x.shape)  # (2, 3)
print(x.dtype)  # float32
print(x.ndim)   # 2
print(x.size)   # 6
```

| 속성 | 의미 |
|---|---|
| **shape** | 각 차원의 크기 튜플 |
| **dtype** | 원소 자료형 |
| **ndim** | 차원 수 |
| **size** | 전체 원소 개수 |

**dtype**이 중요한 이유:

- `float32`는 딥러닝에서 매우 흔하다 (속도와 메모리)
- `float64`는 더 정밀하지만 더 무겁다
- `int64`/`int32`는 인덱스·토큰 ID에 자주 쓰인다

```python
ids = np.array([2, 3, 4], dtype=np.int64)
logits = np.array([2.0, 5.0, 1.0], dtype=np.float32)
```

### 3.3 reshape와 차원 다루기

```python
v = np.arange(6)          # shape (6,)
m = v.reshape(2, 3)       # shape (2, 3)
print(m)
```

원소 총개수가 같아야 reshape할 수 있다. `6 → (2,3)`은 가능, `(2,4)`는 불가능.

```python
batch = np.arange(24).reshape(2, 3, 4)
# (batch=2, seq=3, feature=4)처럼 읽을 수 있다
print(batch.shape)
```

`reshape`는 종종 뷰(view)를 반환해 메모리를 공유할 수 있다.  
지금은 “모양을 바꾼다”는 수준으로 충분하고, 복사 여부의 세부는 필요해질 때 다시 본다.

차원 추가:

```python
x = np.array([1, 2, 3])
print(x.shape)                 # (3,)
print(x[np.newaxis, :].shape)  # (1, 3)
print(x[:, np.newaxis].shape)  # (3, 1)
```

배치 차원·컬럼 벡터를 만들 때 자주 쓴다.

### 3.4 인덱싱과 슬라이싱

```python
m = np.array([[10, 20, 30],
              [40, 50, 60],
              [70, 80, 90]])

print(m[0, 1])      # 20
print(m[1])         # [40, 50, 60]
print(m[:, 1])      # [20, 50, 80]  (두 번째 열)
print(m[:2, 1:])    # 상위 2행, 열 1부터
```

불리언 마스크:

```python
print(m[m >= 50])   # 조건을 만족하는 원소들
```

fancy indexing:

```python
rows = np.array([0, 2])
print(m[rows, 1])   # [20, 80]
```

토큰 ID로 embedding 테이블에서 행을 고르는 연산이, 개념적으로 이런 “인덱스로 행 선택”에 해당한다. (실제로는 Tensor 연산으로 구현)

### 3.5 원소별 연산과 집계

```python
a = np.array([1.0, 2.0, 3.0])
b = np.array([10.0, 20.0, 30.0])

print(a + b)     # [11. 22. 33.]
print(a * b)     # [10. 40. 90.]
print(a ** 2)    # [1. 4. 9.]
print(np.sqrt(a))
```

집계:

```python
m = np.array([[1.0, 2.0], [3.0, 4.0]])
print(m.sum())          # 전체 합 10
print(m.sum(axis=0))    # 열 방향 합 [4. 6.]
print(m.sum(axis=1))    # 행 방향 합 [3. 7.]
print(m.mean())
print(m.max())
```

**axis**는 “어느 축을 따라 접을까”이다.  
딥러닝에서 `mean(dim=-1)` 같은 표현을 만날 때, 이미 여기서 연습한 감각이 그대로 쓰인다.

### 3.6 행렬 곱

원소별 곱 `*`와 행렬 곱은 다르다.

```python
A = np.array([[1.0, 2.0],
              [3.0, 4.0]])
B = np.array([[5.0, 6.0],
              [7.0, 8.0]])

print(A * B)        # 원소별
print(A @ B)        # 행렬 곱
print(np.dot(A, B)) # 2D에서는 @와 유사하게 쓰이는 경우 많음
```

손으로 확인 (`A @ B`):

\[
\begin{bmatrix}1&2\\3&4\end{bmatrix}
\begin{bmatrix}5&6\\7&8\end{bmatrix}
=
\begin{bmatrix}19&22\\43&50\end{bmatrix}
\]

선형층의 $y = xW + b$에서 $xW$가 바로 이런 곱의 확장이다.

### 3.7 Broadcasting 입문

**Broadcasting(브로드캐스팅)**은 모양이 다른 배열을, 일정한 규칙 아래에서 마치 모양이 맞는 것처럼 연산하는 기능이다.

```python
m = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])   # (2, 3)
v = np.array([10.0, 20.0, 30.0]) # (3,)

print(m + v)
```

결과:

```text
[[11. 22. 33.]
 [14. 25. 36.]]
```

`v`가 각 행에 더해진 것처럼 동작한다.

또 다른 예 — 편향 더하기:

```python
x = np.ones((2, 3))          # (2, 3)
b = np.array([0.1, 0.2, 0.3])  # (3,)
y = x + b
```

초보자가 기억할 최소 규칙:

1. 끝쪽 차원부터 비교한다
2. 크기가 같거나, 한쪽이 1이면 확장 가능하다
3. 그렇지 않으면 에러

```python
# 실패 예
# np.ones((2, 3)) + np.ones((2,))  → 끝 차원이 3 vs 2
```

Broadcasting은 편하지만, 의도치 않은 확장이 버그가 되기도 한다.  
항상 `shape`를 출력해 확인하는 습관이 필요하다.

### 3.8 복사와 뷰 (짧게)

```python
a = np.arange(4)
b = a[:2]
b[0] = 99
print(a)  # 앞부분이 함께 바뀔 수 있다 (뷰)
```

독립 복사:

```python
c = a.copy()
```

슬라이싱 결과를 수정할 때는 “원본과 메모리를 공유하는가?”를 한 번 생각한다.

## 직관적으로 이해하기
NumPy 배열을 **엑셀 시트 + 고속 계산기**로 생각하자.

| 개념 | 직관 |
|---|---|
| shape | 시트의 행·열·겹수 |
| dtype | 각 칸의 숫자 종류 |
| vectorization | 칸을 하나씩 클릭하지 않고 열 전체에 수식 적용 |
| broadcasting | 짧은 행을 복사해 붙여 넣는 듯한 자동 정렬 |
| axis | “어느 방향으로 합칠까” |

LLM 텐서도 같은 직관을 GPU 위로 옮긴 것이다.

```text
token ids     : (batch, seq)           int
embeddings    : (batch, seq, hidden)   float
attention map : (batch, head, seq, seq)
logits        : (batch, seq, vocab)
```

이름은 PyTorch에서 더 자주 보지만, **모양을 읽고 축을 기준으로 생각**하는 훈련은 NumPy에서 시작한다.

## 수학적으로 조금 더
벡터 $x \in \mathbb{R}^{n}$, 행렬 $W \in \mathbb{R}^{n \times m}$에 대해

\[
y = xW \in \mathbb{R}^{m}
\]

NumPy에서는 보통 `x`를 shape `(n,)` 또는 `(1, n)`으로 두고 `x @ W`를 계산한다.

배치가 있으면

\[
X \in \mathbb{R}^{B \times n}, \quad Y = XW \in \mathbb{R}^{B \times m}
\]

```python
B, n, m = 4, 3, 2
X = np.ones((B, n))
W = np.arange(n * m, dtype=np.float64).reshape(n, m)
Y = X @ W
print(Y.shape)  # (4, 2)
```

이 한 줄이 신경망 “선형 변환”의 최소 형태이다.

## 작은 숫자 예제 — logits에서 최댓값 고르기
배치 크기 2, 어휘 크기 4인 logits:

```text
샘플0: [1.0, 3.0, 0.5, 2.0]
샘플1: [2.0, 0.0, 4.0, 1.0]
```

각 샘플에서 점수가 가장 큰 토큰 ID를 고른다. (argmax)

손으로:

- 샘플0 → index 1 (점수 3.0)
- 샘플1 → index 2 (점수 4.0)

```python
import numpy as np

logits = np.array([
    [1.0, 3.0, 0.5, 2.0],
    [2.0, 0.0, 4.0, 1.0],
], dtype=np.float32)

pred_ids = np.argmax(logits, axis=-1)
print(pred_ids)          # [1 2]
print(pred_ids.shape)    # (2,)
```

생성·평가에서 “가장 높은 점수의 토큰을 고른다(greedy decoding)”의 축소판이다.

## 코드로 종합하기
```python
# numpy_llm_sketch.py
# 목적: 토큰 ID → (가짜) 임베딩 → 선형층 → argmax 까지의 최소 스케치

from __future__ import annotations

import numpy as np

def lookup_embeddings(ids: np.ndarray, table: np.ndarray) -> np.ndarray:
    """ids: (batch, seq) int, table: (vocab, hidden) → (batch, seq, hidden)"""
    return table[ids]

def mean_pool(x: np.ndarray) -> np.ndarray:
    """(batch, seq, hidden) → (batch, hidden)  단순 평균 풀링"""
    return x.mean(axis=1)

def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """x: (batch, in), weight: (in, out), bias: (out,)"""
    return x @ weight + bias  # bias는 broadcasting

def main() -> None:
    rng = np.random.default_rng(0)

    vocab_size = 6
    hidden = 4
    out_size = 6  # 다음 토큰 점수 (여기선 vocab과 같게)

    # 작은 배치: 두 문장, 길이 3
    ids = np.array([
        [2, 3, 4],
        [2, 1, 4],
    ], dtype=np.int64)  # shape (2, 3)

    embed = rng.normal(size=(vocab_size, hidden)).astype(np.float32)
    W = rng.normal(size=(hidden, out_size)).astype(np.float32)
    b = np.zeros((out_size,), dtype=np.float32)

    x = lookup_embeddings(ids, embed)  # (2, 3, 4)
    h = mean_pool(x)                   # (2, 4)
    logits = linear(h, W, b)           # (2, 6)
    pred = np.argmax(logits, axis=-1)  # (2,)

    print("ids shape:", ids.shape)
    print("embed lookup shape:", x.shape)
    print("pooled shape:", h.shape)
    print("logits shape:", logits.shape)
    print("pred ids:", pred)

if __name__ == "__main__":
    main()
```

이 스케치에서 중요한 것은 성능이 아니라 **shape의 흐름**이다.

```text
(batch, seq)
  → embedding lookup → (batch, seq, hidden)
    → mean pool → (batch, hidden)
      → linear → (batch, vocab)
        → argmax → (batch,)
```

실제 Transformer는 mean pool 대신 훨씬 풍부한 Attention·블록을 쓰지만, “배열 모양이 변환된다”는 사고방식은 동일하다.

## 리스트와 NumPy의 속도 감각 (선택 실험)
```python
# speed_feel.py
# 목적: 벡터화가 왜 필요한지 감각적으로 확인한다.
# (환경에 따라 시간 값은 달라진다.)

import time
import numpy as np

def main() -> None:
    n = 1_000_000
    data = list(range(n))

    t0 = time.perf_counter()
    _ = [x * 2 for x in data]
    t_list = time.perf_counter() - t0

    arr = np.arange(n)
    t0 = time.perf_counter()
    _ = arr * 2
    t_np = time.perf_counter() - t0

    print(f"list:  {t_list:.4f}s")
    print(f"numpy: {t_np:.4f}s")

if __name__ == "__main__":
    main()
```

절대 수치보다 “NumPy가 같은 일을 훨씬 짧게 끝내는 경우가 많다”는 경험이면 충분하다.

## 수식 보강 — 브로드캐스팅 · 축소 · Softmax 축

NumPy/PyTorch의 축 연산은 수식으로 쓰면 명확합니다. 벡터 $\mathbf{x}\in\mathbb{R}^{n}$에 대해

$$
\mathrm{sum}(\mathbf{x}) = \sum_{i=1}^{n} x_i,
\quad
\mathrm{mean}(\mathbf{x}) = \frac{1}{n}\sum_{i=1}^{n} x_i,
\quad
\mathrm{max}(\mathbf{x}) = \max_i x_i
$$

행렬 $A\in\mathbb{R}^{m\times n}$에서 `axis=1`(행 방향 합)은

$$
(\mathrm{sum}(A,\ \mathrm{axis}=1))_i = \sum_{j=1}^{n} A_{ij}
\ \in\ \mathbb{R}^{m}
$$

입니다. Softmax를 마지막 축에 적용한다는 말은, logits $\mathbf{z}\in\mathbb{R}^{V}$에 대해

$$
\mathrm{softmax}(\mathbf{z})_k
=
\frac{e^{z_k}}{\sum_{j=1}^{V} e^{z_j}}
$$

를 **각 토큰 위치마다** 독립적으로 계산한다는 뜻입니다. 배치·시퀀스가 있으면 Shape는 `(B, T, V) → (B, T, V)`이고, 합은 $V$축에서만 1이 됩니다.

### 브로드캐스팅 한 줄

$$
A\in\mathbb{R}^{B\times T\times d},\ 
\mathbf{b}\in\mathbb{R}^{d}
\quad\Rightarrow\quad
(A + \mathbf{b})_{b,t,:} = A_{b,t,:} + \mathbf{b}
$$

bias를 더할 때 앞 축 $B,T$가 자동으로 맞춰집니다.

> ⚠️ **주의**
>
> `(3,1) + (3,)` 와 `(3,1) + (3,1)`의 결과는 같을 수 있어도, Shape 실수는 조용히 잘못된 합을 만듭니다. 항상 `.shape`를 확인하세요.

## LLM에서는 어디에 사용될까?
| NumPy 개념 | LLM/PyTorch에서의 대응 |
|---|---|
| ndarray | `torch.Tensor` |
| shape | `tensor.shape` |
| dtype | `torch.float32`, `torch.long` 등 |
| axis | `dim` |
| broadcasting | Tensor broadcasting |
| `x @ W + b` | `nn.Linear`의 핵심 계산 |
| `argmax` | greedy decoding |
| 인덱싱 `table[ids]` | embedding lookup |

실무에서는 학습 루프 대부분이 PyTorch로 돌아가지만,

- 데이터 전처리
- 메트릭 집계
- 디버깅용 작은 수치 실험
- 논문 수식의 빠른 검증

에서는 여전히 NumPy가 강력하다.

또한 NumPy ↔ Tensor 변환이 자주 등장한다. (이후 강의)

```python
# 미리보기 (지금은 실행 안 해도 됨)
# torch.from_numpy(a)
# tensor.numpy()
```

GPU·자동미분이 필요해지는 순간 Tensor로 넘어가고,  
그 전까지는 NumPy로 shape와 연산을 단단히 익히는 것이 효율적이다.

## 실습
### 실습 1 — shape 읽기

다음 배열의 `shape`, `dtype`, `ndim`을 출력한다.

```python
np.zeros((4, 8, 16), dtype=np.float32)
```

이 shape를 `(batch, seq, hidden)`으로 가정하고 각 축의 의미를 한 줄씩 적어 본다.

### 실습 2 — 정규화

`x = np.array([1.0, 2.0, 3.0, 4.0])`에 대해

\[
z = \frac{x - \mathrm{mean}(x)}{\mathrm{std}(x)}
\]

를 계산한다. (`std`가 0이 아니라고 가정)

### 실습 3 — 배치 matmul

`X` shape `(8, 16)`, `W` shape `(16, 32)`일 때 `Y = X @ W`의 shape를 예측하고 코드로 확인한다.

### 실습 4 — broadcasting 편향

`logits` shape `(2, 5)`에 bias shape `(5,)`를 더하고, `(2,)` bias를 더하려 하면 어떻게 되는지 확인한다.

### 실습 5 — 미니 임베딩 lookup

`table` shape `(10, 3)`를 만들고 `ids = np.array([[1, 2], [0, 9]])`로 조회한 결과 shape를 확인한다.

## 자주 하는 실수
1. **`*`와 `@`를 혼동한다**  
   `*`는 원소별, `@`는 행렬 곱이다. 선형층에서는 대개 `@`.

2. **axis 방향을 반대로 접는다**  
   `sum(axis=0)`이 행을 없애는지 열을 없애는지, 작은 배열로 먼저 확인한다.

3. **dtype을 무시한다**  
   int 배열끼리 나누면 의도와 다른 결과가 날 수 있다. 확률·loss는 float로.

4. **broadcasting 에러를 읽지 않는다**  
   `operands could not be broadcast`는 shape 불일치 신호다. 양쪽 `shape`를 출력한다.

5. **1차원 `(n,)`과 2차원 `(n,1)`을 같다고 생각한다**  
   행렬 곱·broadcasting에서 결과가 달라진다.

6. **Python list와 ndarray를 섞어 연산한다**  
   가능해도 의도가 흐려진다. 수치 계산 구간에서는 ndarray로 통일한다.

## 핵심 요약
- NumPy ndarray는 다차원 숫자 배열이며 LLM 수치 계산의 기초이다
- shape/dtype을 읽는 능력이 Tensor 이해의 출발점이다
- 벡터화로 루프 없는 배열 연산을 표현한다
- broadcasting은 다른 모양의 배열을 규칙에 따라 확장해 연산한다
- `x @ W + b`, indexing lookup, argmax는 신경망·생성의 직접 재료이다
- NumPy는 PyTorch Tensor로 개념이 자연스럽게 확장된다

## 용어 사전
| 용어 | 의미 |
|---|---|
| NumPy | 다차원 배열 수치 계산 라이브러리 |
| ndarray | NumPy의 N차원 배열 객체 |
| shape | 각 차원 크기의 튜플 |
| dtype | 배열 원소의 자료형 |
| Vectorization | 배열 단위로 연산을 수행하는 방식 |
| Broadcasting | 다른 모양 배열을 규칙적으로 확장해 연산 |
| axis / dim | 연산을 적용·축소할 축 |
| reshape | 원소 수를 유지한 채 모양 변경 |
| matmul (`@`) | 행렬 곱 |
| argmax | 최댓값의 인덱스 |

## 연습문제
### 문제 1 (개념)

Python list 대신 NumPy를 쓰는 이유를 “규모”와 “표기” 측면에서 두 문장으로 쓰시오.

### 문제 2 (shape)

`np.arange(24).reshape(2, 3, 4)`의 `ndim`과 `size`는?

### 문제 3 (연산)

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
```

`A * B`와 `A @ B`의 결과를 쓰시오.

### 문제 4 (broadcasting)

shape `(3, 1)`과 `(1, 4)`를 더하면 결과 shape는 무엇인가?

### 문제 5 (코드)

`logits` shape `(4, 1000)`에서 각 행의 최댓값 인덱스를 구하는 한 줄을 쓰시오.

### 문제 6 (연결)

Embedding lookup이 `table[ids]` 형태와 닮은 이유와, 이후 Tensor에서 이 연산이 중요한 이유를 설명하시오.

---

## 정답 및 해설
### 문제 1

수백만~수억 원소를 파이썬 루프로 다루기 어렵기 때문이다. 또한 벡터/행렬 연산을 수식에 가까운 짧은 코드로 표현할 수 있다.

### 문제 2

`ndim=3`, `size=24`

### 문제 3

- `A * B` → `[[5, 12], [21, 32]]`
- `A @ B` → `[[19, 22], [43, 50]]`

### 문제 4

`(3, 4)`

### 문제 5

```python
np.argmax(logits, axis=-1)
```

### 문제 6

토큰 ID가 곧 “표에서 몇 번째 행을 가져올까”라는 인덱스이기 때문이다. 언어 모델은 이산 토큰을 연속 벡터로 바꾸기 위해 embedding lookup을 거의 항상 사용한다.

## 다음 강의와 연결
이전 **제7강. 파일 입출력과 데이터 다루기**에서 디스크의 데이터를 불러왔고, 이번 강의에서 그 숫자를 배열로 계산하는 법을 익혔다.

다음 **제9강. Scalar · Vector · Matrix · Tensor**에서는 같은 대상을 수학 언어 다시 정리한다.  
0차원부터 고차원까지, “모양”이 의미를 갖는 이유를 선형대수 관점에서 고정한 뒤, PyTorch Tensor로 넘어간다.

> 배열을 다루게 되었다. 이제 그 배열에 수학 이름을 붙인다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [7강. 파일 입출력과 데이터 다루기](07강_파일_입출력과_데이터_다루기.md)
- **다음 강:** [9강. Scalar, Vector, Matrix, Tensor](09강_Scalar_Vector_Matrix_Tensor.md)

<!-- /LECTURE_NAV -->
