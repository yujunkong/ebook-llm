# 9강. Scalar, Vector, Matrix, Tensor
## 이번 강에서 배우는 내용

- Scalar, Vector, Matrix, Tensor가 각각 무엇인지
- Rank(랭크, 계수/차원 수) 0~4가 무엇을 의미하는지
- Shape(형상)과 Axis(축)를 보고 데이터의 구조를 읽는 법
- LLM이 왜 3D·4D Tensor를 쓰는지 (batch, sequence, embedding, heads)

## 왜 중요한가?
LLM 코드를 열면 거의 모든 곳이 Tensor다.

```text
토큰 ID  → Embedding → 3D Tensor
Attention → Q, K, V 행렬곱 → 다시 Tensor
Loss      → Scalar
Parameter → 수백만~수천억 개의 Scalar가 모인 Tensor
```

즉, Tensor는 LLM의 **기본 자료형**이다.  
Scalar/Vector/Matrix는 Tensor의 특수한 경우로 이해하면, 이후 수식과 코드가 한 줄로 이어진다.

## 선수 개념
다음을 이미 알고 있다고 가정한다.

- Python의 숫자, 리스트, 중첩 리스트
- NumPy의 `ndarray`, `shape`, indexing (8강)
- “배열에 숫자가 들어 있다”는 감각

아직 모르는 것은 괜찮다.

- 내적·행렬곱의 자세한 계산 → 10강
- 미분·Gradient → 11~12강
- 신경망 구조 → 15강 이후

## 핵심 개념
### 3.1 Scalar (스칼라)

**Scalar(스칼라)**는 **하나의 숫자**이다. 방향이 없고, 크기만 있다.

예:

```text
3
-1.5
0
2.718
```

왜 필요한가?

- Loss 값은 보통 Scalar다. “모델이 얼마나 틀렸는가”를 한 숫자로 요약한다.
- Learning Rate도 Scalar다.
- Softmax 직전의 개별 logit 하나도 Scalar다.

수학에서는 보통 소문자 이탤릭으로 쓴다.

$$

a = 3, \quad \ell = 0.42

$$

- $a$: 임의의 스칼라
- $\ell$: Loss를 나타내는 스칼라 (관례적으로 script L 또는 ell 사용)

### 3.2 Vector (벡터)

**Vector(벡터)**는 **같은 종류의 숫자가 한 줄로 나열된 것**이다. 크기와 방향(또는 순서 있는 성분들)을 가진다.

예:

$$

\mathbf{x} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}

$$

- $\mathbf{x}$: 벡터 (굵은 소문자로 표기하는 관례)
- 성분 $x_1=1$, $x_2=2$, $x_3=3$

가로로 쓰기도 한다.

$$

\mathbf{x}^\top = [1,\ 2,\ 3]

$$

- $\mathbf{x}^\top$: $\mathbf{x}$의 **Transpose(전치)** — 행/열을 뒤집은 형태

왜 필요한가?

- 하나의 토큰 Embedding은 보통 Vector다. 예: 768차원 실수 벡터.
- 한 샘플의 특징(feature)도 Vector로 표현한다.
- Gradient도 파라미터가 여러 개면 Vector가 된다.

### 3.3 Matrix (행렬)

**Matrix(행렬)**는 **숫자를 행(row)과 열(column)로 늘어놓은 표**이다. 2차원 격자다.

예:

$$

A = \begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}

$$

- $A$: 행렬 (보통 대문자)
- $A_{11}=1$: 1행 1열
- $A_{12}=2$: 1행 2열
- $A_{21}=3$: 2행 1열
- $A_{22}=4$: 2행 2열

Shape는 $2 \times 2$ (2행 2열)이다.

왜 필요한가?

- Linear Layer의 가중치 $W$는 Matrix다.
- Attention의 $Q K^\top$ 결과도 (시퀀스×시퀀스) Matrix다.
- 배치가 없는 단일 문장의 토큰 Embedding들을 세로로 쌓으면 Matrix가 된다.

### 3.4 Tensor (텐서)

**Tensor(텐서)**는 **다차원 배열**의 일반화이다.

엄밀한 수학의 Tensor(좌표 변환 규칙까지 포함한 대상)와, 딥러닝에서 말하는 Tensor(다차원 숫자 배열)는  overlapping하지만 완전히 같지는 않다.  
이 책에서는 **딥러닝/PyTorch 의미**를 사용한다.

즉:

```text
Scalar = 0D Tensor
Vector = 1D Tensor
Matrix = 2D Tensor
그 이상 = 3D, 4D, ... Tensor
```

왜 필요한가?

- 배치(batch)까지 넣으면 Embedding은 3D가 된다.
- Multi-Head Attention에서는 4D로 재배열하는 일이 흔하다.
- 이미지·비디오 모델은 더 높은 차원을 쓴다. LLM도 내부 연산에서 고차 Tensor를 자주 만든다.

## 직관적으로 이해하기
상자로 비유하자.

| 이름 | 비유 | 예 |
|---|---|---|
| Scalar | 숫자 한 칸 | Loss = 1.2 |
| Vector | 칸이 한 줄 | Embedding `[0.1, -0.3, 0.7]` |
| Matrix | 칸이 표 | 문장 토큰 3개의 Embedding을 쌓은 $3\times d$ |
| 3D Tensor | 표를 여러 장 겹친 책 | 배치 크기 2인 문장들 |
| 4D Tensor | 책을 여러 권 꽂은 선반 | batch × heads × seq × dim |

핵심 감각:

> **차원이 하나 늘 때마다 “같은 구조가 여러 개” 생긴다.**

- Vector 여러 개 → Matrix
- Matrix 여러 개 → 3D Tensor
- 3D Tensor 여러 개 → 4D Tensor

## Rank와 Shape
### 5.1 Rank (랭크)

여기서 **Rank(랭크)**는 선형대수에서 말하는 “행렬의 계수(rank)”와 다른, **배열의 축 개수**를 뜻한다. NumPy/PyTorch 문서에서는 **ndim(number of dimensions)**이라고도 한다.

| Rank | 이름 | Shape 예 | 의미 |
|---:|---|---|---|
| 0 | Scalar | `()` | 숫자 하나 |
| 1 | Vector | `(3,)` | 길이 3 |
| 2 | Matrix | `(2, 4)` | 2행 4열 |
| 3 | 3D Tensor | `(2, 3, 4)` | 배치 2, 길이 3, 특징 4 |
| 4 | 4D Tensor | `(2, 8, 3, 16)` | 배치 2, 헤드 8, 길이 3, 헤드차원 16 |

기호로 쓰면:

- Rank $r$인 Tensor는 $r$개의 축을 가진다.
- Shape를 $(n_1, n_2, \ldots, n_r)$로 쓴다.
- 각 $n_i$는 $i$번째 축의 크기이다.

### 5.2 Axis (축)

**Axis(축)**는 Tensor를 따라 세는 **방향의 번호**이다.

2D Matrix $A$ shape `(2, 3)`:

```text
axis=0  ↓  행 방향 (세로로 늘어남)
axis=1  →  열 방향 (가로로 늘어남)

행0:  a00  a01  a02
행1:  a10  a11  a12
```

3D Tensor shape `(B, T, D)`에서 관례적으로:

- `axis=0`: Batch
- `axis=1`: Time / Sequence(토큰 위치)
- `axis=2`: Feature / Embedding 차원

축을 헷갈리면 `sum`, `mean`, `softmax`가 완전히 다른 결과가 된다. LLM 구현에서 가장 흔한 버그 중 하나다.

## 작은 숫자로 직접 계산하기
### 6.1 Rank 0 — Scalar

$$

s = 5

$$

원소 개수: 1  
Shape: `()`

### 6.2 Rank 1 — Vector

$$

\mathbf{v} = [1,\ 2,\ 3]

$$

- 길이 $n=3$
- $v_0=1$, $v_1=2$, $v_2=3$ (0-based indexing)
- 원소 개수: 3

“첫 번째 성분”을 손으로 확인:

$$

v_0 = 1

$$

### 6.3 Rank 2 — Matrix

$$

M = \begin{bmatrix}
1 & 2 \\
3 & 4 \\
5 & 6
\end{bmatrix}

$$

- Shape: $3 \times 2$
- $M_{2,0} = 5$ (3행 1열, 0-based로는 행 2, 열 0)
- 원소 개수: $3\times 2 = 6$

행 벡터로 읽기:

```text
행0 = [1, 2]
행1 = [3, 4]
행2 = [5, 6]
```

### 6.4 Rank 3 — “행렬 두 장”

배치 크기 2, 시퀀스 길이 2, 임베딩 차원 3인 작은 예를 만든다.

$$

X^{(0)} = \begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix},
\quad
X^{(1)} = \begin{bmatrix}
7 & 8 & 9 \\
1 & 0 & 2
\end{bmatrix}

$$

이를 쌓으면 shape `(2, 2, 3)` Tensor $X$가 된다.

손으로 원소를 집어 보자.

- $X[0, 1, 2] = 6$  
  (배치 0, 두 번째 토큰, 세 번째 차원)
- $X[1, 0, 0] = 7$  
  (배치 1, 첫 토큰, 첫 차원)

원소 개수: $2\times 2\times 3 = 12$

### 6.5 Rank 4 — Attention Head 감각

배치 1, 헤드 2, 시퀀스 2, 헤드 차원 2:

```text
batch 0
  head 0:
    [[1, 2],
     [3, 4]]
  head 1:
    [[5, 6],
     [7, 8]]
```

Shape: `(1, 2, 2, 2)`

- $X[0, 1, 0, 1] = 6$  
  (batch 0, head 1, token 0, dim 1)

이 구조는 Multi-Head Attention에서 Q/K/V를 나눌 때 반복해서 등장한다.

## 코드로 구현하기
8강에서 배운 NumPy를 수학 용어와 연결한다.

```python
# lecture09_tensors.py
# 목표: Scalar/Vector/Matrix/Tensor의 shape·ndim·축을 확인한다.

import numpy as np

# --- Rank 0: Scalar ---
s = np.array(5)
print("scalar:", s, "shape=", s.shape, "ndim=", s.ndim)

# --- Rank 1: Vector ---
v = np.array([1, 2, 3])
print("vector:", v, "shape=", v.shape, "ndim=", v.ndim)
print("v[0] =", v[0])

# --- Rank 2: Matrix ---
M = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
])
print("matrix shape=", M.shape)  # (3, 2)
print("M[2, 0] =", M[2, 0])     # 5

# --- Rank 3: batch of matrices ---
X = np.array([
    [[1, 2, 3],
     [4, 5, 6]],
    [[7, 8, 9],
     [1, 0, 2]],
])
print("X shape=", X.shape)      # (2, 2, 3)
print("X[0, 1, 2] =", X[0, 1, 2])  # 6
print("원소 개수 =", X.size)     # 12

# --- Axis 연습: 어느 축으로 합칠까? ---
# shape (2, 3) 행렬
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
])
print("sum all =", A.sum())           # 21
print("sum axis=0 =", A.sum(axis=0))  # [5, 7, 9]  (행을 접음)
print("sum axis=1 =", A.sum(axis=1))  # [6, 15]    (열을 접음)

# --- Rank 4 작은 예 ---
Q = np.arange(1, 9).reshape(1, 2, 2, 2)
print("Q shape=", Q.shape)
print("Q[0, 1, 0, 1] =", Q[0, 1, 0, 1])  # 6
```

실행하면 shape와 indexing이 앞 절의 손계산과 일치해야 한다.  
일치하지 않으면 축 이해에 구멍이 있는 것이다.

## PyTorch로 맛보기
PyTorch의 기본 자료형은 `torch.Tensor`이다. NumPy와 거의 같은 감각으로 쓴다.

```python
# lecture09_torch_tensor.py
import torch

x = torch.tensor([
    [[1.0, 2.0, 3.0],
     [4.0, 5.0, 6.0]],
])
print(x.shape)   # torch.Size([1, 2, 3]) = (batch, seq, dim)
print(x.ndim)    # 3
print(x[0, 1, 2])  # tensor(6.)

# LLM에서 자주 쓰는 view/reshape
# (batch, seq, dim) → (batch, seq, heads, head_dim)
y = torch.arange(24.0).reshape(2, 3, 4)  # B=2, T=3, D=4
z = y.view(2, 3, 2, 2)                   # heads=2, head_dim=2
print(z.shape)  # torch.Size([2, 3, 2, 2])
```

아직 Autograd는 쓰지 않는다. 지금은 **구조**만 본다.

## LLM에서는 어디에 사용될까?
### 9.1 전형적인 Shape 이름

LLM 코드/논문에서 자주 보는 기호:

| 기호 | 의미 | 예 |
|---|---|---|
| $B$ | Batch size | 한 번에 넣는 문장 수 |
| $T$ 또는 $L$ 또는 $S$ | Sequence length | 토큰 개수 |
| $d$ 또는 $d_{\text{model}}$ | Embedding / hidden 차원 | 768, 4096 등 |
| $h$ | Attention head 수 | 12, 32 등 |
| $d_h = d/h$ | Head 차원 | 64 등 |
| $V$ | Vocabulary size | 토큰 종류 수 |

### 9.2 데이터가 흘러가며 Shape가 바뀌는 모습

아주 단순한 GPT식 흐름:

```text
입력 토큰 ID
  shape: (B, T)                 # 정수, Rank 2

Embedding
  shape: (B, T, d)              # Rank 3

Linear으로 Q, K, V 생성 후 head 분할
  shape: (B, h, T, d_h)         # Rank 4

Attention 점수  QK^T / sqrt(d_h)
  shape: (B, h, T, T)           # Rank 4

출력 투영 후
  shape: (B, T, d)              # 다시 Rank 3

LM Head
  shape: (B, T, V)              # 어휘별 점수

Loss
  shape: ()                     # Scalar, Rank 0
```

이 표만 이해해도, 이후 Attention 강의에서 “왜 transpose를 하지?”라는 질문이 훨씬 쉬워진다.

### 9.3 Parameter도 Tensor다

예: Linear $y = xW + b$

- $W$: shape `(d_in, d_out)` Matrix
- $b$: shape `(d_out,)` Vector
- $x$: 단일 벡터면 `(d_in,)`, 배치면 `(B, d_in)` 또는 `(B, T, d_in)`

LLM의 “수십억 파라미터”란, 결국 이런 Tensor들 안의 Scalar를 모두 센 개수다.

## 수식 보강 — 노름 · 외적 감각 · 텐서 축

벡터의 **유클리드 노름(길이)**은 이후 Softmax 스케일·정규화에서 반복됩니다.

$$
\|\mathbf{x}\|_2 = \sqrt{x_1^2 + x_2^2 + \cdots + x_n^2} = \sqrt{\mathbf{x}\cdot\mathbf{x}}
$$

예: $\mathbf{x}=[3,\ 4]$이면 $\|\mathbf{x}\|_2=5$입니다.

행렬 $A\in\mathbb{R}^{m\times n}$의 Frobenius 노름은 “모든 성분을 펼친 벡터”의 길이입니다.

$$
\|A\|_F = \sqrt{\sum_{i=1}^{m}\sum_{j=1}^{n} A_{ij}^2}
$$

> 💡 **팁**
>
> LLM 가중치 분석·클리핑에서 “행렬이 얼마나 큰가”를 말할 때 $\|W\|_F$를 자주 봅니다.

배치가 있는 임베딩 텐서 $X\in\mathbb{R}^{B\times T\times d}$에서, 토큰 하나를 고르는 것은 축을 고정하는 일입니다.

$$
\mathbf{h}_{b,t} = X_{b,t,:}\ \in\ \mathbb{R}^{d}
$$

Multi-Head로 나누면

$$
X \in \mathbb{R}^{B\times T\times d}
\quad\rightarrow\quad
\mathbb{R}^{B\times T\times h\times d_h}
\quad (d = h\cdot d_h)
$$

처럼 **축을 쪼개고 합치는 것**이 Tensor 조작의 핵심입니다.

### 외적과 행렬 (맛보기)

두 열벡터 $\mathbf{u}\in\mathbb{R}^{m}$, $\mathbf{v}\in\mathbb{R}^{n}$의 외적(outer product)은

$$
\mathbf{u}\mathbf{v}^{\top} \in \mathbb{R}^{m\times n},
\quad
(\mathbf{u}\mathbf{v}^{\top})_{ij} = u_i v_j
$$

입니다. LoRA의 $\Delta W = BA$도 “낮은 랭크 외적들의 합”으로 읽을 수 있습니다 (73강).

## Broadcasting과 Shape 사고
딥러닝 코드는 크기가 다른 Tensor를 더하기도 한다.

예:

```python
import numpy as np

X = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])                 # (2, 3)
b = np.array([10.0, 20.0, 30.0])  # (3,)

Y = X + b
# b가 각 행에 더해짐
# [[11, 22, 33],
#  [14, 25, 36]]
print(Y)
```

직관:

> 작은 Tensor를 **호환되는 축**에 맞춰 확장한 뒤 연산한다.

LLM에서는 bias 더하기, mask 더하기, 위치 인코딩 더하기에서 계속 등장한다.  
Shape가 안 맞으면 런타임 에러가 나거나, 더 무섭게는 **잘못된 축으로 조용히 연산**된다.

## 실습
### 실습 1 — 손으로 Shape 읽기

다음 배열의 Rank, Shape, 원소 개수를 쓰시오.

```text
[[[0, 1],
  [2, 3],
  [4, 5]]]
```

### 실습 2 — Indexing

위 배열에서 값 `4`의 인덱스를 `(?,?,?)` 형태로 쓰시오.

### 실습 3 — NumPy로 3D 만들기

1. `np.arange(24)`로 0~23을 만든다.
2. `(2, 3, 4)`로 reshape한다.
3. `X[1, 2, 3]`이 무엇인지 손계산 후 코드로 확인한다.

### 실습 4 — LLM Shape 스케치

$B=2$, $T=5$, $d=8$, $h=4$일 때 다음 Shape를 쓰시오.

1. Embedding 출력
2. Head로 나눈 Q
3. Attention 점수 $QK^\top$
4. 최종 logits (`V=100` 가정)

### 실습 5 — Axis 실험

`(2, 3)` 행렬에서 `sum(axis=0)`과 `sum(axis=1)`의 Shape를 각각 예측하고 실행으로 검증하시오.

## 자주 하는 실수
1. **Vector의 shape를 `(3)`과 `(3, 1)`로 혼동한다**  
   둘 다 “숫자 3개”처럼 보이지만, 행렬곱에서 결과가 달라진다. 항상 `.shape`를 출력한다.

2. **Rank와 “행렬의 계수(rank)”를 같은 말로 생각한다**  
   이 강의의 Rank는 ndim이다. 선형대수의 matrix rank는 다른 개념이다.

3. **Axis 번호를 거꾸로 센다**  
   Softmax를 `dim=-1`(마지막 축)에 적용하는 이유가 바로 어휘/특징 축이기 때문이다.

4. **배치 차원을 잊는다**  
   학습 코드의 거의 모든 Tensor 앞에 `B`가 붙는다. 논문 수식은 배치를 생략하는 경우가 많아 더 헷갈린다.

5. **4D를 “너무 어렵다”고 피한다**  
   4D는 대개 `(B, h, T, d_h)`처럼 **이름 있는 축 네 개**일 뿐이다. 이름을 붙이면 怖지 않다.

## 핵심 요약
- Scalar는 숫자 하나(Rank 0), Vector는 1D, Matrix는 2D, Tensor는 그 일반화다.
- Shape는 각 축의 크기, Axis는 그 축의 번호다.
- LLM은 `(B, T, d)` 3D와 `(B, h, T, d_h)` 4D를 핵심적으로 사용한다.
- Loss는 Scalar, Parameter는 거대한 Tensor, 연산은 그 사이의 변환이다.
- 코드를 볼 때 값보다 먼저 **Shape를 읽는다.**

## 용어 사전
| 용어 | 의미 |
|---|---|
| Scalar (스칼라) | 하나의 숫자, Rank 0 Tensor |
| Vector (벡터) | 1차원으로 나열된 숫자들, Rank 1 |
| Matrix (행렬) | 행·열로 배열된 숫자들, Rank 2 |
| Tensor (텐서) | 다차원 배열 (딥러닝 의미) |
| Rank / ndim | 축(차원)의 개수 |
| Shape | 각 축의 크기 튜플 |
| Axis / Dim | 연산이 적용되는 축의 번호 |
| Batch size $B$ | 한 번에 처리하는 샘플 수 |
| Sequence length $T$ | 토큰(시퀀스) 길이 |
| Embedding dim $d$ | 토큰 표현 벡터의 길이 |
| Transpose | 축 순서를 바꾸는 연산 (2D에서는 행↔열) |
| Broadcasting | 작은 Tensor를 호환 축에 맞춰 확장해 연산 |

## 연습문제
### 문제 1 (개념)

Scalar, Vector, Matrix, Tensor의 관계를 Rank로 한 문장씩 설명하시오.

### 문제 2 (계산)

다음 Matrix에서 $A_{1,0}$의 값은? (1-based가 아니라 **수학 표기의 1-based**로 답해도 되고, 0-based라면 명확히 표시하시오.)

$$

A = \begin{bmatrix} 9 & 8 \\ 7 & 6 \end{bmatrix}

$$

코드에서는 `A[1, 0]`이 무엇인지 함께 쓰시오.

### 문제 3 (Shape)

Shape `(4, 10, 32)` Tensor가 있다. LLM 관점에서 가능한 해석(B, T, d)을 제시하고, 원소 개수를 구하시오.

### 문제 4 (코드)

아래 코드의 출력 shape를 예측하시오.

```python
import numpy as np
x = np.zeros((2, 5, 16))
y = x.reshape(2, 5, 4, 4)
print(y.shape)
```

### 문제 5 (연결)

Attention에서 점수 행렬이 `(B, h, T, T)`인 이유를, “무엇을 무엇과 비교하는가”의 관점에서 설명하시오.

---

## 정답 및 해설
### 문제 1

- Scalar: Rank 0 Tensor  
- Vector: Rank 1 Tensor  
- Matrix: Rank 2 Tensor  
- Tensor: Rank 0 이상을 포함하는 다차원 배열의 일반 이름 (실무에서는 보통 Rank ≥ 0 전부)

### 문제 2

수학 1-based로 $A_{1,0}$은 잘못된 표기다. 보통 $A_{10}$ 또는 $A_{1,0}$을 쓰지 않고 $A_{1,1}=9$, $A_{2,1}=7$처럼 쓴다.  
NumPy 0-based `A[1, 0] = 7` (2행 1열).

### 문제 3

예: $B=4$, $T=10$, $d=32$ — 문장 4개, 각 10토큰, 임베딩 32차원.  
원소 개수: $4\times 10\times 32 = 1280$.

### 문제 4

`y.shape = (2, 5, 4, 4)`.  
해석 예: batch 2, seq 5, heads 4, head_dim 4 (`d=16=4×4`).

### 문제 5

각 헤드에서 쿼리 위치 $T$개와 키 위치 $T$개를 모두 비교하므로 마지막 두 축이 `(T, T)`가 된다. 배치와 헤드를 앞에 두면 `(B, h, T, T)`.

## 다음 강의와 연결
이번 강의에서 “상자(구조)”를 배웠다.

다음 **제10강. 선형대수 기초 — 내적과 행렬곱**에서는 그 상자 안의 숫자를 **어떻게 섞는지** 배운다.  
내적, 행렬곱, $Wx+b$가 Embedding과 Attention의 엔진이 된다.

> 구조를 읽었으면, 이제 연산을 계산하자.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [8강. NumPy로 배열 다루기](08강_NumPy로_배열_다루기.md)
- **다음 강:** [10강. 선형대수 기초 — 내적과 행렬곱](10강_선형대수_기초_내적과_행렬곱.md)

<!-- /LECTURE_NAV -->
