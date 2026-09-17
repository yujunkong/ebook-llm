# 19강. PyTorch Tensor

## 이번 강에서 배우는 내용

- `torch.tensor`로 텐서를 만드는 방법
- dtype, device, shape를 읽고 바꾸는 방법
- Broadcasting 규칙을 **수식·shape**로 설명
- 행렬곱·배치 matmul과 Linear 층의 수학 대응
- NumPy ↔ PyTorch 변환과 메모리 공유 함정
- LLM 코드에서 텐서 shape를 먼저 보는 습관

## 왜 중요한가?

현대 LLM 구현의 거의 모든 값은 Tensor입니다.

```text
token ids → Embedding Tensor
→ Attention Q/K/V Tensor
→ logits Tensor
→ loss 스칼라 Tensor
```

NumPy만으로도 원리는 되지만, GPU·Autograd·생태계는 PyTorch Tensor를 중심으로 돌아갑니다.  
Tensor를 못 읽으면 모델 코드를 읽어도 “행렬이 어디로 가는지”가 안 보입니다.

> **핵심**
>
> Tensor = 숫자 격자 + shape + dtype + device (+ 다음 강 requires_grad).

## 선수 개념

- ndarray의 shape, dtype, broadcasting (제8~9강)
- 행렬곱과 배치 차원 (제10강, 제16강)
- (설치) `pip install torch` 또는 환경에 이미 포함된 PyTorch

CPU만 있어도 충분합니다. GPU는 `device` 개념만 이해하면 됩니다.

---

## 핵심 개념

### 3.1 Tensor (PyTorch)

**Tensor(텐서)**는 PyTorch에서 다차원 배열을 담는 기본 객체입니다.  
수학의 0D~nD 배열 개념과 같고, 여기에 **장치(device)**와 **자동 미분 추적(requires_grad)** 메타데이터가 붙습니다.

```python
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(x.shape)   # torch.Size([2, 2])
print(x.dtype)   # torch.float32
print(x.device)  # cpu
```

### 3.2 dtype

| dtype | 용도 |
|---|---|
| `torch.float32` | 기본 실수 연산 |
| `torch.float16` / `bfloat16` | 가속·메모리 절약 |
| `torch.float64` | 고정밀 수치 검증 |
| `torch.int64` (`long`) | 토큰 ID, 인덱스 |
| `torch.bool` | 마스크 |

잘못된 dtype은 침묵의 버그를 만듭니다. 토큰 ID는 반드시 정수여야 합니다.

### 3.3 device

- `cpu`
- `cuda:0` (NVIDIA GPU)
- `mps` (Apple Silicon 등, 환경 의존)

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.tensor([1.0, 2.0], device=device)
```

연산은 **같은 device**의 텐서끼리만 가능합니다.

### 3.4 shape / size / dim

- `x.shape` / `x.size()`: 각 축 길이
- `x.dim()`: 차원 수
- `x.numel()`: 전체 원소 수

LLM에서는 `[batch, seq, hidden]` 같은 shape 계약이 코드 가독성의 핵심입니다.

### 3.5 Broadcasting

**Broadcasting**은 shape가 다른 텐서를 논리적으로 확장해 원소별 연산이 되게 하는 규칙입니다.

1. 뒤에서부터 축을 맞춥니다.
2. 길이가 같거나, 한쪽이 1이면 통과합니다.
3. 없는 축은 1로 간주합니다.

### 3.6 view / reshape / transpose / permute / contiguous

| 연산 | 의미 |
|---|---|
| `view` | 동일 메모리에서 shape 재해석 (연속 메모리 필요) |
| `reshape` | 가능하면 view, 아니면 복사 |
| `transpose` / `permute` | 축 순서 변경 (종종 non-contiguous) |
| `contiguous` | 메모리를 다시 연속으로 맞춤 |

### 3.7 NumPy와의 관계

```python
t = torch.from_numpy(a)   # 공유 (CPU)
t2 = torch.tensor(a)      # 복사
b = t.numpy()             # requires_grad=False, CPU만
```

`from_numpy`로 만든 텐서를 바꾸면 NumPy 배열도 바뀝니다.

---

## 직관적으로 이해하기

NumPy ndarray가 “숫자 격자 상자”라면,  
PyTorch Tensor는 그 상자에 **택배 송장**이 붙은 것입니다.

- 상자 모양(shape)
- 숫자 종류(dtype)
- 창고 위치(device)
- (다음 강) “이 상자 추적함”(requires_grad)

CPU 상자와 CUDA 상자를 바로 더할 수 없는 이유가 이것입니다.

---

## 수학적으로 이해하기 — matmul과 broadcast

### 5.1 텐서의 공간

rank-$k$ 텐서:

$$
T \in \mathbb{R}^{n_1 \times n_2 \times \cdots \times n_k}
$$

### 5.2 배치 행렬곱 (18강과 동일 관례)

$$
Y = X W^{\top}
,\quad
X\in\mathbb{R}^{B\times d_{\mathrm{in}}}
,\quad
W\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}
,\quad
Y\in\mathbb{R}^{B\times d_{\mathrm{out}}}
$$

작은 숫자:

$$
X
=
\begin{bmatrix}
1.0 & 0.5 \\
0.0 & 1.0
\end{bmatrix}
,\quad
W
=
\begin{bmatrix}
0.3 & -0.2 \\
0.4 & 0.1
\end{bmatrix}
$$

$$
X W^{\top}
=
\begin{bmatrix}
1.0 & 0.5 \\
0.0 & 1.0
\end{bmatrix}
\begin{bmatrix}
0.3 & 0.4 \\
-0.2 & 0.1
\end{bmatrix}
=
\begin{bmatrix}
0.20 & 0.45 \\
-0.20 & 0.10
\end{bmatrix}
$$

(여기서는 Bias 없는 순수 matmul 연습입니다. 전체 Forward는 아래에서 Bias를 더합니다.)

PyTorch:

```python
Y = X @ W.T
# 또는 torch.matmul(X, W.T)
# nn.Linear는 내부적으로 y = x @ W.T + b
```

### 5.3 Bias 브로드캐스트

$$
Z = X W^{\top} + \mathbf{b}
,\quad
\mathbf{b}\in\mathbb{R}^{d_{\mathrm{out}}}
$$

shape로는

$$
\underbrace{(B\times d_{\mathrm{out}})}_{XW^{\top}}
+
\underbrace{(d_{\mathrm{out}},)}_{\mathbf{b}}
\rightarrow
(B\times d_{\mathrm{out}})
$$

$\mathbf{b}$가 각 행에 더해집니다.

### 5.4 Broadcasting 규칙 — 손으로

예: $(3,1,4)+(2,4)$.

뒤에서 맞춤:

| 축 (뒤←앞) | A | B | 결과 |
|---|---|---|---|
| 마지막 | 4 | 4 | 4 |
| 가운데 | 1 | 2 | 2 |
| 앞 | 3 | (없음→1) | 3 |

결과 shape:

$$
(3,2,4)
$$

실패 예: $(2,3)+(2,4)$ — 마지막 축 3≠4이고 どちら도 1이 아닙니다.

### 5.5 3D 배치 matmul

$$
A\in\mathbb{R}^{B\times M\times K}
,\quad
B_{\mathrm{mat}}\in\mathbb{R}^{B\times K\times N}
\Rightarrow
C\in\mathbb{R}^{B\times M\times N}
$$

$$
C_{b,m,n} = \sum_{k=1}^{K} A_{b,m,k}\,(B_{\mathrm{mat}})_{b,k,n}
$$

Attention에서 자주 나오는 패턴입니다.

---

### 5.6 `nn.Linear`와의 대응

`nn.Linear(in_features, out_features)`는 배치 입력 $X\in\mathbb{R}^{B\times d_{\mathrm{in}}}$에 대해

$$
Y = X W^{\top} + \mathbf{b}
,\quad
W\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}
,\quad
\mathbf{b}\in\mathbb{R}^{d_{\mathrm{out}}}
$$

를 계산합니다. 19강에서 `X @ W.T + b`로 쓰는 식과 동일합니다.

한 샘플 행벡터 $\mathbf{x}^{\top}$라면

$$
\mathbf{y}^{\top} = \mathbf{x}^{\top} W^{\top} + \mathbf{b}^{\top}
= (W\mathbf{x})^{\top} + \mathbf{b}^{\top}
$$

즉 열벡터 관점의 $W\mathbf{x}+\mathbf{b}$와 같은 선형 변환입니다.  
관례만 행/열로 바꿔 적을 뿐, 수학은 하나입니다.

## 작은 숫자로 직접 계산하기 — 16강 Forward를 Tensor로

```text
x  = [1.0, 0.5]
W1 = [[0.3, -0.2], [0.4, 0.1]]
b1 = [0.1, -0.1]
W2 = [[0.5, -0.4]]
b2 = [0.2]
```

단샘플을 행벡터 배치 $(1\times 2)$로 보면

$$
\mathbf{z}^{(1)}
=
\begin{bmatrix}1.0 & 0.5\end{bmatrix}
W^{(1)\top}
+
\begin{bmatrix}0.1 & -0.1\end{bmatrix}
=
\begin{bmatrix}0.30 & 0.35\end{bmatrix}
$$

$$
\mathbf{a}^{(1)}
=
\mathrm{ReLU}(\mathbf{z}^{(1)})
=
\begin{bmatrix}0.30 & 0.35\end{bmatrix}
$$

$$
\hat{y}
=
\mathbf{a}^{(1)} W^{(2)\top} + 0.2
=
0.21
$$

손계산과 코드가 같아야 합니다.

### 6.1 Bias broadcast 숫자

$$
X
=
\begin{bmatrix}
1.0 & 0.5 \\
0.0 & 1.0
\end{bmatrix}
,\quad
\mathbf{b}
=
\begin{bmatrix}0.1 & -0.1\end{bmatrix}
$$

$$
X+\mathbf{b}
=
\begin{bmatrix}
1.1 & 0.4 \\
0.1 & 0.9
\end{bmatrix}
$$

각 행에 같은 bias가 더해진 결과입니다.

### 6.2 outer / matmul 연결

17강의 $\delta x^{\top}$도 Tensor로:

$$
\boldsymbol{\delta}
=
\begin{bmatrix}-0.395\\0.316\end{bmatrix}
,\quad
\mathbf{x}
=
\begin{bmatrix}1.0\\0.5\end{bmatrix}
\Rightarrow
\delta x^{\top}
=
\begin{bmatrix}
-0.395 & -0.1975 \\
0.316 & 0.158
\end{bmatrix}
$$

```python
dW1 = torch.outer(delta, x)   # 또는 delta.unsqueeze(1) @ x.unsqueeze(0)
```

---

## 코드로 구현하기

### 7.1 생성과 속성

```python
"""19강: PyTorch Tensor 기초."""

import torch

def show(t, name="t"):
    print(f"{name}: shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}")
    print(t)

x = torch.tensor([1.0, 0.5])
W1 = torch.tensor([[0.3, -0.2], [0.4, 0.1]])
b1 = torch.tensor([0.1, -0.1])
W2 = torch.tensor([[0.5, -0.4]])
b2 = torch.tensor([0.2])
show(x, "x"); show(W1, "W1")
```

### 7.2 Forward (Autograd 없이)

```python
def forward_torch(x, W1, b1, W2, b2):
    z1 = x @ W1.T + b1
    a1 = torch.relu(z1)
    y_hat = a1 @ W2.T + b2
    return y_hat, {"z1": z1, "a1": a1}

y_hat, cache = forward_torch(x, W1, b1, W2, b2)
print(float(y_hat))         # 0.21
assert torch.allclose(y_hat, torch.tensor(0.21))
```

### 7.3 dtype / device

```python
x64 = x.to(torch.float64)
ids = torch.tensor([10, 20], dtype=torch.long)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
z = x.to(device) @ W1.to(device).T
```

### 7.4 Broadcasting 실험

```python
X = torch.tensor([[1.0, 0.5], [0.0, 1.0]])
b = torch.tensor([0.1, -0.1])
print((X + b).shape)  # (2,2)
# torch.randn(2, 3) + torch.randn(2, 4)  → RuntimeError
```

### 7.5 NumPy 왕복

```python
import numpy as np
np_x = np.array([1.0, 0.5], dtype=np.float32)
t_share = torch.from_numpy(np_x)
t_share[0] = 9.0
print(np_x[0])          # 9.0 — 공유
t_copy = torch.tensor(np.array([1.0, 0.5], dtype=np.float32))
t_copy[0] = 9.0
# 원본 NumPy는 그대로 1.0
```

### 7.6 인덱싱과 마스크

```python
logits = torch.tensor([[2.0, 0.1, -1.0], [0.0, 3.0, 0.2]])
pred = torch.argmax(logits, dim=-1)
mask = logits > 0
```

### 7.7 결정적 난수

```python
torch.manual_seed(0)
a = torch.randn(2, 2)
torch.manual_seed(0)
b = torch.randn(2, 2)
assert torch.allclose(a, b)
```

### 7.8 cat / stack / unsqueeze

```python
a = torch.tensor([[1.0, 2.0]])
b = torch.tensor([[3.0, 4.0]])
print(torch.cat([a, b], dim=0).shape)   # (2,2)
print(torch.stack([a.squeeze(0), b.squeeze(0)], dim=0).shape)
print(x.unsqueeze(0).shape)             # (1,2)
```

### 7.9 einsum 맛보기

```python
B, M, K, N = 2, 3, 4, 5
A = torch.randn(B, M, K)
Bmat = torch.randn(B, K, N)
C1 = torch.matmul(A, Bmat)
C2 = torch.einsum("bmk,bkn->bmn", A, Bmat)
assert torch.allclose(C1, C2, atol=1e-5)
```

### 7.10 전체 Forward 스크립트

```python
def forward_torch_full(x, W1, b1, W2, b2, y=None):
    if x.dim() == 1:
        x = x.unsqueeze(0)
    z1 = x @ W1.T + b1
    a1 = torch.relu(z1)
    y_hat = a1 @ W2.T + b2
    loss = None
    if y is not None:
        y = y.reshape(-1, 1)
        loss = 0.5 * torch.mean((y_hat - y) ** 2)
    return y_hat, loss, {"z1": z1, "a1": a1}

if __name__ == "__main__":
    x = torch.tensor([1.0, 0.5])
    y = torch.tensor([1.0])
    W1 = torch.tensor([[0.3, -0.2], [0.4, 0.1]])
    b1 = torch.tensor([0.1, -0.1])
    W2 = torch.tensor([[0.5, -0.4]])
    b2 = torch.tensor([0.2])
    y_hat, loss, _ = forward_torch_full(x, W1, b1, W2, b2, y)
    print(float(y_hat), float(loss))  # 0.21, 0.31205
```

아직 `requires_grad`는 켜지 않았습니다. 다음 강에서 추적만 켜면 Backprop이 자동으로 붙습니다.

---

## NumPy → Tensor 체크리스트

| NumPy | PyTorch |
|---|---|
| `np.array` | `torch.tensor` |
| `np.maximum(0,z)` | `torch.relu(z)` |
| `A @ B` | `A @ B` |
| `np.mean` | `torch.mean` |
| `np.outer(d,x)` | `torch.outer(d,x)` |
| CPU only | `.to(device)` |
| 수동 Backward | 다음 강 Autograd |

> ⚠️ **주의**
>
> Broadcasting이 에러 없이 성공해도 축 의미가 틀릴 수 있습니다. shape만 보지 말고 의미를 적으세요.

> 💡 **팁**
>
> shape를 먼저 적고 코드를 쓰면, Attention 코드의 `reshape(B,S,H,D)`도 읽기 쉬워집니다.

---

## LLM에서는 어디에 사용될까?

```text
input_ids:      (batch, seq)            int64
embeddings:     (batch, seq, hidden)    float16/bfloat16/float32
attn scores:    (batch, heads, seq, seq)
logits:         (batch, seq, vocab)
```

디버깅 첫 질문:

1. shape가 계약과 같은가?
2. dtype이 인덱스/실수에 맞는가?
3. device가 모델과 같은가?
4. contiguous가 필요한 view 직전인가?

---

## 실습

### 실습 1 — 손계산 재현

`y_hat==0.21`을 `torch.allclose`로 검증합니다.

### 실습 2 — shape 사고

`(2,3) @ (2,4)`를 시도해 에러를 읽고, `(2,3)@(3,4)`로 고칩니다.

### 실습 3 — device 불일치

CUDA가 있으면 한쪽만 GPU로 옮겨 더하기를 시도합니다. 없으면 `.to("cpu")` API만 익힙니다.

### 실습 4 — Broadcasting으로 Bias

`(8,16)+ (16,)`와 `(8,16)+(8,1)` 결과 shape를 각각 적습니다.

### 실습 5 — from_numpy 함정

공유/복사 차이를 출력으로 증명합니다.

### 실습 6 — matmul 손계산

5.2절 $XW^{\top}$ 결과를 `torch`로 재현합니다.

---

## 자주 하는 실수

1. **`torch.Tensor(data)`와 `torch.tensor(data)` 혼동** — 생성은 `torch.tensor` 권장.
2. **정수 텐서로 가중치 연산** — 가중치는 float가 기본.
3. **GPU 텐서에 `.numpy()` 직호출** — `.detach().cpu().numpy()`.
4. **Broadcasting이 성공해서 논리가 틀린 줄 모름** — 축 의미를 단위 테스트하세요.
5. **non-contiguous `view`** — `contiguous()` 또는 `reshape`.
6. **silent dtype 기대를 과신** — 명시적 `.to(dtype)`.

---

## 핵심 요약

- PyTorch Tensor는 shape·dtype·device를 가진 다차원 배열이며 LLM 코드의 기본 화폐입니다.
- Broadcasting·행렬곱·view/reshape 규칙을 읽어야 Attention 코드를 따라갈 수 있습니다.
- NumPy와 왕복할 때 메모리 공유 여부를 의식합니다.
- 같은 Forward를 Tensor로 재현할 수 있으면 Autograd로 넘어갈 준비가 된 것입니다.

## 용어 사전

| 용어 | 의미 |
|---|---|
| Tensor | PyTorch의 다차원 배열 객체 |
| dtype / device / shape | 자료형 / 장치 / 각 차원 크기 |
| Broadcasting | 호환 shape를 확장해 원소별 연산 |
| view / reshape / contiguous | shape 재해석 / 연속 메모리 |
| from_numpy | 버퍼 공유 변환 |
| matmul / `@` | 행렬곱 |
| einsum | 첨자 규칙 텐서 곱 |

## 연습문제

### 문제 1 (개념)

dtype과 device가 각각 무엇을 답하는지 한 줄씩 쓰시오.

### 문제 2 (계산)

$(3,1,4)+(2,4)$의 결과 shape는?

### 문제 3 (코드)

토큰 ID 텐서의 권장 dtype은?

### 문제 4 (코드)

shape `(1,1)`인 `y_hat`을 Python float로 뽑는 방법은?

### 문제 5 (연결)

LLM logits의 전형적 3축은?

### 문제 6 (계산)

6.1절 $X+\mathbf{b}$의 $(2,1)$ 성분(0-index면 `[1,0]`)은?

---

## 정답 및 해설

### 문제 1

- dtype: 숫자 표현 형식  
- device: 연산·저장 위치

### 문제 2

$(3,2,4)$.

### 문제 3

`torch.long` (`torch.int64`).

### 문제 4

`y_hat.item()` 또는 `float(y_hat.reshape(()))`.

### 문제 5

`(batch, seq, vocab)`.

### 문제 6

$0.0+0.1=0.1$.

## 다음 강의와 연결

이번 강의에서 PyTorch의 기본 화폐인 Tensor를 익혔습니다.

다음 **제20강. Autograd — 자동 미분**에서는 `requires_grad=True`로 계산 그래프를 기록하고, `loss.backward()`가 제17~18강의 수동 Backprop을 대신해 `tensor.grad`를 채우는 과정을 다룹니다.

> 상자에 송장을 붙였습니다. 이제 송장이 기울기를 기억하게 합시다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [18강. Backpropagation NumPy 구현](18강_Backpropagation_NumPy_구현.md)
- **다음 강:** [20강. Autograd — 자동 미분](20강_Autograd_자동_미분.md)

<!-- /LECTURE_NAV -->
