# 1권. Python · Tensor · 수학 · PyTorch

## 제19강. PyTorch Tensor

### 1. 이번 강의에서 배울 것

제8~10강에서 NumPy 배열과 Tensor 개념을, 제18강에서 NumPy 학습기를 다뤘다.  
이번 강의부터는 같은 데이터를 **PyTorch Tensor**로 다루며, 이후 Autograd·`nn.Module`의 바닥에 깔린다.

이 강의를 마치면 다음을 말할 수 있어야 한다.

- `torch.tensor` / `torch.Tensor`로 텐서를 만드는 방법
- **dtype**, **device**, **shape**를 읽고 바꾸는 방법
- **Broadcasting(브로드캐스팅)** 규칙을 NumPy와 비교해 설명
- NumPy ↔ PyTorch 변환과 메모리 공유 함정
- LLM 코드에서 텐서 shape를 먼저 보는 습관

학습(Backward)은 다음 강이다. 오늘은 **자료구조와 연산 규칙**에 집중한다.

### 2. 왜 이것을 배우는가

현대 LLM 구현의 거의 모든 값은 Tensor이다.

```text
token ids → Embedding Tensor
→ Attention Q/K/V Tensor
→ logits Tensor
→ loss 스칼라 Tensor
```

NumPy만으로도 원리는 되지만, GPU·Autograd·생태계(transformers, vLLM 등)는 PyTorch Tensor를 중심으로 돌아간다.  
Tensor를 못 읽으면 모델 코드를 읽어도 “행렬이 어디로 가는지”가 안 보인다.

### 3. 먼저 알아야 할 개념

- ndarray의 shape, dtype, broadcasting (제8~9강)
- 행렬곱과 배치 차원 (제10강, 제16강)
- (설치) `pip install torch` 또는 환경에 이미 포함된 PyTorch

CPU만 있어도 충분하다. GPU는 `device` 개념만 이해하면 된다.

### 4. 핵심 개념 설명

#### 4.1 Tensor (PyTorch)

**Tensor(텐서)**는 PyTorch에서 다차원 배열을 담는 기본 객체이다.  
수학의 0D~nD 배열 개념과 같고, 여기에 **장치(device)**와 **자동 미분 추적(requires_grad)** 메타데이터가 붙는다.

**왜 필요한가?**  
하나의 타입으로 CPU/GPU 연산, 미분, 신경망 API를 연결하기 위함이다.

**예제:**

```python
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(x.shape)   # torch.Size([2, 2])
print(x.dtype)   # torch.float32
print(x.device)  # cpu
```

#### 4.2 dtype (데이터 타입)

**dtype**은 텐서 원소의 숫자 형식이다.

| dtype | 용도 |
|---|---|
| `torch.float32` | 기본 실수 연산 (가장 흔함) |
| `torch.float16` / `bfloat16` | 학습·추론 가속, 메모리 절약 |
| `torch.float64` | 고정밀 (수치 검증 등) |
| `torch.int64` | 토큰 ID, 인덱스 |
| `torch.bool` | 마스크 |

**왜 중요한가?**  
잘못된 dtype은 침묵의 버그를 만든다. 예: 토큰 ID를 float로 두면 Embedding 인덱싱이 실패하거나 의미가 깨진다.

```python
ids = torch.tensor([1, 2, 3], dtype=torch.long)  # int64
w = torch.randn(4, 8, dtype=torch.float32)
```

#### 4.3 device

**device**는 텐서가 상주하는 계산 장치이다.

- `cpu`
- `cuda:0` (NVIDIA GPU)
- `mps` (Apple Silicon 등, 환경 의존)

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.tensor([1.0, 2.0], device=device)
# 또는
y = x.to(device)
```

**왜 필요한가?**  
연산은 같은 device의 텐서끼리만 가능하다. LLM Inference를 GPU로 올리는 첫 줄이 종종 `model.to("cuda")`이다.

#### 4.4 shape / size / dim

- `x.shape` 또는 `x.size()`: 각 축 길이
- `x.dim()` / `x.ndim`: 차원 수
- `x.numel()`: 전체 원소 수

```python
t = torch.randn(2, 3, 4)
print(t.shape)   # [2, 3, 4]
print(t.dim())   # 3
print(t.numel()) # 24
```

LLM에서는 `[batch, seq, hidden]` 같은 shape 계약이 코드 가독성의 핵심이다.

#### 4.5 Broadcasting

**Broadcasting(브로드캐스팅)**은 shape가 다른 텐서를, 복사 없이(논리적으로) 확장해 원소별 연산이 되게 하는 규칙이다.

규칙 요약:

1. 뒤에서부터 축을 맞춘다.
2. 길이가 같거나, 한쪽이 1이면 통과.
3. 없는 축은 1로 간주.

```python
a = torch.randn(4, 1, 3)
b = torch.randn(1, 5, 3)
c = a + b          # → (4, 5, 3)
```

**왜 필요한가?**  
Bias 덧셈 `X @ W.T + b`에서 `b` shape `(H,)`가 `(B,H)`로 자동 확장된다.

#### 4.6 view / reshape / transpose / permute / contiguous

| 연산 | 의미 |
|---|---|
| `view` | 동일 메모리 위에서 shape만 재해석 (연속 메모리 필요) |
| `reshape` | 가능하면 view, 아니면 복사 |
| `transpose` / `permute` | 축 순서 변경 (종종 non-contiguous) |
| `contiguous` | 메모리 배열을 다시 연속으로 맞춤 |

```python
x = torch.arange(6)
y = x.view(2, 3)
z = y.t()                 # 전치 → non-contiguous 가능
# z.view(6) 는 실패할 수 있음
z2 = z.contiguous().view(6)
```

Attention 코드에서 `reshape(B, S, H, D)` 패턴이 매우 흔하다.

#### 4.7 NumPy와의 관계

```python
import numpy as np

# NumPy → Tensor
a = np.array([1.0, 2.0], dtype=np.float32)
t = torch.from_numpy(a)   # 메모리 공유 (CPU)
t2 = torch.tensor(a)      # 복사본

# Tensor → NumPy
b = t.numpy()             # requires_grad=False, CPU만
```

**함정:** `from_numpy`로 만든 텐서를 바꾸면 NumPy 배열도 바뀐다.  
GPU 텐서는 `.cpu().numpy()`가 필요하다.

### 5. 직관적으로 이해하기

NumPy ndarray가 “숫자 격자 상자”라면,  
PyTorch Tensor는 그 상자에 **택배 송장**이 붙은 것이다.

송장에는 다음이 적힌다.

- 상자 모양(shape)
- 숫자 종류(dtype)
- 창고 위치(device)
- (다음 강) “이 상자 추적함”(requires_grad)

연산은 송장이 호환될 때만 진행된다.  
CPU 상자와 CUDA 상자를 바로 더할 수 없는 이유가 이것이다.

### 6. 수학적으로 이해하기

Tensor \(T\)의 rank-\(k\)는

\[
T \in \mathbb{R}^{n_1 \times n_2 \times \cdots \times n_k}
\]

배치 행렬곱:

\[
Y = X W^{\top},\quad X\in\mathbb{R}^{B\times d_{\mathrm{in}}},\; W\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}
\]

PyTorch에서는

```python
Y = X @ W.T
# 또는
Y = torch.matmul(X, W.T)
# Linear 층에서는 F.linear(X, W, b) 가 W @ x 관례
```

`torch.nn.Linear`는 내부적으로 \(y = xW^{\top}+b\) (x가 행벡터 배치)를 쓴다.  
18강 NumPy 관례와 맞춰 두면 이식하기 쉽다.

### 7. 작은 숫자로 직접 계산하기

18강과 같은 단샘플 Forward를 Tensor로 재현한다.

```text
x  = [1.0, 0.5]
W1 = [[0.3, -0.2], [0.4, 0.1]]
b1 = [0.1, -0.1]
W2 = [[0.5, -0.4]]
b2 = [0.2]
```

손계산 결과: \(a1=[0.3,0.35]\), \(\hat{y}=0.21\).

### 8. 코드로 구현하기

#### 8.1 생성과 속성

```python
"""19강: PyTorch Tensor 기초."""

import torch

def show(t, name="t"):
    """디버깅용 요약 출력."""
    print(f"{name}: shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}")
    print(t)


x = torch.tensor([1.0, 0.5])                 # float32 기본(실수 리터럴)
W1 = torch.tensor([[0.3, -0.2], [0.4, 0.1]])
b1 = torch.tensor([0.1, -0.1])
W2 = torch.tensor([[0.5, -0.4]])
b2 = torch.tensor([0.2])

show(x, "x")
show(W1, "W1")
```

#### 8.2 Forward (Autograd 없이)

```python
def forward_torch(x, W1, b1, W2, b2):
    """18강과 동일한 2-2-1 Forward."""
    z1 = x @ W1.T + b1                 # broadcasting으로 bias 추가
    a1 = torch.relu(z1)
    y_hat = a1 @ W2.T + b2
    return y_hat, {"z1": z1, "a1": a1}


y_hat, cache = forward_torch(x, W1, b1, W2, b2)
print("y_hat =", float(y_hat))         # 0.21
print("a1    =", cache["a1"])
```

#### 8.3 dtype / device 변환

```python
# dtype 변경
x64 = x.to(torch.float64)
ids = torch.tensor([10, 20], dtype=torch.long)

# device 변경
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W1_d = W1.to(device)
x_d = x.to(device)
# 연산 전 반드시 맞춤
z = x_d @ W1_d.T
```

#### 8.4 Broadcasting 실험

```python
X = torch.tensor([[1.0, 0.5],
                  [0.0, 1.0]])          # (2,2) 배치
b = torch.tensor([0.1, -0.1])           # (2,)
print((X + b).shape)                    # (2,2) — 행마다 bias

# 실패 예시 (주석 해제 시 RuntimeError)
# bad = torch.randn(2, 3) + torch.randn(2, 4)
```

#### 8.5 NumPy 왕복

```python
import numpy as np

np_x = np.array([1.0, 0.5], dtype=np.float32)
t_share = torch.from_numpy(np_x)
t_share[0] = 9.0
print(np_x[0])          # 9.0 — 공유!

np_x2 = np.array([1.0, 0.5], dtype=np.float32)
t_copy = torch.tensor(np_x2)
t_copy[0] = 9.0
print(np_x2[0])         # 1.0 — 복사본
```

#### 8.6 인덱싱과 마스크

```python
logits = torch.tensor([[2.0, 0.1, -1.0],
                       [0.0, 3.0, 0.2]])
pred = torch.argmax(logits, dim=-1)     # 클래스 인덱스
mask = logits > 0
print(pred, mask)
```

LLM에서 `attention_mask`, `loss` ignore index 등이 이 계열 API를 쓴다.

#### 8.7 결정적 난수 (재현성)

```python
torch.manual_seed(0)
a = torch.randn(2, 2)
torch.manual_seed(0)
b = torch.randn(2, 2)
assert torch.allclose(a, b)
```

#### 8.8 cat / stack / unsqueeze

배치·시퀀스를 이어 붙이는 연산은 LLM 전처리에서 자주 나온다.

```python
a = torch.tensor([[1.0, 2.0]])   # (1,2)
b = torch.tensor([[3.0, 4.0]])   # (1,2)

# cat: 기존 축을 따라 이어 붙임
print(torch.cat([a, b], dim=0).shape)   # (2,2)

# stack: 새 축을 만들어 쌓음
print(torch.stack([a.squeeze(0), b.squeeze(0)], dim=0).shape)  # (2,2)

# unsqueeze / squeeze: 축 추가·제거
x = torch.tensor([1.0, 0.5])
print(x.unsqueeze(0).shape)   # (1,2) — 배치 차원 추가
print(x.unsqueeze(-1).shape)  # (2,1)
```

#### 8.9 einsum 맛보기 (선택)

**einsum**은 첨자 규칙으로 텐서 곱을 적는 방법이다. Attention에서 자주 등장한다.

```python
# 배치 행렬곱: (B,M,K) @ (B,K,N) → (B,M,N)
B, M, K, N = 2, 3, 4, 5
A = torch.randn(B, M, K)
Bmat = torch.randn(B, K, N)
C1 = torch.matmul(A, Bmat)
C2 = torch.einsum("bmk,bkn->bmn", A, Bmat)
assert torch.allclose(C1, C2, atol=1e-5)
```

오늘은 “첨자로 shape를 읽을 수 있다” 정도만 기억하면 충분하다. 2권 Attention에서 다시 만난다.

#### 8.10 18강 Forward를 Tensor로 옮긴 전체 스크립트

```python
"""19강: NumPy 2-2-1 Forward를 torch로 이식."""

import torch


def forward_torch(x, W1, b1, W2, b2, y=None):
    """x: (2,) 또는 (B,2)."""
    if x.dim() == 1:
        x = x.unsqueeze(0)                 # (1,2)
    z1 = x @ W1.T + b1
    a1 = torch.relu(z1)
    y_hat = a1 @ W2.T + b2                 # (B,1)
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
    y_hat, loss, cache = forward_torch(x, W1, b1, W2, b2, y)
    print(float(y_hat), float(loss))
    # 0.21 , 0.31205
```

아직 `requires_grad`는 켜지 않았다. 다음 강에서 같은 코드에 추적만 켜면 Backprop이 자동으로 붙는다.

### 9. NumPy 학습기를 Tensor 연산으로 옮길 때 체크리스트

| NumPy | PyTorch |
|---|---|
| `np.array` | `torch.tensor` / `torch.asarray` |
| `np.maximum(0,z)` | `torch.relu(z)` |
| `A @ B` | `A @ B` (동일) |
| `np.mean` | `torch.mean` |
| `dtype=np.float32` | `dtype=torch.float32` |
| CPU only | `.to(device)` |
| 수동 Backward | 다음 강 Autograd |

오늘은 `requires_grad`를 켜지 않아도 Forward는 동일하다.  
켜는 순간부터 계산 그래프가 기록되기 시작한다(제20강).

### 10. 실제 LLM에서는 어떻게 사용하는가

예: Causal LM 배치의 전형적 shape

```text
input_ids:      (batch, seq)            int64, CPU→GPU
position/index: (batch, seq)
embeddings:     (batch, seq, hidden)    float16/bfloat16/float32
attn scores:    (batch, heads, seq, seq)
logits:         (batch, seq, vocab)
```

디버깅 첫 질문:

1. shape가 계약과 같은가?
2. dtype이 인덱스/실수에 맞는가?
3. device가 모델과 같은가?
4. contiguous가 필요한 view 직전인가?

5권 vLLM·최적화로 가면 dtype(FP8 등)과 device 배치가 성능의 중심이 된다. 출발점은 오늘이다.

### 11. 실습

#### 실습 1 — 손계산 재현

8.2 코드로 `y_hat==0.21`을 `torch.allclose`로 검증한다.

#### 실습 2 — shape 사고 실험

`(2,3) @ (2,4)`를 시도해 보고 에러 메시지를 읽는다.  
올바른 쌍 `(2,3)@(3,4)`로 고친다.

#### 실습 3 — device 불일치

CUDA가 있으면 한쪽만 `.cuda()`로 옮겨 더하기를 시도한다.  
에러를 확인한 뒤 둘 다 같은 device로 맞춘다. CUDA가 없으면 CPU에서 `.to("cpu")`로 API만 익힌다.

#### 실습 4 — Broadcasting으로 Bias

`(8,16)` 행렬에 `(16,)` bias를 더하고, `(8,1)` bias를 더하는 경우 shape 결과를 각각 적는다.

#### 실습 5 — from_numpy 함정

공유/복사 차이를 출력으로 증명하는 미니 스크립트를 작성한다.

### 12. 자주 하는 실수

1. **`torch.Tensor(data)`와 `torch.tensor(data)` 혼동**  
   생성은 `torch.tensor` / `torch.as_tensor`를 권장. `Tensor`는 구식 생성자 함정이 있다.

2. **정수 나눗셈·정수 텐서**  
   `torch.tensor([1,2])`는 int64가 될 수 있다. 딥러닝 가중치는 float가 기본.

3. **GPU 텐서에 `.numpy()` 직호출**  
   먼저 `.detach().cpu().numpy()` 패턴이 안전하다(다음 강 `detach` 상세).

4. **Broadcasting이 “성공”해서 논리가 틀린 줄 모름**  
   shape가 에러 없이 커져도 축 의미가 잘못일 수 있다. 단위 테스트를 켠다.

5. **non-contiguous `view`**  
   `transpose` 후 `view` 실패 시 `reshape` 또는 `contiguous()`를 검토.

6. **silent dtype promotion 기대**  
   환경/연산에 따라 규칙이 엄격하다. 명시적 `.to(dtype)`이 디버깅에 유리하다.

### 13. 핵심 정리

- PyTorch Tensor는 shape·dtype·device를 가진 다차원 배열이며 LLM 코드의 기본 화폐이다.
- Broadcasting·행렬곱·view/reshape 규칙을 읽어야 Attention 코드를 따라갈 수 있다.
- NumPy와 왕복할 때 메모리 공유 여부를 의식한다.
- 같은 Forward를 Tensor로 재현할 수 있으면 Autograd로 넘어갈 준비다 된 것이다.

### 14. 핵심 용어

| 용어 | 의미 |
|---|---|
| Tensor | PyTorch의 다차원 배열 객체 |
| dtype | 원소 자료형 |
| device | CPU/GPU 등 상주 장치 |
| shape | 각 차원의 크기 |
| Broadcasting | 호환 shape를 확장해 원소별 연산 |
| view / reshape | 원소 배치를 유지한 shape 재해석 |
| contiguous | 메모리상 원소가 예상 순서로 연속인지 |
| from_numpy | NumPy 배열과 버퍼를 공유하는 변환 |
| matmul / `@` | 행렬곱 |
| torch.relu | ReLU의 PyTorch 구현 |

### 15. 복습 문제

#### 문제 1 (개념)

Tensor의 dtype과 device가 각각 무엇을 답하는지 한 줄씩 쓰시오.

#### 문제 2 (계산)

`(3,1,4) + (2,4)`의 결과 shape는? (가능 여부 포함)

#### 문제 3 (코드)

토큰 ID 텐서를 만들 때 권장 dtype은?

#### 문제 4 (코드)

`y_hat`이 shape `(1,1)`일 때 Python float로 뽑는 안전한 방법 하나를 쓰시오.

#### 문제 5 (연결)

LLM logits 텐서의 전형적 3축은 무엇인가?

---

### 정답 및 해설

#### 문제 1

- dtype: 숫자 표현 형식(float32, int64 등)
- device: 연산·저장 위치(cpu, cuda:0 등)

#### 문제 2

뒤에서 맞춤: `4`와 `4` OK, `1`과 `2` → 2, 없는 축과 `3` → 3.  
결과 `(3,2,4)`.

#### 문제 3

`torch.long` (`torch.int64`).

#### 문제 4

`float(y_hat.reshape(()))` 또는 `y_hat.item()`.

#### 문제 5

`(batch, seq, vocab)` (또는 동등 순서의 배치·시퀀스·어휘).

### 16. 다음 강의와 연결

이번 강의에서 PyTorch의 기본 화폐인 Tensor를 익혔다.

다음 **제20강. Autograd — 자동 미분**에서는 `requires_grad=True`로 계산 그래프를 기록하고, `loss.backward()`가 제17~18강의 수동 Backprop을 대신해 `tensor.grad`를 채우는 과정을 다룬다.  
`detach`, `torch.no_grad`까지 알면 제21강 `nn.Module`로 모델을 조립할 준비가 끝난다.

> 상자에 송장을 붙였다. 이제 송장이 기울기를 기억하게 하자.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [제18강. Backpropagation NumPy 구현](18강_Backpropagation_NumPy_구현.md)
- **다음 강:** [제20강. Autograd — 자동 미분](20강_Autograd_자동_미분.md)

<!-- /LECTURE_NAV -->
