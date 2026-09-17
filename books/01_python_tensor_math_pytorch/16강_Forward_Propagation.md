# 16강. Forward Propagation

## 이번 강에서 배우는 내용

- Forward Propagation(순전파)의 정의와 목적
- 2-2-1 네트워크에서 층마다 $z^{(1)}, a^{(1)}, \hat{y}, L$을 **shape와 함께** 손으로 계산하는 방법
- 중간 활성화 값을 왜 저장해야 하는지 (cache)
- NumPy로 벡터화 Forward를 구현하는 방법
- 배치 Forward의 shape 계약
- LLM Inference가 “학습 없는 Forward의 반복”임을 설명하는 방법

## 왜 중요한가?

학습의 한 스텝은 대략 다음입니다.

```text
1) Forward: 입력 → 예측 → Loss
2) Backward: Loss → 각 파라미터 Gradient
3) Update: 파라미터 -= lr * Gradient
```

Backward를 이해하려면 Forward의 계산 그래프를 먼저 써야 합니다.  
중간 값 $z$, $a$를 모르면 Chain Rule을 어디에 적용할지 모릅니다.

또한 서비스를 하는 LLM Inference는 대개 Backward 없이 **Forward만 반복**합니다.  
생성은 “다음 토큰 점수 계산(Forward) → 토큰 선택 → 다시 Forward”의 루프입니다.

> **핵심**
>
> Forward = 계산 그래프를 **입력 → 출력** 방향으로 평가하는 일.  
> 중간 값(cache)은 다음 강의 Backprop의 연료입니다.

## 선수 개념

- 제15강의 2-2-1 구조와 ReLU
- 행렬-벡터 곱 $W\mathbf{x}$
- MSE 같은 기본 Loss (제13강)
- Chain Rule이 “합성 함수의 미분”이라는 사실 (제14강) — 이번 강의에서는 아직 미분하지 않습니다

---

## 핵심 개념

### 3.1 Forward Propagation

**Forward Propagation(순전파, Forward Pass)**는 입력을 네트워크의 앞쪽 층부터 뒤쪽 층으로 통과시켜 **예측값(및 Loss)**를 계산하는 과정입니다.

```text
x → Layer1 → Layer2 → … → ŷ → (선택) Loss L
```

**왜 필요한가?**

1. 학습: 현재 파라미터가 얼마나 틀렸는지(Loss)를 알기 위해
2. 추론: 새 입력에 대한 예측을 얻기 위해

**예제:** 이미지 분류 모델에 고양이 사진을 넣으면, Forward 끝에 클래스 점수가 나옵니다.  
그 점수만 필요하면 Inference이고, 정답과 비교해 Loss까지 가면 Training step의 전반부입니다.

### 3.2 Computational Graph (계산 그래프) — Forward 관점

**Computational Graph(계산 그래프)**는 연산을 노드로, 값의 흐름을 간선으로 나타낸 그래프입니다.

Forward는 이 그래프를 **입력에서 출력 방향**으로 평가하는 일입니다.

```text
x ─┐
   ├─→ (·) ─→ z1 ─→ ReLU ─→ a1 ─┐
W1─┘                              ├─→ (·) ─→ ŷ ─→ Loss ← y
x ─┐                              │
   ├─→ (·) ─→ z2 ─→ ReLU ─→ a2 ─┘
W1─┘                    W2, b2
```

17강에서는 같은 그래프를 **반대 방향**으로 미분합니다.

### 3.3 Cache / Intermediate Values

Forward에서 나온 중간 값($z^{(l)}, a^{(l)}$)을 저장해 두는 것을 종종 **cache**라고 부릅니다.

**왜 필요한가?**  
Backpropagation은 $\partial L/\partial W$를 구할 때 Forward의 중간 값을 다시 사용합니다.  
예: $\partial L/\partial W^{(1)}$에는 입력 $x$와, ReLU 미분에 필요한 $z^{(1)}$ 정보가 필요합니다.

따라서 잘 짠 Forward 함수는 보통 다음을 반환합니다.

```text
예측값 ŷ + 중간 캐시(z, a, ...)
```

> 💡 **팁**
>
> Inference만 할 때는 cache를 버려도 됩니다.  
> 학습용 Forward라면 cache가 자산입니다.

### 3.4 Batch Forward

실무에서는 샘플 하나를 거의 다루지 않습니다.  
**Batch(배치)**는 여러 샘플을 한 번에 행렬로 묶어 처리하는 단위입니다.

샘플 하나: $\mathbf{x} \in \mathbb{R}^{d}$  
배치: $X \in \mathbb{R}^{B \times d}$ (이 책의 한 관례)

그러면

$$
Z = X W^{\top} + \mathbf{b}
$$

처럼 한 번에 계산합니다.  
오늘은 샘플 하나($B=1$)로 원리를 고정하고, 배치 형태는 뒤에서 shape와 함께 확장합니다.

---

## 직관적으로 이해하기

Forward는 **시험지를 앞에서부터 푸는 과정**입니다.

1. 문제(입력)를 읽습니다.
2. 1번 공식을 적용합니다(첫 층).
3. 중간 답을 메모합니다(activation cache).
4. 다음 공식에 넣습니다.
5. 최종 답(예측)을 적고, 정답과 비교해 감점(Loss)을 계산합니다.

아직 “어디서 틀렸는지 채점 코멘트(Gradient)”는 쓰지 않습니다. 그건 Backward입니다.

> **핵심**
>
> Forward만 100번 반복해도 파라미터가 안 바뀌면 Loss는 그대로입니다.  
> 학습에는 Backward + Update가 필수입니다.

---

## 수학적으로 이해하기 — 층마다 식과 shape

2-2-1 회귀 네트워크의 **전체** 식은 다음과 같습니다.

$$
\begin{aligned}
\mathbf{z}^{(1)} &= W^{(1)}\mathbf{x} + \mathbf{b}^{(1)} \\
\mathbf{a}^{(1)} &= \mathrm{ReLU}\!\left(\mathbf{z}^{(1)}\right) \\
\hat{y} &= W^{(2)}\mathbf{a}^{(1)} + b^{(2)} \\
L &= \frac{1}{2}\left(\hat{y} - y\right)^2
\end{aligned}
$$

### 5.1 기호와 shape 계약표

| 기호 | shape | 의미 |
|---|---|---|
| $\mathbf{x}$ | $(2,)$ | 입력 |
| $W^{(1)}$ | $(2,2)$ | 은닉 Weight (out × in) |
| $\mathbf{b}^{(1)}$ | $(2,)$ | 은닉 Bias |
| $\mathbf{z}^{(1)}, \mathbf{a}^{(1)}$ | $(2,)$ | 은닉 pre-/post-activation |
| $W^{(2)}$ | $(1,2)$ | 출력 Weight |
| $b^{(2)}$ | 스칼라(또는 $(1,)$) | 출력 Bias |
| $\hat{y}, L$ | 스칼라 | 예측과 Loss |

> ⚠️ **주의**
>
> $W$를 `(in, out)`로 두는 코드도 있습니다.  
> 이 책·NumPy 실습은 **`(out, in)`** 관례로 통일합니다. shape를 먼저 확인하세요.

### 5.2 Layer 1 — Affine (행렬-벡터)

입력과 Weight를 성분으로 쓰면

$$
\mathbf{x} = \begin{bmatrix}
x_1 \\
x_2
\end{bmatrix},\quad W^{(1)} = \begin{bmatrix}
W^{(1)}_{11} & W^{(1)}_{12} \\
W^{(1)}_{21} & W^{(1)}_{22}
\end{bmatrix},\quad \mathbf{b}^{(1)} = \begin{bmatrix}
b^{(1)}_1 \\
b^{(1)}_2
\end{bmatrix}
$$

Affine 결과는

$$
\mathbf{z}^{(1)} = W^{(1)}\mathbf{x}+\mathbf{b}^{(1)} = \begin{bmatrix}
W^{(1)}_{11}x_1 + W^{(1)}_{12}x_2 + b^{(1)}_1 \\
W^{(1)}_{21}x_1 + W^{(1)}_{22}x_2 + b^{(1)}_2
\end{bmatrix}
$$

shape 검사:

$$
\underbrace{(2\times 2)}_{W^{(1)}}
\cdot
\underbrace{(2\times 1)}_{\mathbf{x}}
+
\underbrace{(2\times 1)}_{\mathbf{b}^{(1)}} = \underbrace{(2\times 1)}_{\mathbf{z}^{(1)}}
$$

### 5.3 Layer 1 — ReLU

$$
a^{(1)}_i = \max(0,\ z^{(1)}_i)
\quad (i=1,2)
$$

벡터로:

$$
\mathbf{a}^{(1)} = \begin{bmatrix}
\max(0,z^{(1)}_1) \\
\max(0,z^{(1)}_2)
\end{bmatrix}
$$

$z\le 0$이면 그 뉴런의 출력은 0입니다. Forward만 봐도 “꺼진 뉴런”을 읽을 수 있습니다.

### 5.4 Layer 2 — 선형 출력

$$
\hat{y} = W^{(2)}\mathbf{a}^{(1)}+b^{(2)} = w^{(2)}_1 a^{(1)}_1 + w^{(2)}_2 a^{(1)}_2 + b^{(2)}
$$

shape:

$$
\underbrace{(1\times 2)}_{W^{(2)}}
\cdot
\underbrace{(2\times 1)}_{\mathbf{a}^{(1)}}
+
\underbrace{(1)}_{\,b^{(2)}} = \underbrace{(1)}_{\hat{y}}
$$

### 5.5 Loss

$$
L = \frac{1}{2}(\hat{y}-y)^2
$$

$\tfrac12$는 미분 때 계수를 깔끔하게 하려는 선택입니다(제13·17강).

### 5.6 깊은 망으로의 일반화

층이 더 깊어져도 패턴은 같습니다.

$$
\mathbf{a}^{(l)} = \sigma\!\left(W^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}\right)
$$

출력층만 문제 유형에 따라 Activation을 달리합니다.

- 회귀: 선형 출력
- 이진 분류: Sigmoid
- 다중 분류: Softmax (2권에서 본격 학습)

---

## 작은 숫자로 직접 계산하기 — 예제 A (기본)

제15강과 **동일한 파라미터**를 사용합니다.

```text
x  = [1.0, 0.5]
y  = 1.0          # 정답 타깃

W1 = [[0.3, -0.2],
      [0.4,  0.1]]
b1 = [0.1, -0.1]

W2 = [[0.5, -0.4]]
b2 = 0.2
```

행렬로 적으면 다음과 같습니다.

$$
\mathbf{x} = \begin{bmatrix} 1.0 \\ 0.5 \end{bmatrix},\quad W^{(1)} = \begin{bmatrix}
0.3 & -0.2 \\
0.4 & 0.1
\end{bmatrix},\quad \mathbf{b}^{(1)} = \begin{bmatrix} 0.1 \\ -0.1 \end{bmatrix}
$$

$$
W^{(2)} = \begin{bmatrix} 0.5 & -0.4 \end{bmatrix},\quad b^{(2)}=0.2,\quad y=1.0
$$

### 6.1 Step 1 — 은닉 Pre-activation

성분별:

$$
\begin{aligned}
z_1^{(1)} &= 0.3\cdot 1.0 + (-0.2)\cdot 0.5 + 0.1 = 0.30 \\
z_2^{(1)} &= 0.4\cdot 1.0 + 0.1\cdot 0.5 - 0.1 = 0.35
\end{aligned}
$$

$$
\mathbf{z}^{(1)} = \begin{bmatrix} 0.30 \\ 0.35 \end{bmatrix}
$$

### 6.2 Step 2 — ReLU

$$
\mathbf{a}^{(1)} = \begin{bmatrix}
\max(0,0.30) \\
\max(0,0.35)
\end{bmatrix} = \begin{bmatrix} 0.30 \\ 0.35 \end{bmatrix}
$$

두 뉴런 모두 켜져 있습니다 ($z>0$).

### 6.3 Step 3 — 출력

$$
\hat{y} = 0.5\cdot 0.30 + (-0.4)\cdot 0.35 + 0.2 = 0.15 - 0.14 + 0.2 = 0.21
$$

### 6.4 Step 4 — Loss

$$
L = \frac{1}{2}(0.21 - 1.0)^2 = \frac{1}{2}(0.79)^2 = 0.31205
$$

### 6.5 Forward 결과표 (이후 강의용 캐시)

| 이름 | 값 | shape |
|---|---|---|
| $x$ | $[1.0,\ 0.5]$ | $(2,)$ |
| $z^{(1)}$ | $[0.30,\ 0.35]$ | $(2,)$ |
| $a^{(1)}$ | $[0.30,\ 0.35]$ | $(2,)$ |
| $\hat{y}$ | $0.21$ | 스칼라 |
| $y$ | $1.0$ | 스칼라 |
| $L$ | $0.31205$ | 스칼라 |

17강에서는 이 표를 펼쳐 놓고 모든 편미분을 계산합니다.

> 💡 **팁**
>
> 손계산 후에는 반드시 “각 단계의 shape가 표와 같은지”를 한 번 더 확인하세요.  
> 숫자 실수보다 shape 실수가 더 자주 버그를 만듭니다.

---

## 작은 숫자로 직접 계산하기 — 예제 B (ReLU가 꺼지는 경우)

$x = [0.0,\ 1.0]$로 바꿉니다. 파라미터는 그대로입니다.

### 7.1 Affine

$$
\begin{aligned}
z_1^{(1)}
&= 0.3\cdot 0 + (-0.2)\cdot 1 + 0.1
= -0.1 \\
z_2^{(1)}
&= 0.4\cdot 0 + 0.1\cdot 1 - 0.1
= 0.0
\end{aligned}
$$

$$
\mathbf{z}^{(1)} = \begin{bmatrix} -0.1 \\ 0.0 \end{bmatrix}
$$

### 7.2 ReLU

$$
\mathbf{a}^{(1)} = \begin{bmatrix}
\max(0,-0.1) \\
\max(0,0.0)
\end{bmatrix} = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}
$$

### 7.3 출력과 Loss

$$
\hat{y} = 0.5\cdot 0 + (-0.4)\cdot 0 + 0.2 = 0.2
$$

$y=1.0$이면

$$
L = \frac{1}{2}(0.2-1.0)^2 = \frac{1}{2}(0.8)^2 = 0.32
$$

> ⚠️ **주의**
>
> Forward만 보면 “출력이 0.2”로 끝납니다.  
> 그러나 Backward에서는 $a^{(1)}=0$인 경로의 Gradient가 막힐 수 있습니다.  
> 중간 값 $z^{(1)}$을 기록해 두어야 그 현상이 설명됩니다 (제17강).

---

## 배치 Forward — shape를 숫자로

두 샘플을 한 번에 처리합니다.

$$
X = \begin{bmatrix}
1.0 & 0.5 \\
0.0 & 1.0
\end{bmatrix}
\in \mathbb{R}^{2\times 2}
\quad (B=2,\ d=2)
$$

배치 관례 ($W$는 out×in 유지):

$$
\begin{aligned}
Z^{(1)} &= X\,(W^{(1)})^{\top} + \mathbf{b}^{(1)} && (B\times 2) \\
A^{(1)} &= \mathrm{ReLU}(Z^{(1)}) && (B\times 2) \\
\hat{Y} &= A^{(1)}\,(W^{(2)})^{\top} + b^{(2)} && (B\times 1)
\end{aligned}
$$

첫 행은 예제 A, 둘째 행은 예제 B와 같아야 합니다.

$$
Z^{(1)} = \begin{bmatrix}
0.30 & 0.35 \\
-0.10 & 0.00
\end{bmatrix},\quad A^{(1)} = \begin{bmatrix}
0.30 & 0.35 \\
0.00 & 0.00
\end{bmatrix},\quad \hat{Y} = \begin{bmatrix}
0.21 \\
0.20
\end{bmatrix}
$$

평균 MSE ($y_1=y_2=1$이라고 가정하면):

$$
L = \frac{1}{2B}\sum_{n=1}^{B}(\hat{y}_n-y_n)^2 = \frac{1}{4}\left[(0.21-1)^2+(0.20-1)^2\right] = \frac{1}{4}(0.6241+0.64) = 0.316025
$$

> 📘 **심화**
>
> 배치에서도 **샘플 축만 늘었을 뿐**, 샘플마다의 계산은 단샘플과 동일합니다.  
> 벡터화는 속도를 위한 것이지, 수학을 바꾸는 것이 아닙니다.

---

## 코드로 구현하기

### 11.1 샘플 하나 Forward

```python
"""16강: 2-2-1 Forward Propagation (손계산과 동일 숫자)."""

import numpy as np

def relu(z: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0.0, z)

def mse_loss(y_hat: float, y: float) -> float:
    """1/2 * (y_hat - y)^2  — 미분 시 계수 정리용."""
    return 0.5 * (y_hat - y) ** 2

def forward(x, W1, b1, W2, b2, y=None):
    """순전파. cache에 중간 값을 담아 반환합니다.

    Returns
    -------
    y_hat : float
    loss  : float | None
    cache : dict  — Backprop에 필요한 중간 결과
    """
    # (1) 은닉 Affine
    z1 = W1 @ x + b1
    # (2) 은닉 Activation
    a1 = relu(z1)
    # (3) 출력 Affine (선형)
    y_hat = float((W2 @ a1 + b2).reshape(()))

    loss = None
    if y is not None:
        loss = mse_loss(y_hat, float(y))

    # Backprop을 위해 Forward 중간값을 보관
    cache = {
        "x": x,
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "z1": z1,
        "a1": a1,
        "y_hat": y_hat,
        "y": y,
    }
    return y_hat, loss, cache

if __name__ == "__main__":
    x = np.array([1.0, 0.5])
    y = 1.0
    W1 = np.array([[0.3, -0.2], [0.4, 0.1]])
    b1 = np.array([0.1, -0.1])
    W2 = np.array([[0.5, -0.4]])
    b2 = np.array([0.2])

    y_hat, loss, cache = forward(x, W1, b1, W2, b2, y=y)
    print("z1   =", cache["z1"])
    print("a1   =", cache["a1"])
    print("yhat =", y_hat)
    print("loss =", loss)
    # 기대값: z1=[0.3 0.35], a1=[0.3 0.35], yhat=0.21, loss=0.31205
```

### 11.2 배치 Forward

```python
def forward_batch(X, W1, b1, W2, b2):
    """X: shape (B, 2) — 여러 샘플을 한 번에.

    여기서는 W를 (out, in)로 두었으므로
    Z1 = X @ W1.T + b1  형태를 사용합니다.
    """
    # 배치 행렬곱: (B,2) @ (2,2) -> (B,2)
    Z1 = X @ W1.T + b1
    A1 = relu(Z1)
    # (B,2) @ (2,1) -> (B,1)
    Y_hat = A1 @ W2.T + b2
    return Y_hat, {"Z1": Z1, "A1": A1}


# 예제 A+B 동시 검증
X = np.array([[1.0, 0.5], [0.0, 1.0]])
Y_hat, cache = forward_batch(X, W1, b1, W2, b2)
print(Y_hat.ravel())  # ≈ [0.21, 0.20]
```

### 11.3 이진 분류용 Forward (Sigmoid 출력)

회귀가 아니라 확률을 내고 싶다면 출력에 Sigmoid를 붙입니다.

$$
p = \sigma(z^{(2)}) = \frac{1}{1+e^{-z^{(2)}}},\quad z^{(2)} = W^{(2)}\mathbf{a}^{(1)}+b^{(2)}
$$

```python
def sigmoid(z):
    """수치적으로 단순한 sigmoid."""
    return 1.0 / (1.0 + np.exp(-z))

def forward_binary(x, W1, b1, W2, b2):
    """은닉 ReLU + 출력 Sigmoid."""
    z1 = W1 @ x + b1
    a1 = relu(z1)
    z2 = float((W2 @ a1 + b2).reshape(()))
    y_hat = sigmoid(z2)  # (0,1) 확률
    return y_hat, {"z1": z1, "a1": a1, "z2": z2}
```

예제 A 숫자로 $z^{(2)}=0.21$이면

$$
p = \sigma(0.21) \approx 0.552
$$

Loss는 Binary Cross Entropy를 쓰는 것이 자연스럽습니다(제13강 연장).  
18강 실습에서 분류 루프를 한 번 돌립니다.

---

## Forward만으로 할 수 있는 일 / 없는 일

| 할 수 있다 | 할 수 없다 |
|---|---|
| 현재 파라미터로 예측 | 파라미터를 개선 |
| Loss 수치 확인 | Loss를 낮추는 방향 계산 |
| Inference / 생성 루프 | Weight 업데이트 |
| 버그 확인(NaN, shape) | “어느 연결이 잘못됐는지” 귀인 |

학생이 자주 하는 착각은 “Forward를 여러 번 돌리면 학습된다”는 것입니다.  
같은 파라미터로 Forward를 100번 해도 Loss는 그대로입니다.

---

## 수식 보강 — 층별 Forward

$$
\mathbf{z}^{(l)}=W^{(l)}\mathbf{a}^{(l-1)}+\mathbf{b}^{(l)},\quad
\mathbf{a}^{(l)}=\sigma(\mathbf{z}^{(l)})
$$

$\mathbf{a}^{(0)}=\mathbf{x}$. 출력층 활성은 문제에 따라 softmax/항등입니다.

## LLM에서는 어디에 사용될까?

### 13.1 Training step의 전반부

LLM 학습의 한 step:

```text
tokens → Embedding → Transformer Blocks → LM Head → logits
logits + labels → Cross Entropy Loss
```

여기까지가 Forward입니다. 그다음 `loss.backward()`가 Backward입니다.

### 13.2 Inference = Forward 반복

텍스트 생성:

```text
[문맥 토큰들]
  → Forward 한 번 → 다음 토큰 분포
    → 토큰 하나 선택
      → 문맥에 붙이기
        → 다시 Forward …
```

KV Cache 같은 최적화(5권)는 “매 스텝 전체 Forward를 덜 반복하게” 만드는 기법이지, Forward 자체를 없애지는 않습니다.

### 13.3 Teacher Forcing

학습 시에는 정답 다음 토큰을 입력으로 넣어 **한 번의 Forward로 여러 위치의 Loss**를 계산합니다.  
이것도 Forward Propagation의 배치·시퀀스 확장입니다.

전형적인 logits shape:

$$
\text{logits} \in \mathbb{R}^{B \times S \times V}
$$

($B$=배치, $S$=시퀀스 길이, $V$=어휘 크기)

---

## 실습

### 실습 1 — 손계산 검증

예제 A 숫자를 가리고 다시 손으로 계산한 뒤, 11.1 코드 결과와 비교합니다.  
한 자리라도 다르면 shape 또는 Activation 위치 오류입니다.

### 실습 2 — cache 없이 구현해 보기

`forward`가 `y_hat`만 반환하게 바꾼 뒤, “나중에 $W1$ Gradient를 구하려면 무엇이 부족한가?”를 문장으로 적습니다.

### 실습 3 — 입력 변경

$x=[2.0,-1.0]$으로 바꿔 $z1,a1,\hat{y},L$을 손과 코드로 맞춥니다.  
(기대: $a^{(1)}=[0.9,0.6]$, $\hat{y}=0.41$, $L=0.17405$)

### 실습 4 — 배치 두 샘플

```python
X = np.array([
    [1.0, 0.5],
    [0.0, 1.0],
])
```

`forward_batch`로 두 예측을 한 번에 구하고, 예제 A·B 손계산과 비교합니다.

### 실습 5 — Loss 곡면 맛보기 (선택)

$W2_{11}$만 $-1.0$부터 $1.0$까지 0.1 간격으로 바꾸며 Loss를 기록해 봅니다.  
Forward만으로도 파라미터–Loss 지형을 엿볼 수 있습니다.

---

## 자주 하는 실수

1. **Activation 순서를 바꿉니다**  
   $W(\mathrm{ReLU}(x))$와 $\mathrm{ReLU}(Wx)$는 다릅니다. 표준 MLP는 Affine → Activation입니다.

2. **Bias를 행렬곱 뒤에 안 더합니다**  
   `W @ x`만 하고 끝내면 표현력이 줄어듭니다.

3. **shape 불일치**  
   `(2,2) @ (2,)`는 되지만 `(2,2) @ (1,2)`는 아닙니다. 배치를 넣으면 전치 위치가 달라질 수 있습니다.

4. **Loss 없는 Forward와 있는 Forward를 혼동합니다**  
   Inference는 보통 Loss가 없습니다. Training Forward는 정답 텐서가 함께 있습니다.

5. **중간 값을 버려 버립니다**  
   학습용 Forward라면 cache가 자산입니다. 17~18강에서 바로 씁니다.

6. **부동소수점 기대를 과도하게 합니다**  
   `0.31205`는 이 예제의 유한 소수입니다. 더 복잡한 망에서는 근사 비교(`np.allclose`)가 필요합니다.

---

## 핵심 요약

- Forward Propagation은 입력→층→예측(→Loss)으로 계산 그래프를 앞에서 뒤로 평가하는 과정입니다.
- 2-2-1에서 $z^{(1)}\to a^{(1)}\to\hat{y}\to L$ 순서를 **식·shape·숫자**로 일치시켜야 합니다.
- 중간 값 cache는 Backpropagation의 연료입니다.
- 배치 Forward는 샘플 축만 늘린 동일 계산입니다.
- Forward만으로는 예측과 Loss 확인만 가능하고, 학습에는 Backward가 필요합니다.
- LLM의 학습 전반부와 생성 Inference 모두 Forward에 기반합니다.

## 용어 사전

| 용어 | 의미 |
|---|---|
| Forward Propagation | 입력을 앞→뒤로 통과시켜 예측/Loss를 계산하는 과정 |
| Forward Pass | Forward Propagation과 같은 말 |
| Computational Graph | 연산과 값의 의존 관계를 나타낸 그래프 |
| Cache / Intermediate | Backward를 위해 보관하는 Forward 중간 값 |
| Pre-activation $z$ | Affine 결과, Activation 직전 |
| Activation $a$ | 비선형 통과 후 다음 층 입력 |
| Prediction $\hat{y}$ | 네트워크 출력 |
| Loss $L$ | 예측과 정답의 차이를 스칼라로 측정한 값 |
| Batch | 여러 샘플을 묶어 한 번에 Forward하는 단위 |
| Inference | 학습 없이 예측만 수행하는 실행 모드 |
| Shape 계약 | 각 텐서의 허용 차원·길이 약속 |

## 연습문제

### 문제 1 (개념)

Forward Propagation을 “계산 그래프” 관점에서 한 문장으로 정의하시오.

### 문제 2 (계산)

예제 A에서 $b2$만 $0.2$ → $0.0$으로 바꾸면 $\hat{y}$와 $L$은?

### 문제 3 (계산)

$W2=[0.5,\ -0.4]$ 대신 $W2=[1.0,\ 0.0]$이면 $\hat{y}$는? ($a^{(1)}$은 그대로 $[0.3,0.35]$, $b2=0.2$)

### 문제 4 (계산)

예제 B에서 $\mathbf{z}^{(1)}$과 $\mathbf{a}^{(1)}$을 행렬로 쓰시오.

### 문제 5 (코드)

`cache`에 `z1`을 넣지 않으면, ReLU 구간의 Backprop에서 어떤 정보가 빠지는가?

### 문제 6 (연결)

LLM 텍스트 생성 10토큰을 만들 때 Forward는 최소 몇 번 개념적으로 필요한가? (KV Cache 없는 단순 모델 가정)

### 문제 7 (shape)

$X\in\mathbb{R}^{4\times 2}$, $W^{(1)}\in\mathbb{R}^{3\times 2}$일 때 $Z^{(1)}=X(W^{(1)})^{\top}$의 shape는?

---

## 정답 및 해설

### 문제 1

계산 그래프의 입력 노드에서 출력(Loss) 노드 방향으로 각 연산을 차례로 평가해 예측과 Loss를 얻는 과정입니다.

### 문제 2

$\hat{y}=0.15-0.14+0=0.01$,  
$L=\frac12(0.01-1)^2=\frac12(0.99)^2=0.49005$.

### 문제 3

$\hat{y}=1.0\cdot0.3 + 0.0\cdot0.35 + 0.2 = 0.5$.

### 문제 4

$$
\mathbf{z}^{(1)}=\begin{bmatrix}-0.1\\0.0\end{bmatrix},\quad \mathbf{a}^{(1)}=\begin{bmatrix}0\\0\end{bmatrix}
$$

### 문제 5

ReLU의 미분은 $z>0$에서 1, $z\le0$에서 0입니다. $z1$(또는 동등한 마스크)이 없으면 어느 뉴런이 켜져 있었는지 알 수 없습니다.

### 문제 6

매 토큰마다 한 번의 Forward가 필요하므로 개념적으로 **10번**입니다. (실제 구현은 캐시로 중복 계산을 줄입니다.)

### 문제 7

$(4\times 2)\,(2\times 3)=(4\times 3)$.

## 다음 강의와 연결

이번 강의에서 신호가 **앞에서 뒤로** 흐르는 길을 고정했습니다.

다음 **제17강. Backpropagation 직접 계산하기**에서는, 같은 2-2-1 숫자와 같은 cache를 사용해 Loss에서부터 **모든 Weight·Bias의 편미분**을 손으로 구합니다.  
제14강 Chain Rule이 네트워크 위에서 실제로 작동하는 장면을 보게 됩니다.

> 순방향 계산표를 덮지 말고 옆에 두자. 곧 거꾸로 내려옵니다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [15강. Neural Network의 구조](15강_Neural_Network의_구조.md)
- **다음 강:** [17강. Backpropagation 직접 계산하기](17강_Backpropagation_직접_계산하기.md)

<!-- /LECTURE_NAV -->
