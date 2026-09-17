# 17강. Backpropagation 직접 계산하기

## 이번 강에서 배우는 내용

- Backpropagation이 Computational Graph 위에서 Chain Rule을 체계적으로 적용하는 알고리즘임을 설명
- Loss → 출력층 → 은닉층 순으로 Gradient가 흐르는 이유
- 2-2-1 네트워크에서 $\partial L/\partial W^{(2)}, \partial L/\partial b^{(2)}, \partial L/\partial W^{(1)}, \partial L/\partial b^{(1)}$을 **끝까지** 직접 계산
- Local Gradient와 Upstream Gradient의 곱으로 각 노드의 Gradient를 얻는 습관
- ReLU가 꺼진 샘플에서 Gradient가 어떻게 막히는지
- `loss.backward()`가 내부적으로 하는 일의 손계산 버전

## 왜 중요한가?

딥러닝 프레임워크는 `loss.backward()` 한 줄로 Gradient를 채워 줍니다.  
하지만 그 한 줄이 하는 일은 결국 다음뿐입니다.

```text
Loss
  → Computational Graph를 거꾸로
    → Chain Rule로 각 텐서의 Gradient 계산
      → parameter.grad에 저장
```

손계산을 한 번이라도 끝까지 해 보면,

1. shape가 맞는지
2. Activation 미분이 어디에 곱해지는지
3. Weight Gradient가 “입력 × upstream” 형태인 이유

가 몸이 기억합니다. 18강의 NumPy 구현과 20강의 Autograd는 이 손계산의 자동화입니다.

> **핵심**
>
> Backprop = Forward cache + Upstream×Local을 출력→입력으로 반복.

## 선수 개념

### 2.1 Chain Rule 복습 (제14강)

합성 $L = f(g(w))$이면

$$
\frac{dL}{dw} = \frac{dL}{dg}\cdot\frac{dg}{dw}
$$

여러 경로가 있으면 경로별 기여를 **더합니다**.

### 2.2 Upstream / Local Gradient

어떤 노드 $v$에서

- **Upstream Gradient**: 뒤쪽(Loss 쪽)에서 흘러 온 $\partial L/\partial v$
- **Local Gradient**: 그 노드 연산의 순간 기울기 $\partial v/\partial(\text{입력})$

이면

$$
\frac{\partial L}{\partial(\text{입력})}
=
\underbrace{\frac{\partial L}{\partial v}}_{\text{upstream}}
\cdot
\underbrace{\frac{\partial v}{\partial(\text{입력})}}_{\text{local}}
$$

입니다.

### 2.3 Forward cache (16강 결과)

| 이름 | 값 |
|---|---|
| $x$ | $[1.0,\ 0.5]$ |
| $W^{(1)}$ | $\begin{bmatrix}0.3 & -0.2\\0.4 & 0.1\end{bmatrix}$ |
| $b^{(1)}$ | $[0.1,\ -0.1]$ |
| $z^{(1)}$ | $[0.30,\ 0.35]$ |
| $a^{(1)}$ | $[0.30,\ 0.35]$ |
| $W^{(2)}$ | $[0.5,\ -0.4]$ |
| $b^{(2)}$ | $0.2$ |
| $\hat{y}$ | $0.21$ |
| $y$ | $1.0$ |
| $L$ | $0.31205$ |

---

## 핵심 개념

### 3.1 Backpropagation

**Backpropagation(역전파, Backprop)**는 Loss에서 시작해 계산 그래프를 **출력→입력 방향**으로 따라가며, Chain Rule로 각 파라미터의 Gradient를 효율적으로 계산하는 알고리즘입니다.

파라미터가 9개인 2-2-1에서도 수치 미분은 최소 9번의 추가 Forward가 필요합니다. Backprop은 Forward 한 번 + Backward 한 번으로 **모든 파라미터 Gradient**를 얻습니다. LLM처럼 파라미터가 수십억이면 차이는 절대적입니다.

### 3.2 전체 파이프라인

```text
1. Forward로 L 계산, 중간값 cache
2. ∂L/∂ŷ 부터 시작
3. 출력층 Weight/Bias Gradient
4. 은닉 Activation을 지나 은닉 Affine으로
5. 은닉 Weight/Bias Gradient
6. θ ← θ - η · ∇θ L
```

PyTorch에서는 2~5가 `loss.backward()`에 해당합니다(제20강).

### 3.3 Error Signal / Delta

$$
\delta^{(l)} = \frac{\partial L}{\partial \mathbf{z}^{(l)}}
$$

를 종종 **delta(δ, 오차 신호)**라고 부릅니다. 가중치 Gradient는 대체로

$$
\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)}\,(\mathbf{a}^{(l-1)})^{\top}
$$

형태입니다. 오늘은 성분별로 풀어서 이 공식의 출처를 확인합니다.

---

## 직관적으로 이해하기

Forward가 “시험 답안을 앞에서 뒤로 작성”이라면,  
Backprop은 **채점 코멘트를 뒤에서 앞으로 전달**하는 일입니다.

1. 최종 점수(Loss)를 봅니다. “0.21인데 정답은 1.0, 너무 작다.”
2. 출력층 손잡이($W2, b2$)에게 “출력을 키우는 방향”을 알려 줍니다.
3. 그 메시지를 은닉 뉴런에게 번역해 전달합니다.
4. 꺼져 있던 ReLU 뉴런($z\le0$)에게는 “지금은 손을 대지 마(Gradient 0)”라고 합니다.
5. 모든 손잡이에 “얼마나, 어느 방향으로 돌릴지” 쪽지가 붙습니다.

쪽지가 곧 Gradient입니다.

---

## 수학적으로 이해하기 — 준비 미분

### 5.1 Loss 미분

$$
L = \frac{1}{2}(\hat{y} - y)^2
$$

$$
\frac{\partial L}{\partial \hat{y}} = (\hat{y} - y)
$$

$1/2$와 제곱의 2가 상쇄됩니다.

### 5.2 출력층

$$
\hat{y} = w^{(2)}_1 a^{(1)}_1 + w^{(2)}_2 a^{(1)}_2 + b^{(2)}
$$

$$
\frac{\partial \hat{y}}{\partial w^{(2)}_i} = a^{(1)}_i,\quad
\frac{\partial \hat{y}}{\partial b^{(2)}} = 1,\quad
\frac{\partial \hat{y}}{\partial a^{(1)}_i} = w^{(2)}_i
$$

### 5.3 ReLU 미분

$$
\frac{d}{dz}\mathrm{ReLU}(z) =
\begin{cases}
1 & z > 0 \\
0 & z < 0 \\
\text{보통 0} & z = 0
\end{cases}
$$

코드에서는 `(z > 0).astype(float)` 마스크를 곱합니다.

### 5.4 은닉 Affine

$$
z^{(1)}_i = \sum_j W^{(1)}_{ij} x_j + b^{(1)}_i
$$

$$
\frac{\partial z^{(1)}_i}{\partial W^{(1)}_{ij}} = x_j,\quad
\frac{\partial z^{(1)}_i}{\partial b^{(1)}_i} = 1
$$

---

## 작은 숫자로 직접 계산하기 — 모든 편미분

목표: 숫자로 $\nabla L$을 전부 구합니다.

### 6.1 Step A — $\partial L/\partial\hat{y}$

$$
\frac{\partial L}{\partial\hat{y}} = \hat{y} - y = 0.21 - 1.0 = -0.79
$$

해석: Loss를 줄이려면 $\hat{y}$를 **키워야** 합니다.  

$$
Gradient Descent는 $w \leftarrow w - \eta \frac{\partial L}{\partial w}$이므로, $\partial L/\partial\hat{y}<0$이면 $\hat{y}$를 늘리는 쪽 파라미터가 강화됩니다.
$$

### 6.2 Step B — 출력층 Bias

$$
\frac{\partial L}{\partial b^{(2)}}
= \frac{\partial L}{\partial\hat{y}}\cdot 1
= -0.79
$$

### 6.3 Step C — 출력층 Weight

$$
\begin{aligned}
\frac{\partial L}{\partial w^{(2)}_1}
&= (-0.79)\cdot 0.30 = -0.237 \\
\frac{\partial L}{\partial w^{(2)}_2}
&= (-0.79)\cdot 0.35 = -0.2765
\end{aligned}
$$

$$
\frac{\partial L}{\partial W^{(2)}}
=
\begin{bmatrix}-0.237 & -0.2765\end{bmatrix}
=
\delta^{(2)}\,(\mathbf{a}^{(1)})^{\top}
,\quad
\delta^{(2)}=-0.79
$$

### 6.4 Step D — 은닉 Activation으로 흘리기

$$
\begin{aligned}
\frac{\partial L}{\partial a^{(1)}_1}
&= (-0.79)\cdot 0.5 = -0.395 \\
\frac{\partial L}{\partial a^{(1)}_2}
&= (-0.79)\cdot (-0.4) = 0.316
\end{aligned}
$$

$$
\frac{\partial L}{\partial \mathbf{a}^{(1)}}
=
\begin{bmatrix}-0.395 \\ 0.316\end{bmatrix}
$$

### 6.5 Step E — ReLU → Pre-activation

$z^{(1)}_1=0.30>0$, $z^{(1)}_2=0.35>0$이므로 ReLU 미분은 둘 다 1.

$$
\boldsymbol{\delta}^{(1)}
=
\begin{bmatrix}-0.395 \\ 0.316\end{bmatrix}
\odot
\begin{bmatrix}1 \\ 1\end{bmatrix}
=
\begin{bmatrix}-0.395 \\ 0.316\end{bmatrix}
$$

만약 $z_1\le0$이었다면 $\delta^{(1)}_1=0$이 되어, 그 뉴런으로 가는 Weight 학습이 이번 스텝에서 멈춥니다.

### 6.6 Step F — 은닉 Bias

$$
\frac{\partial L}{\partial \mathbf{b}^{(1)}}
=
\boldsymbol{\delta}^{(1)}
=
\begin{bmatrix}-0.395 \\ 0.316\end{bmatrix}
$$

### 6.7 Step G — 은닉 Weight (모든 성분)

$$
\frac{\partial L}{\partial W^{(1)}_{ij}} = \delta^{(1)}_i\, x_j
$$

$$
\begin{aligned}
\frac{\partial L}{\partial W^{(1)}_{11}} &= (-0.395)\cdot 1.0 = -0.395 \\
\frac{\partial L}{\partial W^{(1)}_{12}} &= (-0.395)\cdot 0.5 = -0.1975 \\
\frac{\partial L}{\partial W^{(1)}_{21}} &= (0.316)\cdot 1.0 = 0.316 \\
\frac{\partial L}{\partial W^{(1)}_{22}} &= (0.316)\cdot 0.5 = 0.158
\end{aligned}
$$

$$
\frac{\partial L}{\partial W^{(1)}}
=
\begin{bmatrix}
-0.395 & -0.1975 \\
0.316 & 0.158
\end{bmatrix}
=
\boldsymbol{\delta}^{(1)}\,\mathbf{x}^{\top}
=
\begin{bmatrix}-0.395\\0.316\end{bmatrix}
\begin{bmatrix}1.0 & 0.5\end{bmatrix}
$$

### 6.8 (참고) 입력으로의 Gradient

$$
\frac{\partial L}{\partial \mathbf{x}}
=
(W^{(1)})^{\top}\boldsymbol{\delta}^{(1)}
$$

$$
\begin{aligned}
\frac{\partial L}{\partial x_1}
&= (-0.395)(0.3)+(0.316)(0.4)=0.0079 \\
\frac{\partial L}{\partial x_2}
&= (-0.395)(-0.2)+(0.316)(0.1)=0.1106
\end{aligned}
$$

LLM에서는 여기를 더 앞 블록(Attention, Embedding)으로 계속 흘립니다.

### 6.9 결과 요약표

| 파라미터 | Gradient |
|---|---|
| $b^{(2)}$ | $-0.79$ |
| $W^{(2)}$ | $\begin{bmatrix}-0.237 & -0.2765\end{bmatrix}$ |
| $b^{(1)}$ | $\begin{bmatrix}-0.395\\0.316\end{bmatrix}$ |
| $W^{(1)}$ | $\begin{bmatrix}-0.395 & -0.1975\\ 0.316 & 0.158\end{bmatrix}$ |

### 6.10 한 스텝 업데이트 (맛보기)

학습률 $\eta=0.1$이면

$$
b^{(2)} \leftarrow 0.2 - 0.1\cdot(-0.79) = 0.279
$$

$$
w^{(2)}_1 \leftarrow 0.5 - 0.1\cdot(-0.237) = 0.5237
$$

새 파라미터로 Forward하면 $\hat{y}$가 0.21보다 커져 $y=1$에 조금 가까워지고, Loss가 줄어드는 것이 정상입니다.  
18강에서 반복 루프로 확인합니다.

> 💡 **팁**
>
> 표의 숫자를 외울 필요는 없습니다.  
> **순서**(Loss→출력→ReLU→은닉)와 **패턴**($\partial L/\partial W=\delta a^{\top}$)만 몸에 남기면 됩니다.

---

## 경로별로 Chain Rule 펼치기 (한 Weight만 확대)

$W^{(1)}_{11}$ 하나만 끝까지 추적합니다.

```text
W11 → z1 → a1 → ŷ → L
```

$$
\begin{aligned}
\frac{\partial L}{\partial W_{11}}
&=
\frac{\partial L}{\partial\hat{y}}
\cdot
\frac{\partial\hat{y}}{\partial a_1}
\cdot
\frac{\partial a_1}{\partial z_1}
\cdot
\frac{\partial z_1}{\partial W_{11}} \\
&=
(-0.79)\cdot(0.5)\cdot(1)\cdot(1.0)
=
-0.395
\end{aligned}
$$

6.7의 결과와 같습니다.  
Backprop의 장점은, 이 곱셈을 노드마다 **한 번만** 정리해 재사용하는 것입니다.  
$\partial L/\partial a_1=-0.395$를 한 번 구해 두면 $W_{11}, W_{12}, b_1$이 공유합니다.

---

## 예제 B — ReLU가 꺼진 샘플의 Backprop

16강 예제 B: $x=[0,1]$, $z^{(1)}=[-0.1,0]$, $a^{(1)}=[0,0]$, $\hat{y}=0.2$, $y=1$.

$$
L=\frac12(0.2-1)^2=0.32
,\quad
\frac{\partial L}{\partial\hat{y}}=0.2-1=-0.8
$$

출력층:

$$
\frac{\partial L}{\partial W^{(2)}}
=
(-0.8)\,
\begin{bmatrix}0 & 0\end{bmatrix}
=
\begin{bmatrix}0 & 0\end{bmatrix}
,\quad
\frac{\partial L}{\partial b^{(2)}}=-0.8
$$

은닉으로:

$$
\frac{\partial L}{\partial \mathbf{a}^{(1)}}
=
(-0.8)
\begin{bmatrix}0.5\\-0.4\end{bmatrix}
=
\begin{bmatrix}-0.4\\0.32\end{bmatrix}
$$

그러나 ReLU 마스크가 $[0,0]$이므로

$$
\boldsymbol{\delta}^{(1)}
=
\begin{bmatrix}-0.4\\0.32\end{bmatrix}
\odot
\begin{bmatrix}0\\0\end{bmatrix}
=
\begin{bmatrix}0\\0\end{bmatrix}
$$

따라서

$$
\frac{\partial L}{\partial W^{(1)}}
=
\begin{bmatrix}0 & 0\\0 & 0\end{bmatrix}
,\quad
\frac{\partial L}{\partial \mathbf{b}^{(1)}}
=
\begin{bmatrix}0\\0\end{bmatrix}
$$

> ⚠️ **주의**
>
> 은닉이 전부 꺼지면 **은닉 Weight는 이번 스텝에서 학습되지 않습니다**.  
> 출력 Bias만 $-0.8$ 방향으로 움직입니다. Dying ReLU 논의의 씨앗입니다.

---

## 수치 미분으로 검증하기

$$
\frac{\partial L}{\partial \theta}
\approx
\frac{L(\theta+\epsilon)-L(\theta-\epsilon)}{2\epsilon}
$$

```python
"""17강: 수치 미분으로 ∂L/∂W2[0] 검증."""

import numpy as np

def relu(z):
    return np.maximum(0.0, z)

def forward_loss(x, y, W1, b1, W2, b2):
    a1 = relu(W1 @ x + b1)
    y_hat = float(W2 @ a1 + b2)
    return 0.5 * (y_hat - y) ** 2

x = np.array([1.0, 0.5]); y = 1.0
W1 = np.array([[0.3, -0.2], [0.4, 0.1]])
b1 = np.array([0.1, -0.1])
W2 = np.array([0.5, -0.4]); b2 = 0.2

eps = 1e-6
e0 = np.array([1.0, 0.0])
L_plus = forward_loss(x, y, W1, b1, W2 + eps * e0, b2)
L_minus = forward_loss(x, y, W1, b1, W2 - eps * e0, b2)
num_grad = (L_plus - L_minus) / (2 * eps)
print("numeric ∂L/∂W2_1 =", num_grad)  # ≈ -0.237
print("analytic         =", -0.237)
```

모든 파라미터에 대해 이 검사를 돌리는 습관이, 18강 구현의 버그를 잡는 가장 확실한 방법입니다.

---

## Computational Graph에서의 Backward 스케치

```text
Forward:
x,W1,b1 → z1 → a1 →(+W2,b2)→ ŷ → L

Backward (화살표는 Gradient 흐름):
∂L/∂L = 1
   ↓
∂L/∂ŷ = -0.79
   ↓
  ┌─ ∂L/∂W2 = -0.79 * a1
  ├─ ∂L/∂b2 = -0.79
  └─ ∂L/∂a1 = -0.79 * W2
         ↓
     ∂L/∂z1 = ∂L/∂a1 ⊙ ReLU'(z1)
         ↓
      ┌─ ∂L/∂W1 = δ1 x^T
      ├─ ∂L/∂b1 = δ1
      └─ ∂L/∂x  = W1^T δ1
```

이 그림이 곧 Autograd 엔진의 할 일입니다.

---

## 코드로 “계산 순서”만 구현하기 (완성본은 18강)

```python
"""17강: Backprop 단계를 명시적으로 풀어 쓴 버전."""

import numpy as np

x = np.array([1.0, 0.5]); y = 1.0
W1 = np.array([[0.3, -0.2], [0.4, 0.1]])
b1 = np.array([0.1, -0.1])
W2 = np.array([0.5, -0.4]); b2 = 0.2

# ----- Forward + cache -----
z1 = W1 @ x + b1
a1 = np.maximum(0.0, z1)
y_hat = float(W2 @ a1 + b2)
L = 0.5 * (y_hat - y) ** 2

# ----- Backward -----
dL_dyhat = y_hat - y                    # -0.79
dL_dW2 = dL_dyhat * a1                  # [-0.237, -0.2765]
dL_db2 = dL_dyhat                       # -0.79
dL_da1 = dL_dyhat * W2                  # [-0.395, 0.316]

relu_mask = (z1 > 0).astype(float)      # [1, 1]
dL_dz1 = dL_da1 * relu_mask             # delta^(1)

dL_db1 = dL_dz1
dL_dW1 = np.outer(dL_dz1, x)            # δ1 x^T
dL_dx = W1.T @ dL_dz1

print("L=", L)
print("dW2=", dL_dW2)
print("dW1=\n", dL_dW1)
```

출력은 6.9 표와 같아야 합니다.

> 📘 **심화** — 업데이트 후 Loss가 줄어드는지
>
> $\eta=0.1$로 $b^{(2)},W^{(2)}$만 갱신한 뒤 Forward를 다시 하면
> $\hat{y}$가 커지고 $L$이 $0.31205$보다 작아집니다.  
> 전체 파라미터 갱신·반복은 18강에서 합니다.

---

## LLM에서는 어디에 사용될까?

LLM 학습의 Backward도 원리는 동일합니다. 다만 그래프가 깁니다.

```text
Loss (Cross Entropy)
  ← LM Head Weight
    ← 마지막 Transformer Block
      ← … 여러 Block …
        ← Embedding
```

각 블록 안에서는 Residual 경로 + Attention 경로 + FFN 경로로 Gradient가 **더해지며** 흐릅니다.

| 증상 | 오늘 개념과의 연결 |
|---|---|
| Dying ReLU | Activation 미분이 0 (예제 B) |
| Vanishing Gradient | local gradient ≪ 1이 곱누적 |
| Exploding Gradient | local gradient ≫ 1이 곱누적 |
| Gradient Checkpointing | Forward cache를 버려 메모리를 줄이고, Backward 때 재계산 |
| `loss.backward()` | 오늘 손계산의 자동 실행 |

---

## 실습

### 실습 1 — 표 다시 쓰기

책을 덮고 6.9 요약표를 빈칸으로 다시 채웁니다.

### 실습 2 — ReLU 꺼짐 시나리오

예제 B의 Gradient 표를 손으로 완성합니다. $W^{(2)}$ Gradient가 0인지 확인합니다.

### 실습 3 — 수치 미분 전수 검사

위의 코드를 확장해 $W1, b1, W2, b2$ 모든 성분에 대해 `|analytic - numeric| < 1e-6`인지 검사합니다.

### 실습 4 — 학습률 한 스텝

$\eta=0.1$로 모든 파라미터를 한 번 업데이트한 뒤 Loss를 다시 계산합니다.  
$0.31205$보다 작아지는지 확인합니다.

### 실습 5 — Sigmoid 은닉으로 바꿔 보기 (도전)

은닉을 Sigmoid로 바꾸고 $\sigma'(z)=\sigma(z)(1-\sigma(z))$를 넣어 δ를 다시 계산합니다.

---

## 자주 하는 실수

1. **Forward 값과 Gradient를 혼동합니다** — $a=0.3$과 $\partial L/\partial a=-0.395$는 다른 객체입니다.
2. **ReLU 마스크를 안 곱합니다** — $z\le0$인데도 Gradient를 흘리면 구현 버그입니다.
3. **outer product 방향을 뒤집습니다** — $\delta x^{\top}$와 $x\delta^{\top}$는 shape가 다릅니다.
4. **Loss의 $1/2$를 미분에서 또 넣습니다** — $\partial L/\partial\hat{y}=\hat{y}-y$입니다.
5. **여러 경로의 합을 빠뜨립니다** — Residual이 있으면 경로 Gradient를 더합니다.
6. **업데이트 부호를 반대로 합니다** — 반드시 $-\eta\nabla L$입니다.

---

## 핵심 요약

- Backpropagation은 Loss에서 시작해 Chain Rule로 모든 파라미터 Gradient를 효율적으로 계산합니다.
- 순서는 항상 Loss → 출력층 → (Activation 미분) → 은닉층 → … 입니다.
- 핵심 패턴: $\partial L/\partial W = \delta\, a_{\mathrm{prev}}^{\top}$, $\partial L/\partial b = \delta$.
- ReLU는 마스크로 Gradient를 통과/차단합니다.
- 수치 미분은 해석적 Backprop의 단위 테스트입니다.
- `loss.backward()`와 Autograd는 이 손계산의 일반화입니다.

## 용어 사전

| 용어 | 의미 |
|---|---|
| Backpropagation | 계산 그래프를 역으로 순회하며 Gradient를 구하는 알고리즘 |
| Upstream Gradient | 그래프 뒤쪽(Loss 쪽)에서 전달된 Gradient |
| Local Gradient | 해당 연산 자체의 미분 |
| Delta (δ) | Pre-activation에 대한 $\partial L/\partial z$ |
| Numerical / Analytic Gradient | $\epsilon$ 섭동 근사 / 식으로 구한 Gradient |
| Vanishing / Exploding Gradient | 곱누적으로 Gradient가 소멸/폭발하는 현상 |
| Dying ReLU | $z\le0$이 지속되어 Gradient·학습이 멈추는 현상 |

## 연습문제

### 문제 1 (개념)

Backpropagation을 Chain Rule과 Computational Graph 단어로 정의하시오.

### 문제 2 (계산)

$\hat{y}=0.21$, $y=1$일 때 $\partial L/\partial\hat{y}$는? ($L=\frac12(\hat{y}-y)^2$)

### 문제 3 (계산)

$\partial L/\partial w^{(2)}_2$를 다시 구하시오.

### 문제 4 (계산)

$z^{(1)}_1=-0.5$라고만 가정하면(사고 실험) $\partial L/\partial W^{(1)}_{11}$은?

### 문제 5 (응용)

학습률 0.1로 $b^{(2)}$만 한 번 갱신한 값을 구하시오.

### 문제 6 (연결)

`loss.backward()` 이후 `W2.grad`에 들어가야 할 값은?

### 문제 7 (계산)

예제 B에서 $\partial L/\partial b^{(2)}$와 $\partial L/\partial W^{(2)}$는?

---

## 정답 및 해설

### 문제 1

Loss에서 시작해 계산 그래프를 입력 방향으로 순회하며, Upstream과 Local Gradient를 Chain Rule로 곱(필요 시 합)하여 모든 학습 파라미터의 Gradient를 구하는 알고리즘입니다.

### 문제 2

$0.21-1.0=-0.79$.

### 문제 3

$(-0.79)\times 0.35=-0.2765$.

### 문제 4


$$
ReLU 미분이 0이므로 $\delta^{(1)}_1=0$, 따라서 $\partial L/\partial W^{(1)}_{11}=0$.
$$

### 문제 5

$b^{(2)}\leftarrow 0.2 - 0.1\times(-0.79)=0.279$.

### 문제 6

$\partial L/\partial W^{(2)}=\begin{bmatrix}-0.237 & -0.2765\end{bmatrix}$.

### 문제 7

$\partial L/\partial b^{(2)}=-0.8$, $\partial L/\partial W^{(2)}=\begin{bmatrix}0 & 0\end{bmatrix}$.

## 다음 강의와 연결

이번 강의에서 Backprop을 **손으로 완전 분해**했습니다.

다음 **제18강. Backpropagation NumPy 구현**에서는, 오늘 식을 함수로 옮기고 **학습 루프**로 묶어 Loss가 실제로 내려가는 장면을 만듭니다.  
회귀와 작은 분류 예제를 훈련하며, 수치 미분 검증까지 코드로 자동화합니다.

> 표의 숫자를 외울 필요는 없습니다. 흐르는 순서만 몸에 남기면 됩니다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [16강. Forward Propagation](16강_Forward_Propagation.md)
- **다음 강:** [18강. Backpropagation NumPy 구현](18강_Backpropagation_NumPy_구현.md)

<!-- /LECTURE_NAV -->
