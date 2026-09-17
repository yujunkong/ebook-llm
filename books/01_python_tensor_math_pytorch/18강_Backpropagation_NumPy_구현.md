# 18강. Backpropagation NumPy 구현

## 이번 강에서 배우는 내용

- Forward / Backward / Update를 한 파일에서 연결하는 방법
- 17강 손계산 식을 NumPy 코드와 **한 줄씩 대응**시키는 방법
- 단샘플·배치 학습 루프의 차이
- 회귀(MSE)와 이진 분류(BCE)에서 Loss·출력층만 어떻게 바뀌는지
- 수치 미분으로 구현을 검증하는 방법 (relative error)
- Loss가 실제로 감소하는 실험을 해석하는 방법

## 왜 중요한가?

프레임워크 없이 학습이 한 바퀴 도는 코드를 갖고 있으면,

1. `loss.backward()`가 무엇을 대체하는지 명확해지고
2. shape 버그를 스스로 찾을 수 있으며
3. LLM 학습 루프(3권)의 축소판을 이미 이해하게 됩니다

오늘의 목표는 화려한 모델이 아니라, **Loss가 줄어드는 최소 학습기**입니다.

> **핵심**
>
> 검증(17강 숫자 · 수치 미분) → 그다음 학습 루프.  
> 순서가 바뀌면 버그를 “학습”으로 덮어 버립니다.

## 선수 개념

- 제15~17강의 2-2-1 구조, Forward cache, δ 공식
- NumPy 행렬곱, `np.outer`, 브로드캐스팅 (제8~10강)
- Gradient Descent: $\theta \leftarrow \theta - \eta \nabla_\theta L$ (제12강)

---

## 핵심 개념

### 3.1 Training Loop (학습 루프)

**Training Loop(학습 루프)**는 데이터를 반복해 보며 파라미터를 갱신하는 바깥쪽 순환입니다.

```text
for epoch in range(E):
    for batch in data:
        1) Forward → loss
        2) Backward → grads
        3) Update → params
```

**Epoch(에폭)**는 학습 데이터를 한 바퀴 도는 단위입니다.

### 3.2 SGD (Stochastic Gradient Descent)

**SGD**는 전체 데이터가 아니라 **일부 샘플(배치)**의 Gradient로 업데이트하는 방법입니다.

- Batch size = 1: variance가 큰 추정
- Full batch: 안정적이지만 데이터가 크면 비쌈
- Mini-batch: 실무 기본

### 3.3 Gradient Verification

$$
\text{relative error}
=
\frac{\|\nabla_{\mathrm{ana}}-\nabla_{\mathrm{num}}\|}
{\max(\|\nabla_{\mathrm{ana}}\|,\|\nabla_{\mathrm{num}}\|,\epsilon)}
$$

보통 `1e-6` 이하면 안심할 수 있습니다.

수치 미분의 정의는

$$
\frac{\partial L}{\partial \theta_i}
\approx
\frac{L(\theta_i+\epsilon)-L(\theta_i-\epsilon)}{2\epsilon}
$$

입니다.

### 3.4 회귀 vs 이진 분류

| | 회귀 | 이진 분류 |
|---|---|---|
| 출력 | 선형 $\hat{y}$ | Sigmoid 확률 $p$ |
| Loss | MSE $\frac12(\hat{y}-y)^2$ | BCE $-(y\log p+(1-y)\log(1-p))$ |
| $\partial L/\partial z_{\mathrm{out}}$ | $\hat{y}-y$ | $p-y$ (Sigmoid+BCE의 단순형) |

은닉층의 Backprop 패턴은 동일합니다. **머리(출력·Loss)만 갈아 끼웁니다.**

---

## 직관적으로 이해하기

수동 Backprop 학습기는 다음 세 서랍으로 이루어집니다.

1. **예측 서랍 (Forward)** — 시험 보기  
2. **채점 서랍 (Backward)** — 빨간 펜으로 기울기 쪽지  
3. **수정 서랍 (Update)** — 손잡이를 아주 조금 돌리기  

세 서랍을 `for`로 묶으면 “공부하는 기계”가 됩니다.  
PyTorch는 2번을 Autograd가, 3번을 Optimizer가 맡습니다. 오늘은 셋 다 우리가 씁니다.

---

## 수학적으로 이해하기 — 배치 식과 코드 대응

배치 크기 $B$, 입력 차원 2, 은닉 2, 출력 1.  
관례: $X\in\mathbb{R}^{B\times 2}$, Weight는 `(out, in)`.

### 5.1 Forward

$$
\begin{aligned}
Z^{(1)} &= X (W^{(1)})^{\top} + \mathbf{b}^{(1)} && (B\times 2)\\
A^{(1)} &= \mathrm{ReLU}(Z^{(1)}) && (B\times 2)\\
\hat{Y} &= A^{(1)} (W^{(2)})^{\top} + b^{(2)} && (B\times 1)\\
L &= \frac{1}{2B}\sum_{n=1}^{B}(\hat{y}_n - y_n)^2
\end{aligned}
$$

코드 대응:

```text
Z1 = X @ W1.T + b1
A1 = relu(Z1)
Y_hat = A1 @ W2.T + b2
loss = 0.5 * mean((Y_hat - y)**2)
```

### 5.2 Backward

$$
\begin{aligned}
\frac{\partial L}{\partial \hat{Y}} &= \frac{1}{B}(\hat{Y}-Y) && (B\times 1)\\
\frac{\partial L}{\partial W^{(2)}} &= \Big(\frac{\partial L}{\partial \hat{Y}}\Big)^{\top} A^{(1)} && (1\times 2)\\
\frac{\partial L}{\partial b^{(2)}} &= \sum_{n}\frac{\partial L}{\partial \hat{y}_n} && (1,)\\
\frac{\partial L}{\partial A^{(1)}} &= \frac{\partial L}{\partial \hat{Y}}\, W^{(2)} && (B\times 2)\\
\frac{\partial L}{\partial Z^{(1)}} &= \frac{\partial L}{\partial A^{(1)}} \odot \mathbf{1}_{Z^{(1)}>0} && (B\times 2)\\
\frac{\partial L}{\partial W^{(1)}} &= \Big(\frac{\partial L}{\partial Z^{(1)}}\Big)^{\top} X && (2\times 2)\\
\frac{\partial L}{\partial b^{(1)}} &= \sum_{n}\frac{\partial L}{\partial z^{(1)}_n} && (2,)
\end{aligned}
$$

코드 대응:

```text
dY  = (Y_hat - y) / B
dW2 = dY.T @ A1
db2 = sum(dY, axis=0)
dA1 = dY @ W2
dZ1 = dA1 * relu_grad(Z1)
dW1 = dZ1.T @ X
db1 = sum(dZ1, axis=0)
```

### 5.3 단샘플은 $B=1$

$B=1$이면 17강 식과 일치합니다.

$$
\frac{\partial L}{\partial W^{(2)}}
=
\delta^{(2)}\,(\mathbf{a}^{(1)})^{\top}
,\quad
\frac{\partial L}{\partial W^{(1)}}
=
\delta^{(1)}\,\mathbf{x}^{\top}
$$

배치의 `dZ.T @ X`는 단샘플 `outer(δ, x)`의 일반화입니다.

### 5.4 Update

$$
\theta \leftarrow \theta - \eta\,\frac{\partial L}{\partial \theta}
$$

```text
params[k] = params[k] - lr * grads[k]
```

---

## 작은 숫자로 직접 계산하기 — 구현 직전 점검

17강과 동일 단샘플에서 Backward 함수가 내야 할 값:

$$
\frac{\partial L}{\partial W^{(2)}}
=
\begin{bmatrix}-0.237 & -0.2765\end{bmatrix}
,\quad
\frac{\partial L}{\partial b^{(2)}}=-0.79
$$

$$
\frac{\partial L}{\partial W^{(1)}}
=
\begin{bmatrix}
-0.395 & -0.1975 \\
0.316 & 0.158
\end{bmatrix}
,\quad
\frac{\partial L}{\partial \mathbf{b}^{(1)}}
=
\begin{bmatrix}-0.395 \\ 0.316\end{bmatrix}
$$

$B=1$이므로

$$
\frac{\partial L}{\partial \hat{Y}}
=
\hat{y}-y
=
-0.79
$$

입니다 (`/B`가 그대로).

> 💡 **팁**
>
> 코드를 짜면 **먼저 이 숫자와 대조**합니다.  
> 학습 루프는 그다음입니다.

### 6.1 배치 두 샘플 손점검 (미리보기)

$X$가 16강 예제 A+B이고 $y=[1,1]^{\top}$이면 Forward는

$$
\hat{Y}
=
\begin{bmatrix}0.21\\0.20\end{bmatrix}
,\quad
\frac{\partial L}{\partial\hat{Y}}
=
\frac{1}{2}
\begin{bmatrix}-0.79\\-0.80\end{bmatrix}
=
\begin{bmatrix}-0.395\\-0.400\end{bmatrix}
$$

($L$이 평균 $\frac12$ MSE일 때).  
구현 후 이 값과 `dY`를 비교하면 배치 평균 버그를 바로 잡을 수 있습니다.

---

## 단샘플 outer와 배치 행렬곱이 같은 이유

$B=1$일 때 $\partial L/\partial\hat{Y}$는 스칼라 $\delta^{(2)}$, $A^{(1)}$은 행벡터입니다.

$$
\frac{\partial L}{\partial W^{(2)}}
=
\begin{bmatrix}\delta^{(2)}\end{bmatrix}
\begin{bmatrix}a_1 & a_2\end{bmatrix}
=
\delta^{(2)}\,(\mathbf{a}^{(1)})^{\top}
$$

숫자로 $\delta^{(2)}=-0.79$, $\mathbf{a}^{(1)}=[0.30,0.35]$이면

$$
\frac{\partial L}{\partial W^{(2)}}
=
\begin{bmatrix}-0.237 & -0.2765\end{bmatrix}
$$

은닉도 같습니다.

$$
\frac{\partial L}{\partial W^{(1)}}
=
\begin{bmatrix}\delta_1\\\delta_2\end{bmatrix}
\begin{bmatrix}x_1 & x_2\end{bmatrix}
=
\begin{bmatrix}
-0.395 & -0.1975 \\
0.316 & 0.158
\end{bmatrix}
$$

배치에서는 여러 샘플의 외적을 **행 방향으로 합산**한 것이 `dZ.T @ X`입니다.

## 코드로 구현하기

### 7.1 유틸과 모델 파라미터

```python
"""18강: NumPy로 2-2-1 Backprop 학습기."""

from __future__ import annotations
import numpy as np

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)

def relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(z.dtype)

def init_params(seed: int = 0):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, 0.3, size=(2, 2)),
        "b1": np.zeros(2),
        "W2": rng.normal(0, 0.3, size=(1, 2)),
        "b2": np.zeros(1),
    }

def fixed_params_for_check():
    """17강 손계산과 동일한 파라미터."""
    return {
        "W1": np.array([[0.3, -0.2], [0.4, 0.1]]),
        "b1": np.array([0.1, -0.1]),
        "W2": np.array([[0.5, -0.4]]),
        "b2": np.array([0.2]),
    }
```

### 7.2 Forward

```python
def forward(X: np.ndarray, params: dict, y: np.ndarray | None = None):
    """X: (B, 2), y: (B, 1) 또는 (B,)."""
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    Z1 = X @ W1.T + b1          # (B,2)  ← 식의 Z^(1)
    A1 = relu(Z1)               # (B,2)
    Y_hat = A1 @ W2.T + b2      # (B,1)
    loss = None
    if y is not None:
        y = y.reshape(-1, 1)
        loss = 0.5 * np.mean((Y_hat - y) ** 2)
    cache = {"X": X, "Z1": Z1, "A1": A1, "Y_hat": Y_hat, "y": y, **params}
    return Y_hat, loss, cache
```

### 7.3 Backward

```python
def backward(cache: dict) -> dict:
    X, Z1, A1 = cache["X"], cache["Z1"], cache["A1"]
    Y_hat, y, W2 = cache["Y_hat"], cache["y"].reshape(-1, 1), cache["W2"]
    B = X.shape[0]

    dY = (Y_hat - y) / B                # ∂L/∂Y_hat
    dW2 = dY.T @ A1                     # (1,2)
    db2 = np.sum(dY, axis=0)            # (1,)
    dA1 = dY @ W2                       # (B,2)
    dZ1 = dA1 * relu_grad(Z1)           # δ^(1)
    dW1 = dZ1.T @ X                     # (2,2)
    db1 = np.sum(dZ1, axis=0)           # (2,)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
```

### 7.4 Update

```python
def sgd_update(params: dict, grads: dict, lr: float) -> None:
    for k in params:
        params[k] = params[k] - lr * grads[k]
```

### 7.5 손계산 검증

```python
def check_against_lesson17():
    params = fixed_params_for_check()
    X = np.array([[1.0, 0.5]])
    y = np.array([[1.0]])
    _, loss, cache = forward(X, params, y)
    grads = backward(cache)
    assert abs(loss - 0.31205) < 1e-8
    assert np.allclose(grads["W2"], [[-0.237, -0.2765]])
    assert np.allclose(grads["b2"], [-0.79])
    assert np.allclose(grads["W1"], [[-0.395, -0.1975], [0.316, 0.158]])
    assert np.allclose(grads["b1"], [-0.395, 0.316])
    print("17강 대조 통과")
```

### 7.6 수치 미분 검증기

```python
def numerical_gradients(params, X, y, eps=1e-6):
    grads = {k: np.zeros_like(v) for k, v in params.items()}
    for name, arr in params.items():
        flat = arr.reshape(-1)
        gflat = grads[name].reshape(-1)
        for i in range(flat.size):
            old = flat[i]
            flat[i] = old + eps
            params[name] = flat.reshape(arr.shape)
            _, Lp, _ = forward(X, params, y)
            flat[i] = old - eps
            params[name] = flat.reshape(arr.shape)
            _, Lm, _ = forward(X, params, y)
            gflat[i] = (Lp - Lm) / (2 * eps)
            flat[i] = old
            params[name] = flat.reshape(arr.shape)
        grads[name] = gflat.reshape(arr.shape)
    return grads

def rel_error(a, b, eps=1e-12):
    return np.max(np.abs(a - b) / np.maximum(eps, np.maximum(np.abs(a), np.abs(b))))

def verify_gradients():
    params = fixed_params_for_check()
    X = np.array([[1.0, 0.5]]); y = np.array([[1.0]])
    _, _, cache = forward(X, params, y)
    ana = backward(cache)
    num = numerical_gradients(dict(params), X, y)  # 복사본 권장
    for k in ana:
        e = rel_error(ana[k], num[k])
        print(k, "rel_error =", e)
        assert e < 1e-6
```

> ⚠️ **주의**
>
> 수치 미분 루프가 `params`를 제자리 수정하면 이후 analytic과 어긋날 수 있습니다.  
> 검증 전에는 `fixed_params_for_check()`를 다시 부르거나 deep copy를 쓰세요.

### 7.7 회귀 학습 루프

합성 데이터: $y = 2x_1 - 3x_2 + 0.5$에 작은 노이즈.

```python
def make_regression_data(n=64, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, 2))
    y = (2 * X[:, 0] - 3 * X[:, 1] + 0.5).reshape(-1, 1)
    y += rng.normal(0, 0.05, size=y.shape)
    return X, y

def train_regression(epochs=400, lr=0.05, batch_size=16, seed=0):
    X, y = make_regression_data()
    params = init_params(seed)
    history = []
    n = X.shape[0]
    rng = np.random.default_rng(42)
    for epoch in range(epochs):
        idx = rng.permutation(n)
        Xo, yo = X[idx], y[idx]
        for start in range(0, n, batch_size):
            xb = Xo[start:start + batch_size]
            yb = yo[start:start + batch_size]
            _, _, cache = forward(xb, params, yb)
            grads = backward(cache)
            sgd_update(params, grads, lr)
        _, full_loss, _ = forward(X, params, y)
        history.append(full_loss)
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:4d}  loss={full_loss:.6f}")
    return params, history
```

**성공 기준:** 마지막 Loss가 첫 Loss보다 명확히 작습니다.  
특정 초깃값·환경에 묶인 “정확 숫자 벤치마크”는 주장하지 않습니다.

### 7.8 이진 분류 버전

Sigmoid + BCE에서

$$
\frac{\partial L}{\partial Z^{(2)}} = \frac{1}{B}(P - Y)
$$

이 되는 것이 핵심입니다.

```python
def sigmoid(z):
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))

def forward_clf(X, params, y=None):
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    Z1 = X @ W1.T + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2.T + b2
    P = sigmoid(Z2)
    loss = None
    if y is not None:
        y = y.reshape(-1, 1)
        eps = 1e-9
        loss = -np.mean(y * np.log(P + eps) + (1 - y) * np.log(1 - P + eps))
    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "P": P, "y": y, **params}
    return P, loss, cache

def backward_clf(cache):
    X, Z1, A1, P, y = cache["X"], cache["Z1"], cache["A1"], cache["P"], cache["y"]
    W2 = cache["W2"]; B = X.shape[0]; y = y.reshape(-1, 1)
    dZ2 = (P - y) / B
    dW2 = dZ2.T @ A1
    db2 = np.sum(dZ2, axis=0)
    dA1 = dZ2 @ W2
    dZ1 = dA1 * relu_grad(Z1)
    dW1 = dZ1.T @ X
    db1 = np.sum(dZ1, axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

def make_classification_data(n=80, seed=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, 2))
    logits = 1.5 * X[:, 0] - 2.0 * X[:, 1] + 0.1
    y = (logits > 0).astype(float).reshape(-1, 1)
    return X, y

def train_classification(epochs=300, lr=0.1, batch_size=20, seed=0):
    X, y = make_classification_data()
    params = init_params(seed)
    n = X.shape[0]
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            sl = idx[start:start + batch_size]
            _, _, cache = forward_clf(X[sl], params, y[sl])
            sgd_update(params, backward_clf(cache), lr)
        P, loss, _ = forward_clf(X, params, y)
        acc = np.mean((P > 0.5) == y)
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:4d}  loss={loss:.4f}  acc={acc:.3f}")
    return params
```

### 7.9 메인

```python
if __name__ == "__main__":
    print("== 17강 대조 =="); check_against_lesson17()
    print("== 수치 미분 =="); verify_gradients()
    print("== 회귀 =="); train_regression()
    print("== 분류 =="); train_classification()
```

---

## 식 ↔ 코드 체크리스트 (한 장 요약)

| 수학 | NumPy | shape |
|---|---|---|
| $Z^{(1)}=X W^{(1)\top}+b^{(1)}$ | `X @ W1.T + b1` | $(B,2)$ |
| $A^{(1)}=\mathrm{ReLU}(Z^{(1)})$ | `relu(Z1)` | $(B,2)$ |
| $\hat{Y}=A^{(1)}W^{(2)\top}+b^{(2)}$ | `A1 @ W2.T + b2` | $(B,1)$ |
| $\partial L/\partial\hat{Y}=(\hat{Y}-Y)/B$ | `(Y_hat-y)/B` | $(B,1)$ |
| $\partial L/\partial W^{(2)}=(\partial L/\partial\hat{Y})^{\top}A^{(1)}$ | `dY.T @ A1` | $(1,2)$ |
| $\partial L/\partial Z^{(1)}=(\partial L/\partial A^{(1)})\odot 1_{Z>0}$ | `dA1 * relu_grad(Z1)` | $(B,2)$ |
| $\partial L/\partial W^{(1)}=(\partial L/\partial Z^{(1)})^{\top}X$ | `dZ1.T @ X` | $(2,2)$ |
| $\theta\leftarrow\theta-\eta\nabla L$ | `p - lr * g` | 동일 |

> 📘 **심화**
>
> Loss를 `sum`으로 정의하면 Backward의 `/B`를 빼야 스케일이 맞습니다.  
> **Forward의 집계와 Backward의 계수를 항상 짝지어** 생각하세요.

---

## 학습이 “됐다”는 신호를 읽는 법

1. **Loss 곡선**: 초반 하락 → 완만. 진동이 크면 `lr` 감소 또는 `batch_size` 증가.
2. **회귀**: 예측과 $y$의 산점이 대각선에 가까워짐.
3. **분류**: accuracy가 chance(0.5)를 넘어 안정.
4. Loss가 안 줄면: Update 부호, `relu_grad`, `dW` shape, `lr` 크기부터 점검.

---

## LLM에서는 어디에 사용될까?

| 오늘 (NumPy) | LLM 학습 |
|---|---|
| `forward` | model(tokens) → logits |
| MSE/BCE | Cross Entropy (next token) |
| `backward` | `loss.backward()` |
| `sgd_update` | Adam/AdamW `optimizer.step` |
| `verify_gradients` | 드물게 unit test / 정밀도 디버깅 |

분산 학습에서도 Backward로 구한 Gradient를 GPU 간에 평균 냅니다.  
원천은 여전히 “Loss에서 온 δ를 Weight에 외적”입니다.

---

## 실습

### 실습 1 — 17강 숫자 통과

`check_against_lesson17()`에 assert를 걸어 오차 `1e-10` 수준으로 통과시킵니다.

### 실습 2 — 수치 미분 전 항목

`verify_gradients()`에서 모든 relative error `< 1e-6`.  
고의로 `relu_grad`를 제거해 error가 커지는지 봅니다.

### 실습 3 — 회귀 Loss 하락

`history[0]` 대비 `history[-1]` 비율을 출력합니다. 목표: **명확한 감소**.

### 실습 4 — 학습률 스윕

`lr ∈ {0.001, 0.05, 0.5, 2.0}`로 각각 100 epoch. 발산·정체·양호를 표로 정리합니다.

### 실습 5 — 은닉 너비 변경

`hidden=8` 인자로 shape를 일반화해 봅니다 (출력은 1 유지).

### 실습 6 — 분류 정확도

최종 acc가 chance를 명확히 넘는지 확인합니다.

---

## 자주 하는 실수

1. **배치 평균을 Forward/Backward에서 서로 다르게 둡니다.**
2. **`np.outer`와 배치 `T @`를 혼용하다 shape 사고.**
3. **검증 없이 학습부터 돌립니다.**
4. **in-place로 cache를 오염시킵니다.** Forward→Backward→Update 순서를 지키세요.
5. **분류에서 MSE를 그대로 씁니다.** BCE+Sigmoid가 더 자연스럽습니다.
6. **난수 시드를 안 고정해 디버깅이 안 됩니다.**

---

## 핵심 요약

- NumPy Backprop은 Forward cache → δ → $dW=\delta a^{\top}$ → SGD의 반복입니다.
- 먼저 17강 숫자·수치 미분으로 검증하고, 그다음 학습 루프를 돕니다.
- 회귀와 분류는 출력/Loss 머리만 다르고 은닉 Backprop 패턴은 같습니다.
- Loss 감소는 구현이 살아 있다는 최소 증거입니다.
- 이 코드가 곧 PyTorch 학습 루프의 뼈대입니다.

## 용어 사전

| 용어 | 의미 |
|---|---|
| Training Loop | Forward-Backward-Update 반복 |
| Epoch / Mini-batch | 전체 1회 순회 / Gradient용 작은 묶음 |
| SGD | 배치 Gradient로 파라미터 갱신 |
| Gradient Verification | 해석적·수치 Gradient 비교 |
| Relative Error | 두 Gradient의 상대 차이 |
| BCE / Logit | 이진 로그 손실 / Sigmoid 직전 점수 |
| Initialization | 학습 전 Weight 세팅 |

## 연습문제

### 문제 1 (개념)

학습 루프 한 step의 세 단계를 순서대로 쓰시오.

### 문제 2 (식)

$\partial L/\partial W^{(2)} = (\partial L/\partial\hat{Y})^{\top} A^{(1)}$이 되는 이유를 한 문장으로 설명하시오.

### 문제 3 (구현)

`relu_grad`를 곱하지 않으면 어떤 버그가 생기는가?

### 문제 4 (실험)

Loss가 폭발할 때 가장 먼저 의볼 하이퍼파라미터는?

### 문제 5 (연결)

오늘 `sgd_update`가 PyTorch에서 보통 어떤 API 조합으로 대체되는가?

### 문제 6 (계산)

17강 단샘플에서 `B=1`일 때 `dY`의 값은?

### 문제 7 (계산)

평균 MSE에서 $B=2$, $\hat{Y}-Y=[-0.79,-0.80]^{\top}$이면 $\partial L/\partial\hat{Y}$는?

---

## 정답 및 해설

### 문제 1

Forward → Backward → Update.

### 문제 2

$\hat{Y}=A^{(1)}W^{(2)\top}+\cdots$이므로 $W^{(2)}$ Gradient는 앞 층 활성과 출력 δ의 행렬곱(단샘플에서는 외적)으로 모입니다.

### 문제 3

$z\le0$ 뉴런에도 Gradient가 흘러 ReLU 정의와 다른 업데이트가 수행됩니다.

### 문제 4

학습률 `lr` (너무 큼).

### 문제 5

`loss.backward()` 후 `optimizer.step()` (보통 `optimizer.zero_grad()` 포함).

### 문제 6

$-0.79$.

### 문제 7

$$
\frac{1}{2}\begin{bmatrix}-0.79\\-0.80\end{bmatrix}
=
\begin{bmatrix}-0.395\\-0.400\end{bmatrix}
$$

## 다음 강의와 연결

이번 강의에서 “수동 딥러닝 엔진”을 NumPy로 완성했습니다.

다음 **제19강. PyTorch Tensor**에서는, 같은 숫자를 `torch.tensor`로 다루기 시작합니다.  
dtype, device, shape, broadcasting, NumPy와의 차이를 익히면, 제20강 Autograd가 오늘 `backward()` 함수를 자동으로 채워 주는 장면을 볼 준비가 됩니다.

> 엔진을 한 번 손으로 조립해 보았습니다. 이제 공장 도구(PyTorch)로 넘어갑니다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [17강. Backpropagation 직접 계산하기](17강_Backpropagation_직접_계산하기.md)
- **다음 강:** [19강. PyTorch Tensor](19강_PyTorch_Tensor.md)

<!-- /LECTURE_NAV -->
