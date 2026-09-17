# 18강. Backpropagation NumPy 구현
## 이번 강에서 배우는 내용

- Forward / Backward / Update를 한 파일에서 연결하는 방법
- 단샘플·배치 학습 루프의 차이
- 회귀(MSE)와 이진 분류(BCE)에서 Loss·출력층만 어떻게 바뀌는지
- 수치 미분으로 구현을 검증하는 방법
- Loss가 실제로 감소하는 실험을 해석하는 방법

## 왜 중요한가?
프레임워크 없이 학습이 한 바퀴 도는 코드를 갖고 있으면,

1. `loss.backward()`가 무엇을 대체하는지 명확해지고
2. shape 버그를 스스로 찾을 수 있으며
3. LLM 학습 루프(3권)의 축소판을 이미 이해하게 된다

오늘의 목표는 화려한 모델이 아니라, **Loss가 줄어드는 최소 학습기**이다.

## 선수 개념
- 제15~17강의 2-2-1 구조, Forward cache, δ 공식
- NumPy 행렬곱, `np.outer`, 브로드캐스팅 (제8~10강)
- Gradient Descent: $\theta \leftarrow \theta - \eta \nabla_\theta L$ (제12강)

## 핵심 개념
### 3.1 Training Loop (학습 루프)

**Training Loop(학습 루프)**는 데이터를 반복해 보며 파라미터를 갱신하는 바깥쪽 순환이다.

```text
for epoch in range(E):
    for batch in data:
        1) Forward → loss
        2) Backward → grads
        3) Update → params
```

**Epoch(에폭)**는 학습 데이터를 한 바퀴 도는 단위이다.  
데이터가 아주 작으면 epoch와 step이 거의 같다.

### 3.2 SGD (Stochastic Gradient Descent)

**SGD**는 전체 데이터가 아니라 **일부 샘플(배치)**의 Gradient로 업데이트하는 방법이다.

- Batch size = 1: 가장 시끄러운(variance 큰) 추정
- Full batch: 안정적이지만 데이터가 크면 비쌈
- Mini-batch: 실무 기본

오늘은 먼저 단샘플로 원리를 굳힌 뒤, 미니배치로 확장한다.

### 3.3 Gradient Verification

구현이 맞는지 확인하려면 해석적 Gradient와 수치 Gradient를 비교한다.

\[
\text{relative error}
=
\frac{\|\nabla_{\mathrm{ana}}-\nabla_{\mathrm{num}}\|}
{\max(\|\nabla_{\mathrm{ana}}\|,\|\nabla_{\mathrm{num}}\|,\epsilon)}
\]

보통 `1e-6` 이하면 안심할 수 있다.

### 3.4 회귀 vs 이진 분류

| | 회귀 | 이진 분류 |
|---|---|---|
| 출력 | 선형 $\hat{y}$ | Sigmoid 확률 $p$ |
| Loss | MSE $\frac12(\hat{y}-y)^2$ | BCE $-(y\log p+(1-y)\log(1-p))$ |
| $\partial L/\partial z_{\mathrm{out}}$ | $\hat{y}-y$ | $p-y$ (Sigmoid+BCE 조합의 유명한 단순형) |

은닉층의 Backprop 패턴은 동일하다. **머리(출력·Loss)만 갈아 끼운다.**

## 직관적으로 이해하기
수동 Backprop 학습기는 다음 세 서랍으로 이루어진다.

1. **예측 서랍 (Forward)** — 시험 보기  
2. **채점 서랍 (Backward)** — 빨간 펜으로 기울기 쪽지  
3. **수정 서랍 (Update)** — 손잡이를 아주 조금 돌리기  

세 서랍을 `for`로 묶으면 “공부하는 기계”가 된다.  
PyTorch는 2번 서랍을 Autograd가, 3번을 Optimizer가 맡는다. 오늘은 셋 다 우리가 쓴다.

## 수학적으로 이해하기 — 배치 식
배치 크기 $B$, 입력 차원 2, 은닉 2, 출력 1.

관례: $X\in\mathbb{R}^{B\times 2}$

\[
\begin{aligned}
Z^{(1)} &= X W^{(1)\top} + \mathbf{b}^{(1)} && (B\times 2)\\
A^{(1)} &= \mathrm{ReLU}(Z^{(1)}) \\
\hat{Y} &= A^{(1)} W^{(2)\top} + b^{(2)} && (B\times 1)\\
L &= \frac{1}{2B}\sum_{n=1}^{B}(\hat{y}_n - y_n)^2
\end{aligned}
\]

Backward (평균 MSE 기준):

\[
\begin{aligned}
\frac{\partial L}{\partial \hat{Y}} &= \frac{1}{B}(\hat{Y}-Y) \\
\frac{\partial L}{\partial W^{(2)}} &= \left(\frac{\partial L}{\partial \hat{Y}}\right)^{\top} A^{(1)} \\
\frac{\partial L}{\partial b^{(2)}} &= \sum_{n}\frac{\partial L}{\partial \hat{y}_n} \\
\frac{\partial L}{\partial A^{(1)}} &= \frac{\partial L}{\partial \hat{Y}}\, W^{(2)} \\
\frac{\partial L}{\partial Z^{(1)}} &= \frac{\partial L}{\partial A^{(1)}} \odot \mathbf{1}_{Z^{(1)}>0} \\
\frac{\partial L}{\partial W^{(1)}} &= \left(\frac{\partial L}{\partial Z^{(1)}}\right)^{\top} X \\
\frac{\partial L}{\partial b^{(1)}} &= \sum_{n}\frac{\partial L}{\partial z^{(1)}_n}
\end{aligned}
\]

단샘플은 $B=1$인 특수 경우이며, 17강 식과 일치한다.

## 작은 숫자로 직접 계산하기 — 구현 직전 점검
17강과 동일 단샘플에서 Backward 함수가 내야 할 값:

```text
dW2 = [-0.237, -0.2765]
db2 = -0.79
dW1 = [[-0.395, -0.1975],
       [ 0.316,  0.158 ]]
db1 = [-0.395, 0.316]
```

코드를 짜면 **먼저 이 숫자와 대조**한다.  
학습 루프는 그다음이다. 순서가 바뀌면 버그를 학습으로 덮어 버린다.

## 코드로 구현하기
### 7.1 유틸과 모델 파라미터

```python
"""18강: NumPy로 2-2-1 Backprop 학습기."""

from __future__ import annotations

import numpy as np

def relu(z: np.ndarray) -> np.ndarray:
    """ReLU."""
    return np.maximum(0.0, z)

def relu_grad(z: np.ndarray) -> np.ndarray:
    """d ReLU / dz 마스크."""
    return (z > 0).astype(z.dtype)

def init_params(seed: int = 0):
    """작은 난수로 Weight 초기화. Bias는 0."""
    rng = np.random.default_rng(seed)
    # 17강 고정 숫자로 검증할 때는 아래 주석의 값을 사용
    W1 = rng.normal(0, 0.3, size=(2, 2))
    b1 = np.zeros(2)
    W2 = rng.normal(0, 0.3, size=(1, 2))
    b2 = np.zeros(1)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

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
    """X: (B, 2), y: (B, 1) 또는 (B,).

    Returns y_hat (B,1), loss(float|None), cache(dict)
    """
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]

    # Affine → ReLU
    Z1 = X @ W1.T + b1          # (B,2)
    A1 = relu(Z1)               # (B,2)
    Y_hat = A1 @ W2.T + b2      # (B,1)

    loss = None
    if y is not None:
        y = y.reshape(-1, 1)
        # 평균 MSE with 1/2
        loss = 0.5 * np.mean((Y_hat - y) ** 2)

    cache = {"X": X, "Z1": Z1, "A1": A1, "Y_hat": Y_hat, "y": y, **params}
    return Y_hat, loss, cache
```

### 7.3 Backward

```python
def backward(cache: dict) -> dict:
    """cache의 Forward 결과로 Gradient 딕셔너리 계산."""
    X = cache["X"]
    Z1 = cache["Z1"]
    A1 = cache["A1"]
    Y_hat = cache["Y_hat"]
    y = cache["y"].reshape(-1, 1)
    W2 = cache["W2"]
    B = X.shape[0]

    # ∂L/∂Y_hat  (평균 1/2 MSE → 평균 (Y_hat - y))
    dY = (Y_hat - y) / B                # (B,1)

    # 출력층
    dW2 = dY.T @ A1                     # (1,2)
    db2 = np.sum(dY, axis=0)            # (1,)
    dA1 = dY @ W2                       # (B,2)

    # ReLU
    dZ1 = dA1 * relu_grad(Z1)           # (B,2)

    # 은닉층
    dW1 = dZ1.T @ X                     # (2,2)
    db1 = np.sum(dZ1, axis=0)           # (2,)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
```

### 7.4 Update

```python
def sgd_update(params: dict, grads: dict, lr: float) -> None:
    """In-place Gradient Descent."""
    for k in params:
        params[k] = params[k] - lr * grads[k]
```

### 7.5 손계산 검증

```python
def check_against_lesson17():
    """17강 표와 대조."""
    params = fixed_params_for_check()
    X = np.array([[1.0, 0.5]])
    y = np.array([[1.0]])
    _, loss, cache = forward(X, params, y)
    grads = backward(cache)

    print("loss", loss)                 # ~0.31205
    print("dW2", grads["W2"])           # ~[[-0.237, -0.2765]]
    print("db2", grads["b2"])           # ~[-0.79]
    print("dW1\n", grads["W1"])
    print("db1", grads["b1"])
```

### 7.6 수치 미분 검증기

```python
def numerical_gradients(params, X, y, eps=1e-6):
    """모든 파라미터에 대해 수치 Gradient."""
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
    X = np.array([[1.0, 0.5]])
    y = np.array([[1.0]])
    _, _, cache = forward(X, params, y)
    ana = backward(cache)
    num = numerical_gradients(params, X, y)
    for k in ana:
        print(k, "rel_error =", rel_error(ana[k], num[k]))
```

### 7.7 회귀 학습 루프 — Loss가 떨어질 때까지

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
        # 미니배치 섞기
        idx = rng.permutation(n)
        Xo, yo = X[idx], y[idx]
        for start in range(0, n, batch_size):
            xb = Xo[start:start + batch_size]
            yb = yo[start:start + batch_size]
            _, loss, cache = forward(xb, params, yb)
            grads = backward(cache)
            sgd_update(params, grads, lr)
        # epoch 끝 full-batch loss 기록
        _, full_loss, _ = forward(X, params, y)
        history.append(full_loss)
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:4d}  loss={full_loss:.6f}")
    return params, history
```

실행 예(환경에 따라 수치 차이는 있을 수 있음):

```text
epoch    0  loss=...
epoch   50  loss=...
...
epoch  399  loss=...   # 초깃값보다 명확히 감소
```

**성공 기준:** 마지막 Loss가 첫 Loss보다 한 자릿수 이상 작거나, 적어도 단조에 가깝게 감소 추세.

### 7.8 이진 분류 버전

```python
def sigmoid(z):
    # 안정성을 위해 clip
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))

def forward_clf(X, params, y=None):
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    Z1 = X @ W1.T + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2.T + b2          # logit
    P = sigmoid(Z2)              # 확률
    loss = None
    if y is not None:
        y = y.reshape(-1, 1)
        # Binary Cross Entropy
        eps = 1e-9
        loss = -np.mean(y * np.log(P + eps) + (1 - y) * np.log(1 - P + eps))
    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "P": P, "y": y, **params}
    return P, loss, cache

def backward_clf(cache):
    X, Z1, A1, P, y = cache["X"], cache["Z1"], cache["A1"], cache["P"], cache["y"]
    W2 = cache["W2"]
    B = X.shape[0]
    y = y.reshape(-1, 1)

    # Sigmoid + BCE 의 ∂L/∂Z2 = (P - y) / B
    dZ2 = (P - y) / B
    dW2 = dZ2.T @ A1
    db2 = np.sum(dZ2, axis=0)
    dA1 = dZ2 @ W2
    dZ1 = dA1 * relu_grad(Z1)
    dW1 = dZ1.T @ X
    db1 = np.sum(dZ1, axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

def make_classification_data(n=80, seed=2):
    """직선으로 대충 나뉘는 2D 점."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, 2))
    logits = 1.5 * X[:, 0] - 2.0 * X[:, 1] + 0.1
    y = (logits > 0).astype(float).reshape(-1, 1)
    return X, y

def train_classification(epochs=300, lr=0.1, batch_size=20, seed=0):
    X, y = make_classification_data()
    params = init_params(seed)
    history = []
    n = X.shape[0]
    rng = np.random.default_rng(0)

    for epoch in range(epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            sl = idx[start:start + batch_size]
            _, _, cache = forward_clf(X[sl], params, y[sl])
            grads = backward_clf(cache)
            sgd_update(params, grads, lr)
        P, loss, _ = forward_clf(X, params, y)
        acc = np.mean((P > 0.5) == y)
        history.append((loss, acc))
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:4d}  loss={loss:.4f}  acc={acc:.3f}")
    return params, history
```

분류에서도 Loss 감소 + 정확도 상승을 관찰하는 것이 목표이다.

### 7.9 메인

```python
if __name__ == "__main__":
    print("== 17강 대조 ==")
    check_against_lesson17()
    print("\n== 수치 미분 검증 ==")
    verify_gradients()
    print("\n== 회귀 학습 ==")
    train_regression()
    print("\n== 분류 학습 ==")
    train_classification()
```

## 학습이 “됐다”는 신호를 읽는 법
1. **Loss 곡선**: 초반 빠르게 하락 → 완만. 진동이 크면 `lr` 감소 또는 `batch_size` 증가.
2. **회귀**: 예측 $\hat{y}$와 $y$의 산점이 대각선에 가까워짐.
3. **분류**: accuracy가 chance(0.5)를 넘어 안정.
4. **과적합 예고**: 지금은 데이터가 단순해 잘 맞지만, 표현력이 큰 망 + 적은 데이터면 train Loss만 줄고 일반화가 깨진다(제24강).

Loss가 전혀 안 줄면 체크리스트:

- Gradient 부호/Update 부호
- `relu_grad` 적용 여부
- `dW` shape와 `W` shape 일치
- `lr`이 터무니없이 크거나 작음
- 입력 정규화 여부(오늘은 표준정규라 비교적 안전)

## LLM에서는 어디에 사용될까?
NumPy 루프와 LLM Trainer의 대응:

| 오늘 (NumPy) | LLM 학습 |
|---|---|
| `forward` | model(tokens) → logits |
| MSE/BCE | Cross Entropy (next token) |
| `backward` | `loss.backward()` |
| `sgd_update` | Adam/AdamW optimizer.step |
| epoch/batch | step, global batch, grad accumulation |
| `verify_gradients` | 드물게 unit test, 보통 mixed precision 이슈 디버깅 |

또한 분산 학습에서는 Backward로 구한 Gradient를 GPU 간에 평균 낸다.  
원천은 여전히 “Loss에서 온 δ를 Weight에 외적”이다.

## 실습
### 실습 1 — 17강 숫자 통과

`check_against_lesson17()`가 표와 오차 `1e-10` 수준으로 일치하는지 assert를 걸어라.

### 실습 2 — 수치 미분 전 항목

`verify_gradients()`에서 모든 relative error `< 1e-6`을 확인한다.  
일부러 `relu_grad`를 제거해 보면 error가 커지는지 본다(버그 주입 실험).

### 실습 3 — 회귀 Loss 하락

`train_regression`을 실행하고, `history[0]` 대비 `history[-1]` 비율을 출력한다.  
목표: **명확한 감소**.

### 실습 4 — 학습률 스윕

`lr ∈ {0.001, 0.05, 0.5, 2.0}`로 각각 100 epoch.  
발산·정체·양호를 표로 정리한다.

### 실습 5 — 은닉 너비 변경

은닉 뉴런을 2 → 8로 늘리려면 `init_params`와 행렬 shape를 일반화해야 한다.  
`hidden=8` 인자로 리팩터링해 보라. (출력은 1 유지)

### 실습 6 — 분류 정확도

`train_classification` 최종 acc가 0.9 이상이 되는지 확인한다. 안 되면 epoch/`lr`을 조정한다.

## 자주 하는 실수
1. **배치 평균을 Forward/Backward에서 서로 다르게 한다**  
   Loss는 `mean`인데 Backward에서 `/B`를 빼먹으면 Gradient 스케일이 커진다.

2. **`np.outer`와 배치 `T @`를 혼용하다 shape 사고**  
   단샘플은 outer, 배치는 `dZ.T @ X`로 통일하는 편이 안전하다.

3. **검증 없이 학습부터 돌린다**  
   잘못된 Backward + 운 좋은 데이터는 “이상하지만 줄어 보이는” 곡선을 만들 수 있다. 항상 17강 숫자 또는 수치 미분 먼저.

4. **in-place 연산으로 cache 오염**  
   `params`를 바꾼 뒤 옛 `cache`로 Backward하면 안 된다. Forward→Backward→Update 순서를 지킨다.

5. **분류에서 MSE를 그대로 쓴다**  
   동작은 할 수 있으나, BCE+Sigmoid가 확률 학습에 더 자연스럽고 Gradient도 잘 작동한다.

6. **난수 시드를 안 고정해 디버깅이 안 된다**  
   검증 단계에서는 고정 파라미터/`seed`를 사용한다.

## 핵심 요약
- NumPy Backprop 구현은 Forward cache → δ 계산 → $dW=\delta a^{\top}$ → SGD 업데이트의 반복이다.
- 먼저 17강 숫자·수치 미분으로 검증하고, 그다음 학습 루프를 돈다.
- 회귀와 분류는 출력/Loss 머리만 다르고 은닉 Backprop 패턴은 같다.
- Loss 감소는 구현이 살아 있다는 최소 증거이다.
- 이 코드가 곧 PyTorch 학습 루프의 뼈대이다.

## 용어 사전
| 용어 | 의미 |
|---|---|
| Training Loop | Forward-Backward-Update를 반복하는 학습 순환 |
| Epoch | 전체 학습 데이터를 한 번 순회하는 단위 |
| Mini-batch | Gradient 추정에 쓰는 작은 샘플 묶음 |
| SGD | 배치 Gradient로 파라미터를 갱신하는 기본 옵티마이저 |
| Gradient Verification | 해석적/수치 Gradient 비교로 구현을 검증 |
| BCE | 이진 분류용 로그 손실 |
| Logit | Sigmoid/Softmax 직전의 원시 점수 $z$ |
| In-place Update | 기존 파라미터 배열을 직접 갱신하는 방식 |
| Relative Error | 두 Gradient 벡터의 상대 차이 |
| Initialization | 학습 전 Weight를 작은 난수 등으로 세팅하는 일 |

## 연습문제
### 문제 1 (개념)

학습 루프 한 step의 세 단계를 순서대로 쓰시오.

### 문제 2 (식)

배치 MSE에서 $\partial L/\partial W^{(2)} = (\partial L/\partial\hat{Y})^{\top} A^{(1)}$이 되는 이유를 한 문장으로 설명하시오.

### 문제 3 (구현)

`backward`에서 `relu_grad`를 곱하지 않으면 어떤 버그가 생기는가?

### 문제 4 (실험)

Loss가 폭발적으로 커질 때 가장 먼저 의볼 하이퍼파라미터는?

### 문제 5 (연결)

오늘 `sgd_update`가 PyTorch에서 보통 어떤 API 조합으로 대체되는가? (이름만)

### 문제 6 (계산)

17강 단샘플에서 `B=1`일 때 `dY = (Y_hat - y)/B`의 값은?

---

## 정답 및 해설
### 문제 1

Forward(예측·Loss) → Backward(Gradient) → Update(파라미터 갱신).

### 문제 2

$\hat{Y}=A^{(1)}W^{(2)\top}+\cdots$이므로 $W^{(2)}$에 대한 Gradient는 앞 층 활성 $A^{(1)}$과 출력 쪽 δ의 외적(배치에서는 행렬곱)으로 모인다.

### 문제 3

$z\le0$인 뉴런에도 Gradient가 흘러, ReLU의 정의와 다른 잘못된 업데이트가 수행된다.

### 문제 4

학습률 `lr` (너무 큼). 그다음 Gradient 스케일/버그.

### 문제 5

`loss.backward()` 후 `optimizer.step()` (그리고 보통 `optimizer.zero_grad()`).

### 문제 6

$0.21-1.0=-0.79$.

## 다음 강의와 연결
이번 강의에서 “수동 딥러닝 엔진”을 NumPy로 완성했다.

다음 **제19강. PyTorch Tensor**에서는, 같은 숫자를 `torch.tensor`로 다루기 시작한다.  
dtype, device, shape, broadcasting, NumPy와의 차이를 익히면, 제20강 Autograd가 오늘 `backward()` 함수를 자동으로 채워 주는 장면을 볼 준비가 된다.

> 엔진을 한 번 손으로 조립해 보았다. 이제 공장 도구(PyTorch)로 넘어간다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [17강. Backpropagation 직접 계산하기](17강_Backpropagation_직접_계산하기.md)
- **다음 강:** [19강. PyTorch Tensor](19강_PyTorch_Tensor.md)

<!-- /LECTURE_NAV -->
