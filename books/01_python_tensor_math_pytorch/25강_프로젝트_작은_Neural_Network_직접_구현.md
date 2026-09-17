# 25강. 프로젝트 — 작은 Neural Network 직접 구현
## 이번 강에서 배우는 내용

- 2층 MLP의 forward / loss / backward / update를 NumPy로 끝까지 구현한다.
- 동일한 데이터·구조를 `nn.Module` + DataLoader + Optimizer로 재현한다.
- `model.py` / `train.py` 형태의 파일 분리 습관을 갖는다.
- 학습 곡선과 예측 결과를 보고 버그를 스스로 좁힌다.

## 왜 중요한가?
지금까지의 강의는 부품 단위였다.

| 부품 | 강의 |
|---|---|
| 행렬·내적 | 제10강 |
| Gradient Descent | 제12강 |
| Loss | 제13강 |
| Chain Rule / Backprop | 제14·17·18강 |
| PyTorch Tensor / Autograd | 제19·20강 |
| Module / DataLoader / Loop / 정규화 | 제21~24강 |

부품만 알면 “조립할 수 있다”는 착각이 생긴다. 실제로 조립해 보면 다음이 드러난다.

- shape 실수
- 학습률 감각 부족
- train/val 분리 누락
- NumPy 부호 실수 vs PyTorch API 오용

LLM으로 가는 길에서도 패턴은 같다. 2권 Mini Transformer, 3권 Mini GPT는 이번 프로젝트의 **확대 버전**이다. 지금 작은 MLP로 “끝까지 학습되는 파이프라인”을 한 번 완주해야 한다.

## 프로젝트 개요
### 2.1 문제 정의

2차원 입력 $(x_1, x_2)$를 보고 두 클래스 중 하나를 예측하는 **이진 분류**를 한다.

결정 규칙은 비선형이다.

$$

y = \mathbf{1}\big[x_1^2 + x_2 > 0\big]

$$

직선 하나로 나누기 어렵기 때문에, 은닉층이 있는 MLP가 필요하다.

### 2.2 모델 구조

$$

\begin{aligned}
z_1 &= X W_1 + b_1 \\
a_1 &= \mathrm{ReLU}(z_1) \\
z_2 &= a_1 W_2 + b_2 \\
\hat{p} &= \sigma(z_2)
\end{aligned}

$$

- 입력 차원: 2  
- 은닉 차원: 16  
- 출력 차원: 1 (확률)  
- 손실: Binary Cross Entropy (BCE)

NumPy 버전에서는 수식 그대로 손으로 backward 한다.  
PyTorch 버전에서는 `BCEWithLogitsLoss` + Autograd를 쓴다.

### 2.3 권장 디렉터리 구조

두 트랙을 나란히 둔다.

```text
ch25_mini_nn/
├── data.py                 # 공통 데이터 생성
├── numpy_mlp/
│   ├── model.py            # forward / backward / params
│   └── train.py            # 학습 루프
├── torch_mlp/
│   ├── model.py            # nn.Module
│   └── train.py            # DataLoader + Optimizer
└── README.md               # 실행 방법·결과 메모 (선택)
```

한 파일에 모두 넣어도 동작은 한다. 다만 이후 Transformer 프로젝트는 파일이 늘어나므로, **지금 분리 습관을 고정**한다.

## 선수 개념
시작 전 다음을 체크한다.

- [ ] 행렬곱 shape: `(N, in) @ (in, out) → (N, out)`
- [ ] ReLU와 그 미분(마스크)
- [ ] Sigmoid / logit
- [ ] BCE Loss의 의미
- [ ] Gradient Descent 업데이트
- [ ] `nn.Module`, DataLoader, `zero_grad → backward → step`

부족하면 해당 강의를 짧게 재독하고 돌아온다. 프로젝트 도중 재독도 정상이다.

## 공통 데이터 — `data.py`
```python
# ch25_mini_nn/data.py
"""장난감 이진 분류 데이터셋 생성."""

from __future__ import annotations

import numpy as np

def make_toy_binary(
    n: int = 400,
    seed: int = 0,
    noise: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """x:(n,2), y:(n,1) with y in {0,1}."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    # 비선형 경계 + 소량 라벨 노이즈
    y = (x[:, 0] ** 2 + x[:, 1] > 0).astype(np.float64).reshape(-1, 1)
    flip = rng.random(n) < noise
    y[flip] = 1.0 - y[flip]
    return x.astype(np.float64), y

def train_val_split(
    x: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.25,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    idx = rng.permutation(n)
    n_val = int(n * val_ratio)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]

if __name__ == "__main__":
    x, y = make_toy_binary()
    print(x.shape, y.shape, y.mean())
```

예상 출력 예:

```text
(400, 2) (400, 1) 0.48...
```

클래스 비율이 극단적이지 않은지 `y.mean()`으로 확인한다.

## Part A — NumPy MLP
### 5.1 수학 정리

배치 $X \in \mathbb{R}^{N \times 2}$에 대해

$$

\begin{aligned}
Z_1 &= X W_1 + b_1 \\
A_1 &= \mathrm{ReLU}(Z_1) \\
Z_2 &= A_1 W_2 + b_2 \\
\hat{P} &= \sigma(Z_2)
\end{aligned}

$$

BCE:

$$

L = -\frac{1}{N}\sum_{i=1}^{N}\left[
  y_i\log(\hat{p}_i+\varepsilon)
  +(1-y_i)\log(1-\hat{p}_i+\varepsilon)
\right]

$$

출력층에서 sigmoid + BCE를 같이 쓰면, 로짓 $Z_2$에 대한 기울기가 깔끔해진다.

$$

\frac{\partial L}{\partial Z_2} = \frac{1}{N}(\hat{P} - Y)

$$

은닉층:

$$

\begin{aligned}
\frac{\partial L}{\partial W_2} &= A_1^\top \frac{\partial L}{\partial Z_2} \\
\frac{\partial L}{\partial b_2} &= \sum_{n}\frac{\partial L}{\partial Z_2} \\
\frac{\partial L}{\partial A_1} &= \frac{\partial L}{\partial Z_2} W_2^\top \\
\frac{\partial L}{\partial Z_1} &= \frac{\partial L}{\partial A_1} \odot \mathbf{1}[Z_1>0] \\
\frac{\partial L}{\partial W_1} &= X^\top \frac{\partial L}{\partial Z_1} \\
\frac{\partial L}{\partial b_1} &= \sum_{n}\frac{\partial L}{\partial Z_1}
\end{aligned}

$$

제17·18강에서 한 계산의 압축판이다.

### 5.2 `numpy_mlp/model.py`

```python
# ch25_mini_nn/numpy_mlp/model.py
from __future__ import annotations

import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    # 안정적인 sigmoid
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))

class NumpyMLP:
    def __init__(self, d_in=2, d_hidden=16, d_out=1, seed=0):
        rng = np.random.default_rng(seed)
        # He-like 초기화
        self.W1 = rng.normal(scale=np.sqrt(2 / d_in), size=(d_in, d_hidden))
        self.b1 = np.zeros((1, d_hidden))
        self.W2 = rng.normal(scale=np.sqrt(2 / d_hidden), size=(d_hidden, d_out))
        self.b2 = np.zeros((1, d_out))

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = np.maximum(self.Z1, 0.0)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.P = sigmoid(self.Z2)
        return self.P

    def loss(self, Y: np.ndarray, eps: float = 1e-8) -> float:
        p = np.clip(self.P, eps, 1.0 - eps)
        return float(-np.mean(Y * np.log(p) + (1.0 - Y) * np.log(1.0 - p)))

    def backward(self, Y: np.ndarray) -> None:
        n = Y.shape[0]
        dZ2 = (self.P - Y) / n
        self.dW2 = self.A1.T @ dZ2
        self.db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (self.Z1 > 0.0)
        self.dW1 = self.X.T @ dZ1
        self.db1 = np.sum(dZ1, axis=0, keepdims=True)

    def step(self, lr: float) -> None:
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2

    def predict_label(self, X: np.ndarray) -> np.ndarray:
        p = self.forward(X)
        return (p >= 0.5).astype(np.float64)
```

### 5.3 `numpy_mlp/train.py`

```python
# ch25_mini_nn/numpy_mlp/train.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import make_toy_binary, train_val_split  # noqa: E402
from numpy_mlp.model import NumpyMLP  # noqa: E402

def accuracy(model: NumpyMLP, X, Y) -> float:
    pred = model.predict_label(X)
    return float(np.mean(pred == Y))

def iterate_minibatches(X, Y, batch_size, rng):
    n = X.shape[0]
    idx = rng.permutation(n)
    for start in range(0, n, batch_size):
        batch = idx[start:start + batch_size]
        yield X[batch], Y[batch]

def main():
    x, y = make_toy_binary(n=400, seed=0, noise=0.05)
    x_tr, y_tr, x_va, y_va = train_val_split(x, y, val_ratio=0.25, seed=0)

    model = NumpyMLP(d_in=2, d_hidden=16, d_out=1, seed=1)
    lr = 0.1
    epochs = 80
    batch_size = 32
    rng = np.random.default_rng(0)

    best_val = -1.0
    best_params = None

    for epoch in range(1, epochs + 1):
        losses = []
        for xb, yb in iterate_minibatches(x_tr, y_tr, batch_size, rng):
            model.forward(xb)
            losses.append(model.loss(yb))
            model.backward(yb)
            model.step(lr)

        tr_acc = accuracy(model, x_tr, y_tr)
        va_acc = accuracy(model, x_va, y_va)
        print(
            f"[numpy] epoch {epoch:03d} | "
            f"loss={np.mean(losses):.4f} | "
            f"train_acc={tr_acc:.3f} | val_acc={va_acc:.3f}"
        )

        if va_acc > best_val:
            best_val = va_acc
            best_params = (model.W1.copy(), model.b1.copy(), model.W2.copy(), model.b2.copy())

    model.W1, model.b1, model.W2, model.b2 = best_params
    print(f"[numpy] best val_acc={best_val:.3f}")

if __name__ == "__main__":
    main()
```

### 5.4 NumPy 예상 결과

환경에 따라 숫자는 달라질 수 있다. 성공의 기준은 다음과 같다.

```text
[numpy] epoch 001 | loss=0.69.. | train_acc=0.5x | val_acc=0.5x
...
[numpy] epoch 040 | loss=0.2x  | train_acc=0.85+ | val_acc=0.80+
...
[numpy] best val_acc=0.85 전후 (대략 0.80~0.95)
```

관찰 포인트:

1. 초반 loss가 ~0.69 근처(랜덤 이진 분류)에서 시작하는가?
2. epoch가 진행되며 train/val acc가 **함께** 올라가는가?
3. train만 치솟고 val이 무너지면 제24강 정규화를 적용해 본다. (과제)

## Part B — PyTorch MLP
### 6.1 `torch_mlp/model.py`

```python
# ch25_mini_nn/torch_mlp/model.py
from __future__ import annotations

import torch
import torch.nn as nn

class TorchMLP(nn.Module):
    def __init__(self, d_in: int = 2, d_hidden: int = 16, d_out: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),  # logit 출력
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

NumPy 버전은 확률 `P`를 직접 저장했지만, PyTorch에서는 **logit**을 내고 `BCEWithLogitsLoss`에 맡기는 편이 수치적으로 안전하다.

### 6.2 `torch_mlp/train.py`

```python
# ch25_mini_nn/torch_mlp/train.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import make_toy_binary, train_val_split  # noqa: E402
from torch_mlp.model import TorchMLP  # noqa: E402

@torch.no_grad()
def accuracy(model: TorchMLP, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    for xb, yb in loader:
        logit = model(xb)
        pred = (torch.sigmoid(logit) >= 0.5).float()
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return correct / total

def main():
    torch.manual_seed(0)
    x_np, y_np = make_toy_binary(n=400, seed=0, noise=0.05)
    x_tr, y_tr, x_va, y_va = train_val_split(x_np, y_np, val_ratio=0.25, seed=0)

    train_ds = TensorDataset(
        torch.tensor(x_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(x_va, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    model = TorchMLP()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)

    best_val = -1.0
    best_state = None
    patience, wait = 20, 0

    for epoch in range(1, 101):
        model.train()
        losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logit = model(xb)
            loss = criterion(logit, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        tr = accuracy(model, train_loader)
        va = accuracy(model, val_loader)
        print(
            f"[torch] epoch {epoch:03d} | "
            f"loss={np.mean(losses):.4f} | "
            f"train_acc={tr:.3f} | val_acc={va:.3f}"
        )

        if va > best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"[torch] early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    print(f"[torch] best val_acc={best_val:.3f}")
    torch.save(best_state, ROOT / "torch_mlp" / "best.pt")
    print("saved:", ROOT / "torch_mlp" / "best.pt")

if __name__ == "__main__":
    main()
```

### 6.3 PyTorch 예상 결과

```text
[torch] epoch 001 | loss=0.6x | train_acc=0.6x | val_acc=0.6x
...
[torch] epoch 020 | loss=0.2x | train_acc=0.88+ | val_acc=0.85+
...
[torch] best val_acc=0.88 전후
saved: .../torch_mlp/best.pt
```

Adam + 적절한 lr 덕분에 NumPy SGD보다 빨리 안정되는 경우가 많다. 그렇다고 PyTorch가 “다른 수학”을 쓰는 것은 아니다. **같은 gradient update를 자동·고수준 API로 수행**할 뿐이다.

## 두 구현을 나란히 비교하기
| 항목 | NumPy | PyTorch |
|---|---|---|
| 파라미터 | `W1,b1,W2,b2` 직접 보관 | `nn.Linear`가 등록 |
| Forward | 손 코딩 | `model(x)` |
| Backward | 손 코딩 chain rule | `loss.backward()` |
| Update | `W -= lr * dW` | `optimizer.step()` |
| Batch | 직접 인덱스 | `DataLoader` |
| 저장 | `np.savez` 등 | `state_dict` |
| 이득 | 원리 투명 | 확장·GPU·생산성 |

이상적인 학습 목표:

> NumPy로 “왜 돌아가는지”를 설명하고, PyTorch로 “어떻게 키울지”를 구현한다.

LLM 코드는 PyTorch 쪽에 가깝다. 그러나 Attention 버그를 잡을 때는 NumPy로 했던 **shape·미분 추적 습관**이 그대로 필요하다.

## 한 파일로 최소 실행하고 싶을 때
디렉터리 구성이 부담되면, 먼저 아래 통합 스크립트로 성공 경험을 만든 뒤 분리해도 된다.

```python
# ch25_mini_nn/run_both_quick.py
"""빠른 스모크 테스트용. 상세 버전은 numpy_mlp/ torch_mlp/ 참고."""

import numpy as np
import torch
import torch.nn as nn

from data import make_toy_binary, train_val_split

def numpy_smoke():
    from numpy_mlp.model import NumpyMLP

    x, y = make_toy_binary(n=200, seed=0)
    x_tr, y_tr, x_va, y_va = train_val_split(x, y, seed=0)
    model = NumpyMLP(seed=0)
    for _ in range(50):
        model.forward(x_tr)
        model.backward(y_tr)
        model.step(0.2)
    pred = model.predict_label(x_va)
    print("numpy val_acc", float(np.mean(pred == y_va)))

def torch_smoke():
    x, y = make_toy_binary(n=200, seed=0)
    x_tr, y_tr, x_va, y_va = train_val_split(x, y, seed=0)
    model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    crit = nn.BCEWithLogitsLoss()
    xt = torch.tensor(x_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    for _ in range(50):
        opt.zero_grad()
        loss = crit(model(xt), yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(x_va, dtype=torch.float32)))
        acc = float(((p >= 0.5).float().numpy() == y_va).mean())
    print("torch val_acc", acc)

if __name__ == "__main__":
    numpy_smoke()
    torch_smoke()
```

## 실제 LLM에서는 이 프로젝트가 어떻게 확장되는가
이번 프로젝트의 각 칸이 이후 권에서 이렇게 커진다.

| 지금 (1권 25강) | 이후 |
|---|---|
| 2D 숫자 특징 | 토큰 ID → Embedding 벡터 (2권) |
| Linear + ReLU MLP | Attention + FFN Transformer Block (2권) |
| BCE 이진 분류 | Next-token Cross Entropy (2·3권) |
| NumPy backward | Autograd (+ 필요 시 수동 검증) |
| 단일 스크립트 학습 | 분산·혼합정밀·스케줄러 (3권) |
| val accuracy | perplexity / 생성 품질 / 벤치마크 (3~5권) |

따라서 “장난감 분류를 했으니 LLM과 무관하다”가 아니다.  
**데이터가 들어가고, 파라미터가 갱신되고, 검증으로 고르는 루프**를 완주한 것이다.

## 실습 — 필수 미션
### 미션 1 — NumPy 학습 완주

1. 위의 `data.py`, `numpy_mlp/model.py`, `numpy_mlp/train.py`를 작성한다.
2. 실행해 val_acc ≥ 0.80을 목표로 한다.
3. 실패하면 lr, epoch, hidden size 순으로 조절한다.

### 미션 2 — PyTorch 재현

1. `torch_mlp/`를 작성·실행한다.
2. `best.pt`가 저장되는지 확인한다.
3. 새 프로세스에서 `load_state_dict` 후 val_acc를 재측정한다.

### 미션 3 — 결과 보고서 (짧게)

다음을 노트에 남긴다.

- 최종 best val_acc (NumPy / PyTorch)
- 사용한 lr, batch, hidden
- 가장 오래 막힌 버그 한 가지와 해결법

## 도전 과제 (심화)
시간 되는 만큼 도전한다. 전부 할 필요는 없다.

### 도전 1 — Softmax 다중 클래스

경계를 바꿔 3클래스 분류로 확장한다. NumPy에서는 softmax + cross entropy 미분을, PyTorch에서는 `CrossEntropyLoss`를 사용한다.

### 도전 2 — Dropout 추가

NumPy에서 학습 시 마스크를 직접 만들고, 평가 시 끈다. PyTorch `nn.Dropout`과 val 곡선을 비교한다.

### 도전 3 — 초기화 민감도

`seed`만 바꿔 5번 학습해 val_acc 분산을 기록한다. 딥러닝 실험에 반복이 필요한 이유를 체감한다.

### 도전 4 — 결정 경계 시각화

격자 위의 예측을 산점도와 함께 그려, 모델이 $x_1^2 + x_2=0$ 곡선을 근사하는지 본다. (matplotlib)

### 도전 5 — 성능 대조 표

동일 epoch·동일 SGD lr로 NumPy와 PyTorch를 맞추어, 구현 차이가 아닌 **알고리즘 차이**를 줄인 비교를 시도한다. (PyTorch도 `torch.optim.SGD` 사용)

## 자주 하는 실수와 디버깅 가이드
1. **shape 불일치**  
   `W1`을 `(hidden, in)`으로 두었는데 `X @ W1`을 기대한 경우. 매 텐서에 `assert`를 넣는다.

2. **BCE에 확률과 logit을 혼동**  
   `BCELoss`에는 확률, `BCEWithLogitsLoss`에는 logit. 섞으면 학습이 이상해진다.

3. **backward에서 `1/N` 평균을 빼먹음**  
   배치 크기를 바꾸면 업데이트 스케일이 달라져 lr 튜닝이 불가능해진다.

4. **ReLU 마스크를 $A_1$ 기준이 아니라 잘못된 텐서에 적용**  
   $Z_1>0$ 마스크인지 확인한다.

5. **검증 없이 과적합 모델을 “성공”으로 착각**  
   반드시 val_acc를 본다.

6. **PyTorch에서 `model.train()`/`eval()` 누락**  
   Dropout을 넣은 뒤 특히 치명적이다.

디버깅 순서 추천:

```text
1) forward shape 인쇄
2) 초기 loss가 랜덤 수준인지 확인
3) 한 배치만으로 loss가 내려가는지 overfit 테스트
4) 전체 데이터 + val 모니터링
```

“한 배치 암기 테스트”는 구현 버그를 찾는 강력한 방법이다. 충분히 큰 모델이 배치 16개를 암기조차 못 하면 backward/update에 버그가 있을 확률이 높다.

## 수식 보강 — 작은 네트의 완전 학습 목표

입력 $x$, 정답 $y$, 모델 $f_\theta$에 대해

$$
\min_\theta\ \frac{1}{N}\sum_{i=1}^N \ell\left(f_\theta(x_i), y_i\right)
$$

을 SGD/Adam으로 푸는 것이 미니 프로젝트의 수학적 뼈대입니다. 이후 LLM도 같은 뼈대에 데이터·모델 규모만 키운 것입니다.

## LLM에서는 어디에 사용될까?

이번 25강에서 배운 개념은 이후 Transformer · GPT · 서빙 강의에서 반복해서 등장합니다. 각 수식·코드 블록을 “실제 모델의 어느 단계인가”와 연결해 다시 읽어 보세요.

## 핵심 요약
- 1권 프로젝트는 NumPy MLP와 PyTorch MLP로 **같은 학습 문제**를 두 번 푸는 것이다.
- 파일 분리(`data` / `model` / `train`)는 이후 Transformer·GPT 프로젝트의 예행 연습이다.
- NumPy는 원리를, PyTorch는 확장 가능한 공학을 담당한다.
- 성공 기준은 완벽한 숫자보다, **재현 가능한 학습 곡선 + val 기반 모델 선택**이다.
- 이 루프가 곧 LLM 학습 루프의 축소판이다.

## 용어 사전
| 용어 | 의미 |
|---|---|
| MLP | 여러 Linear·활성화로 구성된 전결합 신경망 |
| Logit | sigmoid/softmax 직전의 원시 점수 |
| BCE | 이진 분류용 교차 엔트로피 손실 |
| `BCEWithLogitsLoss` | logit 입력용 안정적 BCE |
| He Initialization | ReLU 네트워크에 맞춘 분산 스케일 초기화 |
| Mini-batch SGD | 배치 Gradient로 파라미터를 갱신하는 학습 |
| Checkpoint | 좋은 val 시점의 `state_dict` 스냅샷 |
| Smoke Test | 짧은 실행으로 파이프라인 생존을 확인하는 테스트 |

## 연습문제
### 문제 1 (개념)

이 프로젝트에서 NumPy 구현과 PyTorch 구현을 모두 요구하는 이유를 두 가지 쓰시오.

### 문제 2 (계산)

배치 $N=4$, $\hat{p}=[0.9,0.1,0.8,0.2]$, $y=[1,0,1,0]$일 때 BCE(평균, $\varepsilon$ 무시)의 대략적 크기가 작은지 큰지 직관적으로 판단하고 근거를 쓰시오.

### 문제 3 (코드)

NumPy `backward`에서 `dZ2 = (P - Y) / n`인 이유를 “sigmoid+BCE 합성 미분” 관점에서 설명하시오.

### 문제 4 (실험)

한 배치 overfit 테스트에서 loss가 전혀 안 내려간다. 점검할 항목 세 가지를 쓰시오.

### 문제 5 (연결)

2권 Mini Transformer 프로젝트로 넘어가면, 이번 `model.py`의 Linear 자리가 어떤 모듈들로 대체되는지 예상해 쓰시오.

---

## 정답 및 해설
### 문제 1

(1) 미분·형상 원리를 투명하게 검증하기 위해 (2) 이후 LLM 코드가 올라갈 공학 스택(Module/Autograd/Optimizer)을 동일 문제에 연결하기 위해.

### 문제 2

예측이 정답과 잘 맞으므로 BCE는 작은 편이다. 예: $0.9$↔1, $0.1$↔0 등은 자신 있는 올바른 예측이라 손실 기여가 낮다.

### 문제 3

$\sigma$와 BCE를 합성하면 $\partial L/\partial z = \hat{p}-y$ 형태가 되고, 배치 평균 Loss를 쓰므로 $1/n$이 곱해진다. (제13·17강 흐름)

### 문제 4

예시: 학습률 과소/과대, Gradient 부호·평균 누락, 라벨 shape `(N,)` vs `(N,1)` 불일치, 옵티마이저 step 누락, 잘못된 loss 함수(logit/확률 혼동).

### 문제 5

토큰 Embedding, Self-Attention(QKV Linear 포함), FFN(두 개의 Linear), LayerNorm, Residual 연결 등으로 대체·확장된다. 최종 출력은 vocab 크기 logit.

## 다음 강의와 연결
작은 신경망이 **실제로 학습되는 것**을 두 스택으로 확인했다.

다음 **제26강. 1권 총정리 — LLM으로 가는 다리**에서는 1~25강의 개념을 LLM 파이프라인 지도 위에 다시 배치하고, 2권 첫 관문인 Tokenizer로 건너갈 준비 상태를 점검한다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [24강. 과적합과 정규화](24강_과적합과_정규화.md)
- **다음 강:** [26강. 1권 총정리 — LLM으로 가는 다리](26강_1권_총정리_LLM으로_가는_다리.md)

<!-- /LECTURE_NAV -->
