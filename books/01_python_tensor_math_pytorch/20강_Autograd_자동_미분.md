# 20강. Autograd — 자동 미분
## 이번 강에서 배우는 내용

- Autograd(자동 미분)가 Computational Graph를 기록·역순회하는 엔진임을 설명
- `requires_grad`, `loss.backward()`, `tensor.grad`의 관계
- Loss → Graph → Chain Rule → Gradient → `Parameter.grad` 파이프라인
- `detach`, `torch.no_grad()`, inplace 함정의 의미
- 제17강 손계산 숫자와 `backward()` 결과가 일치함을 검증
- 수동 NumPy Backprop과 Autograd의 대응표

## 왜 중요한가?
LLM 학습 코드의 핵심 한 줄은 종종 다음이다.

```python
loss.backward()
```

이 한 줄 뒤에서 실제로 일어나는 일은 제17강의 전부이다.

```text
Loss
  → Computational Graph를 거꾸로 탐색
    → 각 연산의 Local Gradient 규칙 적용 (Chain Rule)
      → 잎(leaf) 텐서의 .grad에 누적
```

원리를 모르면

- Gradient가 `None`인 이유
- `no_grad` 때문에 학습이 안 되는 이유
- `detach` 후 그래프가 끊기는 이유
- 두 번 `backward`할 때 생기는 오류

를 해석할 수 없다. 오늘은 그 해석력을 만든다.

## 선수 개념
- Chain Rule (제14강)
- Forward cache와 δ (제16~17강)
- NumPy 수동 `backward` (제18강)
- `torch.tensor`의 shape/dtype/device (제19강)

## 핵심 개념
### 3.1 Autograd

**Autograd(Automatic Differentiation, 자동 미분)**는 연산이 실행될 때 계산 그래프를 기록하고, 나중에 `backward`로 각 입력에 대한 Gradient를 자동 계산하는 시스템이다.

PyTorch는 주로 **Reverse-mode AD**(역모드 자동 미분)를 쓴다. 스칼라 Loss 하나에 대해 모든 파라미터 Gradient를 한 번에 구하기에 딥러닝과 잘 맞는다.

**왜 필요한가?**  
모델을 바꿀 때마다 수동으로 $\partial L/\partial W$를 유도할 수 없다. Autograd는 “연산을 정의하면 미분도 따라온다.”

**예제:** `y = (a*x + b)**2`만 적어도 `y.backward()`가 `x.grad`, `a.grad`를 채운다.

### 3.2 requires_grad

**`requires_grad`**는 해당 텐서를 출발점으로 하는 연산을 추적할지 여부이다.

```python
w = torch.tensor([0.5], requires_grad=True)
```

- `True`: 이 텐서에 의존하는 연산이 그래프에 기록된다.
- `False`(기본): 추적하지 않는다. Inference용 입력 등에 적합.

학습 파라미터는 보통 `True`이다. `nn.Parameter`는 기본이 `requires_grad=True`이다(제21강).

### 3.3 Computational Graph (동적)

PyTorch는 **Define-by-Run**이다. Forward를 실행하는 순간 그래프가 만들어진다.

```text
leaf(w, b, x) → mul/add → … → loss(scalar)
```

그래프의 중간 노드는 함수 객체(`AddBackward0` 등)로 연결된다.  
`loss.backward()`는 이 링크를 타고 내려간다.

### 3.4 backward와 grad

**`loss.backward()`**는 스칼라 `loss`에 대해 $\partial L/\partial(\text{leaf})$를 계산해 leaf 텐서의 **`.grad`**에 저장(기본은 누적)한다.

```text
Loss (스칼라)
  → backward()
    → Chain Rule
      → w.grad, b.grad, …
```

**왜 스칼라인가?**  
역모드 AD는 출력 차원 1일 때 “모든 입력 Gradient”를 효율적으로 준다.  
벡터 출력이면 보통 `backward(gradient=...)`로 시드 Gradient를 넘겨야 한다. 학습 Loss는 스칼라로 만드는 것이 표준이다.

### 3.5 Leaf / Non-leaf

- **Leaf Tensor**: 사용자가 만든 텐서. `requires_grad=True`이면 `.grad`가 채워진다.
- **Non-leaf**: 연산 결과. 기본적으로 `.grad`는 안 차고, 필요하면 `retain_grad()`를 쓴다.

```python
a = torch.tensor(1.0, requires_grad=True)  # leaf
b = a * 2                                  # non-leaf
```

### 3.6 detach

**`detach()`**는 그래프에서 끊긴 **같은 값의 텐서**를 반환한다.

```python
y = model(x)
target = y.detach()   # 더 이상 y 경로로 Gradient 안 흐름
```

**왜 필요한가?**

- Teacher signal을 그래프 밖에 둘 때
- 메트릭 계산 시 불필요한 그래프 성장 방지
- 일부 GAN/RL 패턴에서 경로 선택적 차단

`tensor.data`를 직접 만지는 구식 패턴보다 `detach()`가 안전하다.

### 3.7 torch.no_grad / inference_mode

**`torch.no_grad()`**는 블록 안에서 그래프 기록을 끈다.

```python
with torch.no_grad():
    y = model(x)   # 빠르고 메모리 적음, 학습 경로 없음
```

**`torch.inference_mode()`**는 추론에 더 특화된 유사 API이다.

**왜 필요한가?**  
Validation/Inference에서 Autograd는 비용만 늘린다. LLM 생성 루프는 기본적으로 no_grad 영역이다.

### 3.8 zero_grad와 누적

`.grad`는 기본 **누적(accumulate)** 된다.  
매 step 시작에 비워야 한다.

```python
optimizer.zero_grad()  # 또는 w.grad = None / w.grad.zero_()
loss.backward()
optimizer.step()
```

고의로 누적하는 기법도 있다: **Gradient Accumulation**(큰 배치를 흉내, 3권).

### 3.9 Loss → Graph → Chain Rule → Gradient → Parameter.grad

한 줄로 고정한다.

```text
1) Forward: 연산이 Graph에 기록됨 (requires_grad=True 경로)
2) Loss 스칼라 획득
3) loss.backward():
     - Loss에서 시작
     - Graph를 역방향으로 순회
     - 각 연산의 VJP(Vector-Jacobian Product) = Chain Rule 적용
4) Leaf parameter의 .grad에 ∂L/∂θ 저장
5) Optimizer가 θ ← θ - η · θ.grad (또는 Adam 등)
```

제17강의 손계산이 곧 3)~4)이다.

## 직관적으로 이해하기
Autograd를 **블랙박스 카메라**로 생각하자.

1. `requires_grad=True`인 재료를 쓰면, Forward를 찍을 때 **촬영(그래프 기록)**이 켜진다.
2. 마지막에 Loss라는 한 장의 성적표를 얻는다.
3. `backward()`는 필름을 **거꾸로 재생**하며 “어느 손잡이가 성적에 영향 줬는지” 적는다.
4. 쪽지가 `.grad`이다.
5. `detach`/`no_grad`는 “이 구간은 촬영하지 마”라는 플래카드다.

NumPy 수동 구현은 카메라 없이 직접 쪽지를 쓰는 일이다. 결과는 같아야 한다.

## 수학적으로 이해하기
스칼라 $L(w)$, Forward에서 중간 $u=f(w)$, $L=g(u)$이면 Autograd는

\[
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial u}
\cdot
\frac{\partial u}{\partial w}
\]

를 연산 노드마다 적용한다.

일반적 **Vector-Jacobian Product (VJP)**:

\[
\mathbf{v}^{\top} \frac{\partial \mathbf{f}}{\partial \mathbf{x}}
\]

`backward`가 위에서 내려보내는 upstream $\mathbf{v}$와, 로컬 Jacobian을 곱해 아래 입력을 갱신한다.  
제17강의 “Upstream × Local”과 동일하다.

다중 경로가 있으면 기여분을 **더한다**. Residual Connection이 많은 Transformer에서 이 합 규칙이 핵심이다.

## 작은 숫자로 직접 계산하기 — 17강과 대조
동일 설정:

```text
x=[1.0, 0.5], y=1.0
W1=[[0.3, -0.2],[0.4, 0.1]], b1=[0.1, -0.1]
W2=[[0.5, -0.4]], b2=[0.2]
은닉 ReLU, Loss = 1/2 (ŷ - y)^2
```

Forward 복습:

\[
z^{(1)}=[0.30, 0.35],\quad a^{(1)}=[0.30, 0.35],\quad \hat{y}=0.21,\quad L=0.31205
\]

해석적 Gradient (제17강):

```text
∂L/∂ŷ   = -0.79
dW2     = [[-0.237, -0.2765]]
db2     = [-0.79]
dW1     = [[-0.395, -0.1975],
           [ 0.316,  0.158 ]]
db1     = [-0.395, 0.316]
```

Autograd가 같은 값을 `.grad`에 넣는지 8절에서 확인한다.

### 6.1 초미니 손미분 (워밍업)

$x=2,\; w=3,\; b=1$, $y=wx+b=7$, $L=(y-10)^2=9$.

\[
\begin{aligned}
\frac{\partial L}{\partial y} &= 2(y-10) = -6 \\
\frac{\partial L}{\partial w} &= -6\cdot x = -12 \\
\frac{\partial L}{\partial x} &= -6\cdot w = -18 \\
\frac{\partial L}{\partial b} &= -6
\end{aligned}
\]

8.1 코드 결과와 일치해야 한다.

## 코드로 구현하기
### 7.1 최소 예제

```python
"""20강: Autograd 최소 예제."""

import torch

x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# Forward
y = w * x + b          # 7
L = (y - 10) ** 2      # 9

# Backward: Loss → Graph → Chain Rule → .grad
L.backward()

print(float(x.grad))  # -18
print(float(w.grad))  # -12
print(float(b.grad))  # -6
```

### 7.2 2-2-1을 Autograd로

```python
import torch

x = torch.tensor([1.0, 0.5])  # 입력은 보통 requires_grad=False
y = torch.tensor(1.0)

W1 = torch.tensor([[0.3, -0.2], [0.4, 0.1]], requires_grad=True)
b1 = torch.tensor([0.1, -0.1], requires_grad=True)
W2 = torch.tensor([[0.5, -0.4]], requires_grad=True)
b2 = torch.tensor([0.2], requires_grad=True)

# Forward (그래프 기록)
z1 = x @ W1.T + b1
a1 = torch.relu(z1)
y_hat = (a1 @ W2.T + b2).squeeze()
loss = 0.5 * (y_hat - y) ** 2

print("y_hat", float(y_hat))   # 0.21
print("loss ", float(loss))    # 0.31205

# Backward
loss.backward()

print("W2.grad", W2.grad)      # ~ [[-0.237, -0.2765]]
print("b2.grad", b2.grad)      # ~ [-0.79]
print("W1.grad\n", W1.grad)
print("b1.grad", b1.grad)
```

검증:

```python
assert torch.allclose(W2.grad, torch.tensor([[-0.237, -0.2765]]), atol=1e-6)
assert torch.allclose(b2.grad, torch.tensor([-0.79]), atol=1e-6)
assert torch.allclose(
    W1.grad,
    torch.tensor([[-0.395, -0.1975], [0.316, 0.158]]),
    atol=1e-6,
)
assert torch.allclose(b1.grad, torch.tensor([-0.395, 0.316]), atol=1e-6)
print("17강 손계산과 일치")
```

### 7.3 수치 미분과 교차 검증

```python
def autograd_vs_numeric():
    """leaf 파라미터마다 해석적 vs 수치 Gradient 비교."""
    params = {
        "W1": torch.tensor([[0.3, -0.2], [0.4, 0.1]], requires_grad=True),
        "b1": torch.tensor([0.1, -0.1], requires_grad=True),
        "W2": torch.tensor([[0.5, -0.4]], requires_grad=True),
        "b2": torch.tensor([0.2], requires_grad=True),
    }
    x = torch.tensor([1.0, 0.5])
    y = torch.tensor(1.0)

    def forward_loss():
        z1 = x @ params["W1"].T + params["b1"]
        a1 = torch.relu(z1)
        y_hat = (a1 @ params["W2"].T + params["b2"]).squeeze()
        return 0.5 * (y_hat - y) ** 2

    loss = forward_loss()
    loss.backward()

    eps = 1e-6
    for name, t in params.items():
        g_ana = t.grad.detach().clone()
        g_num = torch.zeros_like(t)
        for i in range(t.numel()):
            orig = float(t.view(-1)[i].detach())
            t.view(-1)[i].data = orig + eps
            lp = float(forward_loss().detach())
            t.view(-1)[i].data = orig - eps
            lm = float(forward_loss().detach())
            g_num.view(-1)[i] = (lp - lm) / (2 * eps)
            t.view(-1)[i].data = orig
        # 다음 비교를 위해 grad 비우기
        for p in params.values():
            if p.grad is not None:
                p.grad.zero_()
        denom = g_ana.abs().max().clamp_min(1e-12)
        rel = (g_ana - g_num).abs().max() / denom
        print(name, "max_rel", float(rel))
```

### 7.4 detach

```python
a = torch.tensor([1.0, 2.0], requires_grad=True)
b = a * 3
c = b.detach()          # 그래프 끊김, requires_grad=False
print(c.requires_grad)  # False

loss = (a * 3).sum()
loss.backward()
print(a.grad)           # tensor([3., 3.])
```

실무 패턴:

```python
# 값은 쓰되 미분 경로는 차단
pred = model(x)
target_like = pred.detach()
```

### 7.5 no_grad와 파라미터 갱신

```python
# 올바른 추론
with torch.no_grad():
    y_eval = torch.relu(x @ W1.T + b1) @ W2.T + b2

# 수동 SGD 한 스텝
loss = 0.5 * ((torch.relu(x @ W1.T + b1) @ W2.T + b2).squeeze() - y) ** 2
loss.backward()
lr = 0.1
with torch.no_grad():
    W1 -= lr * W1.grad
    b1 -= lr * b1.grad
    W2 -= lr * W2.grad
    b2 -= lr * b2.grad
for t in (W1, b1, W2, b2):
    t.grad.zero_()
```

업데이트 자체는 그래프가 필요 없으므로 `no_grad` 안에서 수행한다.  
`optimizer.step()`도 같은 철학이다.

### 7.6 누적과 zero_grad

```python
W = torch.tensor([1.0], requires_grad=True)
for i in range(2):
    loss = (W * 2).sum()
    loss.backward()
    print("step", i, "grad", W.grad.clone())  # 2.0 그다음 4.0 (누적)
W.grad.zero_()
print("after zero", W.grad)
```

### 7.7 retain_graph / create_graph

```python
w = torch.tensor(2.0, requires_grad=True)
loss = w ** 2
loss.backward(retain_graph=True)   # 그래프 유지
loss.backward()                    # 두 번째 가능 (grad 누적)

# 이계 미분이 필요하면 create_graph=True
w = torch.tensor(2.0, requires_grad=True)
loss = w ** 2
loss.backward(create_graph=True)
# 초보 단계에서는 거의 불필요. 에러 메시지에 자주 등장하므로 이름만 기억
```

### 7.8 inplace 함정

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x * 2
# x += 1   # inplace — RuntimeError 가능
x = x + 1  # 새 텐서 할당이 안전
```

Autograd는 Forward 때 값을 저장해 Backward에 쓴다. 그 값을 inplace로 덮으면 실패한다.

### 7.9 수동 NumPy vs Autograd 대응표

| NumPy (18강) | Autograd (20강) |
|---|---|
| `cache` 딕셔너리 | 엔진이 자동 저장 |
| `backward(cache)` 함수 | `loss.backward()` |
| `grads["W2"]` | `W2.grad` |
| `sgd_update` | `with no_grad: W -= lr*W.grad` 또는 Optimizer |
| `relu_grad` 수동 | `torch.relu`의 Backward 규칙 내장 |
| 수치 미분 검증 | 동일하게 사용 가능 |

## 학습 미니 루프 (Autograd판)
```python
def train_toy_regression(steps=200, lr=0.05):
    """합성 회귀를 Autograd SGD로 학습."""
    torch.manual_seed(0)
    X = torch.randn(64, 2)
    y = (2 * X[:, 0] - 3 * X[:, 1] + 0.5).unsqueeze(1)

    W1 = torch.randn(2, 2) * 0.3
    b1 = torch.zeros(2)
    W2 = torch.randn(1, 2) * 0.3
    b2 = torch.zeros(1)
    for t in (W1, b1, W2, b2):
        t.requires_grad_(True)

    history = []
    for step in range(steps):
        # 1) Forward → Loss
        pred = torch.relu(X @ W1.T + b1) @ W2.T + b2
        loss = 0.5 * torch.mean((pred - y) ** 2)
        # 2) Backward → .grad
        loss.backward()
        # 3) Update
        with torch.no_grad():
            W1 -= lr * W1.grad
            b1 -= lr * b1.grad
            W2 -= lr * W2.grad
            b2 -= lr * b2.grad
            for p in (W1, b1, W2, b2):
                p.grad.zero_()
        history.append(float(loss.detach()))
        if step % 40 == 0 or step == steps - 1:
            print(f"step {step:4d}  loss={history[-1]:.6f}")
    return history

if __name__ == "__main__":
    train_toy_regression()
```

**성공 기준:** 마지막 Loss가 초깃값보다 명확히 작다.  
이것이 제18강 NumPy 루프의 Autograd 번역본이다.

## Autograd가 채우는 것과 안 채우는 것
| 채운다 | 안 채운다 / 비어 있음 |
|---|---|
| `requires_grad=True` leaf의 `.grad` | 입력 토큰 ID (보통 False) |
| Loss에 연결된 경로의 파라미터 | `detach`된 경로 |
| | `no_grad` 블록에서 만든 연산 |
| | freeze한 층 (`requires_grad=False`) |

디버깅 질문: “이 텐서가 Loss까지 **미분 가능한 길로 연결**돼 있는가?”

## LLM에서는 어디에 사용될까?
전형적인 학습 조각:

```python
outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
loss = outputs.loss                 # 스칼라
optimizer.zero_grad()
loss.backward()                     # 오늘 배운 전부
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

생성(Inference):

```python
model.eval()
with torch.no_grad():
    logits = model(input_ids)
```

추가 연결:

| LLM 기법 | Autograd 관점 |
|---|---|
| Mixed Precision | 스케일된 Loss로 backward 후 unscale |
| Gradient Checkpointing | Forward 저장↓, Backward 때 재계산 |
| LoRA | 일부 파라미터만 `requires_grad=True` |
| RLHF / PPO | old policy·reward는 `detach`/`no_grad` |
| Grad Accumulation | `backward` 여러 번 누적 후 한 번 `step` |

## 실습
### 실습 1 — 17강 assert

8.2의 네 파라미터 Gradient를 `torch.allclose`로 모두 통과시킨다.

### 실습 2 — requires_grad 끄기

`W2.requires_grad_(False)` 후 `backward`하면 `W2.grad is None`인지 확인한다.  
층을 얼리는(freeze) 감각이다.

### 실습 3 — detach 실험

`pred.detach()`를 타깃처럼 쓰는 장난 코드를 만들고, 학습 파라미터로 Gradient가 안 흘러감을 확인한다.

### 실습 4 — no_grad 평가

학습 루프 매 40 step마다 `no_grad`로 full-batch Loss를 출력한다.

### 실습 5 — zero_grad 잊기

고의로 `zero_grad`를 제거하고 `.grad`가 커지는 장면을 본 뒤 복구한다.

### 실습 6 — NumPy와 교차

제18강 `backward` 결과 배열과 Autograd `.grad.numpy()`를 같은 초기값에서 비교한다.

### 실습 7 — 한 스텝 후 Loss

$\eta=0.1$로 한 번 업데이트한 뒤 Loss가 $0.31205$보다 작아지는지 확인한다(제17강 맛보기와 동일).

## 자주 하는 실수
1. **스칼라 아닌 Tensor에 `.backward()`**  
   `mean`/`sum`으로 줄인다.

2. **`zero_grad` 누락**  
   Gradient가 step마다 합쳐져 학습이 폭주한다.

3. **`no_grad` 안에서 학습 Forward**  
   `.grad`가 비거나 backward가 실패한다.

4. **non-leaf의 `.grad` 기대**  
   중간값이 필요하면 `retain_grad()` 또는 `torch.autograd.grad`.

5. **GPU 텐서를 바로 numpy**  
   `.detach().cpu().numpy()`.

6. **inplace 연산으로 그래프 파괴**  
   추적 텐서에 `+=`, `relu_`를 함부로 쓰지 않는다.

7. **두 번 backward without retain_graph**  
   그래프는 기본적으로 free된다.

8. **Loss를 Python float로 바꿔 버린다**  
   `float(loss)` 이후에는 그래프가 없다. 로그는 `float(loss.detach())`.

## 자주 하는 디버깅 체크리스트
1. `loss` shape가 `()` 인가?
2. 파라미터 `requires_grad`가 True인가?
3. Forward가 `no_grad` 밖인가?
4. `backward` 후 `.grad`가 존재하는가?
5. `zero_grad` 타이밍이 step 시작인가?
6. dtype/device가 섞이지 않았는가?
7. 17강 fixture assert가 통과하는가?

## 핵심 요약
- Autograd는 Reverse-mode 자동 미분으로, Forward 때 그래프를 기록하고 `backward`로 Chain Rule을 적용한다.
- 핵심 파이프라인: **Loss → Computational Graph → Chain Rule → `.grad` → 파라미터 갱신**.
- `requires_grad`가 추적 스위치, `detach`/`no_grad`가 추적 차단, `zero_grad`가 누적 초기화이다.
- 제17강 손계산·제18강 NumPy 결과와 Autograd는 같은 숫자를 내야 한다.
- LLM 학습·추론의 `backward`/`no_grad`는 이 강의의 직접 확장이다.

## 용어 사전
| 용어 | 의미 |
|---|---|
| Autograd | 연산 그래프 기반 자동 미분 엔진 |
| requires_grad | 텐서 추적 여부 플래그 |
| Computational Graph | Forward 연산을 연결한 의존 그래프 |
| backward() | Loss에서 leaf로 Gradient를 전파하는 호출 |
| .grad | leaf 파라미터에 저장된 Gradient |
| Leaf Tensor | 사용자가 생성한, grad가 쌓이는 텐서 |
| detach | 그래프를 끊은 텐서 반환 |
| torch.no_grad | 기록 없는 실행 컨텍스트 |
| zero_grad | Gradient 버퍼 초기화 |
| VJP | Vector-Jacobian Product, 역모드 AD의 단위 연산 |
| retain_graph | backward 후 그래프 유지 |
| inplace | 기존 저장공간을 덮어쓰는 연산 (Autograd와 충돌 가능) |
| Reverse-mode AD | 스칼라 출력 기준 모든 입력 Gradient를 효율 계산하는 방식 |

## 연습문제
### 문제 1 (개념)

`loss.backward()`가 수행하는 일을 Loss, Graph, Chain Rule, `.grad` 단어로 한 문장 정리하시오.

### 문제 2 (계산)

$x=2,w=3,b=1$, $L=(wx+b-10)^2$일 때 $w.grad$와 $x.grad$는?

### 문제 3 (개념)

Validation 루프에서 `torch.no_grad()`를 쓰는 이유 두 가지를 쓰시오.

### 문제 4 (코드)

같은 파라미터로 `backward`를 두 step 돌릴 때 `zero_grad`를 안 하면 두 번째 `.grad`는 어떻게 되는가?

### 문제 5 (연결)

제18강 NumPy `grads["W1"]`는 Autograd에서 무엇에 대응하는가?

### 문제 6 (응용)

LoRA처럼 일부만 학습할 때 Autograd 관점에서 무엇을 끄면 되는가?

### 문제 7 (계산)

2-2-1 예제에서 `loss.backward()` 후 `b2.grad`의 기댓값은?

---

## 정답 및 해설
### 문제 1

스칼라 Loss에서 시작해 기록된 Computational Graph를 역방향으로 순회하며 Chain Rule(VJP)을 적용해 leaf 텐서의 `.grad`에 Gradient를 저장한다.

### 문제 2

$w.grad=-12$, $x.grad=-18$.

### 문제 3

(1) 불필요한 그래프 기록으로 인한 메모리·시간 낭비 방지  
(2) 실수로 학습 경로가 생기지 않게 분리

### 문제 4

이전 Gradient 위에 더해져 값이 커진다(누적).

### 문제 5

`W1.grad`.

### 문제 6

학습하지 않을 파라미터의 `requires_grad=False`(freeze). LoRA 어댑터만 True로 둔다.

### 문제 7

$-0.79$.

## 다음 강의와 연결
이번 강의에서 `backward()`의 정체를 수동 Backprop과 나란히 놓았다.

다음 **제21강. nn.Module로 모델 만들기**에서는 Weight를 일일이 변수로 들고 다니지 않고, `nn.Linear`, `nn.ReLU`, `nn.Module` 서브클래스로 모델을 조립한다.  
파라미터 등록, `model.parameters()`, `model.train()/eval()`이 Autograd와 어떻게 붙는지 이어서 본다.

> 손계산으로 엔진을 이해했고, Autograd로 엔진키를 받았다. 이제 차체(`nn.Module`)를 얹자.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [19강. PyTorch Tensor](19강_PyTorch_Tensor.md)
- **다음 강:** [21강. nn.Module로 모델 만들기](21강_nn_Module로_모델_만들기.md)

<!-- /LECTURE_NAV -->
