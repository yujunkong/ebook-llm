# 21강. nn.Module로 모델 만들기
## 이번 강에서 배우는 내용

- `nn.Module`이 무엇인지, 왜 상속해서 모델을 만드는지 설명한다.
- `nn.Linear`로 한 층을 만들고, `forward`에서 순전파를 정의한다.
- `parameters()`와 `state_dict()`로 학습 가능한 가중치를 확인하고 저장·불러오기 감각을 잡는다.
- 작은 MLP를 `nn.Module` 클래스로 직접 작성한다.

## 왜 중요한가?
제15~18강에서는 NumPy로 Neural Network를 직접 조립했다. 가중치 행렬을 직접 만들고, forward와 backward를 손으로 이었다.

그 방식은 원리를 배우는 데 최고다. 그러나 모델이 조금만 커져도 다음이 고통스러워진다.

- 층이 늘어날 때마다 파라미터 목록을 직접 관리해야 한다.
- 저장·불러오기 형식을 매번 새로 정해야 한다.
- GPU로 옮길 때 텐서를 하나하나 옮겨야 한다.
- 학습/평가 모드(예: Dropout)를 직접 스위치해야 한다.

**`torch.nn.Module`**은 이 반복 작업을 표준화한 **모델의 기본 블록**이다. LLM의 Transformer도, GPT도, 결국 `nn.Module`을 쌓아 만든 큰 클래스다.

## 선수 개념
이번 강의 전에 다음이 준비되어 있어야 한다.

1. **Tensor** — `requires_grad`가 있는 다차원 배열 (제19강)
2. **Autograd** — `loss.backward()`로 기울기를 얻는 흐름 (제20강)
3. **Linear 변환** — $y = xW^\top + b$ 형태의 한 층 (제10·15강)
4. **Forward Propagation** — 입력이 층을 지나 출력이 되는 과정 (제16강)

아직 Optimizer와 DataLoader는 몰라도 된다. 그것은 제22~23강에서 다룬다.

## 핵심 개념
### 3.1 nn.Module이란 무엇인가

**`nn.Module`(모듈)**은 PyTorch에서 신경망의 **구성 단위**를 나타내는 기본 클래스이다.

- 한 개의 Linear 층도 Module이다.
- Linear 여러 개를 묶은 MLP도 Module이다.
- Transformer 전체도 Module이다.

새 모델을 만들 때는 보통 다음 패턴을 쓴다.

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 층을 등록한다

    def forward(self, x):
        # 순전파를 정의한다
        return x
```

핵심은 두 메서드다.

| 메서드 | 역할 |
|---|---|
| `__init__` | 층·파라미터를 **등록**한다 |
| `forward` | 입력이 어떻게 흘러 출력이 되는지 **계산**한다 |

모델을 호출할 때는 `model.forward(x)`를 직접 쓰기보다 **`model(x)`**를 쓴다. `nn.Module`의 `__call__`이 내부에서 `forward`를 호출하고, hook·모드 처리 같은 부가 작업을 함께 수행한다.

### 3.2 nn.Linear — 가장 기본적인 층

**`nn.Linear`(선형 층)**은 입력을 선형 변환하는 층이다.

입력이 크기 $d_{\text{in}}$이고 출력이 크기 $d_{\text{out}}$이면,

$$

y = x W^\top + b

$$

여기서

- $W$는 shape `(out_features, in_features)`인 **weight**
- $b$는 shape `(out_features,)`인 **bias** (기본값으로 존재)

```python
import torch
import torch.nn as nn

layer = nn.Linear(in_features=4, out_features=2)
x = torch.randn(3, 4)   # 배치 3, 특징 4
y = layer(x)            # 배치 3, 특징 2
print(y.shape)          # torch.Size([3, 2])
```

`nn.Linear` 자체도 `nn.Module`의 하위 클래스다. 따라서 더 큰 모델의 `__init__` 안에서 속성으로 두면, 그 가중치가 자동으로 부모 모델에 **등록**된다.

### 3.3 파라미터 등록이란 무엇인가

**파라미터 등록(Parameter registration)**이란, 학습되어야 할 `nn.Parameter` 텐서가 모델의 관리 목록에 들어가는 것을 말한다.

등록이 되면 다음이 가능해진다.

- `model.parameters()`로 Optimizer에 넘길 수 있다.
- `model.to(device)`로 GPU/CPU를 한 번에 옮길 수 있다.
- `model.state_dict()`로 저장·복원할 수 있다.

흔한 실수는 `self.weight = torch.randn(...)`처럼 일반 텐서만 만드는 것이다. 이렇게 하면 Autograd 그래프에는 들어갈 수 있어도, Module의 파라미터 목록에는 **안 잡히는** 경우가 많다. 층을 쓸 때는 `nn.Linear`처럼 Module API를 쓰거나, 직접 만들 때는 `nn.Parameter`로 감싸야 한다.

### 3.4 parameters()와 named_parameters()

**`parameters()`**는 모델에 등록된 학습 가능 파라미터를 iterator로 돌려준다.

```python
for p in model.parameters():
    print(p.shape, p.requires_grad)
```

이름까지 보고 싶으면 **`named_parameters()`**를 쓴다.

```python
for name, p in model.named_parameters():
    print(name, tuple(p.shape))
```

출력 예:

```text
fc1.weight (8, 4)
fc1.bias (8,)
fc2.weight (2, 8)
fc2.bias (2,)
```

LLM처럼 파라미터가 수억~수천억 개여도, 원리는 같다. “이름이 붙은 텐서들의 집합”이 모델이다.

### 3.5 state_dict — 가중치의 스냅샷

**`state_dict`(상태 사전)**는 모델(또는 Optimizer)의 현재 상태를 `dict`로 담은 것이다. 모델의 경우 보통 각 파라미터의 이름 → 텐서 값 매핑이다.

```python
sd = model.state_dict()
print(sd.keys())
torch.save(sd, "model.pt")

model2 = MyModel()
model2.load_state_dict(torch.load("model.pt", weights_only=True))
```

학습을 이어가려면 Optimizer의 `state_dict`까지 함께 저장하는 것이 일반적이다. 그 패턴은 제23강·3권 Checkpoint 강의에서 더 깊게 다룬다. 지금은 “모델 = 구조(코드) + 가중치(state_dict)”라는 분리만 확실히 잡자.

## 직관적으로 이해하기
`nn.Module`을 레고 블록으로 생각하면 쉽다.

- `nn.Linear`는 작은 벽돌 하나
- `MyMLP`는 벽돌을 붙인 작은 집
- GPT는 같은 규칙으로 쌓은 아주 큰 건물

중요한 규칙은 두 가지다.

1. **벽돌을 `__init__`에서 선반에 올려 둔다** (등록)
2. **손님이 오면 `forward`에서 어떤 순서로 지나가는지 정한다** (계산 경로)

손님이 집 안으로 들어오는 호출은 `model(x)`다. 집 안 동선을 직접 부르는 `model.forward(x)`보다, 현관(`__call__`)을 통하는 습관을 들인다.

## 수학적으로 이해하기
2층 MLP를 예로 든다. 입력 $x \in \mathbb{R}^{d}$, 은닉 크기 $h$, 출력 크기 $c$라고 하자.

$$

\begin{aligned}
z_1 &= x W_1^\top + b_1 \\
a_1 &= \mathrm{ReLU}(z_1) \\
z_2 &= a_1 W_2^\top + b_2
\end{aligned}

$$

PyTorch 코드의 `forward`는 위 식의 순서와 1:1로 대응한다.

$$

W_1 \in \mathbb{R}^{h \times d},\quad
W_2 \in \mathbb{R}^{c \times h}

$$

학습은 Loss $L$에 대해 $\partial L / \partial W_1$, $\partial L / \partial b_1$, … 를 Autograd가 계산하고, Optimizer가 값을 갱신한다. Module은 “어떤 텐서가 파라미터인지”를 표시해 주는 명찰 역할이다.

## 작은 숫자로 직접 계산하기
입력이 하나라고 가정한다.

$$

x = [1.0,\ 0.5]

$$

첫 번째 Linear: `in=2, out=2`, bias 없음으로 단순화.

$$

W_1 = \begin{bmatrix} 1 & 0 \\ 0 & 2 \end{bmatrix}

$$

PyTorch의 `y = x @ W.T`이므로

$$

z_1 = [1.0,\ 1.0]

$$

ReLU 후에도 $[1.0,\ 1.0]$. 두 번째 Linear `in=2, out=1`,

$$

W_2 = \begin{bmatrix} 0.5 & 0.5 \end{bmatrix}

$$

이면 출력은 $1.0$이다.

코드에서 같은 숫자를 넣어 보면, `forward`가 수식과 같은지 바로 검증할 수 있다. “작은 숫자 검증”은 이후 Transformer 구현에서도 같은 습관으로 반복한다.

## 코드로 구현하기 — 최소 MLP
```python
# tiny_mlp.py
import torch
import torch.nn as nn

class TinyMLP(nn.Module):
    """입력 d → 은닉 h → 출력 c 인 작은 다층 퍼셉트론."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

def main():
    torch.manual_seed(0)
    model = TinyMLP(d_in=4, d_hidden=8, d_out=2)

    x = torch.randn(5, 4)          # 배치 크기 5
    y = model(x)                   # forward 호출
    print("output shape:", y.shape)

    print("--- parameters ---")
    total = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        print(f"{name:12s} shape={tuple(p.shape)} numel={n}")
    print("total parameters:", total)

    print("--- state_dict keys ---")
    print(list(model.state_dict().keys()))

if __name__ == "__main__":
    main()
```

예상 출력의 핵심:

```text
output shape: torch.Size([5, 2])
fc1.weight   shape=(8, 4) numel=32
fc1.bias     shape=(8,) numel=8
fc2.weight   shape=(2, 8) numel=16
fc2.bias     shape=(2,) numel=2
total parameters: 58
```

파라미터 수 검산:

$$

8\cdot4 + 8 + 2\cdot8 + 2 = 32 + 8 + 16 + 2 = 58

$$

LLM 논문을 읽을 때 나오는 “7B parameters”도 같은 방식으로, 모든 층의 `numel()` 합이다.

## PyTorch로 구현하기 — Sequential과 수동 forward
간단한 층 나열은 `nn.Sequential`로도 만들 수 있다.

```python
model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 2),
)
```

다만 다음이 필요해지면 **클래스를 직접 쓰는 편**이 낫다.

- skip connection (잔차 연결)
- 여러 입력을 받는 forward
- 중간 활성화 값을 반환
- 층마다 다른 마스킹·정규화

Transformer Block은 Residual + LayerNorm + Attention + FFN이 얽히므로, 실무·이 책의 후반에서는 대부분 `nn.Module` 서브클래스를 직접 정의한다.

가중치를 고정된 값으로 넣어 수식과 대조하는 예:

```python
import torch
import torch.nn as nn

layer = nn.Linear(2, 2, bias=False)
with torch.no_grad():
    layer.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 2.0]]))

x = torch.tensor([[1.0, 0.5]])
print(layer(x))  # tensor([[1., 1.]])
```

`torch.no_grad()` 안에서 가중치를 직접 고치면, 불필요한 그래프 추적을 피할 수 있다.

## LLM에서는 어디에 사용될까?
LLM 코드베이스를 열어 보면 반복되는 패턴이 보인다.

```text
class CausalSelfAttention(nn.Module): ...
class MLP(nn.Module): ...
class Block(nn.Module): ...
class GPT(nn.Module): ...
```

각 블록은

1. `__init__`에서 `nn.Linear`, Embedding, LayerNorm 등을 등록하고
2. `forward`에서 Attention·FFN·Residual을 연결한다.

학습 스크립트는 대략 다음과 같다.

```python
model = GPT(config)
model.to(device)
logits = model(input_ids)          # forward
loss = loss_fn(logits, targets)
loss.backward()
optimizer.step()
```

또한 체크포인트는 사실상 `state_dict`의 저장이다. Hugging Face의 `model.save_pretrained`도 내부적으로는 구조 설정과 가중치 텐서를 분리해 둔다.

2권에서 Self-Attention을 구현할 때도, 이번 강의의 `nn.Module` 뼈대 위에 Q/K/V Linear를 올리게 된다.

## 실습
### 실습 1 — 파라미터 수 검산

**목표:** `named_parameters()`와 손 계산이 일치하는지 확인한다.

1. `TinyMLP(16, 32, 10)`을 만든다.
2. 총 파라미터 수를 손으로 계산한다.
3. `sum(p.numel() for p in model.parameters())`와 비교한다.

**검산식:**

$$

32\cdot16 + 32 + 10\cdot32 + 10

$$

### 실습 2 — state_dict 저장과 복원

**목표:** 구조와 가중치의 분리를 체감한다.

1. 모델을 만들고 임의 입력에 대한 출력을 저장한다.
2. `torch.save(model.state_dict(), "tiny.pt")`로 저장한다.
3. **새 인스턴스**를 만든 뒤 `load_state_dict`로 불러온다.
4. 같은 입력에 대해 출력이 같은지 `torch.allclose`로 확인한다.

### 실습 3 — forward 경로 바꾸기

**목표:** `__init__` 등록과 `forward` 계산의 차이를 이해한다.

`TinyMLP`에 은닉층을 하나 더 넣어 3층으로 확장한다. 단, 층을 추가만 하고 `forward`에서 빼먹으면 어떻게 되는지 실험한 뒤, 올바르게 연결한다.

**추가 도전:** `nn.Sequential` 버전과 클래스 버전의 파라미터 수가 같은지 비교한다.

## 자주 하는 실수
1. **`forward` 안에서 `nn.Linear(...)`를 새로 만든다**  
   매 forward마다 새 가중치가 생긴다. 층은 `__init__`에서 한 번만 만든다.

2. **`model.forward(x)`만 습관적으로 호출한다**  
   동작은 하지만 hook·모드 처리가 건너뛰어질 수 있다. `model(x)`를 쓴다.

3. **일반 텐서를 속성으로 두고 학습되길 기대한다**  
   `nn.Parameter` 또는 `nn.Linear` 같은 Module로 등록해야 `parameters()`에 나타난다.

4. **`state_dict`만 저장하고 클래스 정의를 잃어버린다**  
   가중치는 숫자일 뿐이다. 같은 구조의 클래스가 있어야 복원된다.

5. **입출력 shape를 확인하지 않는다**  
   `(batch, features)` 관례를 깨면 Linear가 바로 실패한다. 항상 `print(x.shape)`로 확인한다.

## 핵심 요약
- `nn.Module`은 PyTorch 모델의 기본 블록이다. `__init__`에서 등록하고 `forward`에서 계산한다.
- `nn.Linear`는 $y = xW^\top + b$를 수행하는 가장 흔한 층이다.
- `parameters()` / `named_parameters()`로 학습 가중치를 순회한다.
- `state_dict`는 가중치 스냅샷이며, 저장·불러오기의 표준 형식이다.
- LLM의 모든 블록도 같은 Module 규칙 위에 올라간다.

## 용어 사전
| 용어 | 의미 |
|---|---|
| `nn.Module` | 신경망 구성 단위의 기본 클래스 |
| `nn.Linear` | 선형 변환 층 ($xW^\top + b$) |
| `forward` | 순전파 계산을 정의하는 메서드 |
| Parameter | 학습 대상으로 등록된 텐서 |
| `parameters()` | 등록된 파라미터 iterator |
| `state_dict` | 이름→텐서 매핑의 모델 상태 사전 |
| `nn.Sequential` | 층을 순서대로 쌓는 간단한 컨테이너 |
| `nn.ReLU` | $\max(0, x)$ 활성화 Module |

## 연습문제
### 문제 1 (개념)

`nn.Module`을 상속할 때 `__init__`과 `forward`의 역할을 구분하여 설명하시오.

### 문제 2 (개념)

`model(x)`와 `model.forward(x)`의 차이를 간단히 쓰시오.

### 문제 3 (계산)

`nn.Linear(20, 5)`와 `nn.Linear(5, 3)`를 잇는 MLP(편향 포함)의 총 파라미터 수를 구하시오.

### 문제 4 (코드)

다음 코드의 문제점을 찾고 고치시오.

```python
class Bad(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.fc = nn.Linear(4, 2)  # ?
        return self.fc(x)
```

### 문제 5 (연결)

GPT의 Attention 블록도 `nn.Module`인 이유가 무엇인지, 파라미터 관리 관점에서 설명하시오.

---

## 정답 및 해설
### 문제 1

`__init__`은 Linear 같은 하위 Module·Parameter를 등록한다. `forward`는 등록된 층을 어떤 순서로 적용해 출력을 만들지 정의한다.

### 문제 2

`model(x)`는 `__call__`을 거쳐 `forward`를 호출하므로 추가 훅/모드 처리가 포함된다. `model.forward(x)`는 순전파 본체만 직접 호출한다. 일반적으로 `model(x)`를 사용한다.

### 문제 3

$$

(20\cdot5 + 5) + (5\cdot3 + 3) = 105 + 18 = 123

$$

### 문제 4

`forward` 안에서 `nn.Linear`를 생성하면 호출마다 새 파라미터가 생긴다. `self.fc = nn.Linear(4, 2)`를 `__init__`으로 옮긴다.

### 문제 5

Attention의 Q/K/V 가중치, 출력 투영 등이 모두 학습 파라미터다. `nn.Module`로 묶어야 `parameters()`, `to(device)`, `state_dict`로 일괄 관리할 수 있다. GPT는 그런 Module을 쌓은 큰 Module이다.

## 다음 강의와 연결
모델의 **골격**은 만들었다. 이제 그 골격에 **데이터 묶음**을 넣어 줄 도구가 필요하다.

다음 **제22강. Dataset과 DataLoader**에서는 샘플을 정의하고, 배치로 묶어, 학습 루프에 공급하는 방법을 배운다. LLM이 왜 문장을 토큰 배치로 학습하는지도 여기서 연결된다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [20강. Autograd — 자동 미분](20강_Autograd_자동_미분.md)
- **다음 강:** [22강. Dataset과 DataLoader](22강_Dataset과_DataLoader.md)

<!-- /LECTURE_NAV -->
