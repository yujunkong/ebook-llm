# 제12강. Gradient와 Gradient Descent

> **학습 목표**
> - Gradient(기울기 벡터)의 정의와 기하 직관
> - Gradient Descent(경사하강법) 업데이트 식
> - Learning Rate(학습률)가 하는 일
> - 1변수·2변수 예제를 손으로 한 스텝 이상 계산
> - LLM 학습이 본질적으로 같은 절차의 대규모 버전임

---
## 1. 왜 이것을 배우는가

모델을 학습시킨다는 말을 수식으로 쓰면 대략 이렇다.

$$

\theta^{\star} = \arg\min_{\theta} L(\theta)

$$

- $\theta$: Parameter 전체
- $L(\theta)$: Loss
- $\arg\min$: Loss를 최소로 만드는 $\theta$를 찾아라

직접 최솟값 공식을 푸는 대신, 현장에서 쓰는 기본 전략은:

```text
현재 위치에서 Gradient를 재고
 → Loss가 줄어드는 쪽으로 한 걸음
 → 반복
```

이것이 Gradient Descent다.

## 2. 먼저 알아야 할 개념

- 미분 / 편미분 (11강)
- Vector (9강)
- $L(w)=(wx-y)^2$ 같은 간단한 Loss 감각 (11강)
- 내적은 아직 “방향이 같은지” 직관만 있으면 충분 (10강)

## 3. 핵심 개념 설명

### 3.1 Gradient (그래디언트, 기울기 벡터)

**Gradient(그래디언트)**는 다변수 함수의 모든 편미분을 **하나의 벡터로 모은 것**이다.

$$

f(x_1, x_2, \ldots, x_n)

$$

에 대해

$$

\nabla f
=
\begin{bmatrix}
\dfrac{\partial f}{\partial x_1} \\
\dfrac{\partial f}{\partial x_2} \\
\vdots \\
\dfrac{\partial f}{\partial x_n}
\end{bmatrix}

$$

기호 설명:

- $\nabla$: 나블라(del) 기호. Gradient 연산자
- $\nabla f$: $f$의 Gradient
- 각 성분: 해당 변수 방향의 편미분
- $\nabla f$의 Shape: 입력 변수 개수와 같은 길이의 벡터 (파라미터가 행렬이면 같은 Shape의 Tensor로 일반화)

왜 필요한가?

- 한 줄로 “어느 방향으로 함수값이 가장 빠르게 증가하는가”를 나타낸다.
- 따라서 $-\nabla f$는 **가장 빠르게 감소하는 방향**이다.
- Loss를 줄이려면 파라미터를 $-\nabla L$ 쪽으로 밀면 된다.

### 3.2 Gradient Descent (경사하강법)

**Gradient Descent(GD, 경사하강법)**는 Gradient의 반대 방향으로 파라미터를 반복 갱신해 Loss를 낮추는 최적화 방법이다.

업데이트 식:

$$

\theta \leftarrow \theta - \eta \nabla_{\theta} L(\theta)

$$

기호:

- $\theta$: 현재 파라미터 (스칼라·벡터·행렬·텐서 모두 가능)
- $\leftarrow$: 오른쪽으로 계산한 값으로 갱신
- $\eta$: **Learning Rate(학습률, step size)**. 양수 Scalar
- $\nabla_{\theta} L$: $L$을 $\theta$로 미분한 Gradient
- 빼기: 증가 방향의 반대 = 하강

한 스텝의 의미:

```text
새 위치 = 옛 위치 − (학습률) × (가장 가파른 오르막 방향)
```

### 3.3 Learning Rate

**Learning Rate $\eta$**는 한 걸음의 크기다.

| $\eta$ | 현상 |
|---|---|
| 너무 작음 | 학습이 매우 느림. Loss가 거의 안 움직임 |
| 적절 | Loss가 안정적으로 감소 |
| 너무 큼 | 최솟값을 지나쳐 진동하거나 발산 |

LLM 학습에서는 $\eta$를 스케줄링(warmup, cosine decay 등)하는 일이 많다.  
원리는 같다: **너무 성급하게 크게 움직이지 말 것**.

### 3.4 Local Minimum과 현실

GD는 전역 최솟값(global minimum)을 항상 보장하지 않는다.  
비선형 신경망 Loss 표면은 복잡하다.

실무적으로는:

- 완벽한 전역 최소보다 **충분히 낮은 Loss + 좋은 일반화**가 목표
- Stochastic Gradient Descent(SGD), Adam 등 변형이 더 자주 쓰임
- 그래도 모두 “Gradient 반대 방향으로 간다”는 친척이다

이 강의는 그 가족의 조상인 기본 GD를 손으로 정복한다.

## 4. 직관적으로 이해하기

안개 낀 산에서 계곡으로 내려간다고 하자.

- 발밑에서 기울기를 재면, 가장 가파른 오르막이 $\nabla L$
- 눈을 감고 반대로 한 걸음 내딛는 것이 GD
- 보폭이 $\eta$

1D면 왼쪽/오른쪽만 선택하면 된다.  
2D면 지도 위의 화살표 방향이 생긴다.  
수십억 차원이면 상상은 어렵지만 **식은 동일**하다.

## 5. 작은 숫자로 직접 계산하기 — 1D

목표: $L(w)=(2w-4)^2$를 최소로.

이미 11강에서

$$

L'(w)=8(w-2)

$$

임을 알았다. (1변수라 Gradient는 곧 이 스칼라 미분)

### 5.1 초기값 $w_0=0$, $\eta=0.05$

$$

L'(0)=8(0-2)=-16

$$

$$

w_1 = w_0 - \eta L'(w_0) = 0 - 0.05\cdot(-16) = 0.8

$$

$$

L(0)=(0-4)^2=16,\quad L(0.8)=(1.6-4)^2=5.76

$$

Loss가 줄었다.

### 5.2 한 스텝 더

$$

L'(0.8)=8(0.8-2)=-9.6

$$

$$

w_2=0.8 - 0.05\cdot(-9.6)=0.8+0.48=1.28

$$

$$

L(1.28)=(2.56-4)^2=2.0736

$$

계속하면 $w\to 2$, $L\to 0$으로 간다.

### 5.3 Learning Rate가 클 때

$\eta=0.3$, $w_0=0$:

$$

w_1=0-0.3\cdot(-16)=4.8

$$

$$

L(4.8)=(9.6-4)^2=31.36

$$

Loss가 **커졌다**. 너무 큰 보폭으로 반대편으로 뛰어넘은 것이다.

$$

L'(4.8)=8(4.8-2)=22.4

$$

$$

w_2=4.8-0.3\cdot22.4=4.8-6.72=-1.92

$$

진동·발산 위험이 보인다. $\eta$ 선택의 중요성이다.

## 6. 작은 숫자로 직접 계산하기 — 2D

11강 8.3절의 Loss:

$$

L(w_1,w_2)=(w_1+w_2-5)^2

$$

$$

\nabla L =
\begin{bmatrix}
2(w_1+w_2-5) \\
2(w_1+w_2-5)
\end{bmatrix}

$$

초기값 $\mathbf{w}=(2,1)$, $\eta=0.1$.

$$

w_1+w_2-5=2+1-5=-2

$$

$$

\nabla L = \begin{bmatrix}-4\\-4\end{bmatrix}

$$

업데이트:

$$

\begin{bmatrix}w_1\\w_2\end{bmatrix}
\leftarrow
\begin{bmatrix}2\\1\end{bmatrix}
-
0.1
\begin{bmatrix}-4\\-4\end{bmatrix}
=
\begin{bmatrix}2\\1\end{bmatrix}
+
\begin{bmatrix}0.4\\0.4\end{bmatrix}
=
\begin{bmatrix}2.4\\1.4\end{bmatrix}

$$

새 Loss:

$$

L(2.4,1.4)=(2.4+1.4-5)^2=(-1.2)^2=1.44

$$

이전 $L=4$에서 $1.44$로 감소.

한 스텝 더:

$$

\nabla L = \begin{bmatrix}2(-1.2)\\2(-1.2)\end{bmatrix}=\begin{bmatrix}-2.4\\-2.4\end{bmatrix}

$$

$$

\mathbf{w}\leftarrow\begin{bmatrix}2.4\\1.4\end{bmatrix}-0.1\begin{bmatrix}-2.4\\-2.4\end{bmatrix}
=\begin{bmatrix}2.64\\1.64\end{bmatrix}

$$

$$

L=(2.64+1.64-5)^2=(-0.72)^2=0.5184

$$

$w_1+w_2$가 5에 가까워진다. (이 Loss는 $w_1+w_2=5$인 모든 쌍이 최솟값 0 — 해가 하나로 정해지지 않는 예. 직관용으로 충분하다.)

## 7. Vector / Matrix Parameter로 일반화

파라미터가 행렬 $W$여도 식은 같다.

$$

W \leftarrow W - \eta \frac{\partial L}{\partial W}

$$

- $\partial L/\partial W$: $W$와 **같은 Shape**의 Gradient Tensor
- 빼기는 요소별

예: $W=\begin{bmatrix}1&2\\3&4\end{bmatrix}$, 어떤 Loss에 대해

$$

\frac{\partial L}{\partial W}=\begin{bmatrix}0.1&-0.2\\0.0&0.5\end{bmatrix},
\quad \eta=0.1

$$

이면

$$

W_{\text{new}}=
\begin{bmatrix}
1-0.01 & 2+0.02 \\
3-0.00 & 4-0.05
\end{bmatrix}
=
\begin{bmatrix}
0.99 & 2.02 \\
3.00 & 3.95
\end{bmatrix}

$$

LLM의 거대한 $W$도 **같은 규칙**으로 업데이트된다. 차이는 규모와, Gradient를 효율적으로 구하는 방법(역전파)뿐이다.

## 8. 코드로 구현하기

```python
# lecture12_gradient_descent.py
import numpy as np

def L_1d(w: float) -> float:
    return (2 * w - 4) ** 2

def grad_1d(w: float) -> float:
    return 8 * (w - 2)

def gd_1d(w0: float, eta: float, steps: int):
    w = w0
    history = [(w, L_1d(w))]
    for _ in range(steps):
        w = w - eta * grad_1d(w)
        history.append((w, L_1d(w)))
    return history

print("=== 1D eta=0.05 ===")
for w, loss in gd_1d(0.0, 0.05, 8):
    print(f"w={w:.4f}, L={loss:.4f}")

print("=== 1D eta=0.3 (너무 큼) ===")
for w, loss in gd_1d(0.0, 0.3, 6):
    print(f"w={w:.4f}, L={loss:.4f}")

def L_2d(w: np.ndarray) -> float:
    return float((w[0] + w[1] - 5) ** 2)

def grad_2d(w: np.ndarray) -> np.ndarray:
    s = 2 * (w[0] + w[1] - 5)
    return np.array([s, s], dtype=float)

def gd_2d(w0, eta, steps):
    w = np.array(w0, dtype=float)
    hist = [(w.copy(), L_2d(w))]
    for _ in range(steps):
        w = w - eta * grad_2d(w)
        hist.append((w.copy(), L_2d(w)))
    return hist

print("=== 2D ===")
for w, loss in gd_2d([2.0, 1.0], 0.1, 5):
    print(f"w={w}, L={loss:.4f}")
```

실행하면 6~7절의 손계산과 같은 패턴이 보여야 한다.

## 9. PyTorch로 같은 1D 문제

```python
# lecture12_torch_gd.py
import torch

w = torch.tensor(0.0, requires_grad=True)
eta = 0.05

for step in range(8):
    L = (2 * w - 4) ** 2
    L.backward()
    with torch.no_grad():
        w -= eta * w.grad
        w.grad.zero_()  # 다음 스텝을 위해 Gradient 누적 값 비우기
    print(f"step={step+1}, w={w.item():.4f}, L={L.item():.4f}")
```

핵심 패턴:

1. Forward: Loss 계산
2. `backward()`: Gradient 계산
3. `w -= eta * w.grad`: 업데이트
4. `grad.zero_()`: 누적 초기화

실제 학습 루프의 축소판이다.

## 10. SGD와 미니배치 (미리보기)

전체 데이터 Loss의 정확한 Gradient 대신, **일부 샘플(batch)**만으로 Gradient를 추정하는 방법이 **SGD / Mini-batch GD**다.

$$

\theta \leftarrow \theta - \eta \nabla_{\theta} L_{\text{batch}}(\theta)

$$

LLM Pretraining은 사실상 거대한 미니배치 GD(와 Adam 계열)다.

- 배치가 클수록 Gradient 추정은 안정적
- 너무 크면 계산 비용·일반화 특성이 달라질 수 있음

지금은 “전체 GD와 같은 식, 다만 Gradient가 배치 평균” 정도만 기억한다.

## 11. 실제 LLM에서는 어떻게 사용하는가

| 개념 | LLM에서의 모습 |
|---|---|
| $\theta$ | Embedding, Attention, MLP, LM Head의 모든 weight/bias |
| $L$ | 주로 Next-Token Cross Entropy (13강) |
| $\nabla L$ | Autograd/역전파로 계산 |
| $\eta$ | 스케줄러가 조절하는 학습률 |
| 한 스텝 | Optimizer step (AdamW 등) |

Optimizer가 Adam이어도, 이해하기 위한 1원리는 GD다.

> Gradient가 “어느 쪽으로 Loss가 커지는지”를 말하면, Optimizer가 “그럼 어떻게 움직일지”를 정한다.

디버깅 시에도:

- Loss가 `nan`이면 Gradient 폭주 의심
- Loss가 안 줄면 $\eta$·데이터·버그 의심
- 특정 층 Gradient가 0에 가까우면 죽은 경로 의심

모두 이번 강의의 언어로 읽힌다.

## 12. 실습

### 실습 1 — 손계산 1D

$L(w)=(w-3)^2$, $w_0=0$, $\eta=0.2$로 두 스텝을 손계산하시오. 매 스텝 $L'(w)$, $w$, $L$을 기록하시오.

### 실습 2 — 발산 재현

같은 문제에서 $\eta=2.0$으로 두 스텝을 계산하고 현상을 설명하시오.

### 실습 3 — 2D 한 스텝

$L=(w_1+w_2-5)^2$, $\mathbf{w}=(0,0)$, $\eta=0.05$로 한 스텝 갱신하시오.

### 실습 4 — NumPy GD

9절 코드를 수정해 `steps=50`, `eta=0.05`로 1D를 돌리고 최종 $w$가 2에 가까운지 확인하시오.

### 실습 5 — 행렬 업데이트

8절의 숫자로 $W$ 업데이트를 NumPy로 구현하시오.

## 13. 자주 하는 실수

1. **Gradient 방향으로 더한다**  
    ascents가 된다. Loss 최소화는 **빼기**.

2. **`grad.zero_()`를 잊는다**  
   PyTorch는 기본적으로 Gradient를 누적한다. 안 지우면 이전 스텝과 섞인다.

3. **Learning Rate만 만지며 버그를 가린다**  
   수식·Shape·타깃이 틀리면 $\eta$로는 고칠 수 없다.

4. **1D 직관을 다차원에 무비판적으로 확장한다**  
   방향은 벡터다. 성분마다 부호가 다를 수 있다.

5. **한 스텝에 너무 많은 의미를 부여한다**  
   학습은 수천·수만 스텝의 누적이다. 한 스텝은 작은 개선이면 충분하다.

## 14. 핵심 정리

- Gradient는 모든 편미분을 모은 벡터이며, 가장 가파른 증가 방향이다.
- GD: $\theta \leftarrow \theta - \eta\nabla L$.
- Learning Rate는 보폭이다. 너무 크면 발산, 너무 작으면 정체.
- 1D·2D 손계산이 그대로 거대한 LLM 업데이트의 축소판이다.
- Autograd + Optimizer step = 이 강의의 자동화.

## 15. 핵심 용어

| 용어 | 의미 |
|---|---|
| Gradient $\nabla f$ | 편미분을 모은 벡터/텐서 |
| Gradient Descent | Gradient 반대 방향으로 파라미터 갱신 |
| Learning Rate $\eta$ | 업데이트 보폭 |
| Parameter $\theta$ | 학습되는 변수 집합 |
| Loss Landscape | Loss가 만드는 고차원 “산세” |
| SGD / Mini-batch GD | 배치로 Gradient를 추정하는 GD |
| Optimizer | GD 계열 업데이트 알고리즘 구현 |

## 16. 연습 문제
### 문제 1 (개념)

왜 Loss를 줄이기 위해 Gradient를 **빼는가**?

### 문제 2 (계산)

$L(w)=w^2$, $w=4$, $\eta=0.1$일 때 한 스텝 후 $w$는?

### 문제 3 (계산)

$L(w_1,w_2)=w_1^2 + w_2^2$, $\mathbf{w}=(1,2)$, $\eta=0.1$일 때 한 스텝 후 $\mathbf{w}$는?

### 문제 4 (코드)

다음 중 GD 한 스텝으로 올바른 것은?

```python
w = w - eta * grad
# 또는
w = w + eta * grad
```

### 문제 5 (연결)

LLM에서 Embedding 행렬 $E$도 $E\leftarrow E-\eta\,\partial L/\partial E$로 업데이트된다. 이 사실이 “단어 벡터가 학습된다”는 말과 어떻게 연결되는지 두 문장으로 쓰시오.

---

## 정답 및 해설

### 문제 1

$\nabla L$는 Loss가 가장 빠르게 **증가**하는 방향이다. 최소화하려면 반대 방향인 $-\nabla L$로 가야 하므로 뺀다.

### 문제 2

$L'(w)=2w$, $L'(4)=8$.  
$w\leftarrow 4-0.1\cdot8=3.2$.

### 문제 3

$\nabla L=[2w_1, 2w_2]=[2,4]$.  
$\mathbf{w}\leftarrow[1,2]-0.1[2,4]=[0.8, 1.6]$.

### 문제 4

`w = w - eta * grad`가 맞다.

### 문제 5

Loss를 줄이는 방향으로 Embedding 행렬의 각 성분이 조금씩 바뀐다.  
그 결과 각 토큰 ID에 대응되는 벡터 표현이 과제(다음 토큰 예측 등)에 맞게 재배치된다.

## 17. 다음 강의와 연결

이번 강의에서 “어떻게 내려가는지”를 배웠다.

다음 **제13강. Loss Function**에서는 “무엇을 내려가는지”, 즉 Loss 자체를 정의한다.  
MSE와 Cross Entropy의 직관을 숫자로 보고, 왜 LLM이 Cross Entropy를 쓰는지 연결한다.

> 하강법을 알았으면, 이제 내려갈 언덕(Loss)의 높이를 정의하자.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [11강. 미분과 편미분](11강_미분과_편미분.md)
- **다음 강:** [13강. Loss Function](13강_Loss_Function.md)

<!-- /LECTURE_NAV -->
