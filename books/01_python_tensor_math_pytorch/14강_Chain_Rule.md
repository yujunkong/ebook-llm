# 1권. Python · Tensor · 수학 · PyTorch

## 제14강. Chain Rule

### 1. 이번 강의에서 배울 것

11~13강에서 미분, Gradient Descent, Loss를 배웠다. 신경망은 이 Loss를 **여러 함수가 겹친 합성**으로 만든다. 합성함수를 미분하는 규칙이 **Chain Rule(연쇄법칙)**이다.

이 강의를 마치면 다음을 말할 수 있어야 한다.

- \(f(g(x))\)의 미분을 손으로 계산하기
- 다변수·다단계로 Chain Rule이 확장되는 감각
- **Computational Graph(계산 그래프)**로 Forward/Backward를 읽기
- Backpropagation이 Chain Rule의 체계적 적용임을 이해하기
- LLM처럼 깊은 모델에서 “Gradient가 뒤로 흐른다”는 말의 의미

15강 Neural Network 구조, 이후 Backpropagation 본 강의의 수학적 출발점이다.

### 2. 왜 이것을 배우는가

한 층의 Linear만 있으면 \(L(W)\)를 직접 미분해도 된다.  
그러나 Transformer는 Embedding → 수십~수백 Block → LM Head처럼 **깊이**가 있다.

각 단계는 국소적으로 간단하다.

```text
덧셈, 곱셈, matmul, Softmax, LayerNorm, Residual, ...
```

전체를 한 번에 미분 전개하면 불가능에 가깝다.  
Chain Rule은 “국소 미분을 곱해 연결”하면 전체 미분이 나온다는 규칙이다.

Autograd가 마법처럼 보여도, 내부에서 하는 일은 이 규칙의 반복이다.

### 3. 먼저 알아야 할 개념

- 도함수 \(f'(x)\) (11강)
- 편미분 (11강)
- Gradient / GD / Loss (12~13강)
- 합성: 출력이 다음 입력으로 들어감

### 4. 핵심 개념 설명

#### 4.1 Chain Rule (연쇄법칙) — 1변수

바깥 함수 \(f\), 안쪽 함수 \(g\),

$$
y = f(g(x))
$$

이면

$$
\frac{dy}{dx} = f'(g(x))\cdot g'(x)
$$

다른 표기:

$$
\frac{dy}{dx} = \frac{dy}{du}\cdot \frac{du}{dx}
\quad \text{where } u=g(x),\ y=f(u)
$$

기호 설명:

- \(u=g(x)\): 중간값
- \(\frac{dy}{du}=f'(u)\): 바깥 함수가 중간에 얼마나 민감한지
- \(\frac{du}{dx}=g'(x)\): 안쪽 함수가 \(x\)에 얼마나 민감한지
- 곱: 민감도가 **연쇄적으로 증폭/감쇠**

왜 필요한가?

- 깊은 네트워크의 \(\partial L/\partial W\)는 수많은 국소 미분의 곱(및 합)이다.
- 한 층만 빼먹어도 Gradient가 틀어진다.

#### 4.2 직관

자동차 변속처럼 생각해 보자.

- 페달을 조금 밟으면 엔진 회전이 변하고 (\(du/dx\))
- 엔진 회전이 변하면 속도가 변한다 (\(dy/du\))
- 페달→속도 전체 민감도는 둘의 곱

Loss ← … ← 층2 ← 층1 ← Parameter 도 같다.

#### 4.3 세 겹 합성

$$
y=f(g(h(x)))
$$

$$
\frac{dy}{dx}=\frac{dy}{dv}\cdot\frac{dv}{du}\cdot\frac{du}{dx}
$$

여기서 \(u=h(x)\), \(v=g(u)\), \(y=f(v)\).

깊이가 늘어날수록 곱하는 항이 늘어난다.  
이것이 이후 **Gradient Vanishing/Exploding** 이야기의 씨앗이다. (곱이 계속 작아지거나 커짐)

#### 4.4 다변수 Chain Rule (필요한 최소)

\(z=f(x,y)\), 그런데 \(x=x(t)\), \(y=y(t)\)이면

$$
\frac{dz}{dt}
=
\frac{\partial f}{\partial x}\frac{dx}{dt}
+
\frac{\partial f}{\partial y}\frac{dy}{dt}
$$

중간 변수가 여러 갈래로 \(t\)에 의존하면 **경로별 기여를 더한다**.

신경망에서도 한 파라미터가 여러 경로로 Loss에 영향을 주면, Backward에서 그 경로 Gradient를 **합산**한다. (PyTorch의 Gradient 누적과 같은 정신)

### 5. 작은 숫자로 직접 계산하기

#### 5.1 기본 예 \(y=(2x-1)^2\)

안쪽 \(u=2x-1\), 바깥 \(y=u^2\).

$$
\frac{dy}{du}=2u,\quad \frac{du}{dx}=2
$$

$$
\frac{dy}{dx}=2u\cdot 2=4u=4(2x-1)
$$

\(x=1\)이면 \(u=1\), \(dy/dx=4\).  
11강에서 전개로 얻은 \(8x-4\)에 \(x=1\)을 넣은 값과 같다 (\(8-4=4\)).

#### 5.2 숫자로 Forward / Backward

\(x=3\), \(y=(2x-1)^2\).

**Forward (앞으로 계산):**

$$
u=2\cdot3-1=5,\quad y=25
$$

**Backward (국소 미분 곱):**

$$
\frac{dy}{du}=2u=10,\quad \frac{du}{dx}=2,
\quad \frac{dy}{dx}=10\cdot 2=20
$$

공식 \(4(2x-1)=4\cdot5=20\). 일치.

#### 5.3 Loss까지 한 줄로

간단한 모델:

$$
\hat{y}=wx,\quad L=\frac12(\hat{y}-y)^2
$$

정답 \(y\)는 상수. \(w\)에 대한 미분:

$$
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial \hat{y}}
\cdot
\frac{\partial \hat{y}}{\partial w}
=
(\hat{y}-y)\cdot x
$$

수치: \(w=1\), \(x=2\), \(y=4\)

$$
\hat{y}=2,\quad
\frac{\partial L}{\partial \hat{y}}=2-4=-2,\quad
\frac{\partial L}{\partial w}=-2\cdot 2=-4
$$

12강 GD라면

$$
w\leftarrow w-\eta(-4)=w+4\eta
$$

\(w\)를 키우는 방향 — 직관과 맞다 (\(wx\)가 4에 못 미침).

#### 5.4 두 층 미니 네트워크

$$
h = w_1 x, \quad \hat{y}=w_2 h, \quad L=\frac12(\hat{y}-y)^2
$$

Forward 예: \(x=2\), \(y=4\), \(w_1=1\), \(w_2=1\)

$$
h=2,\ \hat{y}=2,\ L=\frac12(2-4)^2=2
$$

국소 미분:

$$
\frac{\partial L}{\partial \hat{y}}=\hat{y}-y=-2
$$

$$
\frac{\partial \hat{y}}{\partial w_2}=h=2,
\quad
\frac{\partial \hat{y}}{\partial h}=w_2=1
$$

$$
\frac{\partial h}{\partial w_1}=x=2
$$

Chain Rule:

$$
\frac{\partial L}{\partial w_2}
=
\frac{\partial L}{\partial \hat{y}}
\cdot
\frac{\partial \hat{y}}{\partial w_2}
=(-2)\cdot 2=-4
$$

$$
\frac{\partial L}{\partial h}
=
\frac{\partial L}{\partial \hat{y}}
\cdot
\frac{\partial \hat{y}}{\partial h}
=(-2)\cdot 1=-2
$$

$$
\frac{\partial L}{\partial w_1}
=
\frac{\partial L}{\partial h}
\cdot
\frac{\partial h}{\partial w_1}
=(-2)\cdot 2=-4
$$

두 가중치 Gradient가 모두 -4.  
\(\eta=0.05\)면 각각 \(+0.2\)만큼 증가한다.

이 계산이 **수동 Backpropagation**이다.

#### 5.5 덧셈 노드의 분기

$$
u=a+b,\quad L=\frac12 u^2
$$

$$
\frac{\partial L}{\partial u}=u,\quad
\frac{\partial u}{\partial a}=1,\quad
\frac{\partial u}{\partial b}=1
$$

$$
\frac{\partial L}{\partial a}=u\cdot 1=u,
\quad
\frac{\partial L}{\partial b}=u
$$

덧셈은 Gradient를 **그대로 복사해 양쪽**으로 보낸다.

곱셈 \(u=ab\)면

$$
\frac{\partial L}{\partial a}=\frac{\partial L}{\partial u}\cdot b,
\quad
\frac{\partial L}{\partial b}=\frac{\partial L}{\partial u}\cdot a
$$

서로 상대방 값을 곱해 보낸다.  
계산 그래프의 “국소 규칙”이 이렇게 단순하다.

### 6. Computational Graph (계산 그래프)

**Computational Graph(계산 그래프)**는 연산을 노드로, 값의 흐름을 엣지로 나타낸 그림이다.

5.4절 예:

```text
x ──┐
    × ── h ──┐
w1 ─┘        × ── ŷ ── (−y) ── □²/2 ── L
          w2 ─┘
```

#### 6.1 Forward Pass

입력과 파라미터로 중간값·출력을 **왼쪽→오른쪽** 계산.

저장해 두는 이유: Backward에 \(h\), \(w_2\) 같은 값이 필요하기 때문이다.

#### 6.2 Backward Pass

Loss에서 시작해 **오른쪽→왼쪽**으로

$$
\frac{\partial L}{\partial (\text{각 노드})}
$$

를 Chain Rule로 전달한다.

이것이 **Backpropagation(오차역전파)**의 그래프 관점 정의다.

#### 6.3 표로 보는 Forward / Backward

| 노드 | Forward 값 | 필요한 국소 미분 | Backward로 구한 \(\partial L/\partial\) |
|---|---:|---|---:|
| \(L\) | 2 | — | 1 |
| \(\hat{y}\) | 2 | \(\partial L/\partial\hat{y}=\hat{y}-y\) | -2 |
| \(w_2\) | 1 | \(\times h\) | -4 |
| \(h\) | 2 | \(\times w_2\) | -2 |
| \(w_1\) | 1 | \(\times x\) | -4 |

### 7. 코드로 구현하기 — 수동 역전파

```python
# lecture14_manual_backprop.py
# 2층 미니 네트워크를 Chain Rule로 직접 미분한다.

x = 2.0
y = 4.0
w1 = 1.0
w2 = 1.0
eta = 0.05

# ----- Forward -----
h = w1 * x
y_hat = w2 * h
L = 0.5 * (y_hat - y) ** 2
print(f"Forward: h={h}, y_hat={y_hat}, L={L}")

# ----- Backward (Chain Rule) -----
dL_dyhat = y_hat - y          # d/dy_hat of 0.5*(y_hat-y)^2
dL_dw2 = dL_dyhat * h         # y_hat = w2 * h
dL_dh = dL_dyhat * w2
dL_dw1 = dL_dh * x            # h = w1 * x

print(f"Backward: dL/dw1={dL_dw1}, dL/dw2={dL_dw2}")

# ----- Gradient Descent step -----
w1 = w1 - eta * dL_dw1
w2 = w2 - eta * dL_dw2
print(f"Updated: w1={w1}, w2={w2}")

# 한 스텝 후 Loss 확인
h = w1 * x
y_hat = w2 * h
L_new = 0.5 * (y_hat - y) ** 2
print(f"New L={L_new}")  # 2보다 작아야 함
```

수치 미분으로 검증:

```python
# lecture14_check_grad.py
eps = 1e-5
# 위에서 구한 analytic dL/dw1 와 비교


def loss_of(w1, w2):
    h = w1 * x
    y_hat = w2 * h
    return 0.5 * (y_hat - y) ** 2


x, y = 2.0, 4.0
w1, w2 = 1.0, 1.0
num_dw1 = (loss_of(w1 + eps, w2) - loss_of(w1 - eps, w2)) / (2 * eps)
num_dw2 = (loss_of(w1, w2 + eps) - loss_of(w1, w2 - eps)) / (2 * eps)
print("numeric:", num_dw1, num_dw2)
print("analytic:", -4.0, -4.0)
```

### 8. PyTorch Autograd와 같은 예

```python
# lecture14_torch_chain.py
import torch

x = torch.tensor(2.0)
y = torch.tensor(4.0)
w1 = torch.tensor(1.0, requires_grad=True)
w2 = torch.tensor(1.0, requires_grad=True)

h = w1 * x
y_hat = w2 * h
L = 0.5 * (y_hat - y) ** 2
L.backward()

print(w1.grad, w2.grad)  # tensor(-4.), tensor(-4.)
```

수동 Chain Rule과 Autograd가 같은 Gradient를 준다.  
이제 `backward()`의 정체를 말할 수 있다.

> Autograd = 계산 그래프를 만든 뒤 Chain Rule로 Backward.

### 9. 여러 경로가 합쳐지는 예

$$
a = w,\quad b=w,\quad u=a+b,\quad L=\frac12 u^2
$$

\(w=3\)이면 \(u=6\), \(L=18\).

경로:

$$
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial u}\frac{\partial u}{\partial a}\frac{\partial a}{\partial w}
+
\frac{\partial L}{\partial u}\frac{\partial u}{\partial b}\frac{\partial b}{\partial w}
=
6\cdot1\cdot1 + 6\cdot1\cdot1 = 12
$$

또는 \(u=2w\), \(L=\frac12(2w)^2=2w^2\), \(dL/dw=4w=12\).

Residual connection처럼 같은 텐서가 여러 곳으로 연결되면, Backward에서 Gradient가 **더해진다**.

### 10. 실제 LLM에서는 어떻게 사용하는가

#### 10.1 깊고 긴 연쇄

대략적 흐름:

```text
token id
 → Embedding
 → Block1 (Attention + MLP + Residual + Norm)
 → Block2
 → ...
 → Block N
 → LM Head logits
 → Cross Entropy Loss
```

Loss에서 Embedding까지 Gradient가 가려면, 그 사이 모든 연산의 국소 미분이 Chain Rule로 곱·합되어야 한다.

#### 10.2 Attention도 예외가 아니다

\(QK^\top\), Softmax, \(\times V\)도 각각 미분 가능 연산이다.  
전체 \(\partial L/\partial W_Q\)는 그 연쇄의 결과다.

#### 10.3 왜 Residual이 도움이 되나 (예고)

Residual \(x+F(x)\)는 Backward 시 Gradient에 **+1 경로**를 남긴다.  
깊은 곱셈 연쇄만 있을 때보다 Gradient가 사라지기 어렵다.  
세부 논의는 네트워크/Transformer 강의에서 이어진다. 지금은 Chain Rule 관점의 힌트만 잡는다.

#### 10.4 학습 루프와의 대응

| 단계 | 수학 | 코드 |
|---|---|---|
| Forward | 합성함수 값 계산 | `logits = model(x)` / `loss = ...` |
| Backward | Chain Rule로 \(\partial L/\partial\theta\) | `loss.backward()` |
| Update | \(\theta\leftarrow\theta-\eta\nabla L\) | `optimizer.step()` |

14강까지로 이 표의 세 줄이 모두 언어화되었다.

### 11. 실습

#### 실습 1 — 손계산

\(y=\sin(3x)\)에서 \(\dfrac{dy}{dx}\)를 Chain Rule로 쓰시오. (\(\sin\)의 도함수는 \(\cos\))  
\(x=0\)에서의 값도 구하시오.

#### 실습 2 — Forward/Backward

\(y=(3x+1)^3\), \(x=1\)에서 Forward 중간값과 \(\dfrac{dy}{dx}\)를 구하시오.

#### 실습 3 — 두 층 네트워크

5.4절에서 \(w_1=0.5\), \(w_2=2\), \(x=2\), \(y=4\)로 Forward와 \(\partial L/\partial w_1\), \(\partial L/\partial w_2\)를 다시 계산하시오. (\(L=\frac12(\hat{y}-y)^2\))

#### 실습 4 — 코드 검증

실습 3 결과를 `lecture14_manual_backprop.py`를 수정해 검증하고, 수치 미분과도 비교하시오.

#### 실습 5 — 그래프 그리기

\(L=\frac12(w_2(w_1 x)-y)^2\)의 계산 그래프를 종이에 그리고, Backward 화살표 방향을 표시하시오.

### 12. 자주 하는 실수

1. **바깥 미분만 하고 안쪽을 빼먹는다**  
   \((2x-1)^2\)를 \(2(2x-1)\)로만 두면 안 된다. \(\times 2\)가 더 필요하다.

2. **Forward 값을 저장하지 않는다**  
   Backward의 국소 미분은 종종 Forward 활성화에 의존한다.

3. **분기 경로를 한 쪽만 미분한다**  
   같은 변수가 두 번 쓰이면 기여를 더해야 한다.

4. **Chain Rule과 GD를 혼동한다**  
   Chain Rule은 Gradient를 **구하는** 규칙, GD는 그 Gradient로 **갱신**하는 규칙.

5. **Autograd만 믿고 손계산을 안 한다**  
   버그는 작은 그래프를 손으로 풀어봐야 빨리 잡힌다.

### 13. 핵심 정리

- Chain Rule: \(\dfrac{dy}{dx}=\dfrac{dy}{du}\dfrac{du}{dx}\).
- 깊은 모델의 Gradient는 국소 미분의 곱과 합이다.
- Computational Graph에서 Forward는 값, Backward는 민감도를 전달한다.
- Backpropagation = Chain Rule의 체계적·효율적 적용.
- Autograd 결과는 수동 Chain Rule·수치 미분으로 검증할 수 있다.

### 14. 핵심 용어

| 용어 | 의미 |
|---|---|
| Chain Rule (연쇄법칙) | 합성함수 미분 규칙 |
| Intermediate Variable | \(u=g(x)\) 같은 중간값 |
| Computational Graph | 연산을 노드로 나타낸 그래프 |
| Forward Pass | 입력→출력 방향 계산 |
| Backward Pass | Loss→파라미터 방향 Gradient 전달 |
| Backpropagation | 그래프 위 Chain Rule로 모든 \(\partial L/\partial\theta\) 계산 |
| Local Gradient | 한 연산 노드의 국소 미분 |
| Autograd | 자동으로 그래프를 만들고 Backward하는 시스템 |

### 15. 복습 문제

#### 문제 1 (계산)

\(y=(5x+2)^2\)에서 Chain Rule로 \(\dfrac{dy}{dx}\)를 구하고 \(x=1\)에서의 값을 구하시오.

#### 문제 2 (계산)

\(h=w_1 x\), \(\hat{y}=w_2 h\), \(L=\frac12(\hat{y}-y)^2\)일 때 \(\dfrac{\partial L}{\partial w_1}\)을 \(x,y,w_1,w_2\)로 표현하시오.

#### 문제 3 (개념)

Backpropagation이 “그냥 미분”이 아니라 “Chain Rule의 적용”이라고 말하는 이유를 두 문장으로 쓰시오.

#### 문제 4 (코드)

다음 코드의 `w.grad`는?

```python
import torch
x = torch.tensor(3.0)
w = torch.tensor(2.0, requires_grad=True)
L = (w * x) ** 2
L.backward()
print(w.grad)
```

#### 문제 5 (연결)

LLM에서 Loss는 마지막에만 Scalar로 존재한다. Embedding 행렬까지 Gradient가 도달하려면 무엇이 연속으로 적용되어야 하는가?

---

### 정답 및 해설

#### 문제 1

\(u=5x+2\), \(y=u^2\), \(dy/dx=2u\cdot5=10(5x+2)\).  
\(x=1\)이면 \(10\cdot7=70\).

#### 문제 2

$$
\frac{\partial L}{\partial w_1}=(\hat{y}-y)\cdot w_2\cdot x=(w_2 w_1 x - y)\, w_2\, x
$$

#### 문제 3

네트워크는 많은 함수의 합성이라 전체 식을 한 번에 펼쳐 미분하기 어렵다.  
각 연산의 국소 미분을 Chain Rule로 곱·합하면 모든 파라미터 Gradient를 효율적으로 얻을 수 있다.

#### 문제 4

\(L=(wx)^2=w^2 x^2\), \(\partial L/\partial w=2w x^2=2\cdot2\cdot9=36\).  
`w.grad`는 `tensor(36.)`.

#### 문제 5

Loss에서 LM Head, 각 Transformer Block, Embedding에 이르는 모든 연산에 대해 Chain Rule(Backward Pass)이 연속 적용되어야 한다.

### 16. 다음 강의와 연결

1권의 수학 핵심 레일 — Tensor, 행렬곱, 미분, Gradient Descent, Loss, Chain Rule — 이 한 줄로 이어졌다.

다음 **제15강. Neural Network 구조**에서는 이 레일 위에 **층(Layer), 활성화 함수, 다층 퍼셉트론**을 올려 실제 “네트워크”를 조립한다.  
Forward는 행렬곱의 반복이고, Backward는 이번 강의의 Chain Rule이다.

> 연쇄법칙까지 왔다면, 이제 신경망이라는 기계를 설계할 차례다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [제13강. Loss Function](13강_Loss_Function.md)
- **다음 강:** [제15강. Neural Network의 구조](15강_Neural_Network의_구조.md)

<!-- /LECTURE_NAV -->
