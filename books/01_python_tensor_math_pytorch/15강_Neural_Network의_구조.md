# 15강. Neural Network의 구조
## 이번 강에서 배우는 내용

- Neuron(뉴런), Layer(층), Weight(가중치), Bias(편향)가 무엇인지
- Activation Function(활성화 함수)이 왜 필요한지
- MLP(Multi-Layer Perceptron)가 어떻게 층을 쌓는지
- 작은 네트워크(2→2→1)의 구조를 그림과 수식으로 설명할 수 있는지
- LLM 안의 Feed-Forward Network가 같은 부품으로 보이기 시작하는지

## 왜 중요한가?
LLM을 열면 거대한 파라미터가 나온다. 그 숫자들의 정체는 대부분 다음과 같다.

```text
입력 벡터
  → 행렬 곱(Weight) + Bias
    → 비선형 변환(Activation)
      → 다음 층 …
```

즉 Neural Network는 “마법의 상자”가 아니라 **선형 변환과 비선형 변환의 반복**이다.  
구조를 모르면 Loss·Gradient·Optimizer를 배워도 “어디에 기울기를 흘릴지”가 보이지 않는다.

이 강의의 목적은 이후 Forward/Backward가 꽂힐 **골격**을 만드는 것이다.

## 선수 개념
다음을 이미 알고 있다고 가정한다.

- **Vector / Matrix / Tensor** (제9강)
- **내적과 행렬곱** (제10강)
- **편미분과 Gradient** (제11~12강)
- **Loss Function** (제13강)
- **Chain Rule** (제14강)

아직 코드로 학습 루프를 돌리지 않아도 된다. 오늘은 구조와 용어가 우선이다.

## 핵심 개념
### 3.1 Neural Network

**Neural Network(신경망)**는 입력에 Weight와 Bias를 적용하고, Activation을 거쳐 출력을 만드는 **계층적 계산 모델**이다.

이름은 생물 뉴런에서 왔지만, 현대 딥러닝에서 중요한 것은 생물학적 유사성보다 **미분 가능한 합성 함수**라는 점이다.

```text
입력 x → 네트워크 f_θ → 출력 ŷ
```

여기서 **θ(theta, 세타)**는 학습 가능한 파라미터 전체(주로 Weight와 Bias)를 가리킨다.

### 3.2 Neuron (뉴런, 노드)

**Neuron(뉴런)**은 네트워크의 최소 계산 단위이다. 한 뉴런은 대략 다음을 한다.

1. 여러 입력 $x_1, x_2, \ldots, x_n$을 받는다.
2. 각 입력에 Weight $w_1, w_2, \ldots, w_n$을 곱해 더한다.
3. Bias $b$를 더한다.
4. Activation Function $\sigma$를 통과시킨다.

수식으로 쓰면 다음과 같다.

\[
z = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b
\]

\[
a = \sigma(z)
\]

- **$z$**: Pre-activation(활성화 전 값). Affine 변환의 결과.
- **$a$**: Activation(활성화 후 값). 다음 층으로 전달되는 신호.

**왜 필요한가?**  
뉴런 하나가 “특징 하나”를 감지하는 작은 감지기다. 층을 쌓으면 특징이 조합된다.

**예제:** 입력이 $(x_1, x_2) = (1.0, 2.0)$, Weight가 $(0.5, -0.3)$, Bias가 $0.1$이면

\[
z = 0.5\cdot 1.0 + (-0.3)\cdot 2.0 + 0.1 = 0.0
\]

이다. Activation이 ReLU라면 $a = \max(0, 0) = 0$이다.

### 3.3 Weight (가중치)

**Weight(가중치)**는 입력이 출력에 미치는 영향의 크기를 나타내는 **학습 가능한 파라미터**이다.

- 양수 Weight: 해당 입력이 커질수록 $z$가 커지는 방향
- 음수 Weight: 해당 입력이 커질수록 $z$가 작아지는 방향
- 절댓값이 큰 Weight: 영향이 큼

**왜 필요한가?**  
데이터에서 “어떤 입력이 중요한지”를 사람이 규칙으로 쓰지 않고, 학습으로 찾게 하기 위함이다.

행렬로 모으면 Layer 전체의 Weight는 $W$가 된다.

### 3.4 Bias (편향)

**Bias(편향)**는 입력과 무관하게 $z$를 위아래로 옮기는 **학습 가능한 상수**이다.

직관적으로는 직선 $y = wx$를 $y = wx + b$로 평행 이동시키는 역할과 같다.  
Bias가 없으면 항상 원점 근처를 지나야 해서, 표현력이 줄어든다.

**예제:** $z = 2x + b$에서 $b = -1$이면, $x = 0$일 때 $z = -1$이다.  
문턱을 옮기는 스위치처럼 동작한다.

### 3.5 Layer (층)

**Layer(층)**는 같은 단계에서 병렬로 계산되는 뉴런들의 묶음이다.

- **Input Layer(입력층)**: 원시 입력을 받는 자리. 보통 Weight가 없다.
- **Hidden Layer(은닉층)**: 입력과 출력 사이에서 표현을 변환하는 층.
- **Output Layer(출력층)**: 최종 예측을 만드는 층.

층을 하나 지나면 대략 다음이 일어난다.

\[

$$
\mathbf{a}^{(l)} = \sigma\!\left(W^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}\right)
\]
$$

위 첨자 $(l)$은 $l$번째 층을 뜻한다.

### 3.6 Activation Function (활성화 함수)

**Activation Function(활성화 함수)**는 Pre-activation $z$를 비선형으로 변환하는 함수이다.

**왜 필요한가?**  
Activation이 없으면 층을 아무리 쌓아도 전체가 하나의 큰 선형 변환으로 붕괴한다.

\[
W_2(W_1 x + b_1) + b_2 = (W_2 W_1)x + (W_2 b_1 + b_2)
\]

비선형성이 있어야 곡선 경계, XOR 같은 문제, 언어의 복잡한 패턴을 표현할 수 있다.

이 강의에서는 두 가지를 먼저 소개한다.

#### ReLU

**ReLU(Rectified Linear Unit)**는 다음으로 정의된다.

\[

$$
\mathrm{ReLU}(z) = \max(0, z)
\]
$$

- $z > 0$이면 그대로 통과
- $z \le 0$이면 0

장점: 계산이 싸고, 깊은 네트워크에서 Gradient가 잘 흐르는 편이다.  
단점: $z < 0$인 뉴런은 Gradient가 0이 되어 “죽어” 버릴 수 있다(Dying ReLU).

#### Sigmoid

**Sigmoid(시그모이드)**는 출력을 0과 1 사이로 압축한다.

\[

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
\]
$$

- 큰 양수 → 1에 가까움
- 큰 음수 → 0에 가까움
- 0 근처 → 0.5 근처

장점: 확률처럼 보이는 출력이 필요할 때 직관적이다.  
단점: 양 끝에서 Gradient가 매우 작아져 **Vanishing Gradient(기울기 소실)**가 생기기 쉽다. 깊은 은닉층에는 잘 쓰지 않는다.

현대 LLM의 은닉층에서는 ReLU 계열·GELU·SiLU 등이 더 흔하고, Sigmoid는 게이트나 일부 확률 출력에 쓰인다. 원리는 같다: **비선형성을 넣는다**.

### 3.7 MLP (Multi-Layer Perceptron)

**MLP(Multi-Layer Perceptron, 다층 퍼셉트론)**는 전결합(Fully Connected) 은닉층을 여러 개 쌓은 기본 Neural Network이다.

“전결합”이란 이전 층의 모든 뉴런이 다음 층의 모든 뉴런과 Weight로 연결된다는 뜻이다.

가장 작은 실용 예는 다음과 같다.

```text
입력 2개 → 은닉 2개 → 출력 1개
```

이 구조를 이 책의 미니 네트워크로 반복해서 쓴다. Forward(16강), Backprop 손계산(17강), NumPy 구현(18강)이 같은 골격을 공유한다.

## 직관적으로 이해하기
공장 라인으로 비유하자.

1. **입력층**: 원재료(숫자 특징)가 들어온다.
2. **Weight/Bias**: 각 작업대에서 “얼마나 섞을지”를 정한다.
3. **Activation**: “쓸모 없는 신호는 끄고, 강한 신호만 남긴다” 같은 비선형 필터.
4. **다음 층**: 앞 공정의 반제품을 받아 더 복잡한 부품을 만든다.
5. **출력층**: 최종 제품(예측값)을 내보낸다.

학습이란, Loss가 작아지도록 각 작업대의 손잡이(Weight/Bias)를 조금씩 돌리는 일이다.  
손잡이를 어느 방향으로 돌릴지는 Gradient가 알려 주고, 그 계산법이 Backpropagation이다.

## 수학적으로 이해하기
### 5.1 한 층의 Affine 변환

입력 벡터 $\mathbf{x} \in \mathbb{R}^{n_{\mathrm{in}}}$,  

$$
Weight 행렬 $W \in \mathbb{R}^{n_{\mathrm{out}} \times n_{\mathrm{in}}}$,  
Bias 벡터 $\mathbf{b} \in \mathbb{R}^{n_{\mathrm{out}}}$에 대해

\[
\mathbf{z} = W\mathbf{x} + \mathbf{b}
\]
$$

이다. 성분으로 풀면

\[

$$
z_i = \sum_{j=1}^{n_{\mathrm{in}}} W_{ij} x_j + b_i
\]
$$

이다. $W_{ij}$는 “입력 $j$가 출력 뉴런 $i$에 미치는 가중치”이다.

### 5.2 활성화

성분별(elementwise)로 Activation을 적용한다.

\[
\mathbf{a} = \sigma(\mathbf{z})
\]

ReLU라면 $a_i = \max(0, z_i)$이다.

### 5.3 두 은닉이 있는 MLP

입력 $\mathbf{x}$, 은닉1, 은닉2, 출력이 있으면

\[

$$
\begin{aligned}
\mathbf{z}^{(1)} &= W^{(1)}\mathbf{x} + \mathbf{b}^{(1)} \\
\mathbf{a}^{(1)} &= \sigma(\mathbf{z}^{(1)}) \\
\mathbf{z}^{(2)} &= W^{(2)}\mathbf{a}^{(1)} + \mathbf{b}^{(2)} \\
\mathbf{a}^{(2)} &= \sigma(\mathbf{z}^{(2)}) \\
\mathbf{z}^{(3)} &= W^{(3)}\mathbf{a}^{(2)} + \mathbf{b}^{(3)} \\
\hat{y} &= \mathbf{z}^{(3)} \quad \text{(회귀에서는 출력 Activation을 생략하기도 함)}
\end{aligned}
\]
$$

이 책의 2-2-1 예제는 은닉 한 층만 둔다.

\[

$$
\begin{aligned}
\mathbf{z}^{(1)} &= W^{(1)}\mathbf{x} + \mathbf{b}^{(1)} \\
\mathbf{a}^{(1)} &= \mathrm{ReLU}(\mathbf{z}^{(1)}) \\
\hat{y} &= W^{(2)}\mathbf{a}^{(1)} + b^{(2)}
\end{aligned}
\]
$$

## 작은 숫자로 직접 계산하기
### 6.1 미니 네트워크 명세 (2-2-1)

이 절의 숫자는 16~18강에서도 재사용한다.

```text
입력: x = [x1, x2] = [1.0, 0.5]

은닉층 Weight W1 (2×2):
  [[0.3, -0.2],
   [0.4,  0.1]]

은닉층 Bias b1:
  [0.1, -0.1]

은닉 Activation: ReLU

출력층 Weight W2 (1×2):
  [[0.5, -0.4]]

출력층 Bias b2:
  [0.2]

출력 Activation: 없음 (회귀용 선형 출력)
```

ASCII 구조는 다음과 같다.

```text
        w1_11=0.3          w2_11=0.5
   x1 ------------→ (h1) --------------→
        \          /   \                \
         \        /     \                \
          \      /       \                +----→ ŷ
           \    /         \              /
            \  /           \            /
             \/             \          /
             /\              \        /
            /  \              \      /
           /    \              \    /
          /      \              \  /
         /        \              \/
   x2 ------------→ (h2) --------------→
        w1_21=0.4          w2_12=-0.4
               (+b1)              (+b2)
```

그림 15-1. 2-2-1 MLP의 연결 구조.

### 6.2 은닉층 계산

\[

$$
\begin{aligned}
$$
z_1 &= 0.3\cdot 1.0 + (-0.2)\cdot 0.5 + 0.1 = 0.3 - 0.1 + 0.1 = 0.3 \\
z_2 &= 0.4\cdot 1.0 + 0.1\cdot 0.5 + (-0.1) = 0.4 + 0.05 - 0.1 = 0.35
\end{aligned}
\]

ReLU:

\[

$$
\begin{aligned}
$$
a_1 &= \max(0, 0.3) = 0.3 \\
a_2 &= \max(0, 0.35) = 0.35
\end{aligned}
\]

### 6.3 출력층 계산

\[
\hat{y} = 0.5\cdot 0.3 + (-0.4)\cdot 0.35 + 0.2 = 0.15 - 0.14 + 0.2 = 0.21
\]

정답 타깃이 $y = 1.0$이고 MSE Loss를 쓴다면

\[

$$
L = \frac{1}{2}(\hat{y} - y)^2 = \frac{1}{2}(0.21 - 1.0)^2 = \frac{1}{2}(0.79)^2 = 0.31205
\]
$$

이다. (계수 $1/2$는 미분을 예쁘게 만들기 위한 관례이다.)

지금은 Loss 숫자만 확인한다. Gradient는 17강에서 전부 펼친다.

## 코드로 구현하기
구조만 코드로 적으면 다음과 같다. 학습은 아직 하지 않는다.

```python
"""15강: 2-2-1 Neural Network의 구조와 한 번 forward."""

import numpy as np

def relu(z: np.ndarray) -> np.ndarray:
    """ReLU 활성화: 음수는 0, 양수는 그대로."""
    return np.maximum(0.0, z)

def forward_2_2_1(x: np.ndarray, W1, b1, W2, b2):
    """2-2-1 MLP의 Forward Propagation.

    Parameters
    ----------
    x : shape (2,)
    W1 : shape (2, 2)  — 은닉층 weight
    b1 : shape (2,)
    W2 : shape (1, 2)  — 출력층 weight
    b2 : shape (1,)
    """
    # 은닉 Pre-activation
    z1 = W1 @ x + b1
    # 은닉 Activation
    a1 = relu(z1)
    # 출력 (선형)
    y_hat = (W2 @ a1 + b2)[0]
    return {
        "z1": z1,
        "a1": a1,
        "y_hat": y_hat,
    }

if __name__ == "__main__":
    x = np.array([1.0, 0.5])
    W1 = np.array([[0.3, -0.2], [0.4, 0.1]])
    b1 = np.array([0.1, -0.1])
    W2 = np.array([[0.5, -0.4]])
    b2 = np.array([0.2])

    out = forward_2_2_1(x, W1, b1, W2, b2)
    print("z1   =", out["z1"])      # [0.3  0.35]
    print("a1   =", out["a1"])      # [0.3  0.35]
    print("yhat =", out["y_hat"])   # 0.21
```

실행하면 손계산과 같은 값이 나와야 한다.

Sigmoid도 한 줄로 둘 수 있다.

```python
def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid: 출력을 (0, 1) 구간으로 압축."""
    return 1.0 / (1.0 + np.exp(-z))
```

은닉에 Sigmoid를 쓰면 같은 Weight라도 출력이 달라진다. 직접 바꿔 비교해 보라.

## 파라미터 개수를 세는 습관
딥러닝에서는 “모델이 얼마나 큰가”를 파라미터 수로 말한다.

2-2-1 네트워크:

| 위치 | shape | 개수 |
|---|---|---:|
| $W^{(1)}$ | $2 \times 2$ | 4 |
| $\mathbf{b}^{(1)}$ | $2$ | 2 |
| $W^{(2)}$ | $1 \times 2$ | 2 |
| $b^{(2)}$ | $1$ | 1 |
| **합계** |  | **9** |

파라미터가 9개라는 말은, 학습이 이 9개 숫자를 움직이는 문제라는 뜻이다.

LLM은 같은 원리를 수천억 개로 확장한 것이다. 구조의 원자 단위는 여전히 Weight와 Bias(또는 Bias 없는 Linear)이다.

## LLM에서는 어디에 사용될까?
Transformer 블록 안에는 **Feed-Forward Network(FFN, 피드포워드 네트워크)**가 있다.  
이름은 거창하지만, 본질은 **토큰마다 적용되는 MLP**이다.

전형적인 형태는 다음과 같다.

```text
입력 벡터 (d_model)
  → Linear (확장: d_model → 4*d_model)
    → Activation (GELU / SiLU 등)
      → Linear (축소: 4*d_model → d_model)
```

즉

- Attention이 “어떤 토큰을 볼지”를 섞고
- FFN(MLP)이 “각 토큰의 내용을 비선형으로 변환”한다

2권 제45강에서 FFN을 본격적으로 다루지만, 오늘 배운 단어만으로도 이미 뼈대가 보인다.

또한 Embedding, Attention의 Projection(Q/K/V), 출력 LM Head도 전부 **Linear Layer**이다.  
Neural Network의 문법을 알면 LLM 코드의 대부분이 “큰 행렬 곱 + 비선형”으로 읽히기 시작한다.

## 실습
### 실습 1 — ASCII 다이어그램 직접 그리기

**목표:** 구조를 손으로 고정한다.

1. 입력 3, 은닉 4, 출력 2인 MLP를 ASCII로 그린다.
2. Weight 행렬 shape를 모두 적는다.
3. 파라미터 총개수를 계산한다.

**예상:** $W1:(4,3), b1:(4,), W2:(2,4), b2:(2,)$ → $12+4+8+2=26$.

### 실습 2 — ReLU vs Sigmoid

**목표:** 활성화가 출력을 어떻게 바꾸는지 본다.

7절의 Weight를 그대로 두고

1. 은닉 Activation을 ReLU로 둔 출력
2. Sigmoid로 둔 출력

을 각각 계산한다. 어느 쪽이 더 큰지, 왜 그런지 한 문장으로 적는다.

### 실습 3 — 음수 Pre-activation 만들기

**목표:** ReLU의 “꺼짐”을 체험한다.

$x = [1.0, 0.5]$는 유지하고, $W1$의 첫 행을 모두 음수로 바꿔 $z_1 < 0$이 되게 한다.  
$a_1$이 0이 되는지, 출력 $\hat{y}$가 어떻게 바뀌는지 확인한다.

### 실습 4 — 코드 확장

앞의 `forward_2_2_1`에 `activation="relu"|"sigmoid"` 인자를 추가해 선택 가능하게 만든다.

## 자주 하는 실수
1. **Input Layer에 Weight가 있다고 착각한다**  
   입력층은 데이터를 담는 자리이다. 첫 Weight는 “입력 → 첫 은닉”을 잇는다.

2. **Weight shape를 뒤집는다**  
   $W$를 $(n_{\mathrm{in}}, n_{\mathrm{out}})$으로 두면 행렬곱이 깨진다. 이 책은 $W @ x$ 관례로 $(n_{\mathrm{out}}, n_{\mathrm{in}})$을 쓴다. 라이브러리마다 배치 차원 위치가 다를 수 있으니 문서의 shape를 항상 확인한다.

3. **Activation을 “장식”으로 본다**  
   Activation이 없으면 깊은 네트워크의 의미가 사라진다.

4. **Bias를 잊는다**  
   Bias는 작아 보여도 결정력에 중요한 자유도이다.

5. **뉴런 수 = 파라미터 수라고 생각한다**  
   파라미터는 연결(Weight)과 Bias의 총합이다. 뉴런 수보다 훨씬 많다.

6. **Sigmoid를 깊은 은닉층에 기본값처럼 쓴다**  
   역사적으로 중요하지만, 현대 깊은 모델의 은닉 기본값은 ReLU 계열이 더 흔하다.

## 핵심 요약
- Neural Network는 Weight·Bias의 선형 변환과 Activation의 비선형 변환을 층으로 쌓은 합성 함수이다.
- Neuron은 $z = w\cdot x + b$, $a = \sigma(z)$를 계산하는 최소 단위이다.
- Layer는 뉴런의 묶음이며, MLP는 전결합 층을 쌓은 기본 구조이다.
- ReLU와 Sigmoid는 대표적 Activation이며, 비선형성이 깊은 표현력의 핵심이다.
- 2-2-1 미니 네트워크는 이후 Forward/Backprop 강의의 공통 무대이다.
- LLM의 FFN·Linear Projection도 같은 부품의 확장이다.

## 용어 사전
| 용어 | 의미 |
|---|---|
| Neural Network | Weight/Bias/Activation으로 입력을 변환하는 계층적 모델 |
| Neuron | 한 단위의 가중합 + 활성화 계산기 |
| Weight | 입력 영향력을 조절하는 학습 파라미터 |
| Bias | 입력이 0이어도 출력을 이동시키는 학습 파라미터 |
| Layer | 같은 단계의 뉴런 묶음 |
| Hidden Layer | 입력과 출력 사이의 중간 표현 층 |
| Activation Function | Pre-activation을 비선형 변환하는 함수 |
| ReLU | $\max(0,z)$ |
| Sigmoid | $1/(1+e^{-z})$, 출력을 (0,1)로 압축 |
| MLP | 전결합 은닉층을 쌓은 다층 신경망 |
| Parameter (θ) | 학습으로 갱신되는 Weight·Bias 전체 |
| Pre-activation (z) | Activation 직전의 Affine 결과 |
| Activation (a) | Activation 통과 후 다음 층으로 가는 값 |

## 연습문제
### 문제 1 (개념)

Activation Function이 없으면 층을 쌓아도 안 되는 이유를 수식 한 줄과 문장 한 줄로 설명하시오.

### 문제 2 (구조)

입력 5, 은닉 8, 출력 3인 MLP의 파라미터 수를 구하시오. (Bias 포함)

### 문제 3 (계산)

$z = -2$일 때 ReLU와 Sigmoid 값을 각각 구하시오. Sigmoid는 소수 셋째 자리까지 근사해도 된다.

### 문제 4 (계산)

7절의 네트워크에서 $x = [0.0, 1.0]$일 때 $\hat{y}$를 구하시오.

### 문제 5 (연결)

Transformer FFN이 “토큰마다 적용되는 MLP”라는 말이 뜻하는 바를, Attention과의 역할 차이로 한 문장씩 쓰시오.

---

## 정답 및 해설
### 문제 1

선형 변환의 합성은 다시 선형 변환이다.  
$W_2(W_1 x + b_1)+b_2 = W'x + b'$이므로, 비선형성 없이는 깊은 층의 추가 표현력이 사라진다.

### 문제 2

$W1: 8\times5=40$, $b1:8$, $W2:3\times8=24$, $b2:3$ → 총 **75**.

### 문제 3

- ReLU$(-2)=0$
- Sigmoid$(-2)=1/(1+e^{2}) \approx 0.119$

### 문제 4

\[

$$
\begin{aligned}
$$
z_1 &= 0.3\cdot0 + (-0.2)\cdot1 + 0.1 = -0.1 \\
z_2 &= 0.4\cdot0 + 0.1\cdot1 - 0.1 = 0.0 \\
a_1 &= 0,\quad a_2 = 0 \\
\hat{y} &= 0.5\cdot0 + (-0.4)\cdot0 + 0.2 = 0.2
\end{aligned}
\]

### 문제 5

- Attention: 토큰 사이에서 정보를 섞는다(어떤 위치를 참고할지).
- FFN(MLP): 각 토큰 벡터를 위치마다 독립적으로 비선형 변환한다(내용을 가공한다).

## 다음 강의와 연결
이번 강의에서 Neural Network의 **부품과 배치도**를 그렸다.

다음 **제16강. Forward Propagation**에서는, 오늘 만든 2-2-1 네트워크에 숫자를 넣어 **앞에서 뒤로 출력을 계산하는 과정**을 손과 NumPy로 완전히 고정한다.  
중간 값 $z$, $a$를 저장해야 하는 이유도 그때 드러난다. 그 저장이 바로 17강 Backpropagation의 출발점이다.

> 뼈대가 생겼다. 이제 신호가 앞에서 뒤로 흐르게 하자.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [14강. Chain Rule](14강_Chain_Rule.md)
- **다음 강:** [16강. Forward Propagation](16강_Forward_Propagation.md)

<!-- /LECTURE_NAV -->
