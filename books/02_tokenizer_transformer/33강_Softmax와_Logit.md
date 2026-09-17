# 33강. Softmax와 Logit
## 이번 강에서 배우는 내용

- Logit과 Softmax의 정의·동기
- Softmax가 만족하는 확률 성질 ($>0$, 합=1)
- 벡터 `[2, 1, 0]`으로 Softmax를 손계산하기
- Temperature가 분포의 뾰족함을 어떻게 바꾸는지 (미리보기)
- 큰 숫자에서 생기는 수치 불안정과 안전한 계산법
- LLM 생성·학습에서 Softmax가 어디에 붙는지

## 왜 중요한가?
LLM의 매 forward 끝은 대략 이렇게 생긴다.

```text
숨은 상태 h_t ∈ R^d
  → Linear:  W h_t + b  ∈ R^{|V|}
    → Softmax → 확률 ∈ R^{|V|}
```

이 확률로:

- 학습: 정답 토큰의 $-\log p$를 줄인다 (34강)
- 추론: argmax / multinomial / top-p 등으로 다음 id를 고른다

Softmax를 “라이브러리 한 줄”로만 알면, Temperature·수치 오버플로·LogSoftmax 같은 실전 이슈에서 길을 잃는다.  
오늘은 공식을 **손으로 한 바퀴** 돌린다.

## 선수 개념
- 지수함수 $e^x$의 성질: 항상 양수, $x$가 커지면 빠르게 증가
- 벡터 / 합 / 나눗셈
- Vocabulary size $|V|$ (30강)
- Next Token Prediction (32강)
- 확률 분포: 각 항목 ≥ 0, 총합 1

미적분의 깊은 내용은 필수가 아니다. 다만 “매끄럽게 커지는 양수 함수”로서 $e^x$만 있으면 충분하다.

## 핵심 개념
### 3.1 Logit (로짓)

**Logit(로짓)**은 Softmax(또는 시그모이드)에 들어가기 **전의 원시 점수(raw score)**이다.

LLM 맥락에서 보통:

$$

\mathbf{z} = W \mathbf{h} + \mathbf{b} \in \mathbb{R}^{|V|}

$$

- $z_i$: 토큰 $i$에 대한 점수
- 범위: 실수 전체 (음수·양수 모두 가능)
- 아직 확률이 아님. 합이 1일 필요도 없음

왜 필요한가?

1. 신경망이 제약 없는 실수 공간을 출력하기 쉽다.
2. Softmax가 그 점수를 확률 simplex로 투영한다.
3. 점수 차이가 클수록 승자 토큰의 확률이 커진다.

용어 주의: 통계학에서 logit은 $\log\frac{p}{1-p}$를 뜻하기도 한다.  
딥러닝 실무·LLM 코드에서는 **Softmax 직전의 벡터**를 logit이라 부르는 경우가 많다. 이 강의도 후자를 쓴다.

### 3.2 Softmax (소프트맥스)

**Softmax(소프트맥스)**는 실수 벡터를 확률 분포로 바꾸는 함수이다.

성분 $i$에 대해:

$$

\operatorname{softmax}(\mathbf{z})_i
=
\frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}

$$

- $K$: 클래스 수 (= vocab size in LM)
- $e^{z_i}$: 점수를 양의 무게로 변환
- 분모: 모든 무게의 합 (정규화)

성질:

1. $\operatorname{softmax}(\mathbf{z})_i > 0$ for all $i$
2. $\sum_i \operatorname{softmax}(\mathbf{z})_i = 1$
3. $z_i$가 상대적으로 클수록 해당 확률이 커진다
4. Softmax는 **상대값**에만 의존한다. (아래 shift 불변성)

### 3.3 Softmax의 Shift Invariance

임의의 상수 $c$에 대해:

$$

\operatorname{softmax}(\mathbf{z} - c\mathbf{1})
=
\operatorname{softmax}(\mathbf{z})

$$

이유: $e^{z_i - c} = e^{z_i}e^{-c}$이므로 분자·분모에서 $e^{-c}$가 약분된다.

실무 함의:

- logit 전체에 같은 수를 더해도 확률 동일
- 수치 안정화를 위해 $\mathbf{z} \leftarrow \mathbf{z} - \max(\mathbf{z})$를 자주 쓴다

### 3.4 Argmax와의 관계

탐욕 디코딩:

$$

\hat{y} = \arg\max_i z_i
=
\arg\max_i \operatorname{softmax}(\mathbf{z})_i

$$

Softmax는 **순서를 보존**한다. (단조 변환)  
따라서 “가장 큰 logit”과 “가장 큰 확률”의 인자는 같다.  
다만 **확률의 격차**는 Softmax·Temperature가 결정한다.

### 3.5 Temperature (온도) 미리보기

**Temperature(템퍼러처, 온도)** $T > 0$는 logit을 $T$로 나눈 뒤 Softmax하는 조절 장치이다.

$$

p_i
=
\frac{e^{z_i / T}}{\sum_j e^{z_j / T}}

$$

직관:

| $T$ | 효과 |
|---|---|
| $T \to 0^+$ | 분포가 뾰족 (거의 argmax) |
| $T = 1$ | 기본 Softmax |
| $T > 1$ | 분포가 평평 (더 랜덤·다양) |

생성 품질·다양성 레버로 쓰인다. 학습 Loss에서는 보통 $T=1$ 기준이다.

## 직관적으로 이해하기
경연 점수 비유:

1. 심사위원이 각 후보에게 실수 점수(logit)를 준다.
2. Softmax는 점수를 “지지도”로 바꿔 **비율**로 만든다.
3. 점수가 조금 높아도 지수 때문에 지지도가 크게 벌어질 수 있다.

또 다른 비유: Softmax는 **부드러운 승자독식**이다.  
하드맥스(argmax one-hot)는 1등만 1이고 나머지는 0이다.  
Softmax는 1등이 가장 크지만, 2등·3등에도 양의 확률을 남긴다. 그래서 미분이 가능하고 학습이 된다.

## 수학적으로 이해하기
벡터 $\mathbf{z}\in\mathbb{R}^{K}$

$$

p_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}, \quad
\mathbf{p} = \operatorname{softmax}(\mathbf{z})

$$

Jacobian의 성분 (참고):

$$

\frac{\partial p_i}{\partial z_k}
=
p_i(\delta_{ik} - p_k)

$$

직관: 자기 점수 $z_i$를 올리면 $p_i$는 늘고, 다른 확률은 상대적으로 줄어든다.

Log-Softmax:

$$

\log p_i
=
z_i - \log\sum_j e^{z_j}

$$

Cross Entropy 구현은 종종 `log_softmax` + NLL 조합을 쓴다. (34강, 수치적으로 더 안정)

## 작은 숫자로 직접 계산하기
문제: $\mathbf{z} = [2, 1, 0]$

### 6.1 지수 무게

$$

e^{2} \approx 7.389

$$

$$

e^{1} \approx 2.718

$$

$$

e^{0} = 1

$$

합:

$$

S = 7.389 + 2.718 + 1 = 11.107

$$

### 6.2 Softmax 확률

$$

p_0 = 7.389 / 11.107 \approx 0.665

$$

$$

p_1 = 2.718 / 11.107 \approx 0.245

$$

$$

p_2 = 1 / 11.107 \approx 0.090

$$

검산: $0.665+0.245+0.090=1.000$

해석:

- 가장 큰 logit 2 → 약 66.5%
- 다음 1 → 약 24.5%
- 0 → 약 9.0%

하드맥스는 `[1,0,0]`이었을 것이다. Softmax는 남은 확률을 나눠 준다.

### 6.3 Temperature $T=2$

$$

\mathbf{z}/T = [1,\ 0.5,\ 0]

$$

$$

e^{1}\approx 2.718,\quad
e^{0.5}\approx 1.649,\quad
e^{0}=1

$$

합 $\approx 5.367$

$$

p \approx [0.507,\ 0.307,\ 0.186]

$$

분포가 더 평평해졌다. (1등의 독식이 완화)

### 6.4 Temperature $T=0.5$

$$

\mathbf{z}/T = [4,\ 2,\ 0]

$$

$$

e^{4}\approx 54.60,\quad
e^{2}\approx 7.389,\quad
e^{0}=1

$$

합 $\approx 62.99$

$$

p \approx [0.867,\ 0.117,\ 0.016]

$$

1등이 더 뾰족해졌다.

### 6.5 수치 안정화 shift

$\mathbf{z}=[2,1,0]$에서 $\max=2$를 빼면 $\mathbf{z}'=[0,-1,-2]$

$$

e^{0}=1,\quad e^{-1}\approx0.368,\quad e^{-2}\approx0.135

$$

합 $\approx 1.503$

$$

p \approx [0.665,\ 0.245,\ 0.090]

$$

동일하다. 큰 logit이 있어도 $e^{z-\max}$는 최대 1이므로 overflow를 완화한다.

## 코드로 구현하기 — 손계산 재현
```python
# softmax_scratch.py
import math
from typing import List

def softmax(zs: List[float]) -> List[float]:
    # 안정화: max subtract
    m = max(zs)
    exps = [math.exp(z - m) for z in zs]
    s = sum(exps)
    return [e / s for e in exps]

def softmax_with_temperature(zs: List[float], T: float) -> List[float]:
    scaled = [z / T for z in zs]
    return softmax(scaled)

if __name__ == "__main__":
    z = [2.0, 1.0, 0.0]
    print("T=1.0", [round(p, 3) for p in softmax(z)])
    print("T=2.0", [round(p, 3) for p in softmax_with_temperature(z, 2.0)])
    print("T=0.5", [round(p, 3) for p in softmax_with_temperature(z, 0.5)])
```

기대 출력 (반올림):

```text
T=1.0 [0.665, 0.245, 0.090]
T=2.0 [0.507, 0.307, 0.186]
T=0.5 [0.867, 0.117, 0.016]
```

## NumPy / PyTorch로 구현하기
```python
# softmax_torch.py
import numpy as np
import torch
import torch.nn.functional as F

def softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

z = np.array([2.0, 1.0, 0.0])
print("numpy:", softmax_np(z))

zt = torch.tensor([2.0, 1.0, 0.0])
print("torch:", F.softmax(zt, dim=-1))

# 배치 × vocab
logits = torch.tensor(
    [
        [2.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],  # 균등
    ]
)
print(F.softmax(logits, dim=-1))
```

균등 logit `[0,0,0]`의 Softmax는 `[1/3, 1/3, 1/3]`이다.  
모델이 “모르겠다”면 logit이 비슷해지고 분포가 평평해질 수 있다.

## LLM 연결 — 한 위치의 Next Token
인과 LM의 한 스텝:

```text
input_ids → Embedding → Transformer → hidden[t]
hidden[t] ∈ R^d
logits[t] = hidden[t] @ W.T (+b)  ∈ R^{|V|}
probs[t]  = softmax(logits[t])
```

학습:

```text
loss_t = -log probs[t][label_t]
```

추론:

```text
next_id = argmax(probs)  or  sample(probs)
```

주의:

1. 학습 코드는 종종 `F.cross_entropy(logits, labels)`를 호출하며, 내부에서 log-softmax를 한다.  
   **수동 Softmax 후 CrossEntropy를 또 적용하면 이중 Softmax**가 된다. (자주 하는 실수)
2. 생성 시 Temperature는 Softmax 앞에 적용한다.
3. Top-k / Top-p는 Softmax 전·후 분포를 잘라 재정규화하는 디코딩 기법이다. (후반 강의)

## 수치 안정성 (Numerical Stability)
### 10.1 Overflow

$z_i = 1000$이면 $e^{1000}$은 float32에서 `inf`가 될 수 있다.  
분모·분자가 모두 `inf`면 `nan`이 된다.

해결: max subtraction.

### 10.2 Underflow

매우 작은 확률의 $\log p$를 직접 계산하면 $-inf$에 가깝다.  
`log_softmax`는

$$

\log p_i = (z_i - \max z) - \log\sum_j e^{z_j-\max z}

$$

형태로 한 번에 계산한다.

```python
import torch.nn.functional as F

logits = torch.tensor([1000.0, 999.0, 998.0])
# 위험: F.softmax(logits).log()
# 안전:
print(F.log_softmax(logits, dim=-1))
```

### 10.3 Mixed Precision

bf16/fp16 학습에서 Softmax는 종종 fp32로 올려 계산한다.  
이유: 지수·합산의 동적 범위.

## Softmax vs Sigmoid vs Hardmax
| 함수 | 출력 | 쓰임 |
|---|---|---|
| Softmax | 다중 클래스 확률 (합1) | LM next-token, 다중 분류 |
| Sigmoid | 각 성분을 독립적으로 (0,1) | 다중 라벨, 게이트 |
| Hardmax / Argmax one-hot | 이산 선택 | 추론 선택, 미분 불가 |

LLM의 vocab 분포는 **상호 배타적 다음 토큰**이므로 Softmax가 맞다.  
(동시에 두 토큰이 “정답”이 아님 — 한 위치에 토큰 하나)

## Attention Softmax와의 관계 (예고)
Self-Attention에서도 Softmax가 등장한다.

$$

\alpha = \operatorname{softmax}\Big(\frac{QK^{\top}}{\sqrt{d}}\Big)

$$

여기는 “vocab 확률”이 아니라 **과거 위치에 대한 가중치**이다.  
수학 객체는 같고, 의미가 다르다.

- LM head Softmax: 다음 토큰 분포
- Attention Softmax: 문맥 혼합 비율

이름을 공유한다고 해서 같은 자리에 있는 것은 아니다. (38강 등)

## 실습
### 실습 1. 손계산

$\mathbf{z}=[3, 0, -2]$의 Softmax를 소수점 셋째 자리까지 계산하시오.  
max-shift로도 같은지 확인하시오.

### 실습 2. Temperature 곡선

`[2,1,0]`에 대해 $T \in \{0.2, 0.5, 1, 2, 5\}$의 $p_0$(1등 확률)를 표로 그리시오.

### 실습 3. 이중 Softmax 함정

PyTorch에서 `F.softmax(logits, dim=-1)` 결과에 `F.cross_entropy`를 적용하면 왜 잘못된지 한 단락으로 설명하고, 올바른 한 줄 호출을 쓰시오.

### 실습 4. 균등 분포

모든 logit이 같으면 확률이 $1/K$임을 수식으로 보이시오.

### 실습 5. (도전) Softmax Jacobian

$K=2$, $\mathbf{z}=[a,b]$일 때 $p_0=(1+e^{b-a})^{-1}$임을 유도하시오. (로지스틱과의 연결)

## 자주 하는 실수
1. **Logit을 이미 확률이라고 착각**  
   합이 1이 아닐 수 있다. Softmax 전이다.

2. **`cross_entropy`에 확률을 넣음**  
   보통 logit을 넣어야 한다. (`BCEWithLogits`도 같은 철학)

3. **Temperature를 Softmax 뒤에 곱함**  
   $p/T$는 확률이 깨진다. logit을 $T$로 나눠야 한다.

4. **큰 logit에서 `exp` 직접 호출**  
   overflow → nan. max subtract / log_softmax 사용.

5. **Attention Softmax와 LM Softmax를 혼동**  
   축(dim)과 의미가 다르다.

6. **확률 0을 기대**  
   Softmax는 이론상 모든 토큰에 양수. 극단적으로 작을 수는 있어도 정확히 0은 아니다. (실수 연산상 0.0이 될 수는 있음)

## LLM에서는 어디에 사용될까?

이번 33강에서 배운 개념은 이후 Transformer · GPT · 서빙 강의에서 반복해서 등장합니다. 각 수식·코드 블록을 “실제 모델의 어느 단계인가”와 연결해 다시 읽어 보세요.

## 핵심 요약
- Logit은 Softmax 전의 실수 점수 벡터이다.
- Softmax는 $e^{z_i}/\sum e^{z_j}$로 확률 분포를 만든다.
- `[2,1,0]` → 약 `[0.665, 0.245, 0.090]`
- Softmax는 상수 shift에 불변; 수치 안정화에 활용한다.
- Temperature는 $z/T$로 분포의 뾰족함을 조절한다.
- LLM에서는 매 시점 vocab logit에 Softmax(또는 log-softmax+CE)가 붙는다.
- 다음 강의 Cross Entropy는 $-\log p_{\text{정답}}$으로 학습 신호를 만든다.

## 용어 사전
| 용어 | 의미 |
|---|---|
| Logit | Softmax 전 원시 점수 |
| Softmax | 점수를 확률 분포로 정규화 |
| Temperature $T$ | logit 스케일 조절 파라미터 |
| Log-Softmax | $\log$와 Softmax의 안정적 결합 |
| Numerical Stability | overflow/underflow를 피하는 계산 |
| Softmax dim | 정규화를 수행하는 축 |
| Hardmax / Argmax | 최대 항목만 선택하는 이산 연산 |
| Probability Simplex | 합1·비음수 벡터의 집합 |

## 연습문제
### 문제 1 (손계산)

$\mathbf{z}=[2,1,0]$의 Softmax를 소수점 셋째 자리까지 쓰시오.

### 문제 2 (개념)

왜 Softmax 결과에 음수가 나올 수 없는가?

### 문제 3 (Temperature)

$T$를 키우면 분포가 어떻게 변하는가? 한 문장.

### 문제 4 (안정성)

$z=[1000, 999]$에서 max subtract를 적용한 뒤의 벡터는?

### 문제 5 (코드)

PyTorch에서 배치 logits `[B, V]`에 Softmax를 적용할 때 `dim`은 보통 무엇인가?

### 문제 6 (LLM 연결)

탐욕 디코딩이 Softmax 없이도 가능한 이유와, 그럼에도 Softmax가 필요한 상황을 구분하시오.

### 문제 7 (함정)

`probs = softmax(logits); loss = cross_entropy(probs, y)`가 잘못된 이유를 쓰시오.

---

## 정답 및 해설
### 문제 1

약 `[0.665, 0.245, 0.090]`

### 문제 2

분자 $e^{z_i}$가 항상 양수이고 분모(양수들의 합)도 양수이므로 비율이 양수이다.

### 문제 3

분포가 더 평평해져(엔트로피↑) 선택 다양성이 커진다.

### 문제 4

`[0, -1]`

### 문제 5

`dim=-1` (마지막 축, vocab 축)

### 문제 6

argmax는 logit과 Softmax에서 동일하므로 탐욕만이면 Softmax 불필요.  
샘플링·확률 임계값·Loss($-\log p$)·불확실도 측정에는 Softmax(또는 log-softmax)가 필요하다.

### 문제 7

`cross_entropy`는 내부에서 log-softmax를 가정/수행하므로, 이미 Softmax된 확률에 다시 적용하면 이중 변환이 되어 Loss·gradient가 틀어진다. logit을 직접 넘겨야 한다.

## 다음 강의와 연결
이번 강의에서 점수가 확률이 되는 문을 열었다.

다음 **제34강. Cross Entropy Loss**에서는, Softmax 확률과 정답 토큰을 묶어

$$

L = -\log p_{y}

$$

로 학습 목표를 수치화한다. 32강의 Next Token Prediction이 비로소 **미분 가능한 Loss**가 된다.

이전 강의: **제32강. Language Model과 Next Token Prediction**  
다음 강의: **제34강. Cross Entropy Loss**

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [32강. Language Model과 Next Token Prediction](32강_Language_Model과_Next_Token_Prediction.md)
- **다음 강:** [34강. Cross Entropy Loss](34강_Cross_Entropy_Loss.md)

<!-- /LECTURE_NAV -->
