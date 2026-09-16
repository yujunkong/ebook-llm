# 2권. Tokenizer와 Transformer

## 제34강. Cross Entropy Loss

### 1. 이번 강의에서 배울 것

33강에서 **Logit(로짓)**을 **Softmax**로 확률로 바꾸는 법을 배웠다. 이번 강의는 그 확률을 **학습 신호**로 바꾸는 단계다.

이 강의를 마치면 다음을 말할 수 있어야 한다.

- **Cross Entropy Loss(교차 엔트로피 손실)**가 무엇인지, 왜 \(-\log p_{\text{correct}}\)인지
- Logits → Softmax → Cross Entropy의 전체 파이프라인
- 작은 vocabulary로 손계산하기
- Next-Token Prediction에서 CE가 어떻게 쓰이는지
- NumPy와 `torch.nn.functional.cross_entropy`로 같은 값을 재현하기

LLM 학습의 기본 나침반은 거의 항상 이 Loss다. Softmax만 알면 “확률”까지고, CE까지 알면 “학습”까지다.

### 2. 왜 이것을 배우는가

Language Model은 매 위치에서 vocabulary 크기 \(V\)개의 점수를 낸다. Softmax로 확률 \(p\)를 만든 뒤, **정답 토큰**에 부여한 확률이 높을수록 좋다.

그런데 “좋다/나쁘다”를 파라미터 업데이트에 쓰려면 **미분 가능한 숫자 하나**가 필요하다. 그 숫자가 Cross Entropy다.

```text
문맥 (이전 토큰들)
  → Transformer (+ Embedding 등)
    → Logits (길이 V)
      → Softmax → 확률 p
        → Cross Entropy: ℓ = -log p[정답]
          → Gradient → 파라미터 업데이트
```

1권 13강에서 CE의 **직관**을 미리 보았다. 이번 강의는 Softmax(33강) 직후에 붙여, **logits에서부터 Loss까지를 끝까지** 계산한다.

### 3. 먼저 알아야 할 개념

- Softmax와 Logit (33강)
- Loss Function 일반 개념 (1권 13강)
- Next-Token Prediction (32강)
- 자연로그 \(\ln\), \(e^x\)
- one-hot 벡터 (정답 위치에만 1)

아직 필요 없는 것:

- Attention 내부 구조 (35강 이후)
- Label smoothing, KL divergence의 깊은 이론

### 4. 핵심 개념 설명

#### 4.1 Cross Entropy Loss란?

**Cross Entropy(교차 엔트로피)**는 “정답이 가리키는 분포”와 “모델이 예측한 확률 분포”가 얼마나 다른지를 재는 척도다. 분류·토큰 예측에서 **Loss Function**으로 쓸 때 **Cross Entropy Loss**라고 부른다.

멀티클래스에서 정답이 one-hot \(\mathbf{y}\), 모델 확률이 \(\mathbf{p}\)이면

$$
\ell = -\sum_{k=1}^{V} y_k \log p_k
$$

one-hot이면 정답 인덱스 \(c\)만 남는다.

$$
\ell = -\log p_c
$$

기호:

- \(V\): vocabulary 크기 (클래스 수)
- \(y_k\): 정답 one-hot의 \(k\)번째 성분 (정답이면 1, 아니면 0)
- \(p_k\): 모델이 클래스 \(k\)에 부여한 확률 (\(0 < p_k \leq 1\), \(\sum_k p_k = 1\))
- \(c\): 정답 클래스(토큰) 인덱스
- \(\log\): 보통 자연로그 \(\ln\) (밑이 \(e\))
- \(\ell\): 한 샘플(또는 한 토큰 위치)의 손실

왜 필요한가?

1. Softmax 확률을 **최소화할 목표**로 바꾼다.
2. 정답 확률이 낮을수록 크게 벌점한다.
3. Softmax와 결합하면 Gradient가 안정적이고 해석이 쉽다.

#### 4.2 Negative Log Likelihood와의 관계

**NLL(Negative Log Likelihood, 음의 로그 우도)**는 정답의 로그 확률에 마이너스를 붙인 것이다.

$$
\mathrm{NLL} = -\log p_c
$$

one-hot 정답에 대한 Cross Entropy는 NLL과 **같다**.  
문헌·라이브러리에서 “CE”, “NLL”, “log loss”가 같은 식을 가리키는 경우가 많다. 혼동하지 말고 **식이 \(-\log p_c\)인지**를 보면 된다.

#### 4.3 Logits에서 시작하는 전체 식

모델이 내는 것은 보통 확률이 아니라 **logits** \(\mathbf{z} = (z_1,\ldots,z_V)\)다.

$$
p_k = \mathrm{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^{V} e^{z_j}}
$$

따라서

$$
\ell = -\log p_c = -\log\left(\frac{e^{z_c}}{\sum_j e^{z_j}}\right) = -z_c + \log\sum_j e^{z_j}
$$

이 형태는 **numerically stable**한 구현의 출발점이다. Softmax를 따로 구한 뒤 \(\log\)를 취하면 \(p_c\)가 아주 작을 때 언더플로가 나기 쉽다.

#### 4.4 배치·시퀀스에서의 평균

LLM 학습에서는 보통 배치 \(B\), 시퀀스 길이 \(T\)에 대해 위치마다 CE를 구한 뒤 **평균**한다.

$$
L = \frac{1}{N}\sum_{(b,t)\in\text{유효 위치}} \ell_{b,t}
$$

- Padding 위치는 mask로 제외하는 경우가 많다.
- \(N\)은 유효 토큰 개수다.

### 5. 직관적으로 이해하기

#### 5.1 왜 \(-\log\)인가?

확률이 1에 가까우면 Loss는 0에 가깝다. 확률이 0에 가까우면 Loss는 커진다.

| \(p_c\) | \(-\ln p_c\) | 감각 |
|---|---|---|
| 1.0 | 0 | 완벽 |
| 0.5 | \(\approx 0.693\) | 반반 |
| 0.1 | \(\approx 2.303\) | 많이 틀림 |
| 0.01 | \(\approx 4.605\) | 심하게 틀림 |
| 0.001 | \(\approx 6.907\) | 거의 못 맞춤 |

\(\log\)는 작은 확률을 **크게 벌점**한다. “정답을 거의 안 찍었는데 다른 곳에 확신을 쏟은” 상황을 강하게 고친다.

앞의 마이너스는 “클수록 좋은 확률”을 “작을수록 좋은 손실”로 뒤집는다. Gradient Descent는 Loss를 **최소화**하기 때문이다.

#### 5.2 “틀린 확신”이 비싼 이유

모델이 정답에 0.01, 오답에 0.9를 주면 Loss가 크다.  
학습은 정답 logit을 올리고 오답 logit을 내리는 쪽으로 민다. Softmax+CE의 Gradient는 대략

$$
\frac{\partial \ell}{\partial z_k} = p_k - y_k
$$

로 요약된다(유도는 심화에서). 즉 Softmax 확률과 one-hot의 **차이**가 Gradient다.

#### 5.3 Next-Token과의 연결

문장 “나는 밥을” 다음 정답이 “먹었다”라면, 모델은 vocabulary 전체에 확률을 뿌린다.  
그중 “먹었다”의 확률이 \(p_c\)이고 Loss는 \(-\log p_c\)다.

한 문장의 학습 신호는 보통 **각 위치의 CE를 평균**한 것이다.

### 6. 수학적으로 이해하기

#### 6.1 Softmax + CE의 한 줄 정의

$$
\ell(\mathbf{z}, c) = -\log \mathrm{softmax}(\mathbf{z})_c = -z_c + \log\sum_{j=1}^{V} e^{z_j}
$$

#### 6.2 왜 Softmax와 잘 맞는가 (Gradient 스케치)

\(\ell = -\log p_c\), \(p_k = e^{z_k}/\sum_j e^{z_j}\)일 때

$$
\frac{\partial \ell}{\partial z_k} = p_k - \mathbf{1}[k=c]
$$

- 정답 위치: \(p_c - 1\) (확률이 1보다 작으면 음수 → logit을 올림)
- 오답 위치: \(p_k\) (양수 → logit을 내림)

해석이 깔끔하다. “예측 확률을 one-hot에 가깝게”가 곧 Gradient다.

#### 6.3 Log-Sum-Exp 안정화

큰 logit이 있으면 \(e^{z}\)가 폭발한다. 관례적으로

$$
\log\sum_j e^{z_j} = m + \log\sum_j e^{z_j - m}, \quad m=\max_j z_j
$$

를 쓴다. Softmax도 같은 트릭을 쓴다(33강).

### 7. 작은 숫자로 직접 계산하기

Vocabulary를 아주 작게 잡는다. \(V=4\), 토큰 이름:

| 인덱스 | 토큰 |
|---|---|
| 0 | `<pad>` |
| 1 | 나 |
| 2 | 밥 |
| 3 | 먹다 |

#### 7.1 예제 A — Softmax부터 CE까지

어떤 위치의 logits:

$$
\mathbf{z} = [1.0,\ 2.0,\ 0.5,\ 3.0]
$$

정답 토큰이 **먹다** (\(c=3\))라고 하자.

**Step 1. Softmax 분모**

$$
\begin{aligned}
e^{1.0} &\approx 2.718\
e^{2.0} &\approx 7.389\
e^{0.5} &\approx 1.649\
e^{3.0} &\approx 20.086\
\sum &\approx 31.842
\end{aligned}
$$

**Step 2. 확률**

$$
\begin{aligned}
p_0 &\approx 2.718/31.842 \approx 0.0854\
p_1 &\approx 7.389/31.842 \approx 0.2321\
p_2 &\approx 1.649/31.842 \approx 0.0518\
p_3 &\approx 20.086/31.842 \approx 0.6308
\end{aligned}
$$

합 \(\approx 1.000\).

**Step 3. Cross Entropy**

$$
\ell = -\ln p_3 \approx -\ln(0.6308) \approx 0.461
$$

정답에 이미 63%를 주고 있으므로 Loss는 비교적 작다.

#### 7.2 예제 B — 같은 logits, 다른 정답

정답이 **밥** (\(c=2\))이면

$$
\ell = -\ln(0.0518) \approx 2.961
$$

같은 예측이라도 정답이 바뀌면 Loss가 크게 달라진다. Loss는 “모델이 얼마나 자신 있는가”가 아니라 **정답에 얼마나 확률을 주었는가**를 본다.

#### 7.3 예제 C — 확신이 틀린 경우

$$
\mathbf{z}' = [0.0,\ 5.0,\ 0.0,\ 0.1], \quad c=3\ (\text{먹다})
$$

대략 \(p_1\)이 압도적으로 크고 \(p_3\)는 매우 작다.

대략 계산:

$$
\begin{aligned}
e^{0}&=1,\quad e^{5}\approx 148.41,\quad e^{0.1}\approx 1.105\
\sum &\approx 1+148.41+1+1.105 = 151.515\
p_3 &\approx 1.105/151.515 \approx 0.00729\
\ell &\approx -\ln(0.00729) \approx 4.92
\end{aligned}
$$

틀린 토큰에 확신이 쏠리면 Loss가 커진다.

#### 7.4 예제 D — 시퀀스 평균 (Next-Token)

세 위치의 \(p_c\)가 \(0.5,\ 0.8,\ 0.2\)라고 하자.

$$
\begin{aligned}
\ell_1 &= -\ln 0.5 \approx 0.693\
\ell_2 &= -\ln 0.8 \approx 0.223\
\ell_3 &= -\ln 0.2 \approx 1.609\
L &= (0.693+0.223+1.609)/3 \approx 0.842
\end{aligned}
$$

LM training loss는 이런 평균이다.

#### 7.5 예제 E — log-sum-exp로 같은 값 검증

예제 A에서 \(c=3\), \(z_3=3.0\), \(\sum e^{z_j}\approx 31.842\)이므로

$$
\ell = -3.0 + \ln(31.842) \approx -3.0 + 3.461 = 0.461
$$

\(-\ln p_3\)와 일치한다.

### 8. 코드로 구현하기 (NumPy)

```python
# lecture34_cross_entropy_numpy.py
# Softmax → Cross Entropy를 밑바닥에서 계산한다.

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Logits → 확률. 축=-1 기준으로 안정화 Softmax."""
    # 최댓값을 빼 오버플로를 막는다 (33강).
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def cross_entropy_from_probs(probs: np.ndarray, target: int) -> float:
    """이미 Softmax된 확률에서 CE = -log p[target]."""
    # 아주 작은 값 clip으로 log(0) 방지.
    p = np.clip(probs[target], 1e-12, 1.0)
    return float(-np.log(p))


def cross_entropy_from_logits(logits: np.ndarray, target: int) -> float:
    """Logits에서 바로 CE (권장 경로)."""
    # ℓ = -z_c + logsumexp(z)
    m = np.max(logits)
    logsumexp = m + np.log(np.sum(np.exp(logits - m)))
    return float(-logits[target] + logsumexp)


def sequence_mean_ce(logits_btv: np.ndarray, targets_bt: np.ndarray) -> float:
    """
    logits: (T, V) 또는 (B, T, V) — 여기선 (T, V) 예제
    targets: (T,) 정답 토큰 ID
    """
    losses = [
        cross_entropy_from_logits(logits_btv[t], int(targets_bt[t]))
        for t in range(logits_btv.shape[0])
    ]
    return float(np.mean(losses))


if __name__ == "__main__":
    z = np.array([1.0, 2.0, 0.5, 3.0], dtype=np.float64)
    p = softmax(z)
    print("probs:", np.round(p, 4))
    print("CE (c=3) from probs:", round(cross_entropy_from_probs(p, 3), 4))
    print("CE (c=3) from logits:", round(cross_entropy_from_logits(z, 3), 4))
    print("CE (c=2) from logits:", round(cross_entropy_from_logits(z, 2), 4))

    # 시퀀스 예: 길이 3, V=4
    logits_seq = np.array(
        [
            [1.0, 2.0, 0.5, 3.0],
            [0.0, 0.0, 0.0, 0.0],  # 균등 → p=0.25, CE=ln4
            [0.0, 5.0, 0.0, 0.1],
        ],
        dtype=np.float64,
    )
    targets = np.array([3, 1, 3])
    print("seq mean CE:", round(sequence_mean_ce(logits_seq, targets), 4))
```

실행하면 손계산(약 0.461)과 맞는지 확인할 수 있다.

### 9. PyTorch로 구현하기

```python
# lecture34_cross_entropy_torch.py
# F.cross_entropy는 logits를 받고 내부에서 log-softmax + NLL을 한다.

import torch
import torch.nn.functional as F


def demo_single():
    # shape: (N, C) — N개 샘플, C개 클래스
    logits = torch.tensor([[1.0, 2.0, 0.5, 3.0]])
    target = torch.tensor([3])  # 클래스 인덱스
    loss = F.cross_entropy(logits, target)  # 평균 (여기선 샘플 1개)
    print("F.cross_entropy:", float(loss))


def demo_sequence_lm():
    # LLM 스타일: (B, T, V) → (B*T, V), target (B*T,)
    B, T, V = 1, 3, 4
    logits = torch.tensor(
        [
            [
                [1.0, 2.0, 0.5, 3.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.1],
            ]
        ]
    )  # (1, 3, 4)
    targets = torch.tensor([[3, 1, 3]])  # (1, 3)

    loss = F.cross_entropy(
        logits.view(B * T, V),
        targets.view(B * T),
    )
    print("LM-style mean CE:", float(loss))


def demo_ignore_index():
    # 패딩 위치는 ignore_index로 Loss에서 제외
    logits = torch.tensor([[1.0, 2.0, 0.5, 3.0], [9.0, 0.0, 0.0, 0.0]])
    targets = torch.tensor([3, 0])  # 두 번째가 pad라고 가정
    loss = F.cross_entropy(logits, targets, ignore_index=0)
    print("with ignore_index=0:", float(loss))


if __name__ == "__main__":
    demo_single()
    demo_sequence_lm()
    demo_ignore_index()
```

핵심:

- `F.cross_entropy` 입력은 **확률조차 Softmax하지 않은 logits**다.
- 이미 Softmax한 확률을 넣으면 잘못된 Loss가 나온다.
- LM에서는 `(B, T, V)`를 `(B*T, V)`로 펼쳐 쓰는 패턴이 흔하다.

### 10. 실제 LLM에서는 어떻게 사용하는가

#### 10.1 학습 목표

Causal LM(GPT 계열)은 위치 \(t\)에서 토큰 \(x_t\)를, 이전 토큰 \(x_{<t}\)로 예측한다.

$$
L = -\frac{1}{N}\sum_{t} \log p_\theta(x_t \mid x_{<t})
$$

이것이 곧 위치별 Cross Entropy의 평균이다.

#### 10.2 Shape

```text
logits:  (B, T, V)
target:  (B, T)     # 보통 입력 시프트(다음 토큰)
loss:    scalar
```

구현에서는 입력이 `[t0, t1, t2, t3]`일 때  
logits의 위치 0은 `t1`을, 위치 1은 `t2`를 예측하도록 **target을 한 칸 shift**한다.

#### 10.3 평가 지표와의 관계

- **Perplexity(퍼플렉서티)** \(\approx \exp(L)\) (평균 CE를 지수로)
- Loss↓ ↔ Perplexity↓ (같은 전제에서)
- 대화 품질·안전성 등은 CE만으로 보장되지 않는다 (3~4권)

#### 10.4 Softmax를 학습 루프에서 따로 쓰지 않는 이유

학습 Loss 계산은 `cross_entropy(logits, target)` 한 번이면 된다.  
생성(추론) 시에만 Softmax/샘플링이 겉으로 드러나는 경우가 많다.

### 11. 실습

#### 실습 1 — 손계산

\(\mathbf{z}=[0, 0, 0, 0]\), \(c=1\)일 때 \(p_c\)와 CE를 구하시오. (\(V=4\))

#### 실습 2 — 손계산

\(\mathbf{z}=[2, 2, 2, 5]\), \(c=3\)일 때 Softmax와 CE를 구하시오. (계산기 OK)

#### 실습 3 — 코드

8절 NumPy 코드로 예제 A를 재현하고, \(c=0,1,2,3\) 네 경우의 CE를 표로 만드시오.

#### 실습 4 — PyTorch 대조

같은 logits/target에 대해 NumPy CE와 `F.cross_entropy` 값이 오차 \(10^{-5}\) 이내인지 확인하시오.

#### 실습 5 — Shift 사고실험

입력 토큰 ID `[10, 20, 30]`을 다음 토큰 예측으로 학습할 때,  
logits 위치와 target 매칭을 적으시오. (예: 위치 0의 target은?)

#### 실습 6 — Perplexity

평균 CE가 \(0.693\)이면 Perplexity \(\approx e^{0.693}\)는 얼마인가?

### 12. 자주 하는 실수

1. **확률을 한 번 더 Softmax한다**  
   `F.cross_entropy`는 logits를 기대한다. Softmax 결과를 넣으면 Loss가 왜곡된다.

2. **\(-\log\)를 logits에 직접 적용한다**  
   \(-\log z_c\)는 의미가 다르다. 반드시 확률 또는 log-softmax 경로를 쓴다.

3. **target을 one-hot으로 넣어 클래스 인덱스를 기대하는 API에 넘긴다**  
   PyTorch `cross_entropy`의 target은 보통 **클래스 인덱스**다 (`NLLLoss` 경로).

4. **시퀀스 shift를 잊는다**  
   같은 위치의 토큰을 그대로 target으로 쓰면 “자기 자신 맞히기”가 되어 LM 목표가 아니다.

5. **padding을 Loss에 포함한다**  
   pad 토큰까지 평균하면 학습 신호가 희석·왜곡된다. `ignore_index` 또는 mask 평균을 쓴다.

6. **Accuracy만 보고 Loss를 무시한다**  
   Softmax 확률의 교정은 CE Gradient가 담당한다. 맞춘 개수만으로는 확신이 안 보인다.

### 13. 핵심 정리

- Cross Entropy Loss는 정답 확률에 \(-\log\)를 취한 값이다: \(\ell=-\log p_c\).
- LLM에서는 Softmax 확률의 \(p_c\)가 “다음 토큰이 정답일 확률”이다.
- 구현은 logits에서 `logsumexp`로 바로 CE를 계산하는 편이 안정적이다.
- Softmax+CE의 Gradient는 \(p - y\)로, 직관과 수학이 잘 맞는다.
- 시퀀스 LM Loss는 위치별 CE의 (유효 토큰) 평균이다.

### 14. 핵심 용어

| 용어 | 의미 |
|---|---|
| Cross Entropy Loss | 정답 분포와 예측 분포의 차이를 재는 손실, 분류/LM 표준 |
| NLL | \(-\log p_c\); one-hot CE와 동일 |
| Logit | Softmax 직전 점수 |
| Softmax | logits → 확률 분포 |
| Next-Token Prediction | 이전 토큰으로 다음 토큰을 예측 |
| Perplexity | \(\exp(\text{평균 CE})\), LM 평가에 자주 사용 |
| ignore_index | Loss 계산에서 특정 라벨(예: pad)을 무시 |
| log-sum-exp | \(\log\sum e^{z}\)의 안정 계산 |

### 15. 복습 문제

#### 문제 1 (계산)

\(p_c=1/e\)일 때 \(\ell=-\ln p_c\)는?

#### 문제 2 (계산)

\(\mathbf{z}=[0, 1]\), \(c=0\)일 때 Softmax 확률과 CE를 구하시오.

#### 문제 3 (개념)

왜 Accuracy 대신 Cross Entropy로 학습하는가?

#### 문제 4 (코드/API)

`F.cross_entropy`에 Softmax 확률을 넣으면 안 되는 이유를 한 문장으로.

#### 문제 5 (연결)

33강 Softmax로 만든 \(p\)가 이번 강의의 어디에 들어가는가? 다음 35강으로 넘어가기 전, “Loss까지 있는 LM”의 남은 큰 구멍이 무엇인지 말해 보시오.

---

### 정답 및 해설

#### 문제 1

\(-\ln(e^{-1})=1\).

#### 문제 2

\(p_0=e^0/(e^0+e^1)=1/(1+e)\approx 0.2689\),  
\(\ell=-\ln p_0=\ln(1+e)\approx 1.313\).  
(또는 \(\ell=-0+\ln(1+e)\).)

#### 문제 3

Accuracy는 맞음/틀림의 거친 신호라 미분이 거의 없거나 불연속이다. CE는 확률에 따라 부드러운 Gradient를 제공한다.

#### 문제 4

해당 API는 내부에서 log-softmax를 수행하므로, 이미 Softmax된 값을 넣으면 이중 변환이 되어 Loss가 틀어진다.

#### 문제 5

\(p\)의 정답 성분이 \(-\log p_c\)의 입력이 된다.  
지금까지는 “한 위치의 점수→확률→Loss”까지다. 남은 큰 구멍은 **여러 토큰 문맥을 어떻게 한 벡터/표현으로 모을 것인가** — 곧 **Attention**이다.

### 16. 다음 강의와 연결

이제 Softmax(33강)와 Cross Entropy(34강)로 **다음 토큰 학습의 출력단**이 완성되었다.

그런데 Transformer의 핵심은 출력단이 아니라, 문맥을 섞는 **Attention**이다.  
다음 **제35강. Attention이 필요한 이유**에서는 RNN·CNN이 긴 문맥에서 겪는 한계를 보고, 왜 언어에 Attention이 필요한지부터 시작한다.

> 확률과 Loss를 얻었으면, 이제 “무엇에 주목할 것인가”를 배우자.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [제33강. Softmax와 Logit](33강_Softmax와_Logit.md)
- **다음 강:** [제35강. Attention이 필요한 이유](35강_Attention이_필요한_이유.md)

<!-- /LECTURE_NAV -->
