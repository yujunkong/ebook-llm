# 1권. Python · Tensor · 수학 · PyTorch

## 제13강. Loss Function

### 1. 이번 강의에서 배울 것

12강에서 Gradient Descent로 파라미터를 내리는 법을 배웠다. 그런데 “무엇을 기준으로 높다/낮다”를 말하려면 **Loss Function**이 필요하다.

이 강의를 마치면 다음을 말할 수 있어야 한다.

- **Loss Function(손실 함수)**이 무엇인지, 왜 학습에 필수인지
- **MSE(Mean Squared Error)**를 손으로 계산하고 미분과 연결하기
- **Cross Entropy(교차 엔트로피)**의 직관 (상세 Softmax는 이후, 여기서는 점수→확률→Loss 감각)
- “좋은 모델”을 Loss 감소로 표현하는 법
- LLM이 왜 Next-Token Cross Entropy를 쓰는지

Loss는 학습의 **나침반**이다. 나침반이 잘못된 방향을 가리키면, GD가 아무리 정확해도 잘못된 곳으로 간다.

### 2. 왜 이것을 배우는가

모델은 숫자를 출력한다. 사람 눈에는 “답이 맞다/틀리다”가 보이지만, 컴퓨터가 파라미터를 업데이트하려면 **틀린 정도를 미분 가능한 숫자 하나(또는 배치 평균)**로 바꿔야 한다.

```text
예측과 정답의 차이
  → Loss (Scalar)
    → Gradient
      → Parameter 업데이트
```

이 파이프라인의 첫 변환이 Loss Function이다.

### 3. 먼저 알아야 할 개념

- Scalar / Vector (9강)
- 평균, 제곱
- 미분·Gradient·GD (11~12강)
- “확률은 0~1이고 합이 1”이라는 초등 직관

아직 Softmax를 엄밀히 몰라도 된다. 이번 강의는 Cross Entropy의 **의미**에 초점을 둔다.

### 4. 핵심 개념 설명

#### 4.1 Loss Function (손실 함수)

**Loss Function(손실 함수, cost function과 비슷하게 쓰이기도 함)**은 모델의 예측이 정답과 **얼마나 다른지**를 하나의 숫자로 요약하는 함수다.

보통

$$
L = L(\hat{y},\ y)
$$

또는 파라미터가 보이게

$$
L(\theta) = L\big(f_{\theta}(x),\ y\big)
$$

기호:

- \(f_{\theta}\): 파라미터 \(\theta\)를 가진 모델
- \(x\): 입력
- \(\hat{y}=f_{\theta}(x)\): 예측
- \(y\): 정답(타깃)
- \(L\): 손실 값 (작을수록 보통 더 좋음)

왜 필요한가?

1. 학습 목표를 수학적으로 정의한다.
2. Gradient Descent가 최소화할 대상을 제공한다.
3. 학습 중 모니터링 지표가 된다. (다만 최종 품질 지표와 항상 같지는 않다)

#### 4.2 MSE — Mean Squared Error

**MSE(Mean Squared Error, 평균 제곱 오차)**는 회귀(연속 값 예측)에서 가장 기본적인 Loss다.

샘플이 \(N\)개일 때

$$
L_{\mathrm{MSE}}
=
\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2
$$

기호:

- \(\hat{y}_i\): \(i\)번째 예측
- \(y_i\): \(i\)번째 정답
- \((\hat{y}_i-y_i)^2\): 오차를 제곱 (부호 제거 + 큰 오차 더 강하게 벌점)
- \(1/N\): 평균

한 샘플만 보면

$$
\ell = (\hat{y}-y)^2
$$

왜 제곱인가?

- 절댓값 오차 \(|\hat{y}-y|\)도 가능하지만, 제곱은 미분이 매끄럽고(0 근처 제외 이슈가 적고), 큰 오차에 민감하다.
- 11~12강에서 이미 써 온 \((wx-y)^2\)가 바로 이것이다.

#### 4.3 Cross Entropy — 직관 미리보기

**Cross Entropy(교차 엔트로피)**는 “정답 분포”와 “모델이 예측한 확률 분포”가 얼마나 다른지를 재는 데 쓰이는 Loss다. 분류 문제와 LLM의 다음 토큰 예측에서 표준이다.

간단한 이진/멀티클래스 직관:

모델이 정답 클래스에 확률 \(p_{\text{correct}}\)를 부여했다고 하자.  
Cross Entropy는 Roughly

$$
\ell = -\log p_{\text{correct}}
$$

기호:

- \(p_{\text{correct}}\): 정답 토큰/클래스에 대한 모델 확률 (\(0<p\leq 1\))
- \(\log\): 보통 자연로그
- 앞의 마이너스: 확률을 **손실**(작을수록 좋음)로 뒤집는다

성질:

| \(p_{\text{correct}}\) | \(-\log p\) | 의미 |
|---|---|---|
| 1.0 | 0 | 완벽히 확신, Loss 0 |
| 0.5 | \(\approx 0.693\) | 반반 |
| 0.1 | \(\approx 2.302\) | 많이 틀림 |
| 0.01 | \(\approx 4.605\) | 심하게 틀림 |

정답에 높은 확률을 줄수록 Loss가 작아진다.  
GD는 그 방향으로 logits/파라미터를 민다.

멀티클래스에서 정답이 one-hot \(\mathbf{y}\)이고 모델 확률이 \(\mathbf{p}\)이면

$$
\ell = - \sum_{k} y_k \log p_k
$$

one-hot이면 정답 인덱스 \(c\)만 남아 \(\ell=-\log p_c\)가 된다.

#### 4.4 Softmax와의 관계 (예고만)

로짓(점수) 벡터 \(\mathbf{z}\)를 확률로 바꾸는 함수가 **Softmax**다.

$$
p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}
$$

LLM LM Head는 각 토큰에 대해 길이 \(V\)인 logits를 내고, Softmax(+Cross Entropy)로 Loss를 계산한다.  
상세 구현은 이후 Softmax/분류 강의에서 다룬다. 지금은 “점수 → 확률 → \(-\log p_{\text{정답}}\)” 파이프라인만 잡는다.

#### 4.5 Loss vs Metric

| | Loss | Metric(지표) |
|---|---|---|
| 목적 | 학습(미분·업데이트) | 평가 |
| 예 | MSE, Cross Entropy | Accuracy, BLEU, 사람 평가 |
| 요구 | 대개 미분 가능 | 미분 불필요해도 됨 |

Accuracy가 높아도 Cross Entropy가 더 정보적일 수 있다.  
확률 0.51로 맞춘 것과 0.99로 맞춘 것을 Accuracy는 같게 보지만 Loss는 구분한다.

### 5. 직관적으로 이해하기

선생님(Loss)이 숙제를 채점한다고 하자.

- MSE 선생님: “숫자 답이 얼마나 멀리 떨어졌나”를 제곱으로 감점
- Cross Entropy 선생님: “정답이라고 얼마나 확신했나”를 확률로 감점

학습은 학생이 선생님 감점을 줄이도록 내부 숫자(파라미터)를 고치는 과정이다.

LLM 관점:

```text
문맥: "오늘은 날씨가"
정답 다음 토큰: "좋다"
모델이 "좋다"에 확률 0.02만 줌 → Loss 큼 → 강하게 수정
모델이 0.80을 줌 → Loss 작음 → 약하게 수정
```

### 6. 작은 숫자로 직접 계산하기 — MSE

#### 6.1 한 샘플

예측 \(\hat{y}=3\), 정답 \(y=5\)

$$
\ell=(3-5)^2=4
$$

#### 6.2 세 샘플의 MSE

| \(i\) | \(\hat{y}_i\) | \(y_i\) | 제곱오차 |
|---|---:|---:|---:|
| 1 | 2 | 2 | 0 |
| 2 | 1 | 3 | 4 |
| 3 | 4 | 5 | 1 |

$$
L=\frac{0+4+1}{3}=\frac{5}{3}\approx 1.667
$$

#### 6.3 선형 모델과 연결

\(x=2\), \(y=4\), \(\hat{y}=wx\)

$$
L(w)=(2w-4)^2
$$

\(w=1\) → \(L=4\), \(w=2\) → \(L=0\).  
12강의 GD가 바로 이 \(L\)을 내려간다.

#### 6.4 MSE의 편미분 (복습 연결)

\(L=\frac{1}{2}(\hat{y}-y)^2\)처럼 \(\frac{1}{2}\)를 붙이는 관례도 흔하다 (미분 시 2가 사라져 깔끔).

$$
\frac{\partial L}{\partial \hat{y}} = \hat{y}-y
$$

(위에서 \(L=\frac12(\hat{y}-y)^2\)일 때)

예측이 정답보다 크면 Gradient가 양수 → 예측을 줄이는 방향이 하강 방향.

### 7. 작은 숫자로 직접 계산하기 — Cross Entropy

#### 7.1 세 클래스, 정답은 클래스 1 (0-based index 1)

모델 확률:

$$
\mathbf{p}=[0.1,\ 0.7,\ 0.2]
$$

정답 one-hot:

$$
\mathbf{y}=[0,\ 1,\ 0]
$$

$$
\ell=-\sum_k y_k\log p_k = -\log 0.7 \approx -(-0.3567)=0.3567
$$

(자연로그 기준. \(\ln 0.7\approx -0.3567\))

#### 7.2 같은 정답, 나쁜 예측

$$
\mathbf{p}=[0.8,\ 0.1,\ 0.1], \quad \ell=-\ln 0.1\approx 2.302
$$

확신이 정답에서 벗어나면 Loss가 커진다.

#### 7.3 LLM 토큰 한 자리

어휘가 `{좋다, 맑다, 의자}`이고 정답이 `좋다`라 하자.

| 토큰 | logit \(z\) (가정) | Softmax 확률 (가정) |
|---|---:|---:|
| 좋다 | 2.0 | 0.72 |
| 맑다 | 1.0 | 0.26 |
| 의자 | 0.0 | 0.02 |

(확률은 예시 값. Softmax로 다시 계산해도 비슷한 경향.)

$$
\ell=-\log 0.72 \approx 0.329
$$

만약 정답이 `의자`인데 모델이 0.02만 줬다면

$$
\ell=-\log 0.02\approx 3.91
$$

학습은 이런 큰 Loss가 나온 자리에서 파라미터를 강하게 수정한다.

#### 7.4 시퀀스 Loss

토큰 \(T\)개의 정답 시퀀스에 대해 흔히

$$
L = \frac{1}{T}\sum_{t=1}^{T} -\log p_{\theta}(y_t \mid y_{<t}, x)
$$

- \(y_t\): \(t\)번째 정답 토큰
- \(p_{\theta}(\cdot)\): 모델이 준 조건부 확률
- 평균: 길이 다른 문장을 비교하기 쉽게

이것이 Next Token Prediction 학습의 Loss다.

### 8. 코드로 구현하기

```python
# lecture13_loss.py
import numpy as np


def mse(y_hat: np.ndarray, y: np.ndarray) -> float:
    """평균 제곱 오차"""
    return float(np.mean((y_hat - y) ** 2))


y_hat = np.array([2.0, 1.0, 4.0])
y = np.array([2.0, 3.0, 5.0])
print("MSE =", mse(y_hat, y))  # 5/3


def cross_entropy(p: np.ndarray, y_index: int) -> float:
    """
    p: 확률 벡터 (합이 1이라고 가정)
    y_index: 정답 클래스 인덱스
    """
    # 수치 안정성을 위해 아주 작은 값 클램프
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    return float(-np.log(p[y_index]))


p_good = np.array([0.1, 0.7, 0.2])
p_bad = np.array([0.8, 0.1, 0.1])
print("CE good =", cross_entropy(p_good, 1))
print("CE bad  =", cross_entropy(p_bad, 1))


def softmax(z: np.ndarray) -> np.ndarray:
    """간단한 Softmax (최대값 빼서 overflow 완화)"""
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


logits = np.array([2.0, 1.0, 0.0])
probs = softmax(logits)
print("probs =", probs)
print("CE for class0 =", cross_entropy(probs, 0))
```

#### 배치 MSE로 GD 한 바퀴

```python
# lecture13_mse_gd.py
import numpy as np

# y = 3x + 1 을 데이터에서 복원한다고 가정
X = np.array([0.0, 1.0, 2.0, 3.0])
Y = 3 * X + 1

w, b = 0.0, 0.0
eta = 0.05

for step in range(40):
    y_hat = w * X + b
    loss = np.mean((y_hat - Y) ** 2)
    # dL/dw, dL/db (MSE)
    dL_dyhat = 2 * (y_hat - Y) / len(X)
    dw = np.sum(dL_dyhat * X)
    db = np.sum(dL_dyhat)
    w -= eta * dw
    b -= eta * db
    if step % 10 == 0 or step == 39:
        print(f"step={step:02d} loss={loss:.4f} w={w:.3f} b={b:.3f}")
```

Loss가 줄며 \(w\to 3\), \(b\to 1\)에 가까워지는지 확인한다.

### 9. PyTorch로 Loss

```python
# lecture13_torch_loss.py
import torch
import torch.nn.functional as F

y_hat = torch.tensor([2.0, 1.0, 4.0])
y = torch.tensor([2.0, 3.0, 5.0])
print(F.mse_loss(y_hat, y))

logits = torch.tensor([[2.0, 1.0, 0.0]])  # (batch=1, C=3)
target = torch.tensor([0])                 # 정답 클래스 0
print(F.cross_entropy(logits, target))     # Softmax+NLL을 한 번에
```

`F.cross_entropy`는 내부에서 Softmax와 \(-\log p_c\)를 **수치적으로 안정적**으로 계산한다.  
실무에서는 확률을 직접 만든 뒤 로그를 취하기보다 이 API를 쓰는 편이 안전하다.

### 10. 실제 LLM에서는 어떻게 사용하는가

#### 10.1 표준 목적

Pretraining / SFT의 기본:

$$
L = - \sum_{t} \log p_{\theta}(y_t \mid \text{context}_t)
$$

- Causal LM: context는 이전 토큰들
- 배치·시퀀스에 대해 평균

#### 10.2 Shape 관점

```text
logits:  (B, T, V)
target:  (B, T)          # 토큰 ID
loss:    ()              # Scalar (무시 마스크 적용 후 평균)
```

Padding 토큰은 mask로 Loss에서 제외하는 경우가 많다.

#### 10.3 Loss만으로 부족할 때

- 낮은 Loss ≠ 항상 좋은 대화 품질
- 그래서 3~4권에서 SFT, Preference Learning, RLHF 등이 **추가 목표**를 얹는다
- 그래도 바닥의 학습 신호는 여전히 “미분 가능한 Loss/보상”이다

### 11. 어떤 Loss를 고를까 (초보 가이드)

| 문제 유형 | 대표 Loss |
|---|---|
| 숫자 회귀 | MSE, MAE |
| 분류 / 토큰 예측 | Cross Entropy |
| 생성 모델 기본 LM | Cross Entropy (NLL) |
| 거리 학습 등 | Contrastive 계열 (후술) |

LLM Volume 1 단계에서는 **MSE로 GD 감각**, **Cross Entropy로 LLM 연결** 두 축이면 충분하다.

### 12. 실습

#### 실습 1 — MSE 손계산

예측 `[0, 2, 4]`, 정답 `[1, 2, 3]`의 MSE를 구하시오.

#### 실습 2 — Cross Entropy 손계산

\(p=[0.2,0.5,0.3]\), 정답 인덱스 0일 때와 1일 때 \(\ell=-\ln p_c\)를 비교하시오. (계산기 사용 OK)

#### 실습 3 — 코드

8절 `mse`와 `cross_entropy`를 실행해 손계산과 맞는지 확인하시오.

#### 실습 4 — GD와 Loss

`lecture13_mse_gd.py`에서 `eta=0.5`로 바꿔 보고 Loss가 어떻게 되는지 관찰하시오.

#### 실습 5 — LLM 문장

문장 정답 토큰이 4개이고 각 위치에서 \(p_{\text{correct}}\)가 `0.5, 0.5, 0.5, 0.5`일 때 평균 Cross Entropy를 구하시오.

### 13. 자주 하는 실수

1. **확률과 logits를 혼동한다**  
   Cross Entropy에 넣을 \(p\)는 확률이다. logits에 바로 \(-\log z\)를 적용하면 안 된다. (`F.cross_entropy`는 logits 입력이 맞음)

2. **\(\log 0\)을 만든다**  
   확률 0에 로그는 \(-\infty\). clip 또는 stable CE 구현 필요.

3. **Loss가 줄면 만사 OK라고 생각한다**  
   과적합·잘못된 데이터면 Loss만 내려가도 모델은 쓸모없을 수 있다.

4. **배치 평균과 합을 혼동한다**  
   평균 여부에 따라 Gradient 스케일이 달라져 \(\eta\) 감각이 바뀐다.

5. **MSE를 토큰 분류에 그대로 쓴다**  
   클래스는 one-hot 회귀로도 가능하지만, Softmax+CE가 표준이며 해석·학습이 더 잘 맞다.

### 14. 핵심 정리

- Loss는 예측과 정답의 차이를 미분 가능한 Scalar로 요약한다.
- MSE는 제곱 오차의 평균으로, 회귀와 GD 연습에 적합하다.
- Cross Entropy는 정답 확률에 \(-\log\)를 취해, 틀린 확신에 큰 벌점을 준다.
- LLM 학습의 기본 Loss는 Next-Token Cross Entropy다.
- GD는 Loss가 정의한 지형을 내려간다. Loss 설계가 곧 학습 목표 설계다.

### 15. 핵심 용어

| 용어 | 의미 |
|---|---|
| Loss Function | 예측 오류를 숫자로 요약하는 함수 |
| MSE | 평균 제곱 오차 |
| Cross Entropy | 정답 분포와 예측 분포의 차이를 재는 손실 |
| NLL (Negative Log Likelihood) | \(-\log p\); CE와 밀접 |
| Softmax | logits → 확률 분포 |
| Logit | Softmax 직전의 점수 |
| Target / Label | 정답 |
| Metric | 평가용 지표 (반드시 Loss는 아님) |

### 16. 복습 문제

#### 문제 1 (계산)

\(\hat{y}=10\), \(y=7\)일 때 제곱 오차는?

#### 문제 2 (계산)

\(p_{\text{correct}}=e^{-1}\approx 0.3679\)일 때 Cross Entropy \(\ell=-\ln p\)는 대략 얼마인가?

#### 문제 3 (개념)

왜 Accuracy만으로 학습하지 않고 Cross Entropy 같은 Loss를 쓰는가? (미분 관점)

#### 문제 4 (코드)

`F.mse_loss`에 예측 `[1,2]`, 정답 `[1,4]`를 넣으면 값은?

#### 문제 5 (연결)

LLM이 문맥 “서울의 수도는” 다음에 “부산”에 높은 확률을 주면 Loss는 커지는가 작아지는가? 정답이 “서울”일 때 기준으로 설명하시오.

---

### 정답 및 해설

#### 문제 1

\((10-7)^2=9\).

#### 문제 2

\(-\ln(e^{-1})=1\).

#### 문제 3

Accuracy는 맞음/틀림의 계단 함수에 가까워 미분이 거의  everywhere 0이거나 불연속이다. Cross Entropy는 확률에 따라 부드러운 기울기를 제공해 GD가 가능하다.

#### 문제 4

평균 \(((0)^2+(2)^2)/2=2\).

#### 문제 5

정답이 “서울”인데 “부산”에 높은 확률(그리고 “서울”에 낮은 확률)을 주면 \(-\log p_{\text{서울}}\)이 커져 Loss가 **커진다**. 학습은 “서울” 확률을 높이는 쪽으로 파라미터를 수정한다.

### 17. 다음 강의와 연결

Loss까지 정의했다. 그런데 실제 네트워크는

$$
L = L\big(f(g(h(x)))\big)
$$

처럼 **함수가 깊게 합성**되어 있다.

다음 **제14강. Chain Rule**에서는 합성함수의 미분법을 배우고, Computational Graph 위에서 Gradient가 뒤로 흐르는 그림을 그린다.  
그것이 Backpropagation의 문이다.

> 나침반(Loss)을 얻었으면, 이제 깊은 산에서 기울기를 전달하는 규칙을 배우자.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [제12강. Gradient와 Gradient Descent](12강_Gradient와_Gradient_Descent.md)
- **다음 강:** [제14강. Chain Rule](14강_Chain_Rule.md)

<!-- /LECTURE_NAV -->
