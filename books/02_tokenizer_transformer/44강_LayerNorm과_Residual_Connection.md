# 제44강. LayerNorm과 Residual Connection

> **학습 목표**
> - Residual Connection (잔차 연결)
> - Layer Normalization (LayerNorm)
> - Residual이 깊은 망에서 하는 역할
> - Transformer에서의 `x + SubLayer(x)` 패턴
> - LayerNorm의 수식과 작은 숫자 예제
> - BatchNorm과 LayerNorm의 차이(개념 수준)

---
## 1. 왜 이것을 배우는가

Attention과 FFN을 아무리 잘 만들어도, 수십·수백 층으로 쌓이면 다음이 생긴다.

- 신호가 너무 커지거나 사라짐
- Gradient가 불안정
- 최적화가 어려움

Residual과 Norm은 “똑똑한 새 연산”이라기보다 **깊게 쌓기 위한 공학적 안전벨트**다.

LLM 연결:

```text
x
 → Norm
 → MHA
 → + x          ← Residual
 → Norm
 → FFN
 → + x          ← Residual
```

제46강에서 Block을 조립할 때 이 두 줄이 뼈대가 된다.

## 2. 먼저 알아야 할 개념

- 평균, 분산, 표준편차
- 요소별 곱·합, 브로드캐스팅
- Gradient가 층을 타고 흐른다는 직관 (제14·17강)
- MHA 출력 shape = 입력 shape (제41강)

## 3. 핵심 개념 설명

### 3.1 Residual Connection

**Residual Connection**은 층이 입력을 완전히 대체하지 않고, **변화량(잔차)**만 학습하도록 길을 여는 연결이다.

$$

y = x + F(x)

$$

- $x$: 입력
- $F(x)$: Attention 또는 FFN 같은 서브층
- $y$: 출력

직관:

```text
“기존 정보를 유지한 채, 필요한 수정을 더한다”
```

장점:

1. **항등 경로**: $F$가 거의 0이어도 $y \approx x$로 정보 보존
2. **Gradient 고속도로**: 역전파 때 $\partial y/\partial x$에 1이 더해져 신호가 잘 흐름
3. **깊은 망 안정화**: ResNet에서 입증된 패턴을 Transformer도 차용

Transformer에서는 보통 두 번 쓴다.

$$

x \leftarrow x + \mathrm{MHA}(\cdot)

$$

$$

x \leftarrow x + \mathrm{FFN}(\cdot)

$$

(정확한 Norm 배치에 따라 $\cdot$의 입력이 `x` 또는 `Norm(x)`.)

### 3.2 LayerNorm

**Layer Normalization**은 **한 토큰 벡터의 특징 차원**을 정규화한다.

토큰 벡터 $x \in \mathbb{R}^{d}$에 대해:

$$

\mu = \frac{1}{d}\sum_{j=1}^{d} x_j

$$

$$

\sigma^2 = \frac{1}{d}\sum_{j=1}^{d} (x_j - \mu)^2

$$

$$

\hat{x}_j = \frac{x_j - \mu}{\sqrt{\sigma^2 + \epsilon}}

$$

$$

\mathrm{LN}(x)_j = \gamma_j \hat{x}_j + \beta_j

$$

- $\epsilon$: 0나눗셈 방지 (예: $1\mathrm{e}{-5}$)
- $\gamma, \beta$: 학습 가능한 scale/bias (차원별)

BatchNorm과의 차이(요지):

| | BatchNorm | LayerNorm |
|---|---|---|
| 통계를 내는 축 | 배치 쪽 | 특징 차원 쪽 |
| NLP 시퀀스 | 배치/길이 변동에 민감할 수 있음 | 토큰별 정규화로 안정적 |
| Transformer | 드묾 | 고전적 기본 선택 |

### 3.3 RMSNorm (신중히)

**RMSNorm**은 LayerNorm에서 **평균을 빼는 단계**를 생략하고, 제곱평균제곱근(RMS)으로만 나누는 정규화다.

$$

\mathrm{RMS}(x) = \sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \epsilon}

$$

$$

\mathrm{RMSNorm}(x)_j = \gamma_j \frac{x_j}{\mathrm{RMS}(x)}

$$

사실:

- LLaMA 등 많은 현대 LLM이 LayerNorm 대신 **RMSNorm**을 사용한다.

설명:

- 평균 중심화를 빼면 연산이 줄고, 실무에서 학습이 잘 되는 경우가 많다는 보고/선택이 있다.
- “RMSNorm이 항상 LayerNorm보다 우수하다”고 단정할 필요는 없다. **설계 선택**이다.
- 이 책의 수식 예제는 LayerNorm을 먼저 익히고, 구현에서 RMSNorm으로 바꾸는 길을 남긴다.

### 3.4 Pre-LN vs Post-LN (예고)

**Post-LN** (원 논문에 가까운 형태):

$$

x = \mathrm{LN}(x + \mathrm{Attn}(x))

$$

**Pre-LN** (많은 현대 구현):

$$

x = x + \mathrm{Attn}(\mathrm{LN}(x))

$$

Pre-LN이 깊은 모델에서 학습이 더 안정적이라는 경험적 보고가 많다.  
제46강은 **Pre-LN Transformer Block**을 기본으로 조립한다.

## 4. 직관적으로 이해하기

Residual:

```text
고속도로(본선) : x가 그대로 흐름
진입 램프     : F(x)가 수정량을 더함
```

LayerNorm:

```text
한 토큰의 좌표들을
“평균 0, 분산 1 근처”로 맞춘 뒤
γ, β로 미세 조정
```

둘이 함께 있으면:

- 본선으로 정보/Gradient가 흐르고
- 각 서브층 입력이 적당한 스케일로 유지된다

## 5. 수학적으로 이해하기

### 5.1 Residual의 역전파

$$

y = x + F(x)

$$

$$

\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial y}
+
\frac{\partial L}{\partial y}\frac{\partial F}{\partial x}

$$

첫 항 덕분에 $F$의 Jacobian이 작아도 Gradient가 완전히 사라지기 어렵다.

### 5.2 LayerNorm 미분은?

실습·프레임워크에서는 Autograd에 맡긴다.  
개념적으로는 $\mu, \sigma$가 $x$에 의존하므로, 단순 요소별 스케일보다 조금 더 얽힌다.  
지금 단계의 목표는 forward 수식과 역할을 고정하는 것이다.

### 5.3 수치 안정성

$\epsilon$이 너무 작으면 저분산 벡터에서 폭발할 수 있고, 너무 크면 정규화가 둔해진다.  
프레임워크 기본값(예: `1e-5`)을 따르는 것이 안전하다.

## 6. 작은 숫자로 직접 계산하기

한 토큰:

$$

x = [2,\ 4,\ 4,\ 6]

$$

$$

\mu = \frac{2+4+4+6}{4} = 4

$$

$$

\sigma^2 = \frac{(2-4)^2+(4-4)^2+(4-4)^2+(6-4)^2}{4}
= \frac{4+0+0+4}{4} = 2

$$

$$

\sigma = \sqrt{2} \approx 1.4142

$$

$\epsilon=0$으로 두면

$$

\hat{x}
=
\frac{[2,4,4,6]-4}{1.4142}
\approx
[-1.4142,\ 0,\ 0,\ 1.4142]

$$

$\gamma=[1,1,1,1],\ \beta=[0,0,0,0]$이면 LN 출력은 $\hat{x}$ 그대로다.

Residual 예:

$$

F(x) = [0.1,\ -0.1,\ 0.2,\ 0.0]

$$

$$

y = x + F(x) = [2.1,\ 3.9,\ 4.2,\ 6.0]

$$

정보가 “통째로 교체”되지 않고 살짝 수정된다.

RMSNorm 비교 (같은 $x$):

$$

\mathrm{RMS}
=
\sqrt{\frac{2^2+4^2+4^2+6^2}{4}}
=
\sqrt{\frac{4+16+16+36}{4}}
=
\sqrt{18}
\approx 4.2426

$$

$$

\frac{x}{\mathrm{RMS}}
\approx
[0.4714,\ 0.9428,\ 0.9428,\ 1.4142]

$$

평균을 빼지 않았으므로 LayerNorm 결과와 다르다.  
이것이 “비슷하지만 같은 연산은 아님”을 보여주는 최소 예제다.

## 7. 코드로 구현하기

```python
# norm_residual.py
from __future__ import annotations

import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5):
    """x: (..., d)"""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)  # ddof=0
    x_hat = (x - mu) / np.sqrt(var + eps)
    return gamma * x_hat + beta

def rms_norm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-5):
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return gamma * (x / rms)

def residual(x: np.ndarray, fx: np.ndarray) -> np.ndarray:
    return x + fx

if __name__ == "__main__":
    x = np.array([2.0, 4.0, 4.0, 6.0])
    gamma = np.ones(4)
    beta = np.zeros(4)
    print("LN", layer_norm(x, gamma, beta, eps=0.0))
    print("RMS", rms_norm(x, gamma, eps=0.0))
```

PyTorch:

```python
import torch
import torch.nn as nn

class PreLNResidual(nn.Module):
    """y = x + sublayer(LN(x)) 패턴의 뼈대."""

    def __init__(self, d_model: int, sublayer: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.sublayer = sublayer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.sublayer(self.norm(x))
```

RMSNorm을 쓰려면 `nn.LayerNorm` 대신 커스텀 모듈을 넣으면 된다.

## 8. 실제 LLM에서는 어떻게 사용하는가

사실:

- 원 논문 Transformer: LayerNorm + Residual (Post-LN에 가깝게 서술)
- 많은 현대 decoder-only LLM: **Pre-LN + RMSNorm** 조합이 흔함

설명:

- Residual은 거의 공통 분모다.
- Norm의 종류·위치는 세대/모델에 따라 다르다.
- 제46강 조립은 교육용으로 Pre-LN + LayerNorm을 기본으로 하고, RMSNorm 교체 지점을 표시한다.

제48강 Causal LM에서도 블록 내부는 결국

```text
Residual + Norm + Attn + Residual + Norm + FFN
```

의 반복이다.

## 9. 실습

1. 제7절 벡터로 LN 손계산과 NumPy 결과가 같은지 확인하라 (`eps=0` 주의).
2. 같은 벡터의 RMSNorm 결과를 계산하라.
3. `x + F(x)`에서 $F=0$이면 출력이 $x$인지 확인하라.
4. `(B,T,C)` 랜덤 텐서에 LayerNorm을 적용해 마지막 축 평균이 대략 0인지 확인하라.
5. Pre-LN 래퍼 클래스에 가짜 `sublayer`를 넣어 shape 보존을 확인하라.

## 10. 자주 하는 실수

1. **정규화 축 착각**  
   LayerNorm은 보통 마지막 특징 차원이다. 배치 평균이 아니다.

2. **Residual shape 불일치**  
   `F(x)`의 shape가 `x`와  alike해야 한다. MHA/FFN이 `d_model`을 보존하는 이유다.

3. **Norm과 Residual 순서 혼동**  
   Pre/Post를 바꾸면 학습 역학이 달라진다. 구현 하나를 끝까지 일관되게 유지한다.

4. **LayerNorm = RMSNorm으로 호칭**  
   관련은 있으나 동일하지 않다. 평균 중심화 유무를 구분한다.

5. **$\gamma, \beta$ 초기화 무시**  
   보통 $\gamma=1, \beta=0$에서 시작해 정체성에 가깝게 출발한다.

## 11. 핵심 정리

- Residual `x + F(x)`는 깊은 망의 정보·Gradient 통로다.
- LayerNorm은 토큰 벡터를 특징 축에서 정규화하고 $\gamma, \beta$로 재스케일한다.
- RMSNorm은 평균 중심화 없는 단순 정규화로, 현대 LLM에 흔하다.
- Pre-LN은 제46강 Block 조립의 기본 배치다.
- Attention/FFN “내용”과 Norm/Residual “안정 장치”가 만나야 Transformer가 된다.

## 12. 핵심 용어

| 용어 | 설명 |
|---|---|
| Residual Connection | $y=x+F(x)$ 형태의 잔차 연결 |
| LayerNorm | 특징 차원 평균·분산 정규화 + affine |
| RMSNorm | RMS로 나누는 정규화(평균 미사용) |
| Pre-LN | 서브층 앞에 Norm |
| Post-LN | 서브층+잔차 뒤에 Norm |
| $\gamma, \beta$ | Norm의 학습 가능 scale/bias |

## 13. 연습 문제
**문제 1.** Residual 연결의 수식을 쓰라.

**문제 2.** $x=[2,4,4,6]$의 평균과(표본이 아닌 모집단) 분산은?

**문제 3.** LayerNorm과 RMSNorm의 핵심 차이는?

**문제 4.** Transformer에서 Residual이 MHA/FFN과 맞물리려면 출력 차원이 왜 보존되어야 하는가?

**문제 5.** Pre-LN의 한 줄 패턴을 쓰라.

### 정답과 해설

1. $y = x + F(x)$.

2. 평균 4, 분산 2.

3. LayerNorm은 평균을 빼고 표준편차로 나누고, RMSNorm은 평균을 빼지 않고 RMS로 나눈다.

4. `x + F(x)`를 하려면 shape가 같아야 한다.

5. $x \leftarrow x + \mathrm{SubLayer}(\mathrm{LN}(x))$.

## 14. 다음 강의와 연결

오늘은 안정 장치다.  
제45강은 블록의 다른 절반, **Feed-Forward Network(MLP)**다.

Attention이 토큰 사이를 섞으면, FFN은 **토큰마다** 비선형 변환을 가한다.  
제46강에서 Norm/Residual/Attn/FFN을 하나의 Block으로 조립한다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [제43강. RoPE](43강_RoPE.md)
- **다음 강:** [제45강. Feed-Forward Network (MLP)](45강_Feed_Forward_Network_MLP.md)

<!-- /LECTURE_NAV -->
