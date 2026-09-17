# 제45강. Feed-Forward Network (MLP)

> **학습 목표**
> - FFN이 두 개의 선형층과 활성화로 이루어짐을 수식으로 쓰기
> - Expansion ratio(확장 비율)와 $d_{\text{ff}}$의 의미
> - ReLU / GeLU / SwiGLU의 위치(사실과 설명 구분)
> - FFN이 시퀀스 길이가 아니라 특징 차원에서 동작한다는 점
> - Attention과 FFN의 역할 분담 직관
> - 제46강 Block 조립에 바로 넣을 수 있는 모듈 스케치

---
## 1. 왜 이것을 배우는가

Attention만 있으면 “누구와 대화할지”는 정해져도, “들은 내용을 얼마나 깊게 재표현할지”가 약하다.  
FFN은 각 위치의 벡터를 더 넓은 은닉 공간으로 보냈다가 다시 접으며 **비선형 특징 변환**을 담당한다.

파라미터 관점에서도 중요하다.  
많은 Transformer에서 **FFN 파라미터가 Attention보다 크다**.  
$d_{\text{ff}} \approx 4\, d_{\text{model}}$이면 선형층 두 장의 무게가 상당하다.

LLM 연결:

```text
Block:
  Attn 경로  — 토큰 간 통신
  FFN 경로  — 토큰 내 계산   ← 오늘
```

제48강 Causal LM의 층 하나하나는 결국 이 두 경로의 반복이다.

## 2. 먼저 알아야 할 개념

- 선형층 $y = xW + b$ (제15~16강)
- 활성화 함수(ReLU 등)
- Residual / Pre-LN 패턴 (제44강)
- shape `(B, T, d_model)` 유지의 필요성

## 3. 핵심 개념 설명

### 3.1 기본 2층 FFN

원 논문 Transformer의 FFN:

$$

\mathrm{FFN}(x) = W_2\,\sigma(W_1 x + b_1) + b_2

$$

시퀀스 표기로는 각 위치 $t$에 독립적으로

$$

\mathrm{FFN}(x_t)
=
\max(0,\ x_t W_1 + b_1)\, W_2 + b_2

$$

(활성화가 ReLU인 경우).

shape:

- 입력: `(..., d_model)`
- 은닉: `(..., d_ff)`
- 출력: `(..., d_model)`

토큰 사이 혼합 없음.  
같은 $W_1, W_2$가 모든 위치에 **공유**된다.  
그래서 “position-wise FFN”이라고 부른다.

### 3.2 Expansion Ratio

**Expansion ratio**는 은닉 확장을 얼마나 할지다.

$$

d_{\text{ff}} = \mathrm{ratio} \times d_{\text{model}}

$$

고전적 Transformer:

- ratio $= 4$ → $d_{\text{model}}=512$이면 $d_{\text{ff}}=2048$

현대 LLM:

- 4가 아니어도 된다.
- SwiGLU를 쓰면 게이트 때문에 중간 차원이 다르게 잡히는 경우가 많다.
- “항상 4배”는 **교육용 기본값**이지 만능 법칙이 아니다.

왜 확장하는가(설명):

- 한 겹 선형만으로는 표현력이 부족하다.
- 넓은 은닉 + 비선형이 특징을 분리·재결합하기 쉽다.
- 다시 $d_{\text{model}}$로 축소해 Residual에 더한다.

### 3.3 활성화 함수들

**ReLU**

$$

\mathrm{ReLU}(z) = \max(0, z)

$$

원 논문 FFN의 기본.

**GeLU (Gaussian Error Linear Unit)**

대략:

$$

\mathrm{GeLU}(z) \approx z \cdot \Phi(z)

$$

$\Phi$는 표준정규 CDF.  
GPT-2 등에서 널리 쓰였다.  
ReLU보다 부드러운 게이트처럼 동작한다.

**SwiGLU**

GLU(Gated Linear Unit) 계열. 단순화된 형태:

$$

\mathrm{SwiGLU}(x)
=
\big(\mathrm{Swish}(x W_{gate}) \odot (x W_{up})\big) W_{down}

$$

- $\odot$: 요소곱
- Swish/SiLU: $z \cdot \sigma(z)$

사실:

- LLaMA 등 다수 현대 LLM이 SwiGLU형 FFN을 사용한다.

설명:

- 게이트가 “어떤 특징을 통과시킬지”를 조절해 표현력이 좋아진다는 해석이 있다.
- 파라미터가 늘어나므로, 중간 차원을 조절해 전체 예산을 맞추는 설계가 흔하다.

이 책의 제46강 기본 Block은 **교육용으로 GeLU 2층 FFN**을 쓰고, SwiGLU는 “현대 LLM 옵션”으로 표시한다.

### 3.4 Attention vs FFN

| | Attention | FFN |
|---|---|---|
| 혼합 축 | 토큰(시간) 축 | 특징 축 |
| 위치 간 통신 | 있음 | 없음(위치별 독립) |
| 비선형 | Softmax 등 | 활성화·게이트 |
| 역할 직관 | 검색·집계 | 가공·재표현 |

둘 다 빠지면 Transformer가 아니다.  
제46강에서 한 블록 안에 나란히 놓는다.

## 4. 직관적으로 이해하기

회의로 비유하면:

- Attention: 누가 누구에게 말할 차례인지, 무엇을 들을지
- FFN: 각 참가자가 들은 내용을 **혼자 노트에 깊게 정리**

노트가 충분히 넓어야( $d_{\text{ff}}$ ) 복잡한 정리가 가능하다.  
정리 후 다시 책상 크기( $d_{\text{model}}$ )로 접어 Residual 서랍에 넣는다.

## 5. 수학적으로 이해하기

### 5.1 배치 행렬형

$X \in \mathbb{R}^{B \times T \times d_{\text{model}}}$

$$

H = \sigma(X W_1 + b_1) \in \mathbb{R}^{B \times T \times d_{\text{ff}}}

$$

$$

Y = H W_2 + b_2 \in \mathbb{R}^{B \times T \times d_{\text{model}}}

$$

### 5.2 파라미터 수 (편향 무시)

기본 FFN:

$$

|\theta|
\approx
d_{\text{model}} \cdot d_{\text{ff}} + d_{\text{ff}} \cdot d_{\text{model}}
=
2 d_{\text{model}} d_{\text{ff}}

$$

ratio 4이면 $8\, d_{\text{model}}^2$ 규모.

MHA의 $W^Q,W^K,W^V,W^O$가 대략 $4\, d_{\text{model}}^2$이므로,  
**FFN이 더 무거운** 경우가 많다.

### 5.3 Pre-LN에서의 위치

$$

x \leftarrow x + \mathrm{FFN}(\mathrm{LN}(x))

$$

Attention residual 이후의 $x$를 받아 한 번 더 가공한다.

## 6. 작은 숫자로 직접 계산하기

설정:

- $d_{\text{model}}=2$, $d_{\text{ff}}=4$, ReLU
- 한 토큰 $x = [1,\ 2]$

$$

W_1 =
\begin{bmatrix}
1 & 0 & -1 & 2 \\
0 & 1 & 1 & -1
\end{bmatrix}
,\quad
b_1 = [0,0,0,0]

$$

(여기서 $W_1$는 $d_{\text{model}} \times d_{\text{ff}}$로 두었다.)

$$

x W_1 = [1,\ 2,\ 1,\ 0]

$$

$$

\mathrm{ReLU}(x W_1) = [1,\ 2,\ 1,\ 0]

$$

$$

W_2 =
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
1 & 0 \\
0 & 1
\end{bmatrix}
,\quad
b_2 = [0,0]

$$

$$

\mathrm{FFN}(x) = [1,\ 2,\ 1,\ 0] W_2 = [2,\ 2]

$$

Residual이면 $x + \mathrm{FFN}(x) = [3,\ 4]$.

확장했다가(4차원) 다시 접어(2차원) 수정량을 만든 것이다.

## 7. 코드로 구현하기

```python
# ffn.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """Position-wise FFN: Linear → GeLU → Linear."""

    def __init__(self, d_model: int, d_ff: int | None = None, dropout: float = 0.0):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SwiGLUFFN(nn.Module):
    """현대 LLM에서 흔한 게이트형 FFN의 교육용 스케치."""

    def __init__(self, d_model: int, d_ff: int | None = None):
        super().__init__()
        # 관례상 SwiGLU는 중간 차원을 2/3 스케일 등으로 조정하기도 함
        d_ff = d_ff or int(8 * d_model / 3)
        # 보통 다중을 맞추기 위해 반올림 규칙을 추가한다
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

if __name__ == "__main__":
    ff = FeedForward(8, 32)
    x = torch.randn(2, 5, 8)
    y = ff(x)
    print(y.shape)  # (2, 5, 8)
```

NumPy 최소 버전:

```python
import numpy as np

def ffn_relu(x, W1, b1, W2, b2):
    h = np.maximum(0, x @ W1 + b1)
    return h @ W2 + b2
```

## 8. 실제 LLM에서는 어떻게 사용하는가

사실:

- 원본 Transformer: ReLU FFN, 확장비 4
- GPT-2: GeLU MLP
- LLaMA류: SwiGLU MLP + RMSNorm 등과 조합

설명:

- 활성화 선택은 세대별 유행·실험 결과의 산물이다.
- FFN이 “사실 저장소”처럼 동작한다는 해석 연구도 있으나, 메커니즘 전체를 그 한 문장으로 환원하진 않는다.

제48강에서 블록을 셀 때, 파라미터의 상당 부분이 이 MLP에 있음을 기억하면 메모리·연산 예상이 쉬워진다.

## 9. 실습

1. `FeedForward(16)`에 랜덤 입력을 넣어 shape 보존을 확인하라.
2. $d_{\text{ff}}=4 d_{\text{model}}$일 때 대략 파라미터 수를 계산하고 `sum(p.numel())`과 비교하라.
3. ReLU/GeLU를 바꿔 같은 입력의 출력 차이를 관찰하라.
4. SwiGLU 스케치의 중간 활성화가 음수도 통과하는지(SiLU 특성) 한 원소로 확인하라.
5. Pre-LN residual 래퍼에 FFN을 넣어 `x + ffn(ln(x))`를 실행하라.

## 10. 자주 하는 실수

1. **FFN을 시퀀스 축으로 섞으려 하기**  
   기본 FFN은 위치별 독립이다.

2. **출력 차원을 d_ff로 남기기**  
   Residual이 깨진다. 반드시 d_model로 돌아온다.

3. **확장비 4를 절대 법칙으로 암기**  
   모델마다 다르다.

4. **SwiGLU를 ‘그냥 GeLU 두 번’으로 이해**  
   게이트 요소곱 구조가 핵심이다.

5. **dropout 위치 무시**  
   구현마다 다르니 하나로 고정한다.

## 11. 핵심 정리

- FFN은 position-wise 2층 MLP로, 토큰 내 비선형 변환을 담당한다.
- 보통 $d_{\text{model}} \to d_{\text{ff}} \to d_{\text{model}}$이며 확장비 4가 고전 기본값이다.
- 활성화는 ReLU → GeLU → SwiGLU로 세대가 진화해 왔다.
- Attention(통신)과 FFN(계산)이 한 블록의 양 날개다.
- 제46강에서 Norm/Residual과 함께 조립한다.

## 12. 핵심 용어

| 용어 | 설명 |
|---|---|
| FFN / MLP | Transformer의 위치별 피드포워드 네트워크 |
| $d_{\text{ff}}$ | FFN 은닉 차원 |
| Expansion Ratio | $d_{\text{ff}} / d_{\text{model}}$ |
| GeLU | GPT류에서 흔한 부드러운 활성화 |
| SwiGLU | 게이트형 FFN. 현대 LLM에 흔함 |
| Position-wise | 모든 위치에 같은 가중치를 독립 적용 |

## 13. 연습 문제
**문제 1.** 기본 FFN의 입출력 차원을 쓰라.

**문제 2.** $d_{\text{model}}=512$, ratio=4일 때 $d_{\text{ff}}$는?

**문제 3.** FFN이 토큰 사이 정보를 직접 섞는가?

**문제 4.** SwiGLU의 핵심 연산 한 가지는?

**문제 5.** Attention과 FFN의 역할 분담을 한 문장으로.

### 정답과 해설

1. $d_{\text{model}} \to d_{\text{ff}} \to d_{\text{model}}$.

2. 2048.

3. 아니다. 위치별 독립이다.

4. 게이트 활성화와 up 투영의 요소곱(후 down 투영).

5. Attention이 토큰 간 정보를 모으고, FFN이 각 토큰 표현을 비선형으로 가공한다.

## 14. 다음 강의와 연결

부품이 모두 모였다.

- MHA (제41강)
- 위치 (제42~43강)
- Norm / Residual (제44강)
- FFN (제45강)

제46강 **Transformer Block 조립**에서 Pre-LN 기준으로 하나로 묶고, 코드를 완성한다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [44강. LayerNorm과 Residual Connection](44강_LayerNorm과_Residual_Connection.md)
- **다음 강:** [46강. Transformer Block 조립](46강_Transformer_Block_조립.md)

<!-- /LECTURE_NAV -->
