# 2권. Tokenizer와 Transformer

## 제41강. Multi-Head Attention

### 1. 이번 강의에서 배울 것

제39강에서 Self-Attention을, 제40강에서 Causal Mask를 익혔다.  
이번 강의는 Attention을 **여러 개의 머리(head)**로 나누어 병렬로 돌리는 **Multi-Head Attention(MHA)**을 다룬다.

이 강의를 마치면 다음을 말할 수 있어야 한다.

- Single-Head Attention과 Multi-Head Attention의 차이
- \(d_{\text{model}}\), \(h\), \(d_k = d_{\text{model}} / h\)의 관계
- 각 head에서 \(Q, K, V\)를 계산하고, 결과를 **concat**한 뒤 \(W^O\)로 섞는 전체 흐름
- 왜 “한 번에 크게” 보는 대신 “여러 관점으로 작게” 보는지
- Causal Mask가 Multi-Head에서도 동일하게 적용되는 방식
- NumPy/PyTorch로 MHA를 스케치하는 방법
- GPT류 LLM에서 MHA(또는 그 변형)가 차지하는 위치

제40강의 Causal Mask는 **한 head의 score 행렬**에 붙는 규칙이었다.  
오늘은 그 규칙을 \(h\)개 head에 동시에 적용한 뒤, 결과를 다시 \(d_{\text{model}}\)로 합친다.

### 2. 왜 이것을 배우는가

Self-Attention 한 번으로도 “토큰끼리 정보를 섞는” 일은 가능하다.  
그런데 한 번의 Attention은 **하나의 유사도 기준**으로 섞는다.

문장 안에서는 보통 여러 종류의 관계가 동시에 필요하다.

- 주어–동사 연결
- 대명사–선행사 연결
- 가까운 이웃 토큰의 지역 패턴
- 문장 끝에서 앞쪽 주제를 되짚는 장거리 의존

한 개의 \(QK^\top\) 공간만으로는 이 관계들을 동시에 잘 담기 어렵다.  
Multi-Head는 **표현 공간을 \(h\)개의 부분 공간으로 나누고**, 각 부분 공간이 서로 다른 Attention 패턴을 학습하도록 만든다.

LLM 관점에서는 더 직접적이다.

```text
토큰 임베딩 (d_model)
  → Multi-Head Self-Attention (+ Causal Mask)
    → Residual + Norm
      → FFN
        → Residual + Norm
          → 다음 블록 …
```

Transformer Block의 첫 핵심 연산이 바로 MHA다.  
제46강에서 블록을 조립할 때, 오늘은 그 안쪽의 “Attention 엔진”을 완성한다.

### 3. 먼저 알아야 할 개념

이미 알고 있어야 하는 것:

- Query / Key / Value (제36강)
- Dot-Product Attention과 Softmax (제37~38강)
- Self-Attention 구현과 shape (제39강)
- Causal Mask: 미래 토큰 score를 \(-\infty\)로 가리기 (제40강)
- 행렬곱과 reshape/transpose (제8~10강, 제19강)

이번 강의에서 새로 고정할 기호:

| 기호 | 의미 |
|---|---|
| \(d_{\text{model}}\) | 토큰 벡터 차원 (모델 폭) |
| \(h\) | head 개수 |
| \(d_k\) | 각 head의 Key/Query 차원. 보통 \(d_{\text{model}}/h\) |
| \(d_v\) | 각 head의 Value 차원. 보통 \(d_k\)와 같게 둠 |
| \(W^Q, W^K, W^V\) | 전체 입력에서 Q/K/V를 만드는 선형 변환 |
| \(W^O\) | head 출력을 합친 뒤 다시 \(d_{\text{model}}\)로 섞는 출력 투영 |

### 4. 핵심 개념 설명

#### 4.1 Single-Head를 한 줄로 복습

입력 행렬 \(X \in \mathbb{R}^{T \times d_{\text{model}}}\)가 있을 때,

$$
Q = X W^Q,\quad K = X W^K,\quad V = X W^V
$$

$$
\mathrm{Attention}(Q,K,V)
=
\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V
$$

여기서 \(T\)는 시퀀스 길이, \(M\)은 Causal Mask(필요 시)이다.

Single-Head에서는 보통 \(d_k = d_{\text{model}}\)로 두고 한 번에 계산한다.

#### 4.2 Multi-Head의 정의

**Multi-Head Attention**은 동일한 \(X\)에서 **서로 다른 선형 변환**으로 \(h\)개의 \((Q_i, K_i, V_i)\)를 만들고, 각 head의 Attention 출력을 이어 붙인 뒤 다시 선형 변환한다.

$$
\mathrm{head}_i
=
\mathrm{Attention}(X W_i^Q,\ X W_i^K,\ X W_i^V)
$$

$$
\mathrm{MultiHead}(X)
=
\mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)\, W^O
$$

각 \(W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}\)이고, \(W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}\)이다.

실무 구현에서는 \(h\)개의 작은 행렬을 따로 두지 않고, **큰 \(W^Q\) 한 장**으로 \(Q\) 전체를 만든 뒤 head 축으로 reshape하는 방식이 흔하다. 수학적으로는 같다.

#### 4.3 차원 설계: \(d_k = d_{\text{model}} / h\)

원 논문(Vaswani et al., 2017)의 핵심 설계 포인트는 다음과 같다.

> head를 늘려도, 전체 계산량·파라미터를 Single-Head와 비슷한 수준으로 유지한다.

방법:

$$
d_k = d_v = \frac{d_{\text{model}}}{h}
$$

예:

- \(d_{\text{model}} = 512\), \(h = 8\) → \(d_k = 64\)
- \(d_{\text{model}} = 768\), \(h = 12\) → \(d_k = 64\) (GPT-2 small류)
- \(d_{\text{model}} = 4096\), \(h = 32\) → \(d_k = 128\) (대형 모델 예시 스케일)

주의: 위 숫자는 **구조를 익히기 위한 대표 값**이다. 실제 공개 모델마다 head 수·head 차원은 다를 수 있다. “모든 LLM이 반드시 이 숫자”가 아니다.

왜 나누는가?

1. **표현력**: 서로 다른 head가 서로 다른 부분 공간을 본다.
2. **비용 제어**: head마다 \(d_k\)가 작아져, Softmax Attention의 내적·가중합 비용이 Single-Head full-\(d_{\text{model}}\)와 비슷해진다.
3. **안정성**: \(\sqrt{d_k}\) 스케일링에서 \(d_k\)가 작을수록 Softmax 입력이 덜 극단적으로 커질 여지가 있다(완전 보장은 아니다).

#### 4.4 Concat과 \(W^O\)의 역할

각 head의 출력은 보통 shape \((T, d_v)\)이다.  
\(h\)개를 이어 붙이면 \((T, h \cdot d_v)\)가 된다.  
\(d_v = d_{\text{model}}/h\)이면 \(h \cdot d_v = d_{\text{model}}\)이므로, concat 직후 차원이 다시 \(d_{\text{model}}\)과 같아진다.

그렇다면 \(W^O\)는 왜 필요한가?

- Concat만 하면 head별 결과가 **단순 나란히 붙은 상태**다.
- \(W^O\)는 head들이 찾은 정보를 **하나의 통합 표현**으로 섞는다.
- Residual Connection(제44강)에 더해지기 전에, 블록 입력과 같은 “언어”로 맞추는 역할도 한다.

비유:

```text
h명의 전문가가 각자 메모 (d_v 차원)
  → 메모를 한 장으로 이어 붙임
    → 편집장(W^O)이 최종 요약문(d_model)으로 재작성
```

#### 4.5 Causal Mask와의 관계

제40강 Causal Mask는 **시간(위치) 축**의 규칙이다, head 축의 규칙이 아니다.

$$
M_{ij} =
\begin{cases}
0 & i \ge j \
-\infty & i < j
\end{cases}
$$

(행 \(i\) = Query 위치, 열 \(j\) = Key 위치. 구현에 따라 부호·방향 표기가 다를 수 있으므로, “미래 Key를 차단”만 기억해도 된다.)

Multi-Head에서는:

- 모든 head가 **같은 Causal Mask**를 공유하는 것이 일반적이다.
- head마다 다른 마스크를 쓰지 않는다(표준 Causal LM 기준).
- 구현상 mask를 `(1, 1, T, T)` 또는 `(1, T, T)`로 브로드캐스트해 `(B, h, T, T)` score에 더한다.

즉,

```text
제40강: “미래의 토큰을 보지 마라”
제41강: “그 규칙을 h개 head 모두에 적용하고, 결과를 합쳐라”
```

제48강 Causal LM은 이 MHA + Causal Mask가 층층이 쌓인 결과물이다.

### 5. 직관적으로 이해하기

#### 5.1 한 눈으로 보는 것과 여러 눈으로 보는 것

Single-Head:

```text
모든 토큰 관계를 하나의 유사도 기준으로 채점
```

Multi-Head:

```text
관계 유형 A를 보는 눈
관계 유형 B를 보는 눈
…
관계 유형 H를 보는 눈
→ 나중에 합친다
```

학습이 끝나면 어떤 head는 지역(local) 패턴에, 어떤 head는 장거리 의존에 더 민감해지는 경우가 관찰되기도 한다.  
다만 **특정 head가 항상 ‘문법 head’다**처럼 단정하는 것은 위험하다. 해석은 사후 분석이지, 구조가 강제하는 법칙이 아니다.

#### 5.2 “나누었다가 합친다”의 정보 흐름

작은 그림:

```text
X  (T × d_model)
 │
 ├─ head1: Q1,K1,V1 → Attn1 (T × d_k)
 ├─ head2: Q2,K2,V2 → Attn2 (T × d_k)
 └─ headh: Qh,Kh,Vh → Attnh (T × d_k)
 │
 Concat → (T × d_model)
 │
 × W^O → (T × d_model)
```

입력 폭과 출력 폭이 같으므로, Residual로 `X + MHA(X)`를 쓰기 좋다(제44·46강).

#### 5.3 파라미터가 늘어나는 지점

큰 \(W^Q,W^K,W^V,W^O\)를 각각 \(d_{	ext{model}}	imes d_{	ext{model}}\)로 두면 Single-Head full-\(d_k\)와 파라미터 규모가 비슷하다. Multi-Head의 이득은 예산을 폭발시키는 것이 아니라 **같은 예산으로 여러 부분 공간을 쓰는 것**에 가깝다.

### 6. 수학적으로 이해하기

#### 6.1 Head별 수식

전체 투영을 한 번에 쓰고 head로 쪼개는 표기:

$$
Q = X W^Q \in \mathbb{R}^{T \times d_{\text{model}}}
$$

\(Q\)를 \(h\)개로 나누면

$$
Q = \mathrm{Concat}(Q_1,\ldots,Q_h),\quad
Q_i \in \mathbb{R}^{T \times d_k}
$$

각 head:

$$
A_i = \mathrm{softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{d_k}} + M\right)
\in \mathbb{R}^{T \times T}
$$

$$
H_i = A_i V_i \in \mathbb{R}^{T \times d_v}
$$

합치기:

$$
H = \mathrm{Concat}(H_1,\ldots,H_h) \in \mathbb{R}^{T \times (h d_v)}
$$

$$
\mathrm{MHA}(X) = H W^O \in \mathbb{R}^{T \times d_{\text{model}}}
$$

#### 6.2 배치·head를 포함한 실전 shape

배치 크기 \(B\)를 포함하면 흔한 4D 텐서는 다음과 같다.

| 텐서 | shape |
|---|---|
| `x` | `(B, T, d_model)` |
| `qkv` 투영 후 | `(B, T, 3*d_model)` 또는 Q/K/V 각각 `(B, T, d_model)` |
| head reshape | `(B, h, T, d_k)` |
| scores | `(B, h, T, T)` |
| attn output | `(B, h, T, d_v)` |
| merge | `(B, T, h*d_v)` → `(B, T, d_model)` |

`transpose` 순서를 잘못 잡으면 가장 흔한 버그가 난다.  
`(B, T, h, d_k)` → `(B, h, T, d_k)`로 바꾸는 패턴을 몸과 손에 익힌다.

### 7. 작은 숫자로 직접 계산하기

목표는 “거대한 모델”이 아니라 **split → attend → concat → \(W^O\)**를 손으로 한 바퀴 도는 것이다.

#### 7.1 설정

- \(T = 2\) (토큰 2개)
- \(d_{\text{model}} = 4\)
- \(h = 2\) → \(d_k = d_v = 2\)
- 배치 없음, Causal Mask 사용

입력:

$$
X =
\begin{bmatrix}
1 & 0 & 1 & 0 \
0 & 1 & 0 & 1
\end{bmatrix}
$$

설명을 위해, 이미 투영된 \(Q, K, V\)가 다음과 같다고 가정한다.  
(실제로는 \(X W^{Q/K/V}\)로 얻는다.)

$$
Q =
\begin{bmatrix}
1 & 0 & 1 & 0 \
0 & 1 & 0 & 1
\end{bmatrix}
,\quad
K = Q,\quad
V = Q
$$

#### 7.2 Head로 쪼개기

head1: 앞 2차원, head2: 뒤 2차원.

$$
Q_1 =
\begin{bmatrix}
1 & 0 \
0 & 1
\end{bmatrix}
,\quad
Q_2 =
\begin{bmatrix}
1 & 0 \
0 & 1
\end{bmatrix}
$$

\(K_1, V_1, K_2, V_2\)도 동일하게 둔다.

#### 7.3 Head1 score

$$
Q_1 K_1^\top
=
\begin{bmatrix}
1 & 0 \
0 & 1
\end{bmatrix}
\begin{bmatrix}
1 & 0 \
0 & 1
\end{bmatrix}
=
\begin{bmatrix}
1 & 0 \
0 & 1
\end{bmatrix}
$$

스케일: \(\sqrt{d_k} = \sqrt{2} \approx 1.414\)

$$
\frac{Q_1 K_1^\top}{\sqrt{2}}
\approx
\begin{bmatrix}
0.707 & 0 \
0 & 0.707
\end{bmatrix}
$$

Causal Mask 적용(미래 차단): 위치 0은 위치 1을 못 봄.

$$
S_1
\approx
\begin{bmatrix}
0.707 & -\infty \
0 & 0.707
\end{bmatrix}
$$

Softmax (행 단위):

- 행0: \(-\infty\) 열은 확률 0 → \([1,\ 0]\)
- 행1: \(\mathrm{softmax}([0,\ 0.707])\)

$$
\mathrm{softmax}([0, 0.707])
=
\left[
\frac{e^{0}}{e^{0}+e^{0.707}},
\frac{e^{0.707}}{e^{0}+e^{0.707}}
\right]
\approx
[0.331,\ 0.669]
$$

$$
A_1
\approx
\begin{bmatrix}
1.000 & 0.000 \
0.331 & 0.669
\end{bmatrix}
$$

$$
H_1 = A_1 V_1
\approx
\begin{bmatrix}
1.000 & 0.000 \
0.331 & 0.669
\end{bmatrix}
\begin{bmatrix}
1 & 0 \
0 & 1
\end{bmatrix}
=
\begin{bmatrix}
1.000 & 0.000 \
0.331 & 0.669
\end{bmatrix}
$$

#### 7.4 Head2

이 예에서는 \(Q_2=K_2=V_2=Q_1\)이므로 \(H_2 = H_1\)이다.  
실전에서는 투영이 달라져 head마다 다른 패턴이 나온다.  
지금은 **concat 절차**를 보는 것이 목적이다.

#### 7.5 Concat

$$
H = \mathrm{Concat}(H_1, H_2)
\approx
\begin{bmatrix}
1.000 & 0.000 & 1.000 & 0.000 \
0.331 & 0.669 & 0.331 & 0.669
\end{bmatrix}
$$

#### 7.6 \(W^O\) 적용

간단한 \(W^O = I_4\)(단위행렬)라면 출력이 곧 \(H\)다.  
조금 더 의미 있게, \(W^O\)가 head 정보를 섞도록

$$
W^O =
\begin{bmatrix}
0.5 & 0.5 & 0 & 0 \
0.5 & 0.5 & 0 & 0 \
0 & 0 & 0.5 & 0.5 \
0 & 0 & 0.5 & 0.5
\end{bmatrix}^\top
$$

처럼 둘 수도 있다. (정확한 값은 구현 예제에서 확인한다.)

핵심 메시지:

1. head 안에서 \((T, d_k)\) Attention이 돈다.
2. concat으로 \((T, d_{\text{model}})\)이 복구된다.
3. \(W^O\)가 최종 혼합을 담당한다.
4. Causal Mask는 각 head의 \((T, T)\)에 동일하게 들어간다.

### 8. 코드로 구현하기 (NumPy 스케치)

아래 코드는 교육용이다. 속도·수치 안정성·패딩 마스크까지 챙긴 상용 구현은 아니다.

```python
# mha_numpy.py
"""교육용 Multi-Head Attention (NumPy)."""

from __future__ import annotations

import numpy as np

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def causal_mask(T: int) -> np.ndarray:
    """True인 곳이 차단(미래). shape (T, T)."""
    return np.triu(np.ones((T, T), dtype=bool), k=1)


def split_heads(x: np.ndarray, n_heads: int) -> np.ndarray:
    """(B, T, C) -> (B, H, T, D)."""
    B, T, C = x.shape
    assert C % n_heads == 0
    D = C // n_heads
    # (B, T, H, D) -> (B, H, T, D)
    return x.reshape(B, T, n_heads, D).transpose(0, 2, 1, 3)


def merge_heads(x: np.ndarray) -> np.ndarray:
    """(B, H, T, D) -> (B, T, H*D)."""
    B, H, T, D = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, T, H * D)


def mha_forward(
    x: np.ndarray,
    Wq: np.ndarray,
    Wk: np.ndarray,
    Wv: np.ndarray,
    Wo: np.ndarray,
    n_heads: int,
    use_causal: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    x:  (B, T, C)
    W*: (C, C)  — 간단히 d_k*h = C 가정
    return: out (B, T, C), attn (B, H, T, T)
    """
    B, T, C = x.shape
    D = C // n_heads

    q = split_heads(x @ Wq, n_heads)  # (B,H,T,D)
    k = split_heads(x @ Wk, n_heads)
    v = split_heads(x @ Wv, n_heads)

    scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(D)  # (B,H,T,T)

    if use_causal:
        mask = causal_mask(T)  # (T,T)
        scores = np.where(mask[None, None, :, :], -1e9, scores)

    attn = softmax(scores, axis=-1)
    heads = attn @ v  # (B,H,T,D)
    concat = merge_heads(heads)  # (B,T,C)
    out = concat @ Wo
    return out, attn


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    B, T, C, H = 1, 2, 4, 2
    x = rng.normal(size=(B, T, C))
    Wq = rng.normal(size=(C, C)) * 0.1
    Wk = rng.normal(size=(C, C)) * 0.1
    Wv = rng.normal(size=(C, C)) * 0.1
    Wo = rng.normal(size=(C, C)) * 0.1

    out, attn = mha_forward(x, Wq, Wk, Wv, Wo, n_heads=H)
    print("out", out.shape)      # (1, 2, 4)
    print("attn", attn.shape)    # (1, 2, 2, 2)
    print("attn[0,0]\n", np.round(attn[0, 0], 3))
```

확인 포인트:

- `attn[b, h, i, j]`: 배치 b, head h에서 query i가 key j에 둔 무게
- Causal이면 `j > i` 위치의 확률이 0에 가까워야 한다
- `out.shape`가 입력과 같아야 Residual에 더하기 쉽다

### 9. PyTorch로 구현하기

```python
# mha_torch.py
"""교육용 Multi-Head Attention (PyTorch)."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # 한 번에 QKV를 만들거나, 분리해도 된다.
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, T, H, D) -> (B, H, T, D)
        def shape(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        q, k, v = shape(q), shape(k), shape(v)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if causal:
            # True = 가릴 위치
            mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(mask[None, None, :, :], float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        heads = attn @ v  # (B, H, T, D)
        concat = heads.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(concat)


if __name__ == "__main__":
    mha = MultiHeadAttention(d_model=8, n_heads=4)
    x = torch.randn(2, 5, 8)
    y = mha(x, causal=True)
    print(y.shape)  # torch.Size([2, 5, 8])
```

`nn.MultiheadAttention`을 쓸 수도 있다.  
다만 이 책은 **내부 reshape·mask·\(W^O\)**를 직접 보는 것이 목적이므로, 위 스케치를 먼저 이해한다.

### 10. 실제 LLM에서는 어떻게 사용하는가

#### 10.1 GPT류 (Decoder-only)

- Self-Attention + **Causal Mask**가 기본
- 거의 항상 Multi-Head (또는 Multi-Query / Grouped-Query 같은 변형)
- 위치 정보는 제42~43강의 PE/RoPE와 결합

사실:

- GPT-2, GPT-3, LLaMA 계열 등은 **decoder-only Transformer**에서 Multi-Head(또는 그 후속 변형) Self-Attention을 사용한다.

설명(해석):

- “head가 각각 역할을 분담한다”는 유용한 직관이지만, 모든 모델·모든 층에서 깔끔히 역할이 분리된다고 보장되지는 않는다.

#### 10.2 변형을 미리 적어 두기

이후에 MQA·GQA처럼 KV head를 공유하는 변형을 만난다. 지금은 MHA만 정확히 알고, 변형은 추론 최적화(5권)로 미룬다.

#### 10.3 제48강으로 가는 다리

```text
제40강 Causal Mask
  → 제41강 Multi-Head로 병렬화
    → 제42~43강 위치 정보
      → 제44~45강 Norm/Residual/FFN
        → 제46강 Block 조립
          → 제47강 Encoder/Decoder 지형도
            → 제48강 Causal LM 전체 구조
```

오늘 만든 `MHA(x, causal=True)`는 제48강에서 **층의 심장**으로 다시 등장한다.

### 11. 실습

1. \(d_{\text{model}}=8,\ h=2\)와 \(h=4\)로 같은 입력에 MHA를 돌려 `out.shape`가 동일한지 확인하라.
2. Causal를 켠 뒤, `attn[0, 0]` 상삼각(미래)이 0에 가까운지 출력하라.
3. `n_heads`가 `d_model`의 약수가 아닐 때 assert가 뜨는지 확인하라.

### 12. 자주 하는 실수

1. **`d_model % n_heads != 0`**  
   head 차원이 정수가 되지 않는다. 설계 단계에서 약수로 맞춘다.

2. **reshape 순서 오류**  
   `(B, T, H, D)`와 `(B, H, T, D)`를 혼동하면 Attention이 의미 없는 축에서 돈다.

3. **스케일을 \(\sqrt{d_{\text{model}}}\)로 나누기**  
   맞혀야 하는 값은 보통 \(\sqrt{d_k}\)이다.

4. **Mask를 head마다 다르게 만들기**  
   표준 Causal LM에서는 공유한다.

5. **Concat 후 \(W^O\) 생략**  
   차원이 맞아도 표현 혼합이 약해진다. 구조상 \(W^O\)를 두는 편이 일반적이다.

6. **Single-Head 결과와 ‘평균’을 비교하려 하기**  
   Multi-Head는 단순 평균이 아니라 부분 공간 병렬 + 출력 투영이다.

### 13. 핵심 정리

- Multi-Head Attention은 Self-Attention을 \(h\)개 부분 공간에서 병렬로 수행한다.
- 보통 \(d_k = d_{\text{model}} / h\)로 두어 비용을 제어한다.
- 각 head 출력을 concat한 뒤 \(W^O\)로 \(d_{\text{model}}\) 표현을 만든다.
- Causal Mask는 제40강과 동일한 “미래 차단” 규칙을 모든 head에 적용한다.
- shape `(B, H, T, D)`를 안정적으로 다루는 것이 구현의 핵심이다.
- LLM의 Transformer Block에서 MHA는 토큰 간 정보 혼합의 중심이다.

### 14. 핵심 용어

| 용어 | 설명 |
|---|---|
| Multi-Head Attention (MHA) | 여러 Attention head를 병렬 수행 후 결합하는 메커니즘 |
| Head | \(d_k\) 차원의 부분 Attention 경로 하나 |
| \(d_{\text{model}}\) | 모델의 토큰 벡터 차원 |
| \(d_k, d_v\) | head별 Query/Key, Value 차원 |
| Concat | head 출력을 마지막 차원으로 이어 붙임 |
| \(W^O\) (Output Projection) | concat 결과를 통합하는 선형층 |
| Causal Mask | 미래 토큰을 보지 못하게 하는 마스크(제40강) |

### 15. 복습 문제

**문제 1.** \(d_{\text{model}}=768\), \(h=12\)일 때 \(d_k\)는?

**문제 2.** 왜 head를 늘리면서 \(d_k\)를 나누는가? 한 가지 이유를 쓰라.

**문제 3.** Causal Mask는 head마다 다른가, 같은가? (표준 Causal LM 기준)

**문제 4.** `(B, T, C)`를 head로 나눌 때 자주 쓰는 4D shape는?

**문제 5.** Concat만 하고 \(W^O\)가 없다면 무엇이 부족한가?

**문제 6.** 제40강과 제41강의 관계를 한 문장으로 쓰라.

#### 정답과 해설

1. \(d_k = 768 / 12 = 64\).

2. 예: 전체 계산량/파라미터를 Single-Head와 비슷한 규모로 유지하면서, 여러 부분 공간의 관계를 학습하기 위해서.

3. 같다. 미래 차단은 위치 규칙이며 head 규칙이 아니다.

4. `(B, h, T, d_k)` (또는 동등한 배치 배치).

5. head별 정보를 하나의 통합 표현으로 섞는 학습 가능한 혼합이 부족하다. Residual에 들어가기 전 표현을 재구성하는 역할도 약해진다.

6. 제40강이 만든 미래 차단 규칙을, 제41강이 \(h\)개 head에 동시에 적용하고 concat+\(W^O\)로 합친다.

### 16. 다음 강의와 연결

MHA는 “토큰 사이 관계를 여러 눈으로 본다”.  
그런데 아직 **토큰의 순서**를 모델이 본질적으로 알지는 못한다. Attention 자체는 집합에 가까운 연산이라, 위치 정보가 빠지면 “누가 먼저인지”가 약해진다.

제42강 **Positional Encoding**에서 사인·코사인으로 위치를 더하는 고전적 방법을 배운다.  
제43강에서는 현대 LLM이 많이 쓰는 **RoPE**로 이어진다.

```text
이번 강: 여러 head로 무엇을 볼지
다음 강: 그 토큰이 어디에 있는지
그다음: 회전으로 상대 위치를 심기 (RoPE)
…
제48강: Causal LM 전체 지도
```

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [제40강. Causal Mask](40강_Causal_Mask.md)
- **다음 강:** [제42강. Positional Encoding](42강_Positional_Encoding.md)

<!-- /LECTURE_NAV -->
