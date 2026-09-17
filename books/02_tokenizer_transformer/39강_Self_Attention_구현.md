# 39강. Self-Attention 구현
## 이번 강에서 배우는 내용

- Self-Attention의 순전파를 함수/클래스로 쓰기
- NumPy로 끝까지 구현하고 Shape을 검증하기
- PyTorch `nn.Module` 스케치로 옮기기
- (선택) 출력 투영 $W_O$까지 포함한 단일 헤드 블록
- Causal Mask를 넣을 위치만 표시해 40강으로 연결하기

## 왜 중요한가?
LLM 코드베이스를 열면 Attention은 최적화된 fused kernel·FlashAttention·테넌트 설정에 가려져 있다.  
그 전에 **느리고 명확한 참조 구현**이 있어야, 최적화본이 같은 일을 하는지 검증할 수 있다.

또한 49~50강 Mini Transformer 프로젝트의 핵심 부품이 바로 오늘 모듈이다.

## 선수 개념
- Q/K/V 투영 (36강)
- Scaled scores (37강)
- Softmax Attention (38강)
- `nn.Module` 감각 (1권 21강)
- Shape: `(B, T, C)` 또는 `(T, C)`

아직 깊이 넣지 않는 것:

- Multi-Head의 reshape (41강)
- RoPE (43강)
- FlashAttention 알고리즘 (5권)

## 핵심 개념
### 3.1 Self-Attention 모듈이 할 일

입력 $X \in \mathbb{R}^{B\times T\times d_{\mathrm{model}}}$에 대해:

1. $Q=XW_Q,\ K=XW_K,\ V=XW_V$
2. $S=QK^\top/\sqrt{d_k}$
3. (옵션) mask 적용
4. $A=\mathrm{softmax}(S)$
5. $O=AV$
6. (옵션) $O\leftarrow O W_O$

단일 헤드에서는 흔히 $d_k=d_v=d_{\mathrm{model}}$로 둔다.  
Multi-Head는 41강에서 $d_k=d_{\mathrm{model}}/h$로 나눈다.

### 3.2 Self의 의미 재확인

Q/K/V가 **동일 입력 시퀀스**에서 나온다.  
Cross-Attention이면 K/V의 출처가 달라질 수 있다. 인터페이스만 조금 바꾸면 된다.

### 3.3 마스크 훅

함수 시그니처에 `mask=None`을 미리 둔다.

```text
S = scores(Q, K)
S = apply_mask(S, mask)   # 40강에서 구현
A = softmax(S)
```

오늘은 `mask=None` 경로를 완성한다.

## 직관적으로 이해하기 — 레고 조립
| 조각 | 강의 | 코드 함수 |
|---|---|---|
| 투영 | 36 | `project_qkv` |
| 점수 | 37 | `attention_scores` |
| Softmax+V | 38 | `softmax`, `A@V` |
| 조립 | 39 (오늘) | `SelfAttention.forward` |

각 함수를 따로 테스트한 뒤 조립하면 디버깅이 쉽다.

## 수학적으로 이해하기 — 단일 헤드
$$

\begin{aligned}
Q,K,V &= XW_Q,\ XW_K,\ XW_V \\
S &= \frac{QK^\top}{\sqrt{d_k}} \\
A &= \mathrm{softmax}(S) \\
O &= AV W_O \quad (\text{또는 } AV)
\end{aligned}

$$

잔차 $X+O$, LayerNorm은 44·46강에서 블록 조립 시 붙인다.  
오늘은 Attention 서브층만.

## 작은 숫자로 모듈 입출력 검증
$B=1,T=2,d=2$, $W_Q=W_K=I$, $W_V=I$, $W_O=I$.

$$

X=\begin{bmatrix}1&0\\0&1\end{bmatrix}

$$

이면 38강/37강과 같은 경로로

$$

S=I/\sqrt{2},\quad
A=\mathrm{softmax}(S)

$$

손계산:

행 점수 $[0.7071,\ 0]$

$$

\alpha \approx [0.6682,\ 0.3318]

$$

(대칭으로 행1은 `[0.3318, 0.6682]`).

$V=X$이므로

$$

O = A X \approx \begin{bmatrix}0.6682&0.3318\\0.3318&0.6682\end{bmatrix}

$$

구현이 이 숫자에 가까우면 조립이 맞은 것이다.

## 코드로 구현하기
```python
# lecture39_self_attention_numpy.py
"""단일 헤드 Self-Attention 참조 구현 (NumPy)."""

from __future__ import annotations

import numpy as np

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    # 안정 Softmax
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

class SelfAttentionNumpy:
    def __init__(self, d_model: int, d_k: int | None = None, d_v: int | None = None,
                 use_out_proj: bool = True, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.d_model = d_model
        self.d_k = d_k or d_model
        self.d_v = d_v or d_model
        # 작은 초기값 — 폭주 방지용 스케일
        scale = 0.02
        self.W_Q = rng.normal(0, scale, size=(d_model, self.d_k))
        self.W_K = rng.normal(0, scale, size=(d_model, self.d_k))
        self.W_V = rng.normal(0, scale, size=(d_model, self.d_v))
        self.use_out_proj = use_out_proj
        if use_out_proj:
            self.W_O = rng.normal(0, scale, size=(self.d_v, d_model))
        else:
            self.W_O = None

    def project(self, X: np.ndarray):
        # X: (B, T, d_model) 또는 (T, d_model)
        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V
        return Q, K, V

    def scores(self, Q: np.ndarray, K: np.ndarray) -> np.ndarray:
        d_k = Q.shape[-1]
        KT = np.swapaxes(K, -1, -2)
        return (Q @ KT) / np.sqrt(d_k)

    def apply_mask(self, S: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        """mask: True/1 위치가 가려질 위치 (40강과 맞출 수 있음). 오늘은 None."""
        if mask is None:
            return S
        # 가릴 위치에 -1e9 (float32 안전) — 40강에서 -inf 논의
        S = np.array(S, copy=True)
        S = np.where(mask, -1e9, S)
        return S

    def forward(self, X: np.ndarray, mask: np.ndarray | None = None,
                return_weights: bool = False):
        Q, K, V = self.project(X)
        S = self.scores(Q, K)
        S = self.apply_mask(S, mask)
        A = softmax(S, axis=-1)
        O = A @ V
        if self.W_O is not None:
            O = O @ self.W_O
        if return_weights:
            return O, A, S
        return O

def demo_hand_numbers():
    """7절 손계산과 같은 고정 가중치."""
    sa = SelfAttentionNumpy(d_model=2, use_out_proj=False, seed=0)
    sa.W_Q = np.eye(2)
    sa.W_K = np.eye(2)
    sa.W_V = np.eye(2)
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    O, A, S = sa.forward(X, return_weights=True)
    print("S:\n", np.round(S, 4))
    print("A:\n", np.round(A, 4))
    print("O:\n", np.round(O, 4))

def demo_batch():
    sa = SelfAttentionNumpy(d_model=8, d_k=8, d_v=8, seed=1)
    X = np.random.default_rng(1).normal(size=(4, 5, 8))  # B=4,T=5
    O = sa.forward(X)
    assert O.shape == (4, 5, 8)
    print("batch OK", O.shape)

if __name__ == "__main__":
    demo_hand_numbers()
    demo_batch()
```

실행해 7절 근사치와 맞는지 본다.

## PyTorch로 구현하기 (스케치)
```python
# lecture39_self_attention_torch.py
"""단일 헤드 Self-Attention (PyTorch 스케치)."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model: int, d_k: int | None = None, d_v: int | None = None,
                 bias: bool = False):
        super().__init__()
        d_k = d_k or d_model
        d_v = d_v or d_model
        self.d_k = d_k
        self.W_Q = nn.Linear(d_model, d_k, bias=bias)
        self.W_K = nn.Linear(d_model, d_k, bias=bias)
        self.W_V = nn.Linear(d_model, d_v, bias=bias)
        self.W_O = nn.Linear(d_v, d_model, bias=bias)

    def forward(self, X: torch.Tensor, mask: torch.Tensor | None = None):
        # X: (B, T, d_model)
        # mask: broadcastable to (B, T, T) or (1, T, T); True=가림 (선택 규약)
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)

        S = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # 40강: masked_fill(-inf)
            S = S.masked_fill(mask, float("-inf"))

        A = F.softmax(S, dim=-1)
        # mask로 행 전체가 -inf면 nan 가능 → 40강에서 다루거나 보호 코드 추가
        O = A @ V
        return self.W_O(O)

class SelfAttentionDebug(SelfAttention):
    """가중치까지 반환하는 학습용 서브클래스."""

    def forward(self, X, mask=None):
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)
        S = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            S = S.masked_fill(mask, float("-inf"))
        A = F.softmax(S, dim=-1)
        O = self.W_O(A @ V)
        return O, A, S

if __name__ == "__main__":
    torch.manual_seed(0)
    m = SelfAttention(d_model=16)
    X = torch.randn(2, 7, 16)
    Y = m(X)
    print(Y.shape)  # (2,7,16)

    # 고정 단위 테스트용 작은 모델 (identity 가중치)
    dbg = SelfAttentionDebug(d_model=2, bias=False)
    with torch.no_grad():
        dbg.W_Q.weight.copy_(torch.eye(2))
        dbg.W_K.weight.copy_(torch.eye(2))
        dbg.W_V.weight.copy_(torch.eye(2))
        dbg.W_O.weight.copy_(torch.eye(2))
    X2 = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    O2, A2, S2 = dbg(X2)
    print("S2", S2)
    print("A2", A2)
    print("O2", O2)
```

`nn.Linear`의 weight shape는 `(out, in)`이다.  
`copy_(torch.eye)`는 `y = x @ W.T` 규약과 맞물려 identity 동작을 만든다.

## LLM에서는 어디에 사용될까?
### 9.1 블록 안의 위치

```text
X
 → Self-Attention (+ Causal Mask)
 → Dropout (선택)
 → Residual: X + AttnOut
 → Norm (위치는 Pre/Post에 따라 다름)
 → FFN
 → Residual + Norm
```

### 9.2 Multi-Head로의 확장

오늘 모듈의 `d_k`를 줄이고 헤드 차원으로 reshape하면 MHA가 된다 (41강).  
점수·Softmax·가중합 원리는 동일하다.

### 9.3 추론 최적화

참조 구현은 $T\times T$를 명시적으로 만든다.  
실서비스는 fused kernel로 Softmax·마스킹을 합치거나 타일링한다.  
수치는 같아야 한다.

## 실습
### 실습 1 — Shape 테스트

`d_model=32`, `T=10`, `B=3`으로 NumPy 모듈을 돌려 출력 Shape을 확인하시오.

### 실습 2 — 손계산 대조

7절 설정으로 `demo_hand_numbers`를 실행해 $A$ 근사치를 확인하시오.

### 실습 3 — identity 투영

$W_Q=W_K=W_V=I$일 때 $S=XX^\top/\sqrt{d}$가 되는지 임의 $X$로 검사하시오.

### 실습 4 — PyTorch 동등성

같은 가중치를 NumPy와 PyTorch에 복사해 출력 차이가 $10^{-5}$ 이하인지 보시오.

### 실습 5 — return_weights

문장 길이 4짜리 난수 입력에서 $A[0]$ 행 합이 1인지 확인하시오.

### 실습 6 — 마스크 자리

`apply_mask`에 “상삼각을 가리는” 마스크를 넣어 보기만 하시오. 값은 40강에서 해석한다.

## 자주 하는 실수
1. **`(T,T)`와 `(T,d)` 곱 순서 오류**  
   `A@V`가 맞다. `V@A`는 관례가 다르면 바로 깨진다.

2. **Linear weight를 NumPy `X@W`와 혼동**  
   PyTorch는 `X @ W.T` (+bias). 이식 시 전치 확인.

3. **배치 없이만 테스트**  
   `swapaxes`/`transpose(-2,-1)` 버그는 3D에서 터진다. 반드시 배치 테스트.

4. **softmax dim 누락**  
   기본값이 기대한 축이 아닐 수 있다. 명시한다.

5. **잔차까지 한 모듈에 몰아넣고 실패 지점을 모름**  
   Attention만 먼저 맞춘다.

6. **mask 규약(True=가림 vs True=허용) 혼동**  
   팀/프레임워크마다 다르다. docstring에 고정한다.

## 핵심 요약
- Self-Attention 모듈 = QKV 투영 + scaled scores + (mask) + Softmax + AV + (W_O).
- NumPy 참조 구현으로 Shape·손계산을 고정한 뒤 PyTorch로 옮긴다.
- `mask` 인자를 미리 두면 Causal LM으로 확장하기 쉽다.
- 단일 헤드 완성이 Multi-Head·Transformer Block의 기초다.
- 느린 명확한 코드가 최적화 코드의 정답지다.

## 용어 사전
| 용어 | 의미 |
|---|---|
| Self-Attention Module | 입력이 곧 QKV 출처인 Attention 블록 |
| Output Projection $W_O$ | Attention 출력을 모델 차원으로 재투영 |
| Reference implementation | 검증용 명확 구현 |
| Forward | 순전파 |
| Mask hook | Softmax 전 점수를 수정하는 자리 |
| `(B,T,C)` | 배치·시간·채널 Shape |

## 연습문제
### 문제 1 (순서)

Self-Attention 순전파 단계를 5단계로 쓰시오.

### 문제 2 (Shape)

`(2, 9, 64)` 입력, `d_k=d_v=64`, `W_O` 포함 시 출력 Shape은?

### 문제 3 (개념)

왜 Softmax 전에 mask를 적용하는가?

### 문제 4 (코드)

NumPy `X @ W_Q`와 `nn.Linear`의 대응을 한 문장으로.

### 문제 5 (연결)

40강에서 채울 `mask`는 GPT에서 무엇을 막는가?

---

## 정답 및 해설
### 문제 1

투영 → 점수 → (마스크) → Softmax → AV → (출력을 투영).

### 문제 2

`(2, 9, 64)`.

### 문제 3

가린 위치가 Softmax 정규화에 참여하지 않게(가중치 0) 하기 위함이다. Softmax 후에 0으로 지우고 재정규화하지 않으면 합이 1이 깨진다.

### 문제 4

같은 선형 변환이지만 PyTorch weight는 `(out,in)`이라 실질 `X @ W.T` (+bias)에 해당한다.

### 문제 5

미래 토큰(아직 예측하면 안 되는 위치)을 보는 것을 막는다.

## 다음 강의와 연결
모듈에 `mask` 자리가 비어 있다.  
다음 **제40강. Causal Mask**에서 삼각형 마스크로 미래를 가리고, GPT가 왜 그것이 필요한지, `masked_fill(-inf)`를 작은 예제로 확인한다.  
이어 **제41강. Multi-Head Attention**에서 헤드를 나눈다.

> 조립이 끝났으면, 이제 언어모델용으로 “앞만 보기” 규칙을 넣자.

---

## 부록 A. 타입·dtype 권장

- 학습: float32 기본, 혼합정밀도는 이후
- 점수에 `-1e9` vs `-inf`: float16에서 `-1e9`가 더 안전한 경우가 있다
- 손계산 대조는 float64 NumPy가 편하다

## 부록 B. Dropout 위치 (예고)

Attention weight dropout (`A`에)과 residual dropout이 문헌·구현에 등장한다.  
오늘은 생략. 프로젝트 단계에서 필요하면 추가한다.

## 부록 C. 파라미터 수

단일 헤드, bias 없음, $d_k=d_v=d$:

$$

|\theta| \approx 4 d^2

$$

($W_Q,W_K,W_V,W_O$). Multi-Head도 보통 같은 오더를 유지하도록 설계한다 (41강).

## 부록 D. Cross-Attention으로 최소 수정

```python
def forward_cross(self, X_q, X_kv, mask=None):
    Q = X_q @ self.W_Q
    K = X_kv @ self.W_K
    V = X_kv @ self.W_V
    ...
```

Encoder–Decoder에서 쓴다. GPT 기본 경로는 Self.

## 부록 E. 단위 테스트 목록

1. Shape 보존 `(B,T,d)->(B,T,d)`
2. 행 합 1
3. identity 가중치 손계산
4. mask=None과 “전부 False mask” 동일
5. gradient smoke test (PyTorch `O.sum().backward()`)

## 부록 F. 흔한 디버그 print

```python
print(Q.shape, K.shape, V.shape, S.shape, A.shape, O.shape)
print(A.sum(axis=-1))
print(S[0,0])
```

Shape만 맞춰도 버그의 절반이 사라진다.

## 부록 G. einsum 버전 (선택)

```python
S = np.einsum("...id,...jd->...ij", Q, K) / np.sqrt(d_k)
O = np.einsum("...ij,...jd->...id", A, V)
```

가독성 대체재. 성능은 행렬곱 GEMM이 보통 낫다.

## 부록 H. Mini Transformer로 가는 길

```text
39 Self-Attention
40 Causal Mask
41 Multi-Head
42–43 Position
44–45 Norm, FFN
46 Block
48 Causal LM
49–50 Project
```

오늘이 척추다.

## 부록 I. 손계산 $A$ 더 정밀히

$s=[1/\sqrt{2}, 0]$:

$$

\begin{aligned}
e^{0.70710678}&\approx 2.028757\\
e^{0}&=1\\
\sum&\approx 3.028757\\
\alpha&\approx [0.66998,\ 0.33002]
\end{aligned}

$$

7절의 0.668은 반올림 차이. 코드와 맞출 때는 이 값을 기준으로 한다.

## 부록 J. `nn.MultiheadAttention`과의 관계

PyTorch 내장 MHA는 오늘+41강+마스크를 한 모듈에 담는다.  
학습 목적상 우리는 분해 구현을 먼저 한다. 이후 내장 API와 수치를 비교하면 좋다.

## 부록 K. 메모리 추정 (아주 거친)

점수 $A,S$만 $B\cdot T\cdot T\cdot 4$바이트(float32).  
$B=8,T=2048$이면 약 $8\times2048^2\times4 \approx 134$MB 수준(하나당).  
헤드·층이 쌓이면 커진다 → 52강.

## 부록 L. 시퀀스 길이 1

$T=1$이면 $S$는 $1\times1$, Softmax는 `[1]`, $O=v_0$ (출력을 투영 전).  
단위 테스트 코너케이스로 유용하다.

## 부록 M. 코드 스타일 체크리스트

- 매직넘버 $\sqrt{d_k}$에 주석
- mask 규약 주석
- forward에 shape docstring
- 학습용 `return_weights` 플래그

## 부록 N. 연습 — 빈 구현 채우기

학생용 스케레톤:

```python
def forward(self, X, mask=None):
    Q = ???
    K = ???
    V = ???
    S = ???
    S = self.apply_mask(S, mask)
    A = ???
    O = ???
    return O @ self.W_O
```

이 강의 본문대로 채운다.

## 부록 O. 다음 강 미리보기 도식

```text
S_ij = q_i · k_j / sqrt(d_k)
if j > i:  S_ij = -inf     # Causal (GPT)
A = softmax(S)
O = A V
```

`j > i` 규칙이 40강이다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [38강. Softmax Attention과 작은 숫자 예제](38강_Softmax_Attention과_작은_숫자_예제.md)
- **다음 강:** [40강. Causal Mask](40강_Causal_Mask.md)

<!-- /LECTURE_NAV -->
