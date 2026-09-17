# 제40강. Causal Mask

> **학습 목표**
> - Causal / Triangular Mask가 무엇인지
> - 왜 Next-Token LM이 미래를 보면 안 되는지
> - Softmax 전 점수에 `-inf`를 넣는 이유 (`masked_fill`)
> - 작은 $T=3,4$ 예제로 마스크된 Softmax를 손계산하기
> - NumPy/PyTorch로 causal mask를 구현하기
> - 41강 Multi-Head로 넘어가며 마스크가 헤드에 공유됨을 예고하기

---
## 1. 왜 이것을 배우는가

학습 때 문장 전체를 병렬로 넣는다.  
마스크가 없으면 위치 $t$의 Attention이 **미래 토큰 $t+1,t+2,\ldots$**의 정보를 볼 수 있다.

그러면 모델은 “다음 토큰 예측”이 아니라 **치트**를 학습한다.  
정답이 이미 문맥에 보이기 때문이다.

```text
입력:  [BOS] 나는 밥을 먹었다
예측:   나는  밥을 먹었다 [EOS] ...
```

위치 “밥을”을 예측할 때, 마스크 없이 “먹었다”를 보면 너무 쉽다.  
Causal Mask는 **왼쪽(과거·현재)만** 보게 강제한다.

## 2. 먼저 알아야 할 개념

- Self-Attention 점수 $S$ (37강)
- Softmax Attention (38강)
- Next-Token Prediction (32강) + CE (34강)
- 상삼각/하삼각 행렬 직관
- 39강의 `mask` 훅

## 3. 핵심 개념 설명

### 3.1 Causal Mask란?

**Causal Mask(인과 마스크, look-ahead mask)**는 위치 $i$가 위치 $j>i$ (미래)를 주목하지 못하도록 막는 마스크다.

허용 조건 (0-index):

$$

\text{attend}(i,j) \iff j \le i

$$

점수 행렬에서 **엄격 상삼각** 부분을 가린다.

### 3.2 Triangular Mask

$T=4$일 때 허용 패턴 (`1=허용`, `0=차단`):

```text
      j=0 j=1 j=2 j=3
i=0    1   0   0   0
i=1    1   1   0   0
i=2    1   1   1   0
i=3    1   1   1   1
```

하삼각(대각 포함)이 허용 영역이다.  
그래서 **triangular mask**라고도 부른다.

### 3.3 `-inf`를 넣는 이유

Softmax 전:

$$

\tilde{s}_{ij} = \begin{cases}
s_{ij} & j\le i \\
-\infty & j>i
\end{cases}

$$

$$

e^{-\infty}=0

$$

이므로 미래 위치 가중치는 0이 되고, 남은 과거·현재끼리 합이 1로 재정규화된다.

유한한 큰 음수(`-1e9`)를 쓰기도 한다. float16 안정성 때문이다.  
이상적 표기는 `-inf`다.

### 3.4 `masked_fill` 패턴 (PyTorch)

```python
S = S.masked_fill(causal_mask, float("-inf"))
A = F.softmax(S, dim=-1)
```

`causal_mask`가 True인 칸이 미래(가릴 곳)가 되게 만드는 규약이 흔하다.  
(반대로 True=허용인 API도 있으니 문서를 읽는다.)

## 4. 직관적으로 이해하기

### 4.1 시험 부정행위 금지

오픈북 시험인데, **뒷페이지는 못 보게** 가린 것과 같다.  
앞 페이지만 읽고 다음 줄을 써야 한다.

### 4.2 생성 시점과의 일치

추론 시 모델은 실제로 왼쪽부터 토큰을 샘플링한다.  
학습도 같은 정보 제약을 주어야 train–test mismatch가 줄어든다.

### 4.3 Encoder와 다르다

BERT류 Encoder는 양방향 Attention(미래도 봄) + Masked LM 목표가 다르다.  
GPT류 Causal LM은 **단방향(자귀 회귀)** 이 본질이다.

## 5. 수학적으로 이해하기

### 5.1 마스크 행렬 $M$

$$

M_{ij} = \begin{cases}
0 & j\le i \\
-\infty & j>i
\end{cases}

$$

$$

A = \mathrm{softmax}(S + M)

$$

또는 `masked_fill`로 동일 효과.

### 5.2 위치 $i$의 정규화

$$

\alpha_{ij} = \frac{e^{s_{ij}}}{\sum_{j'=0}^{i} e^{s_{ij'}}} \quad (j\le i),\quad
\alpha_{ij}=0 \ (j>i)

$$

분모가 $j'=0..i$만 포함한다.

### 5.3 Padding Mask와의 결합

패딩 위치도 가려야 하면

$$

\text{최종 가림} = \text{causal} \lor \text{padding}

$$

논리합으로 합친다. 오늘은 causal만 집중.

## 6. 작은 숫자로 직접 계산하기

### 6.1 예제 A — $T=3$, 단순한 점수

마스크 전 점수 (이미 scaled라고 가정):

$$

S=\begin{bmatrix}
1.0 & 2.0 & 3.0 \\
1.0 & 1.0 & 1.0 \\
0.0 & 0.0 & 0.0
\end{bmatrix}

$$

**Causal 적용 후**

$$

\tilde{S}=\begin{bmatrix}
1.0 & -\infty & -\infty \\
1.0 & 1.0 & -\infty \\
0.0 & 0.0 & 0.0
\end{bmatrix}

$$

**행0 Softmax**

유한 점수만 `[1.0]` → $\alpha_0=[1,\ 0,\ 0]$

**행1 Softmax**

`[1,1]`, $\alpha_1=[0.5,\ 0.5,\ 0]$

**행2 Softmax**

`[0,0,0]`, $\alpha_2=[1/3,\ 1/3,\ 1/3]$

$$

A=\begin{bmatrix}
1 & 0 & 0 \\
0.5 & 0.5 & 0 \\
1/3 & 1/3 & 1/3
\end{bmatrix}

$$

해석:

- 첫 토큰은 자기만 본다.
- 둘째는 첫·둘만 반반.
- 셋째는 전체를 균등 (점수가 모두 0이라서).

마스크가 없다면 행0도 미래 점수 2,3을 보고 크게 기울어진다 — LM에 부적절.

### 6.2 예제 B — 마스크 없을 때 행0이 치트하는 모습

같은 $S$의 행0을 마스크 없이 Softmax:

점수 `[1,2,3]`, max=3

$$

\begin{aligned}
e^{-2}&\approx0.1353,\ e^{-1}\approx0.3679,\ e^{0}=1\\
\sum&\approx1.5032\\
\alpha&\approx[0.0900,\ 0.2447,\ 0.6652]
\end{aligned}

$$

미래(열2)에 66% — 다음 토큰 정보가 새어 들어온다.

### 6.3 예제 C — Value까지

$$

V=\begin{bmatrix}10&0\\0&20\\30&30\end{bmatrix}

$$

예제 A의 $A$로 $O=AV$:

$$

\begin{aligned}
\mathbf{o}_0 &= 1\cdot[10,0]=[10,0]\\
\mathbf{o}_1 &= 0.5[10,0]+0.5[0,20]=[5,10]\\
\mathbf{o}_2 &= \tfrac13[10,0]+\tfrac13[0,20]+\tfrac13[30,30]=[40/3,\ 50/3]
\end{aligned}

$$

$\mathbf{o}_0$에 미래 Value `[30,30]`이 섞이지 않았다. 마스크의 실체적 효과다.

### 6.4 예제 D — $T=2$ 최소

$$

S=\begin{bmatrix}0.0&5.0\\5.0&0.0\end{bmatrix}
\rightarrow
\tilde{S}=\begin{bmatrix}0.0&-\infty\\5.0&0.0\end{bmatrix}

$$

$$

A=\begin{bmatrix}1&0\\ \sigma(5)&\sigma(0)\end{bmatrix}

$$

여기서 $\sigma$는 두 점수의 Softmax:

$$

\alpha_{10}=\frac{e^{5}}{e^{5}+e^{0}}\approx0.9933,\quad
\alpha_{11}\approx0.0067

$$

## 7. 코드로 구현하기 (NumPy)

```python
# lecture40_causal_mask_numpy.py

import numpy as np

def causal_mask(T: int) -> np.ndarray:
    """True = 가릴 위치 (미래). Shape (T, T)."""
    # triu k=1: 엄격 상삼각
    return np.triu(np.ones((T, T), dtype=bool), k=1)

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def apply_causal(S: np.ndarray) -> np.ndarray:
    T = S.shape[-1]
    mask = causal_mask(T)
    out = np.array(S, copy=True, dtype=np.float64)
    # 배치 차원 허용: mask를 뒤 두 축에 브로드캐스트
    out[..., :, :] = np.where(mask, -np.inf, out[..., :, :])
    return out

def attention_with_causal(Q, K, V):
    d_k = Q.shape[-1]
    S = (Q @ np.swapaxes(K, -1, -2)) / np.sqrt(d_k)
    S = apply_causal(S)
    A = softmax(S, axis=-1)
    # -inf 행이 있으면 nan 가능 — 정상 causal이면 각 행에 최소 1개 유한
    O = A @ V
    return O, A, S

if __name__ == "__main__":
    print("mask T=4:\n", causal_mask(4).astype(int))

    S = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    Sc = apply_causal(S)
    A = softmax(Sc)
    print("S tilde:\n", Sc)
    print("A:\n", np.round(A, 4))

    V = np.array([[10.0, 0.0], [0.0, 20.0], [30.0, 30.0]])
    O = A @ V
    print("O:\n", np.round(O, 4))
```

손계산과 `A`, `O`가 일치해야 한다.

## 8. PyTorch로 구현하기

```python
# lecture40_causal_mask_torch.py

import math
import torch
import torch.nn.functional as F

def make_causal_mask(T: int, device=None) -> torch.Tensor:
    # True = 가림. (T, T)
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

def masked_attention(Q, K, V):
    # Q,K,V: (B, T, d)
    B, T, d_k = Q.shape
    S = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    mask = make_causal_mask(T, device=Q.device)  # (T,T)
    S = S.masked_fill(mask, float("-inf"))  # broadcast to (B,T,T)
    A = F.softmax(S, dim=-1)
    return A @ V, A, S

if __name__ == "__main__":
    torch.manual_seed(0)
    T = 3
    S = torch.tensor(
        [[[1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]]
    )
    mask = make_causal_mask(T)
    Sc = S.masked_fill(mask, float("-inf"))
    A = F.softmax(Sc, dim=-1)
    print(A)

    Q = torch.randn(2, 5, 8)
    K = torch.randn(2, 5, 8)
    V = torch.randn(2, 5, 8)
    O, A2, S2 = masked_attention(Q, K, V)
    print(O.shape)
    # 미래 가중치가 0인지
    print("upper sum", A2.triu(diagonal=1).abs().sum().item())
```

`upper sum`이 0(또는 수치상 극소)이면 causal이 작동한 것이다.

## 9. 실제 LLM에서는 어떻게 사용하는가

### 9.1 GPT / Causal LM의 기본 규칙

모든 Self-Attention 층(모든 헤드)에 causal mask를 적용한다.  
목표가 Next-Token CE(34강)이기 때문이다.

### 9.2 학습 병렬화와의 공존

마스크 덕분에 **전체 시퀀스를 한 번에** 넣어도, 각 위치가 미래 정보를 못 본다.  
RNN처럼 한 스텝씩 순차 학습할 필요가 없다(교사 강요 경로).

### 9.3 KV Cache

추론 시 과거 K/V만 캐시하는 것은, causal 구조상 미래 K/V가 애초에 없기 때문이다 (5권).

### 9.4 Prefix-LM / UniLM 등

일부 과제는 프롬프트 구간만 양방향, 생성 구간만 causal인 **부분 마스크**를 쓴다.  
원리는 동일: Softmax 전 점수에 허용 그래프를 새긴다.

## 10. 실습

### 실습 1 — 마스크 그리기

$T=5$ causal 허용 행렬(0/1)을 손으로 그리시오.

### 실습 2 — 손계산

$$

S=\begin{bmatrix}0&10\\0&0\end{bmatrix}

$$

에 causal을 적용한 뒤 Softmax $A$를 구하시오.

### 실습 3 — 코드

8절 코드로 예제 A의 $A$를 재현하시오.

### 실습 4 — 치트 비교

같은 $S$에서 마스크 있음/없음의 행0 가중치를 비교하시오.

### 실습 5 — PyTorch

`upper sum`이 0인지 배치 난수 입력으로 확인하시오.

### 실습 6 — 연결

34강의 shift target과 causal mask가 함께 만드는 “정직한 다음 토큰 학습”을 한 문장으로 설명하시오.

## 11. 자주 하는 실수

1. **하삼각/상삼각을 뒤바꾼다**  
   가려야 할 것은 $j>i$ (미래). `triu(..., diagonal=1)`이 전형적.

2. **Softmax 후에 0으로 지운다**  
   합이 1이 아니게 남거나, 재정규화를 잊는다. 반드시 Softmax **전**.

3. **`-inf` 대신 0을 넣는다**  
   Softmax(0,...)은 균등에 기여한다. 차단이 아니다.

4. **대각까지 가린다 (`diagonal=0`의 triu)**  
   자기 자신조차 못 보면 행이 전부 `-inf` → Softmax `nan`.

5. **헤드마다 다른 causal을 만들려 한다**  
   표준 LM에서는 위치 규칙이라 헤드 공유 (41강).

6. **padding mask와 곱/합 규약을 혼동**  
   bool OR로 “하나라도 가리면 가림”이 안전하다.

## 12. 핵심 정리

- Causal Mask는 위치 $i$가 $j>i$를 못 보게 한다.
- Softmax 전 점수에 `-inf`를 넣어 미래 가중치를 0으로 만든다.
- GPT형 Next-Token 학습·생성과 정보 제약을 일치시키는 장치다.
- 삼각형(하삼각 허용) 패턴으로 구현한다.
- Attention 모듈의 mask 훅에 이 규칙을 넣으면 Causal LM Self-Attention이 된다.

## 13. 핵심 용어

| 용어 | 의미 |
|---|---|
| Causal Mask | 미래 위치를 차단하는 마스크 |
| Look-ahead Mask | Causal Mask의 다른 이름 |
| Triangular Mask | 삼각 형태 허용/차단 패턴 |
| `masked_fill` | 조건 위치를 특정 값으로 채움 |
| Autoregressive / Causal LM | 왼쪽 조건으로 오른쪽을 생성하는 LM |
| Bidirectional Attention | 양방향(미래 포함) 주목 |
| Padding Mask | 패딩 토큰 차단 |

## 14. 연습 문제
### 문제 1 (패턴)

$T=3$에서 가려지는 $(i,j)$ 쌍을 모두 쓰시오.

### 문제 2 (계산)

예제 A의 $\alpha_{10},\alpha_{11},\alpha_{12}$는?

### 문제 3 (개념)

왜 학습 때 문장 전체를 넣는데도 causal이 필요한가?

### 문제 4 (코드)

`torch.triu(..., diagonal=1)`이 True인 칸의 의미는?

### 문제 5 (연결)

41강 Multi-Head에서 causal mask는 head마다 다른가?

---

## 정답 및 해설

### 문제 1

$(0,1),(0,2),(1,2)$.

### 문제 2

$0.5,\ 0.5,\ 0$.

### 문제 3

병렬 학습 중에도 각 위치가 미래 토큰을 보지 못하게 해, 추론 시 자귀 회귀와 같은 정보 제약을 유지하기 위해서.

### 문제 4

엄격 상삼각 — 보통 “미래(가릴 위치)”.

### 문제 5

같다(공유). 미래 차단은 위치 규칙이다 head별 규칙이 아니다.

## 15. 다음 강의와 연결

단일 헤드 + Causal Mask까지 끝났다.  
다음 **제41강. Multi-Head Attention**에서는 표현을 $h$개 헤드로 나누어 병렬 Attention을 수행하고, concat 후 $W_O$로 합친다.  
Causal Mask는 그 모든 헤드의 점수에 동일하게 적용된다.

뒤로 이어지는 길:

```text
41 Multi-Head
42 Positional Encoding → 43 RoPE
44–45 Norm, FFN
46 Transformer Block
48 Causal LM 구조
```

> 앞만 보는 규칙을 익혔으면, 이제 여러 시선(Multi-Head)으로 문맥을 나누어 보자.

---

## 부록 A. 마스크 생성 다른 방법

```python
i = np.arange(T)[:, None]
j = np.arange(T)[None, :]
allow = j <= i
mask_forbid = ~allow
```

`triu`와 동치. 읽기 쉬운 편.

## 부록 B. float16과 `-1e4`

일부 구현은 `-inf` 대신 `-1e4`/`-1e9`를 쓴다.  
Softmax 후 미래 가중치가 실질 0이면 충분하다.  
과도하게 작지 않은 음수는 underflow 이슈를 줄이기도 한다.

## 부록 C. nan 방어

실수로 행 전체를 `-inf`로 만들면 Softmax가 `nan`이 된다.  
대각을 가리지 않았는지 단위 테스트한다.

## 부록 D. 시각화

허용 영역을 히트맵으로 그리면 계단형 삼각형이 보인다 (51강과 연계).  
학습된 $A$도 상삼각이 비어 있어야 한다.

## 부록 E. 실습 2 답

$$

\tilde{S}=\begin{bmatrix}0&-\infty\\0&0\end{bmatrix},\quad
A=\begin{bmatrix}1&0\\0.5&0.5\end{bmatrix}

$$

## 부록 F. Attention Bias 형태

어떤 구현은 bool mask 대신 **덧셈 바이어스** 행렬(0 / -inf)을 캐시한다.  
식 $S+M$과 동일.

## 부록 G. 상대 위치·RoPE와의 관계

Causal Mask는 “누구를 볼 수 있는가”의 **이산 허용 그래프**다.  
RoPE(43강)는 “볼 때 위치에 따라 회전”하는 **연속 위치 인코딩**이다.  
둘은 대체재가 아니라 같이 쓰이는 경우가 많다.

## 부록 H. Encoder-Decoder 마스크

디코더 Self-Attention: causal.  
디코더 Cross-Attention: 인코더 길이 전체에 대해 (보통) 전부 허용 + 패딩 마스크.  
오늘 범위는 디코더 Self의 causal.

## 부록 I. 복잡도

마스크 자체는 $O(T^2)$ 표 또는 커널 내부 브로드캐스트.  
점수 행렬이 이미 $O(T^2)$이므로 점근적으로 같은 오더다.

## 부록 J. 39강 모듈에 붙이기

```python
mask = causal_mask(T)  # (T,T) True=가림
O = self_attention.forward(X, mask=mask)
```

배치면 `mask[None, :, :]`.

## 부록 K. 생성 루프와의 대응

```text
t=0: 오직 x0
t=1: x0,x1
t=2: x0,x1,x2
...
```

학습 시 행렬의 각 행이 위 제약을 동시에 표현한다.

## 부록 L. CE와 함께 보는 정합성

위치 $t$의 logits는 $x_{0:t}$만으로 만들어져야 하고, target은 $x_{t+1}$ (또는 시프트 규약에 따른 다음 토큰).  
Causal Mask가 전자를, 데이터 shift가 후자를 담당한다.

## 부록 M. 체크리스트 퀴즈

1. `diagonal=1`의 triu를 쓰는 이유  
2. Softmax 전/후  
3. 자기 자신은 보는가  
4. 양방향 모델과의 차이  
5. Multi-Head에서 공유?

답: 1) 미래만 가림 2) 전 3) 예 4) BERT류는 양방향+다른 목표 5) 예

## 부록 N. 추가 손계산 — 비균등 점수

$$

\tilde{S}=\begin{bmatrix}2&-\infty&-\infty\\0&3&-\infty\\1&1&1\end{bmatrix}

$$

행1 Softmax: `[0,3]` → $\alpha\approx[0.0474,\ 0.9526,\ 0]$  
행2: 균등 $1/3$.

코드로 재현해 본다.

## 부록 O. 한 줄 요약

> Causal Mask = Softmax 전에 미래를 `-inf`로 가려, GPT가 정직하게 왼쪽만 보고 다음 토큰을 배우게 한다.

## 부록 P. 33→41 연결 지도

```text
33 Softmax와 Logit
34 Cross Entropy Loss
35 Attention이 필요한 이유
36 Query, Key, Value
37 Dot-Product Attention 계산
38 Softmax Attention과 작은 숫자 예제
39 Self-Attention 구현
40 Causal Mask          ← 지금
41 Multi-Head Attention ← 다음
```

출력단(33–34)과 문맥단(35–40)이 만났고, 이제 문맥단을 다중 헤드로 확장한다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [39강. Self-Attention 구현](39강_Self_Attention_구현.md)
- **다음 강:** [41강. Multi-Head Attention](41강_Multi_Head_Attention.md)

<!-- /LECTURE_NAV -->
