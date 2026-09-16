# 2권. Tokenizer와 Transformer

## 제38강. Softmax Attention과 작은 숫자 예제

### 1. 이번 강의에서 배울 것

37강에서 점수 행렬

$$
S = \frac{QK^\top}{\sqrt{d_k}}
$$

까지 만들었다. 이번 강의는 Softmax로 가중치를 만들고 Value를 섞어, **완전한 Attention 출력**을 손과 코드로 구한다.

이 강의를 마치면 다음을 말할 수 있어야 한다.

- 행 Softmax로 \(\alpha_{ij}\)를 구하는 절차
- \(O = \alpha V = \mathrm{softmax}(S)\,V\)의 Shape
- 2~3토큰 예제를 **끝까지** 손계산하기
- 손계산 ↔ NumPy ↔ PyTorch 결과가 같은지 검증하기
- 33강 Softmax와의 연결, 39강 모듈화로 넘어갈 준비

공식 한 줄이 더 이상 “기호 나열”이 아니라, 채워진 숫자 표가 되어야 한다.

### 2. 왜 이것을 배우는가

Attention의 정의는 짧다.

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

짧아서 위험하다. Softmax 축, Value 곱 순서, 마스크 위치를 한 번만 헷갈려도 조용히 틀린다.  
작은 숫자로 한 바퀴 돌리면, 이후 구현·디버깅이 압도적으로 쉬워진다.

### 3. 먼저 알아야 할 개념

- Softmax (33강)
- Q/K/V (36강)
- Scaled scores \(S\) (37강)
- 행렬곱 (1권 10강)
- Cross Entropy는 “출력단”; 오늘은 “문맥 혼합단”

### 4. 핵심 개념 설명

#### 4.1 Softmax Attention이란?

**Softmax Attention**은 점수 \(S\)의 각 행에 Softmax를 적용해 **확률형 가중치** \(\alpha\)를 만든 뒤, Value를 가중합하는 Attention이다.

$$
\alpha_{ij} = \frac{e^{s_{ij}}}{\sum_{j'=1}^{T} e^{s_{ij'}}}
$$

$$
\mathbf{o}_i = \sum_{j=1}^{T} \alpha_{ij}\,\mathbf{v}_j
$$

행렬로:

$$
A = \mathrm{softmax}(S)\in\mathbb{R}^{T\times T},\quad
O = A V \in\mathbb{R}^{T\times d_v}
$$

#### 4.2 왜 Softmax인가?

1. 가중치가 **양수**가 된다 (기본 Softmax).
2. 행 합이 **1** → 가중합·기댓값 해석이 가능.
3. 점수 차이에 **부드러운** 비중을 부여 (미분 가능).
4. 큰 점수에 더 큰 비중 (하지만 스케일·마스크와 함께 조절).

#### 4.3 Softmax의 축

반드시 **Key 방향(열 방향)**으로 Softmax한다.  
즉 “한 Query가 보는 모든 Key”에 대해 정규화한다.

- 행 \(i\) Softmax → 합_j \(\alpha_{ij}=1\)
- 열 Softmax를 하면 의미가 깨진다

#### 4.4 33강 Softmax와의 관계

33강 Softmax는 vocabulary logits → 토큰 확률이었다.  
오늘 Softmax는 **위치 점수 → 주목 가중치**다.

수학 형태는 같다. **적용 대상과 해석**이 다르다.

| | 33강 LM Softmax | 38강 Attention Softmax |
|---|---|---|
| 입력 | 토큰 logits \(z_k\) | 위치 점수 \(s_{ij}\) |
| 축 길이 | \(V\) | \(T\) |
| 출력 의미 | 다음 토큰 확률 | 어느 위치를 볼지 |
| 뒤에 오는 것 | CE Loss (34강) | Value 가중합 |

### 5. 직관적으로 이해하기

#### 5.1 예산을 나누는 일

각 Query 위치는 “주목 예산” 1.0을 가진다.  
Softmax는 그 예산을 Key 위치들에 나눠 준다.

점수가 높은 곳에 더 많은 예산 → 그 위치의 Value가 출력에 더 많이 섞인다.

#### 5.2 가중합이라는 한 그림

```text
α_i0 * v0 + α_i1 * v1 + ... + α_i,T-1 * v,T-1  =  o_i
```

35강에서 본 `weights @ E`가 바로 이것이다.  
오늘은 \(\alpha\)를 Softmax로 **계산**한다.

### 6. 수학적으로 이해하기

#### 6.1 안정 Softmax

행 점수 \(\mathbf{s}\)에 대해

$$
m=\max_j s_j,\quad
\alpha_j=\frac{e^{s_j-m}}{\sum_{j'} e^{s_{j'}-m}}
$$

\(e^{s}\) 오버플로를 막는다. 결과는 수학적으로 동일하다.

#### 6.2 전체 파이프라인

$$
S=\frac{QK^\top}{\sqrt{d_k}},\quad
A=\mathrm{softmax}(S),\quad
O=AV
$$

마스크가 있으면 Softmax 전 \(S\)를 수정 (40강).

#### 6.3 Gradient 감각 (선택)

Attention 가중치는 이후 Loss(34강)까지 이어지는 경로의 일부다.  
지금은 순전파 숫자에 집중한다.

### 7. 작은 숫자로 직접 계산하기

#### 7.1 예제 1 — 2토큰 완전 계산

37강 설정 B를 이어받는다.

$$
Q=\begin{bmatrix}1&2\\0&1\end{bmatrix},\ 
K=\begin{bmatrix}1&0\\1&1\end{bmatrix},\ 
V=\begin{bmatrix}1&0\\0&1\end{bmatrix}
$$

\(d_k=2\), \(\sqrt{2}\approx1.414213562\).

**Step 1. scores**

$$
QK^\top=\begin{bmatrix}1&3\\0&1\end{bmatrix},\quad
S\approx\begin{bmatrix}0.7071&2.1213\\0&0.7071\end{bmatrix}
$$

**Step 2. 행0 Softmax**

점수 \([0.7071,\ 2.1213]\)

안정화: max=2.1213

$$
\begin{aligned}
e^{0.7071-2.1213}&=e^{-1.4142}\approx 0.2431\\
e^{2.1213-2.1213}&=e^{0}=1\\
\sum &\approx 1.2431\\
\alpha_{00}&\approx 0.2431/1.2431\approx 0.1956\\
\alpha_{01}&\approx 1/1.2431\approx 0.8044
\end{aligned}
$$

**Step 3. 행1 Softmax**

점수 \([0,\ 0.7071]\), max=0.7071

$$
\begin{aligned}
e^{0-0.7071}&=e^{-0.7071}\approx 0.4931\\
e^{0}&=1\\
\sum &\approx 1.4931\\
\alpha_{10}&\approx 0.3303\\
\alpha_{11}&\approx 0.6697
\end{aligned}
$$

$$
A\approx\begin{bmatrix}0.1956&0.8044\\0.3303&0.6697\end{bmatrix}
$$

행 합 ≈ 1.

**Step 4. \(O=AV\)**

$$
O\approx\begin{bmatrix}0.1956&0.8044\\0.3303&0.6697\end{bmatrix}
\begin{bmatrix}1&0\\0&1\end{bmatrix}
=\begin{bmatrix}0.1956&0.8044\\0.3303&0.6697\end{bmatrix}
$$

\(V=I\)라서 \(O=A\)가 된다.  
해석: 토큰0 출력은 약 \(0.20\,\mathbf{v}_0 + 0.80\,\mathbf{v}_1\) → 토큰1에 크게 의존.

#### 7.2 예제 2 — 3토큰 완전 계산

37강 설정 C:

$$
Q=K=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},\quad
V=\begin{bmatrix}1&0\\0&1\\2&2\end{bmatrix}
$$

$$
S\approx\begin{bmatrix}
0.7071&0&0.7071\\
0&0.7071&0.7071\\
0.7071&0.7071&1.4142
\end{bmatrix}
$$

**행0 Softmax** (점수 \([0.7071,0,0.7071]\), max=0.7071)

$$
\begin{aligned}
e^{0}&=1\\
e^{0-0.7071}&\approx 0.4931\\
e^{0}&=1\\
\sum &\approx 2.4931\\
\alpha_0 &\approx [0.4011,\ 0.1978,\ 0.4011]
\end{aligned}
$$

**행1 Softmax** (대칭으로)

$$
\alpha_1 \approx [0.1978,\ 0.4011,\ 0.4011]
$$

**행2 Softmax** (점수 \([0.7071,0.7071,1.4142]\), max=1.4142)

$$
\begin{aligned}
e^{0.7071-1.4142}&=e^{-0.7071}\approx 0.4931\\
e^{-0.7071}&\approx 0.4931\\
e^{0}&=1\\
\sum &\approx 1.9863\\
\alpha_2 &\approx [0.2483,\ 0.2483,\ 0.5035]
\end{aligned}
$$

$$
A\approx\begin{bmatrix}
0.4011&0.1978&0.4011\\
0.1978&0.4011&0.4011\\
0.2483&0.2483&0.5035
\end{bmatrix}
$$

**\(O=AV\) 첫 행 손계산**

$$
\begin{aligned}
\mathbf{o}_0 &= 0.4011\cdot[1,0] + 0.1978\cdot[0,1] + 0.4011\cdot[2,2]\\
&= [0.4011,\ 0] + [0,\ 0.1978] + [0.8022,\ 0.8022]\\
&= [1.2033,\ 1.0000]
\end{aligned}
$$

비슷하게 \(\mathbf{o}_2\)는 자기 Value \([2,2]\)에 약 50%를 두므로 값이 커진다.

$$
\begin{aligned}
\mathbf{o}_2 &\approx 0.2483[1,0]+0.2483[0,1]+0.5035[2,2]\\
&= [0.2483,\ 0] + [0,\ 0.2483] + [1.0070,\ 1.0070]\\
&= [1.2553,\ 1.2553]
\end{aligned}
$$

#### 7.3 예제 3 — Softmax 뾰족함 비교

같은 행에서 raw vs scaled (부록 개념 복습):

- Softmax([10,12]) ≈ [0.1192, 0.8808]
- Softmax([10,12]/8 = Softmax([1.25,1.5]) ≈ [0.4378, 0.5622]

스케일이 분포를 완화한다. 37~38강이 맞물리는 지점이다.

#### 7.4 예제 4 — 한 위치에만 큰 점수

\(S=[0,\ 0,\ 5]\), Softmax:

$$
\alpha \approx [0.0067,\ 0.0067,\ 0.9866]
$$

거의 hard selection에 가깝다. Attention이 “거의 argmax”처럼 동작하는 극단이다.

### 8. 코드로 구현하기 (NumPy) — 손계산 대조

```python
# lecture38_softmax_attention.py
# Softmax Attention 전체: scores → weights → output

import numpy as np


def softmax_last(x: np.ndarray) -> np.ndarray:
    """마지막 축 Softmax (안정화)."""
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=-1, keepdims=True)


def scaled_dot_product_attention(Q, K, V):
    """
    Q,K: (T, d_k), V: (T, d_v)
    return O, A, S
    """
    d_k = Q.shape[-1]
    S = (Q @ K.T) / np.sqrt(d_k)
    A = softmax_last(S)
    O = A @ V
    return O, A, S


if __name__ == "__main__":
    # 예제 1
    Q = np.array([[1.0, 2.0], [0.0, 1.0]])
    K = np.array([[1.0, 0.0], [1.0, 1.0]])
    V = np.eye(2)
    O, A, S = scaled_dot_product_attention(Q, K, V)
    print("S:\n", np.round(S, 4))
    print("A:\n", np.round(A, 4))
    print("O:\n", np.round(O, 4))
    print("row sums:", A.sum(axis=1))

    # 예제 2
    Q3 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    K3 = Q3.copy()
    V3 = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    O3, A3, S3 = scaled_dot_product_attention(Q3, K3, V3)
    print("A3:\n", np.round(A3, 4))
    print("O3[0]:", np.round(O3[0], 4))
    print("O3[2]:", np.round(O3[2], 4))
```

손계산 \(\mathbf{o}_0\approx[1.203,\ 1.000]\)과 오차 \(10^{-3}\) 수준인지 확인한다.

### 9. PyTorch로 구현하기

```python
# lecture38_attention_torch.py

import math
import torch
import torch.nn.functional as F


def attention_torch(Q, K, V):
    d_k = Q.size(-1)
    S = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    A = F.softmax(S, dim=-1)  # 마지막 축 = Key 축
    O = A @ V
    return O, A, S


if __name__ == "__main__":
    Q = torch.tensor([[1.0, 2.0], [0.0, 1.0]])
    K = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    V = torch.eye(2)
    O, A, S = attention_torch(Q, K)
    print(O)
    print(A)

    # 공식 API (마스크 없으면 동일 계열)
    # PyTorch 버전에 따라 사용 가능
    try:
        O2 = F.scaled_dot_product_attention(Q, K, V)
        print("sdpa:", O2)
        print("diff:", (O - O2).abs().max().item())
    except Exception as e:
        print("sdpa unavailable:", e)
```

`F.softmax(..., dim=-1)`의 `dim`이 틀리면 즉시 붕괴한다. 항상 Key 축인지 확인한다.

### 10. 실제 LLM에서는 어떻게 사용하는가

#### 10.1 Self-Attention 한 헤드의 본체

오늘 식이 Multi-Head의 헤드 하나다.  
헤드 출력을 concat + \(W_O\) (41강).

#### 10.2 학습과의 연결

\(O\)는 잔차 연결·LayerNorm·FFN을 거쳐 (44~46강) 결국 LM Head → CE(34강)로 이어진다.  
“어디에 주목했는가”가 다음 토큰 확률을 바꾼다.

#### 10.3 추론

생성 중에는 새 토큰 행의 Softmax만 필요할 때가 많다.  
원리는 동일: 점수 → Softmax → Value 가중합.

#### 10.4 시각화

\(A\)의 행을 히트맵으로 보면 주목 패턴이 보인다 (51강).  
오늘 예제의 \(A\)를 손으로 그려 보는 것이 시작이다.

### 11. 실습

#### 실습 1 — 손계산 Softmax

점수 \([1, 1, 1]\)의 Softmax를 구하시오.

#### 실습 2 — 예제 1 재현

7.1의 \(\alpha_{01}\)이 약 0.80인지 계산기로 확인하시오.

#### 실습 3 — \(O\) 손계산

$$
A=\begin{bmatrix}0.5&0.5\\0.2&0.8\end{bmatrix},\ 
V=\begin{bmatrix}2&0\\0&4\end{bmatrix}
$$

일 때 \(O=AV\)를 구하시오.

#### 실습 4 — 코드 대조

NumPy와 PyTorch로 예제 2의 \(O[0]\)을 구해 손계산과 비교하시오.

#### 실습 5 — 축 실험

같은 \(S\)에 `softmax(dim=0)`과 `dim=-1`을 적용해 행합/열합이 어떻게 다른지 관찰하시오.

#### 실습 6 — 연결 문장

33강 Softmax와 38강 Softmax의 차이를 “축의 의미”로 한 문장 쓰시오.

### 12. 자주 하는 실수

1. **Softmax 축 오류**  
   `dim=-1`이 Key 축이 되게 Shape을 맞춰야 한다.

2. **\(VA\) vs \(AV\)**  
   행벡터 토큰 관례에서는 \(O=AV\). 순서를 바꾸면 Shape부터 깨진다.

3. **이미 Softmax한 \(A\)에 다시 Softmax**  
   디버그 중 중복 적용 주의.

4. **행 합이 1인지 검사하지 않는다**  
   구현 직후 `A.sum(axis=-1)`은 기본 테스트다. (마스크+`-inf` 후도 유효 위치 합 1)

5. **손계산과 float 오차를 동일시해 포기**  
   소수점 3~4자리 일치면 충분. 완전 일치는 dtype에 따라 어렵다.

6. **스케일을 Softmax 뒤에 적용**  
   스케일은 Softmax **전** 점수에 적용한다.

### 13. 핵심 정리

- Softmax Attention은 \(A=\mathrm{softmax}(S)\), \(O=AV\)다.
- Softmax는 각 Query 행에서 Key 방향으로 적용한다.
- 가중치 행 합은 1이며, 출력은 Value의 가중합이다.
- 작은 행렬로 손계산↔코드 대조가 Attention 이해의 핵심 훈련이다.
- 33강의 Softmax와 형태는 같고 해석 대상이 다르다.

### 14. 핵심 용어

| 용어 | 의미 |
|---|---|
| Attention weights \(A\) | Softmax 후 주목 가중치 |
| Softmax Attention | Softmax 가중으로 V를 섞는 Attention |
| Context / Output \(O\) | 가중합 결과 |
| Stable Softmax | max 차감 후 exp |
| Row-wise Softmax | 행마다 정규화 |
| Scaled Dot-Product Attention | 스케일+Softmax+V의 표준 세트 |

### 15. 복습 문제

#### 문제 1 (계산)

\(s=[0,0]\)의 Softmax는?

#### 문제 2 (계산)

예제 1에서 \(\alpha_{00}+\alpha_{01}\)은?

#### 문제 3 (개념)

왜 Attention Softmax를 vocabulary Softmax와 혼동하면 안 되는가?

#### 문제 4 (코드)

`F.softmax(S, dim=-1)`에서 `S.shape==(T,T)`일 때 dim=-1은 무엇을 정규화하는가?

#### 문제 5 (연결)

39강에서 모듈로 묶을 때, 오늘 계산의 입력과 출력 텐서는 무엇인가?

---

### 정답 및 해설

#### 문제 1

\([0.5,\ 0.5]\).

#### 문제 2

1 (수치 오차 범위 내).

#### 문제 3

하나는 토큰 확률(길이 \(V\)), 하나는 위치 주목(길이 \(T\))이다. 축·의미·이후 연산이 다르다.

#### 문제 4

각 행의 \(T\)개 점수를 합 1인 가중치로 정규화한다 (Key 방향).

#### 문제 5

입력 \(Q,K,V\) (또는 \(X\)와 투영 포함), 출력 \(O\) (필요 시 \(A\)까지).

### 16. 다음 강의와 연결

이제 식과 숫자와 짧은 함수가 준비되었다.  
다음 **제39강. Self-Attention 구현**에서는 투영·점수·Softmax·출력을 **하나의 모듈**로 묶고, NumPy 구현 후 PyTorch 스케치로 정리한다.

> 손계산이 끝났으면, 이제 재사용 가능한 Self-Attention 블록으로 올리자.

---

### 부록 A. 예제 2의 \(O\) 전체 (코드로 확정할 값)

학습자가 코드를 돌리면 대략:

```text
O3 ≈
[[1.2033 1.0000]
 [1.0000 1.2033]
 [1.2553 1.2553]]
```

대칭적 \(Q=K\)와 \(V\) 설계 때문에 \(\mathbf{o}_0\)와 \(\mathbf{o}_1\)이 성분이 뒤바뀐 대칭을 보인다.

### 부록 B. 온도 \(T_{\mathrm{temp}}\)와의 관계

추론 샘플링의 temperature는 logits/\(T_{\mathrm{temp}}\)다.  
Attention의 \(\sqrt{d_k}\)도 Softmax 입력 스케일이라는 점에서 닮았지만, **학습 중 Attention 스케일**과 **생성 시 토큰 샘플링 온도**는 다른 스위치다. 섞어 부르지 말 것.

### 부록 C. 마스크가 있을 때의 Softmax (예고)

어떤 위치에 \(s=-\infty\)를 넣으면 \(e^{-\infty}=0\)이라 가중치 0.  
남은 유한 점수들만으로 재정규화된다 (40강).

### 부록 D. 이중 루프 검증 코드

```python
def attention_loops(Q, K, V):
    T, d_k = Q.shape
    d_v = V.shape[1]
    scale = np.sqrt(d_k)
    S = np.zeros((T, T))
    for i in range(T):
        for j in range(T):
            S[i, j] = np.dot(Q[i], K[j]) / scale
    A = np.zeros((T, T))
    for i in range(T):
        row = S[i]
        m = row.max()
        ex = np.exp(row - m)
        A[i] = ex / ex.sum()
    O = np.zeros((T, d_v))
    for i in range(T):
        for j in range(T):
            O[i] += A[i, j] * V[j]
    return O, A, S
```

행렬판과 일치하는지 반드시 비교한다.

### 부록 E. 확률 해석

고정 \(i\)에 대해 \(\alpha_{i\cdot}\)는 Key 위치 위의 이산 분포다.  
\(\mathbf{o}_i=\mathbb{E}_{j\sim\alpha_i}[\mathbf{v}_j]\)로 읽을 수 있다.

### 부록 F. 실습 3 답

$$
O=\begin{bmatrix}0.5\cdot2+0.5\cdot0 & 0.5\cdot0+0.5\cdot4\\0.2\cdot2+0.8\cdot0 & 0.2\cdot0+0.8\cdot4\end{bmatrix}
=\begin{bmatrix}1&2\\0.4&3.2\end{bmatrix}
$$

### 부록 G. 실습 1 답

균등 \([1/3,1/3,1/3]\).

### 부록 H. 수치 안정성 체크

큰 점수 `S = [1000, 1001]`:

- 안정 Softmax: 정상
- max 없는 `exp(S)`: overflow → `inf`/`nan`

Attention 구현에서 안정 Softmax(또는 log-sum-exp 계열 fused kernel)가 기본인 이유다.

### 부록 I. Shape 체크리스트

| 단계 | Shape |
|---|---|
| Q | (T, d_k) |
| K | (T, d_k) |
| V | (T, d_v) |
| S | (T, T) |
| A | (T, T) |
| O | (T, d_v) |

배치: 앞에 B. 헤드: `(B,h,T,*)`.

### 부록 J. 자주 그리는 히트맵 읽는 법

- 밝은 칸: 높은 \(\alpha_{ij}\)
- 대각이 밝음: 자기 자신 주목
- 행 i의 밝은 열: i가 참조하는 토큰들

예제 1의 행0은 열1이 더 밝다.

### 부록 K. CE와의 거리감

Attention Softmax 출력은 **Loss에 직접 넣지 않는다**.  
토큰 예측 Softmax(33~34강)와 단계가 다르다.  
중간 표현을 섞는 내부 연산이다.

### 부록 L. 미니 증명 — 행 합 1

$$
\sum_j \alpha_{ij} = \sum_j \frac{e^{s_{ij}}}{\sum_{j'}e^{s_{ij'}}} = 1
$$

가중합 계수가 확률 단순형인 이유다.

### 부록 M. 다음 구현으로 넘길 인터페이스

```text
def self_attention(X, W_Q, W_K, W_V):
    Q, K, V = X@W_Q, X@W_K, X@W_V
    O, A, S = scaled_dot_product_attention(Q, K, V)
    return O
```

39강이 이 함수를 클래스/모듈로 승격한다.

### 부록 N. 자가 점검 질문

1. \(A\)의 한 행 합은?  
2. \(V=I\)이면 \(O\)는?  
3. Softmax 축은?  
4. 스케일은 Softmax 전/후?  
5. 예제 2에서 \(\alpha_{22}\) 대략?

답: 1) 1 2) \(A\) 3) Key 방향(마지막 축) 4) 전 5) ≈0.50

### 부록 O. 한 페이지 요약

```text
S = QK^T / sqrt(d_k)     # 37강
A = softmax(S, axis=-1)  # 오늘
O = A @ V                # 오늘
```

이 세 줄만 손에 익으면 Transformer 책의 절반이 열린다.
