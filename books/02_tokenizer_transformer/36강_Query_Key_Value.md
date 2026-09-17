# 36강. Query, Key, Value
## 이번 강에서 배우는 내용

- Query(질의), Key(키), Value(값) 각각의 역할
- 도서관·검색 비유로 Q/K/V를 설명하기
- Self-Attention에서 Q, K, V가 같은 입력의 선형 변환으로 나온다는 것
- 기호 $Q=XW_Q,\ K=XW_K,\ V=XW_V$를 Shape와 함께 쓰기
- 37강의 $QK^\top$로 넘어갈 준비 마치기

## 왜 중요한가?
많은 설명이 Attention을 한 줄로 압축한다.

$$

\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V

$$

이 한 줄을 외우기만 하면, $Q$와 $K$가 왜 갈라져 있는지, $V$는 왜 별도인지가 안 보인다.  
부품을 모르면 나중에 Multi-Head·RoPE·KV cache를 만날 때 전부 미스터리다.

오늘은 공식을 **역할**로 분해한다.

## 선수 개념
- Embedding / 토큰 벡터 (31강)
- 행렬곱·내적 (1권 10강)
- Softmax (33강) — “점수를 가중치로”
- 35강의 가중 평균 직관
- Linear Layer: $\mathbf{y}=W\mathbf{x}+\mathbf{b}$ (편향은 생략하는 구현이 많음)

## 핵심 개념
### 3.1 세 역할로 나누는 이유

문맥을 섞으려면 질문에 답해야 한다.

1. **지금 나는 무엇을 찾고 있는가?** → Query
2. **각 위치는 무엇으로 검색되는가?** → Key
3. **실제로 가져올 내용은 무엇인가?** → Value

한 벡터가 세 역할을 동시에 하면, “검색용 표현”과 “내용용 표현”을 분리해 학습하기 어렵다.  
그래서 입력을 **세 종류의 선형 투영**으로 나눈다.

### 3.2 Query (질의)

**Query(쿼리, 질의 벡터)**는 “현재 위치가 문맥에 던지는 질문”을 담은 벡터다.

예:

- 동사 위치의 Query: “내 주어·목적어가 어디 있지?”
- 대명사 위치의 Query: “내가 가리키는 명사가 어디 있지?”

기호: 위치 $i$의 Query $\mathbf{q}_i \in \mathbb{R}^{d_k}$.

### 3.3 Key (키)

**Key(키)**는 “각 위치가 검색에 응하는 표지”다.  
도서관으로 치면 책등·카탈로그 번호에 가깝다.

위치 $j$의 Key $\mathbf{k}_j$는 Query와 **내적되어 점수**가 된다.

$$

\mathrm{score}(i,j) = \mathbf{q}_i \cdot \mathbf{k}_j

$$

(스케일 $1/\sqrt{d_k}$는 37강.)

Key는 “내가 어떤 종류의 정보원인가”를 나타낸다.  
Value와 같을 필요는 없다.

### 3.4 Value (값)

**Value(값)**는 실제로 가중합에 들어가는 **내용 벡터**다.  
점수가 높으면 그 위치의 Value가 결과에 더 많이 섞인다.

$$

\mathbf{o}_i = \sum_j \alpha_{ij}\,\mathbf{v}_j

$$

- $\alpha_{ij}$: Softmax로 만든 가중치 (합 1)
- $\mathbf{v}_j$: 위치 $j$의 Value
- $\mathbf{o}_i$: 위치 $i$의 Attention 출력

Key가 “찾아지기 위한 라벨”이라면, Value는 “건네줄 내용”이다.

### 3.5 Self-Attention에서의 출처

**Self-Attention(셀프 어텐션)**은 Query·Key·Value가 **같은 시퀀스**에서 나온다.

입력을 $X \in \mathbb{R}^{T \times d_{\mathrm{model}}}$라 하자.  
$T$: 토큰 수, $d_{\mathrm{model}}$: 모델 차원.

$$

Q = X W_Q,\quad K = X W_K,\quad V = X W_V

$$

기호:

- $W_Q \in \mathbb{R}^{d_{\mathrm{model}}\times d_k}$
- $W_K \in \mathbb{R}^{d_{\mathrm{model}}\times d_k}$
- $W_V \in \mathbb{R}^{d_{\mathrm{model}}\times d_v}$
- 종종 $d_k=d_v=d_{\mathrm{model}}/h$ (헤드 수로 나눔, 41강)

**Cross-Attention**에서는 Query가 디코더, Key/Value가 인코더처럼 **다른 출처**일 수 있다.  
GPT Causal LM의 기본 블록은 Self-Attention이다.

## 직관적으로 이해하기
### 4.1 도서관 비유

당신이 논문을 쓰며 자료를 찾는다고 하자.

| 역할 | 비유 | Attention |
|---|---|---|
| Query | “트랜스포머 장거리 의존 관련 자료 찾아줘”라는 질문 | 현재 토큰이 찾는 것 |
| Key | 각 책의 주제 태그·초록 키워드 | 각 토큰의 검색용 표현 |
| Value | 책 본문에서 실제로 인용할 문장 | 섞여 들어갈 내용 |
| 점수 | 질문과 태그의 관련도 | $\mathbf{q}\cdot\mathbf{k}$ |
| Softmax | 관련도를 비율로 | 가중치 $\alpha$ |
| 출력 | 여러 책에서 발췌를 비율대로 합침 | $\sum \alpha_j \mathbf{v}_j$ |

같은 책이라도 “태그(Key)”와 “본문 발췌(Value)”는 다를 수 있다.  
그래서 $W_K$와 $W_V$를 따로 둔다.

### 4.2 검색창 비유

- Query: 검색창에 입력한 문장
- Key: 각 문서의 인덱스 키워드
- Value: 문서 스니펫
- Softmax: 상위 문서에 얼마나 비중을 둘지

### 4.3 “왜 Q와 K를 나누나?”

내적 $\mathbf{x}_i\cdot\mathbf{x}_j$만 쓰면, “찾는 표현”과 “찾아지는 표현”이 같은 공간·같은 변환에 묶인다.  
$W_Q$와 $W_K$를 나누면:

- 위치 $i$는 “주어를 찾는 질문”으로 투영되고
- 위치 $j$는 “나는 주어 후보”로 투영될 수 있다

비대칭 관계를 학습하기 쉬워진다.  
(동일 행렬로도 이론상 가능하지만, 실무·표준은 분리다.)

### 4.4 “왜 V를 또 나누나?”

관련 있다고 판정된 위치에서 **무엇을 가져올지**는 검색 표지와 다를 수 있다.  
예: Key는 “이 토큰은 개체명”, Value는 “그 개체의 풍부한 의미 벡터”.

## 수학적으로 이해하기
### 5.1 위치별 정의

입력 행벡터(또는 열벡터 관례에 주의)로 위치 $t$의 입력을 $\mathbf{x}_t \in \mathbb{R}^{d_{\mathrm{model}}}$라 하면

$$

\mathbf{q}_t = W_Q^\top \mathbf{x}_t,\quad
\mathbf{k}_t = W_K^\top \mathbf{x}_t,\quad
\mathbf{v}_t = W_V^\top \mathbf{x}_t

$$

행렬 형태로 한 번에:

$$

Q=XW_Q \in \mathbb{R}^{T\times d_k},\quad
K=XW_K \in \mathbb{R}^{T\times d_k},\quad
V=XW_V \in \mathbb{R}^{T\times d_v}

$$

이 책은 행이 토큰인 $X$를 기본으로 한다. (구현·논문에 따라 전치 관례가 다를 수 있다. Shape만 맞으면 된다.)

### 5.2 점수로의 연결 (예고)

위치 $i$가 $j$를 보는 점수:

$$

s_{ij} = \mathbf{q}_i \cdot \mathbf{k}_j = (QK^\top)_{ij}

$$

전체 점수 행렬:

$$

S = QK^\top \in \mathbb{R}^{T\times T}

$$

37강에서 $\sqrt{d_k}$로 나눈다.

### 5.3 출력

$$

\alpha_{ij} = \mathrm{softmax}_j(s_{ij}),\quad
\mathbf{o}_i = \sum_j \alpha_{ij}\mathbf{v}_j

$$

행렬로:

$$

O = \mathrm{softmax}(S) \, V

$$

(행 Softmax 가정.)

### 5.4 Shape 체크리스트

| 텐서 | Shape | 의미 |
|---|---|---|
| $X$ | $(T, d_{\mathrm{model}})$ | 입력 |
| $W_Q$ | $(d_{\mathrm{model}}, d_k)$ | Query 투영 |
| $Q$ | $(T, d_k)$ | 모든 Query |
| $K$ | $(T, d_k)$ | 모든 Key |
| $V$ | $(T, d_v)$ | 모든 Value |
| $QK^\top$ | $(T, T)$ | 모든 쌍 점수 |
| $O$ | $(T, d_v)$ | Attention 출력 |

배치가 있으면 앞에 $B$가 붙는다: $(B,T,d)$.

## 작은 숫자로 직접 계산하기
설정을 극단적으로 작게 잡는다.

- $T=2$ 토큰: `나`, `책`
- $d_{\mathrm{model}}=2$
- $d_k=d_v=2$

### 6.1 입력 $X$

$$

X = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}

$$

행0 = `나`, 행1 = `책`.

### 6.2 가중치 행렬

$$

W_Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix},\quad
W_K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix},\quad
W_V = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}

$$

(이해를 위한 장난 값. 학습된 값이 아니다.)

### 6.3 Q, K, V 계산

$$

Q = XW_Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix},\quad
K = XW_K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}

$$

$$

V = XW_V = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}

$$

해석:

- `나`의 Query/Key = $[1,0]$
- `책`의 Query/Key = $[0,1]$
- `나`의 Value = $[2,0]$, `책`의 Value = $[0,3]$

### 6.4 점수 (맛보기)

$$

QK^\top = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}

$$

- `나`→`나` 점수 1, `나`→`책` 점수 0
- `책`→`책` 점수 1, `책`→`나` 점수 0

이 예에서는 각자 자신에게만 주목하는 극단 케이스다.  
$W_Q, W_K$를 바꾸면 교차 점수가 생긴다 (아래 7.5).

### 6.5 교차 주목이 나오게 $W_Q$만 바꾸기

$$

W_Q' = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}

$$

$$

Q' = XW_Q' = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}

$$

$$

Q'K^\top = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}

$$

이제 `나`는 `책`과 점수 1, `책`은 `나`와 점수 1이다.  
**투영이 바뀌면 누가 누구를 보는지가 바뀐다.** 이것이 $W_Q, W_K$ 학습의 의미다.

### 6.6 Value가 분리된 효과

점수가 Softmax 후 $\alpha = [[0,1],[1,0]]$이 되었다고 가정하면 (이상화된 예),

$$

O = \alpha V = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}\begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix} = \begin{bmatrix} 0 & 3 \\ 2 & 0 \end{bmatrix}

$$

- `나`의 출력 = `책`의 Value $[0,3]$
- `책`의 출력 = `나`의 Value $[2,0]$

검색(점수)과 내용(Value)이 분리되어, “상대 토큰의 내용”을 가져온다.

### 6.7 3토큰으로 한 칸 더

$T=3$, $d=2$, 단순화해 $Q=K=X$, $V=X$라고 두자.

$$

X=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}

$$

$$

QK^\top = XX^\top = \begin{bmatrix}1&0&1\\0&1&1\\1&1&2\end{bmatrix}

$$

손계산:

- $(XX^\top)_{00}=1\cdot1+0\cdot0=1$
- $(XX^\top)_{02}=1\cdot1+0\cdot1=1$
- $(XX^\top)_{22}=1\cdot1+1\cdot1=2$

세 번째 토큰은 자신과의 점수가 가장 크다.  
이후 Softmax하면 가중치가 나온다 (38강).

## 코드로 구현하기
```python
# lecture36_qkv_numpy.py
# 입력 X와 W_Q, W_K, W_V로 Q, K, V를 만든다.

import numpy as np

def project_qkv(X, W_Q, W_K, W_V):
    """
    X: (T, d_model)
    W_*: (d_model, d_k or d_v)
    return Q, K, V
    """
    Q = X @ W_Q  # (T, d_k)
    K = X @ W_K  # (T, d_k)
    V = X @ W_V  # (T, d_v)
    return Q, K, V

def scores(Q, K):
    """스케일 없는 점수 S = Q K^T. 스케일은 37강."""
    return Q @ K.T  # (T, T)

if __name__ == "__main__":
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    W_Q = np.eye(2)
    W_K = np.eye(2)
    W_V = np.array([[2.0, 0.0], [0.0, 3.0]])

    Q, K, V = project_qkv(X, W_Q, W_K, W_V)
    print("Q:\n", Q)
    print("K:\n", K)
    print("V:\n", V)
    print("S = QK^T:\n", scores(Q, K))

    # 교차 주목용 W_Q'
    W_Q2 = np.array([[0.0, 1.0], [1.0, 0.0]])
    Q2, K2, V2 = project_qkv(X, W_Q2, W_K, W_V)
    print("S' with swapped W_Q:\n", scores(Q2, K2))
```

## PyTorch로 구현하기
```python
# lecture36_qkv_torch.py
# nn.Linear 세 개로 Q, K, V 투영 (bias=False가 논문/구현에 흔함)

import torch
import torch.nn as nn

class QKVProjection(nn.Module):
    def __init__(self, d_model: int, d_k: int, d_v: int):
        super().__init__()
        # 각각 다른 가중치 — 역할 분리의 핵심
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_v, bias=False)

    def forward(self, X: torch.Tensor):
        # X: (B, T, d_model) 또는 (T, d_model)
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)
        return Q, K, V

if __name__ == "__main__":
    torch.manual_seed(0)
    layer = QKVProjection(d_model=4, d_k=2, d_v=2)
    X = torch.randn(1, 3, 4)  # 배치1, 토큰3, d=4
    Q, K, V = layer(X)
    print("Q", Q.shape, "K", K.shape, "V", V.shape)
    S = Q @ K.transpose(-2, -1)
    print("S", S.shape)  # (1, 3, 3)
```

실무에서는 Multi-Head를 위해 $d_k=d_{\mathrm{model}}/h$로 나누고,  
때로는 `W` 하나를 크게 만든 뒤 `chunk`로 Q/K/V를 쪼개기도 한다. 원리는 같다.

## 수식 보강 — QKV 투영

$$
Q=XW_Q,\ K=XW_K,\ V=XW_V
$$

$X\in\mathbb{R}^{T\times d}$, $W_Q\in\mathbb{R}^{d\times d_k}$ 등. 점수는 $QK^\top/\sqrt{d_k}$ (37강).

## LLM에서는 어디에 사용될까?
### 9.1 모든 Self-Attention 층의 입구

Transformer 블록마다 (또는 Attention 서브층마다) $W_Q, W_K, W_V$가 있다.  
파라미터가 쌓이는 주요 위치 중 하나다.

### 9.2 KV Cache (예고)

추론 시 과거 토큰의 **K, V**는 재사용하고, 새 토큰의 Q만 새로 계산하는 최적화가 있다 (5권).  
“Key/Value는 저장된 기억, Query는 현재 질문”이라는 역할 분리가 캐시 설계와 맞닿는다.

### 9.3 Cross-Attention

인코더–디코더 모델에서 디코더 Query가 인코더 Key/Value를 본다.  
역할 이름은 동일하고 **출처만** 다르다.

### 9.4 GPT

GPT는 주로 Causal Self-Attention: 같은 시퀀스의 Q/K/V + 미래 마스크 (40강).

## 실습
### 실습 1 — 역할 문장

Q, K, V를 도서관 비유로 각 한 문장씩 쓰시오.

### 실습 2 — Shape

$T=5,\ d_{\mathrm{model}}=16,\ d_k=8,\ d_v=8$일 때 $Q,K,V,QK^\top$ Shape를 쓰시오.

### 실습 3 — 손계산

$$

X=\begin{bmatrix}1&2\\3&4\end{bmatrix},\ 
W_Q=\begin{bmatrix}1&0\\0&1\end{bmatrix}

$$

일 때 $Q=XW_Q$를 구하시오.

### 실습 4 — 코드

8절 코드에서 $W_V=I$로 두고 $V$가 $X$와 같은지 확인하시오.

### 실습 5 — 사고

Self-Attention과 Cross-Attention의 차이를 “Q의 출처 / K,V의 출처”로 비교하시오.

### 실습 6 — 연결

35강의 가짜 `weights @ E`에서, `weights`에 해당하는 것을 만들기려면 무엇이 필요하고, `E`에 해당하는 것은 Q/K/V 중 무엇인가?

## 자주 하는 실수
1. **Q, K, V가 입력 X 그 자체라고 생각한다**  
   표준 Self-Attention에서는 **투영된** 벡터다. ($W=I$인 특수 예외는 설명용)

2. **Key와 Value를 혼동한다**  
   Key는 점수(검색), Value는 가중합(내용).

3. **Shape에서 $d_k$와 $T$를 섞는다**  
   $QK^\top$의 결과는 $(T,T)$이지 $(d_k,d_k)$가 아니다.

4. **배치 차원 transpose 실수**  
   `K.T` 대신 `K.transpose(-2,-1)`처럼 **마지막 두 축**을 전치해야 한다.

5. **세 Linear를 하나로 묶고 역할을 잊는다**  
   구현 최적화와 개념 이해는 별개다. 개념적으로는 세 투영이다.

6. **bias 유무를 과도하게 걱정한다**  
   구현마다 다르다. 원리를 먼저, 세부 플래그는 코드 읽을 때 확인.

## 핵심 요약
- Query는 “무엇을 찾는가”, Key는 “어떻게 찾아지는가”, Value는 “무엇을 전달하는가”.
- Self-Attention에서 $Q=XW_Q,\ K=XW_K,\ V=XW_V$.
- 점수는 Query–Key 유사도(내적)로 만들고, 출력은 Value의 가중합이다.
- 투영을 바꾸면 “누가 누구를 보는지”가 바뀐다.
- 다음 단계는 점수 행렬을 숫자로 계산하는 것 (37강).

## 용어 사전
| 용어 | 의미 |
|---|---|
| Query (Q) | 현재 위치의 질의 벡터 |
| Key (K) | 각 위치의 검색용 표지 벡터 |
| Value (V) | 가중합에 쓰이는 내용 벡터 |
| Projection | $XW$ 형태의 선형 변환 |
| Self-Attention | Q/K/V가 같은 시퀀스에서 나옴 |
| Cross-Attention | Q와 K/V의 출처가 다를 수 있음 |
| $d_{\mathrm{model}}$ | 토큰 표현 차원 |
| $d_k, d_v$ | Key/Query 차원, Value 차원 |
| Score matrix | $QK^\top$, 위치 쌍 점수 |

## 연습문제
### 문제 1 (개념)

Value를 Key와 동일하게 두면 어떤 유연성이 줄어드는가?

### 문제 2 (계산)

$\mathbf{q}=[1,2],\ \mathbf{k}=[3,4]$의 내적은?

### 문제 3 (Shape)

$Q\in\mathbb{R}^{4\times 8},\ K\in\mathbb{R}^{4\times 8}$일 때 $QK^\top$의 Shape는?

### 문제 4 (코드)

`S = Q @ K.transpose(-2, -1)`에서 transpose가 필요한 이유를 한 문장으로.

### 문제 5 (연결)

37강에서 $\sqrt{d_k}$로 나누기 전, 오늘의 $S=QK^\top$는 Attention 파이프라인의 어느 단계인가?

---

## 정답 및 해설
### 문제 1

“관련 있다”는 판정(표지)과 “전달할 내용”을 따로 학습·표현하기 어려워진다.

### 문제 2

$1\cdot3+2\cdot4=11$.

### 문제 3

$(4,4)$.

### 문제 4

행렬곱에서 $K$의 토큰 축과 차원 축을 바꿔 $(..., d_k, T)$로 만들어야 $Q(..., T, d_k)$와 곱해 $(..., T, T)$ 점수가 나온다.

### 문제 5

가중치를 만들기 **직전**의 raw score(유사도) 행렬을 만드는 단계다.

## 다음 강의와 연결
부품 $Q, K, V$를 얻었다.  
다음 **제37강. Dot-Product Attention 계산**에서는

$$

S = \frac{QK^\top}{\sqrt{d_k}}

$$

를 작은 숫자로 한 칸씩 계산한다.  
왜 나누는지, Shape는 어떻게 맞는지, 2~3토큰 예제로 손계산한다.

> 이름을 붙였으면, 이제 내적으로 점수를 매기는 산수로 들어가자.

---

## 부록 A. 한 위치에 대한 펼친 식

위치 $i$만 보면:

$$

\mathbf{q}_i = W_Q^\top \mathbf{x}_i,\quad
s_{ij}=\mathbf{q}_i^\top\mathbf{k}_j,\quad
\alpha_{ij}=\frac{e^{s_{ij}}}{\sum_{j'}e^{s_{ij'}}},\quad
\mathbf{o}_i=\sum_j \alpha_{ij}\mathbf{v}_j

$$

행렬 공식은 이 과정을 모든 $i$에 대해 한꺼번에 쓴 것이다.

## 부록 B. $d_k$와 Multi-Head 예고

헤드가 $h$개이면 흔히 $d_k=d_v=d_{\mathrm{model}}/h$로 둔다.  
각 헤드가 다른 $W_Q^{(h)}, W_K^{(h)}, W_V^{(h)}$를 갖고, 출력을 이어 붙인 뒤 $W_O$로 합친다 (41강).  
오늘은 헤드 1개 관점으로 충분하다.

## 부록 C. 입력에 이미 문맥이 섞여 있는가?

1층 Attention의 입력 $X$는 보통 Embedding(+위치정보)이다.  
2층부터의 입력은 **이전 층이 이미 섞은 표현**이다.  
그래도 매 층 Q/K/V 투영은 다시 적용된다. “더 추상적인 질문/키/값”을 학습한다.

## 부록 D. 용어 혼동 메모

| 말 | 의미 |
|---|---|
| attention weights | Softmax 후 $\alpha_{ij}$ |
| attention scores | Softmax 전 $s_{ij}$ |
| attention output | $O$ 또는 $\mathbf{o}_i$ |
| attention head | 병렬 QKV 세트 하나 (41강) |

점유/가중치/출력을 같은 말로 부르지 않도록 한다.

## 부록 E. 최소 체크 질문

다음을 막힘없이 말하면 36강 통과다.

1. Q/K/V 한 줄 정의
2. $Q=XW_Q$ Shape
3. 점수는 Q·K, 가중합은 V
4. Self vs Cross의 출처 차이
5. $QK^\top$가 $(T,T)$인 이유

## 부록 F. 추가 손계산 — 편향 없는 Linear와 동일성

`nn.Linear(d_model, d_k, bias=False)`의 가중치 shape는 PyTorch에서 `(d_k, d_model)`이고,  
순전파는 $xW^\top$에 해당한다.  
NumPy 예제의 $X @ W$와 **가중치 저장 형태**가 다를 수 있으니,  
“행렬을 오른쪽에 곱하느냐, `F.linear` 규약이냐”만 맞추면 된다.  
개념적으로는 같은 선형 투영이다.

## 부록 G. 세 토큰 문장으로 역할 읽어보기

문장: `고양이 / 가 / 잠자다`

가능한 학습 결과(이상화):

- `잠자다`의 Query: “행위의 주체가 누구?”
- `고양이`의 Key: “나는 동물 주어 후보”
- `고양이`의 Value: “고양이”의 의미 벡터
- Softmax 후 `잠자다`→`고양이` 가중치 ↑

실제 모델이 반드시 이렇게 해석 가능한 가중치를 학습한다는 보장은 없다.  
그러나 Q/K/V 설계가 **그런 종류의 비대칭 참조**를 표현할 여지를 준다는 점은 분명하다.

## 부록 H. 파라미터 수 감각

투영 세 개의 대략 파라미터:

$$

|\theta|_{\mathrm{QKV}} \approx d_{\mathrm{model}}\cdot d_k + d_{\mathrm{model}}\cdot d_k + d_{\mathrm{model}}\cdot d_v

$$

$d_k=d_v=d_{\mathrm{model}}$이면 약 $3 d_{\mathrm{model}}^2$.  
여기에 출력 투영 $W_O$가 더해지는 구현이 많다 (41강 이후).

## 부록 I. “Attention is all you need”와의 연결

원 논문은 Q/K/V Scaled Dot-Product Attention을 표준으로 제시했다.  
이 책의 36~40강은 그 식을 **손계산 가능한 크기**로 분해한 것이다.  
논문 문장을 먼저 외우기보다, 오늘처럼 역할부터 고정하는 편이 LLM 구현으로 이어지기 쉽다.

## 부록 J. 연습 — 빈칸 채우기

1. $K = X\,\_\_\_$
2. 점수 $s_{ij} = \mathbf{q}_i \cdot \_\_\_$
3. 출력 $\mathbf{o}_i = \sum_j \alpha_{ij}\,\_\_\_$
4. Self-Attention에서 Q/K/V의 출처는 모두 \_\_\_

정답: $W_K$,\ $\mathbf{k}_j$,\ $\mathbf{v}_j$,\ 같은 시퀀스(같은 $X$).

## 부록 K. 다음 강 예고 도식

```text
X
├─ W_Q → Q ─┐
├─ W_K → K ─┼─→ S = QK^T / sqrt(d_k)  → Softmax → α
└─ W_V → V ───────────────→ O = α V
```

오늘은 왼쪽 가지(Q/K/V 생성)까지.  
37강은 가운데 점수, 38강은 Softmax와 $O$, 39강은 모듈 구현, 40강은 마스크다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [35강. Attention이 필요한 이유](35강_Attention이_필요한_이유.md)
- **다음 강:** [37강. Dot-Product Attention 계산](37강_Dot_Product_Attention_계산.md)

<!-- /LECTURE_NAV -->
