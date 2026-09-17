# 제31강. Embedding — 토큰을 벡터로

> **학습 목표**
> - Embedding(임베딩)이 무엇인지, 왜 one-hot보다 쓰는가
> - Embedding 행렬 $E \in \mathbb{R}^{|V| \times d}$에서 lookup이 일어나는 방식
> - Embedding이 학습 가능한 파라미터라는 점
> - 작은 숫자 예제로 id → 벡터 변환을 손으로 추적하기
> - LLM에서 토큰 임베딩 + (이후) positional 정보가 첫 층 입력이 되는 위치

---
## 1. 왜 이것을 배우는가

Token id `3797` 자체에는 기하적 의미가 없다. `3797`과 `3798`이 “비슷한 단어”라는 보장도 없다.

모델이 필요한 것은 대략 이런 성질이다.

```text
비슷한 문맥에서 쓰이는 토큰
  → 비슷한 벡터 근처에 배치
다른 역할의 토큰
  → 다른 방향/영역
```

이 배치를 처음부터 사람이 수작업으로 정하지 않는다. **학습**이 Embedding 행렬의 행을 움직인다.

Embedding을 이해하지 못하면:

- `nn.Embedding`이 “마법 API”로 남는다
- vocab size를 바꿀 때 어떤 가중치가 깨지는지 모른다
- Attention 입력이 왜 `[T, d]` 형태인지 설명이 안 된다

## 2. 먼저 알아야 할 개념

- Token ID / Vocabulary size $|V|$ (27·30강)
- Vector / Matrix / 행렬곱 (1권 9·10강)
- `nn.Module`과 학습 파라미터 (1권 21강)
- one-hot 벡터의 직관 (해당 인덱스만 1, 나머지 0)
- Gradient Descent로 행렬 성분을 업데이트한다는 감각 (1권 12강)

아직 몰라도 되는 것: Positional Encoding의 상세 수식, Attention score.

## 3. 핵심 개념 설명

### 3.1 Embedding (임베딩)

**Embedding(임베딩)**은 이산적인 토큰 id를 고정 길이의 **밀집 실수 벡터(dense vector)**로 바꾸는 표현 또는 그 변환 계층이다.

$$

\text{id } k \in \{0,\ldots,|V|-1\}
\quad\mapsto\quad
\mathbf{e}_k \in \mathbb{R}^{d}

$$

- $d$: **Embedding Dimension(임베딩 차원)**
- $\mathbf{e}_k$: 토큰 $k$의 임베딩 벡터

왜 필요한가?

1. 이산 기호를 연속 공간의 점으로 만들어 미분 가능한 연산을 연결한다.
2. 의미를 기하적으로 배치할 여지를 만든다.
3. Transformer 블록이 요구하는 입력 형태(`[seq, d]`)를 제공한다.

### 3.2 One-hot Representation

**One-hot(원-핫)** 표현은 길이 $|V|$ 벡터에서 해당 id 위치만 1, 나머지는 0인 표현이다.

예: $|V|=5$, id=3

$$

\mathbf{o}_3 = [0,0,0,1,0]

$$

특징:

- 모든 토큰이 서로 **직교**하고 같은 길이이다.
- “유사도”가 기본 표현에 없다. (내적이 항상 0)
- 차원이 vocab만큼 커 **희소하고 비효율적**이다.

### 3.3 One-hot × 행렬 = Embedding Lookup

Embedding 행렬을 행 벡터의 모음으로 둔다.

$$

E =
\begin{bmatrix}
— \mathbf{e}_0 — \\
— \mathbf{e}_1 — \\
\vdots \\
— \mathbf{e}_{|V|-1} — \\
\end{bmatrix}
\in \mathbb{R}^{|V| \times d}

$$

one-hot과 곱하면:

$$

\mathbf{o}_k^{\top} E = \mathbf{e}_k

$$

즉, **행렬곱으로 “k번째 행 선택”**이 일어난다.

실무에서는 곱셈을 실제로 하지 않고 **인덱싱(lookup)**으로 구현한다. 수학적으로는 동일하다.

```text
emb = E[token_id]   # lookup
```

이 등가가 중요하다. Embedding은 “사전에서 행을 꺼내는 일”이며, 그 행들이 학습된다.

### 3.4 Trainable Parameters

Embedding 행렬 $E$의 각 성분은 보통 **학습 가능한 파라미터**이다.

학습 전: 랜덤 초기화 (또는 사전학습된 벡터로 초기화)  
학습 중: Loss의 gradient가 각 행 $\mathbf{e}_k$를 업데이트

직관:

```text
토큰 k가 배치에 자주 등장하고
틀린 예측에 기여하면
→ e_k 가 움직인다
```

한 번도 안 나온 토큰의 행은 (이론상) 업데이트되지 않는다.  
희귀 토큰 Embedding이 덜 안정적인 이유이다.

### 3.5 Sequence Embedding

토큰 서열

$$

(x_1, x_2, \ldots, x_T)

$$

각각 lookup하면

$$

(\mathbf{e}_{x_1}, \mathbf{e}_{x_2}, \ldots, \mathbf{e}_{x_T})
\in \mathbb{R}^{T \times d}

$$

배치까지 있으면

$$

\mathbb{R}^{B \times T \times d}

$$

이것이 Transformer 스택의 기본 입력 텐서 형태이다.  
(이후에 positional embedding을 더하는 경우가 많다.)

### 3.6 Weight Tying (가중치 공유) 미리보기

많은 LM은 Embedding $E$와 출력 투영 $W_{\text{lm}}$을 공유한다.

$$

\text{logits} = H W^{\top}, \quad W = E

$$

이유:

- 파라미터 절약 ($|V| \times d$ 한 벌)
- 입력/출력 공간 정렬에 도움이 될 수 있음

필수는 아니다. 구조 선택이다.

## 4. 직관적으로 이해하기

Vocabulary를 도서관 서가, Embedding 벡터를 각 책의 “좌표 카드”라고 보자.

- id는 청구 기호(정수)
- Embedding은 그 책이 의미 공간에서 어디에 꽂혀 있는지(실수 벡터)
- 학습은 대출 기록(문맥)을 보고 좌표 카드를 조금씩 재배치하는 일

one-hot은 “서가 번호만 있고 내용 요약이 없는 상태”에 가깝다.  
Embedding은 “요약 좌표”를 부여한다.

또 다른 비유: 국가 번호를 위도·경도로 바꾸는 것.  
번호 82와 81이 가깝다는 뜻은 없지만, 임베딩 공간에서는 “한국어/일본어 문맥”이 가깝게 묶일 수 있다. (보장 아님, 학습 결과)

## 5. 수학적으로 이해하기

단일 토큰:

$$

\mathbf{e} = E[x] \in \mathbb{R}^{d}

$$

미니배치:

$$

X \in \{0,\ldots,|V|-1\}^{B \times T}

$$

$$

\text{Emb}(X)_{b,t,:} = E[X_{b,t}, :]

$$

파라미터 수:

$$

\#\theta_{\text{emb}} = |V| \cdot d

$$

(`padding_idx`가 있어도 저장 행 수는 보통 $|V|$이나, 해당 행 grad는 0)

Lookup은 미분 가능하다. $x=k$로 선택된 행만 gradient가 흐른다.

$$

\frac{\partial L}{\partial E_{k,:}}
=
\frac{\partial L}{\partial \mathbf{e}_k}
\quad (x=k\text{인 위치들의 합})

$$

## 6. 작은 숫자로 직접 계산하기

설정:

```text
|V| = 4
d   = 3

tokens:
0: <pad>
1: I
2: love
3: cats
```

Embedding 행렬 $E$ (임의 숫자):

$$

E =
\begin{bmatrix}
0.0 & 0.0 & 0.0 \\
1.0 & 0.2 & -0.5 \\
0.3 & 1.1 & 0.4 \\
-0.2 & 0.8 & 1.0 \\
\end{bmatrix}

$$

입력 서열: `"I love cats"` → ids `[1, 2, 3]`

Lookup 결과:

$$

\begin{align*}
E[1] &= [1.0,\ 0.2,\ -0.5] \\
E[2] &= [0.3,\ 1.1,\ 0.4] \\
E[3] &= [-0.2,\ 0.8,\ 1.0]
\end{align*}

$$

행렬로 모으면 shape `[3, 3]`:

$$

\begin{bmatrix}
1.0 & 0.2 & -0.5 \\
0.3 & 1.1 & 0.4 \\
-0.2 & 0.8 & 1.0
\end{bmatrix}

$$

one-hot 경로로 id=2를 꺼내 보자.

$$

\mathbf{o}_2 = [0,0,1,0]

$$

$$

\mathbf{o}_2^{\top} E = [0.3,\ 1.1,\ 0.4] = E[2]

$$

동일하다.

간단한 “유사도” 감각으로 내적을 보면:

$$

E[2]\cdot E[3]
=
0.3(-0.2) + 1.1(0.8) + 0.4(1.0)
=
0.0 + 0.88 + 0.4
=
1.28

$$

$$

E[1]\cdot E[3]
=
1.0(-0.2) + 0.2(0.8) + (-0.5)(1.0)
=
-0.2 + 0.16 - 0.5
=
-0.54

$$

이 숫자 자체에 의미 부여는 이르다. 다만 **벡터 공간에서 비교가 가능해졌다**는 점이 one-hot과의 차이다.

## 7. 코드로 구현하기 — NumPy Lookup

```python
# embedding_numpy.py
import numpy as np

# E shape = [V, d]
E = np.array(
    [
        [0.0, 0.0, 0.0],   # 0 pad
        [1.0, 0.2, -0.5],  # 1 I
        [0.3, 1.1, 0.4],   # 2 love
        [-0.2, 0.8, 1.0],  # 3 cats
    ],
    dtype=np.float32,
)

ids = np.array([1, 2, 3], dtype=np.int64)
vectors = E[ids]  # advanced indexing = lookup
print("shape:", vectors.shape)
print(vectors)

def one_hot_lookup(E: np.ndarray, idx: int) -> np.ndarray:
    V, d = E.shape
    one_hot = np.zeros((V,), dtype=np.float32)
    one_hot[idx] = 1.0
    return one_hot @ E  # (V,) @ (V,d) -> (d,)

print("one-hot path:", one_hot_lookup(E, 2))
print("index  path:", E[2])
```

## 8. PyTorch로 구현하기

```python
# embedding_torch.py
import torch
import torch.nn as nn

class TinyEmbedLM(nn.Module):
    """Embedding만 있는 초소형 모듈 — 인터페이스 확인용."""

    def __init__(self, vocab_size: int, d_model: int, pad_id: int = 0):
        super().__init__()
        self.tok_emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_id,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, T] long
        # returns:   [B, T, d]
        return self.tok_emb(input_ids)

def demo() -> None:
    torch.manual_seed(0)
    model = TinyEmbedLM(vocab_size=4, d_model=3, pad_id=0)

    input_ids = torch.tensor(
        [
            [1, 2, 3, 0],  # I love cats <pad>
            [1, 2, 0, 0],
        ],
        dtype=torch.long,
    )
    out = model(input_ids)
    print("out.shape:", out.shape)  # [2, 4, 3]

    # 파라미터 수
    n = sum(p.numel() for p in model.parameters())
    print("num params:", n)  # 4*3 = 12

    # padding_idx 행은 보통 0
    print("pad row:", model.tok_emb.weight[0].detach())

if __name__ == "__main__":
    demo()
```

학습 루프에서의 위치 (개념):

```python
logits = transformer(embedding(input_ids))  # 이후 강의
loss = cross_entropy(logits, labels)
loss.backward()  # Embedding.weight에도 grad 전달
optimizer.step()
```

## 9. One-hot Softmax 분류와의 관계

고전적인 선형 분류:

$$

\mathbf{o}^{\top} W + \mathbf{b}

$$

토큰 Embedding + 선형층도 구조적으로 비슷하다.  
차이는 $W$의 입력이 “거대한 one-hot”이 아니라, **이미 압축된 d차원 표현**이며, 그 표현이 여러 층에서 재조합된다는 점이다.

LLM에서는:

```text
id
 → Embedding (lookup)
   → Transformer blocks (문맥화)
     → 각 위치가 문맥을 반영한 벡터 h_t
       → Linear to |V| logits
```

첫 Embedding은 **문맥 없음(context-free)** 초기 표현이다.  
같은 `"bank"` 토큰도 문맥에 따라 이후 층에서 다른 벡터가 된다. (Contextualized Representation)

## 10. 실제 LLM에서는 어떻게 사용하는가

대표 설정 예 (대략적 스케일):

| 모델 규모 감각 | \|V\| | d |
|---|---|---|
| 소형 GPT | ~50k | 768 |
| 중형 | ~32k~100k | 2048+ |
| 대형 | 유사/다양 | 4096+ |

저장 파일에서 흔히 보는 이름:

```text
transformer.wte.weight   # word token embedding, [|V|, d]
transformer.wpe.weight   # positional embedding (절대 위치 방식일 때)
```

순전파 스케치 (절대 위치 임베딩을 더하는 경우):

```text
x = wte(input_ids) + wpe(positions)
```

최신 모델은 RoPE 등 **상대/회전 위치 인코딩**을 Attention 쪽에 넣기도 한다.  
그래도 **토큰 Embedding lookup 자체**는 공통이다.

실무 팁:

1. `resize_token_embeddings` 후 새 행은 랜덤 → 해당 토큰을 쓰는 추가 학습 필요
2. Embedding dtype(fp16/bf16)과 안정성
3. 대형 vocab은 메모리 대역폭 이슈가 될 수 있음 (추론 최적화 주제는 후반 권)

## 11. 실습

### 실습 1. Lookup과 행렬곱 등가

NumPy로 `E[ids]`와 one-hot 행렬곱 결과가 같은지 `np.allclose`로 검증하시오.

### 실습 2. 파라미터 수

`|V|=50257`, `d=768`일 때 Embedding 파라미터 수를 계산하고, 전체 모델이 1.2억 파라미터라면 Embedding이 차지하는 비율을 어림하시오.

### 실습 3. padding_idx

`padding_idx=0`인 `nn.Embedding`에서 pad 위치를 forward한 뒤, `loss.backward()` 후에도 `weight.grad[0]`가 0에 가까운지 확인하시오.

### 실습 4. 문맥 없음 확인

같은 토큰 id가 문장 앞·뒤에 와도 **Embedding 직후 벡터는 동일**함을 코드로 보이시오.  
(문맥화는 Transformer 이후라는 점 고정)

## 12. 자주 하는 실수

1. **id를 float 벡터로 착각**  
   `nn.Linear`에 id를 직접 넣지 않는다. 먼저 Embedding.

2. **shape 혼동**  
   입력 `[B,T]` → 출력 `[B,T,d]`. `d`를 빼먹으면 Attention이 깨진다.

3. **vocab 불일치**  
   Tokenizer id ≥ `num_embeddings` 이면 런타임 에러.

4. **Embedding을 freeze한 채 전체 성능을 기대**  
   도메인 적응 시 토큰 벡터도 같이 움직이는 편이 나은 경우가 많다. (전략 문제)

5. **one-hot을 실제로 물질화**  
   $|V|$가 크면 메모리 폭발. 항상 lookup 사용.

6. **유사도 과해석**  
   학습 초기·소량 데이터에서 코사인 유사도를 “의미”로 단정하지 말 것.

## 13. 핵심 정리

- Embedding은 token id를 $d$차원 밀집 벡터로 바꾼다.
- 수학적으로 one-hot × $E$와 같고, 구현은 행 lookup이다.
- $E$는 학습 파라미터이며 크기 $|V|\times d$이다.
- 서열 Embedding의 shape는 `[B, T, d]`이다.
- Embedding 직후 벡터는 문맥 없음; 문맥화는 이후 Transformer가 담당한다.
- LLM 파이프라인에서 Tokenizer 다음, Attention 이전의 필수 계층이다.

## 14. 핵심 용어

| 용어 | 의미 |
|---|---|
| Embedding | id→밀집 벡터 변환/표현 |
| Embedding Matrix $E$ | `[V, d]` 학습 가중치 |
| Embedding Dimension $d$ | 벡터 길이 |
| Lookup | id로 행을 선택 |
| One-hot | 해당 인덱스만 1인 희소 표현 |
| Contextualized Representation | 문맥을 반영한 이후 층 벡터 |
| Weight Tying | Embedding과 LM Head 가중치 공유 |
| `padding_idx` | 패딩 행 학습 억제 옵션 |

## 15. 연습 문제
### 문제 1 (개념)

Embedding이 one-hot보다 선호되는 이유 두 가지를 쓰시오.

### 문제 2 (수학)

$|V|=6$, $d=4$일 때 $E$의 shape와 파라미터 수는?

### 문제 3 (계산)

$E$의 2번째 행(0-index면 id=1)이 `[0.1, -0.2, 0.3]`일 때, id 서열 `[1,1,0]`의 lookup 결과 shape과 첫 벡터를 쓰시오. (`d=3`)

### 문제 4 (코드)

`nn.Embedding(1000, 128)`의 `weight.shape`는?

### 문제 5 (LLM 연결)

왜 같은 토큰이 문장마다 다른 “의미”로 쓰여도, Embedding 테이블 행은 하나뿐인가? 문맥 차이는 어디서 생기는가?

### 문제 6 (등가)

one-hot 벡터와 $E$의 곱이 lookup과 같다는 것을 한 줄로 설명하시오.

---

## 정답 및 해설

### 문제 1

(1) 차원이 $d \ll |V|$로 압축되어 효율적이다.  
(2) 학습으로 유사/관련 토큰을 가까운 벡터로 배치할 수 있다.  
(추가로: 희소 one-hot을 물질화하지 않아도 됨)

### 문제 2

shape `[6, 4]`, 파라미터 $24$

### 문제 3

shape `[3, 3]`, 첫 벡터 `[0.1, -0.2, 0.3]`

### 문제 4

`torch.Size([1000, 128])`

### 문제 5

Embedding은 문맥 없는 초기 표현만 제공한다. 문맥에 따른 차이는 Attention 등 이후 층이 주변 토큰 정보를 섞으며 만든다.

### 문제 6

one-hot이 켜진 위치의 행만 남기므로 $E$의 해당 행을 선택하는 것과 같다.

## 16. 다음 강의와 연결

이번 강의에서 id가 벡터가 되는 통로를 열었다.

다음 **제32강. Language Model과 Next Token Prediction**에서는, 이 벡터 서열을 조건으로 **다음 토큰 확률** $P(x_{t+1}\mid x_{\le t})$을 모델링한다는 목표를 정식화한다. Teacher Forcing의 미리보기도 포함한다.

이전 강의: **제30강. Vocabulary와 Special Tokens**  
다음 강의: **제32강. Language Model과 Next Token Prediction**

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [30강. Vocabulary와 Special Tokens](30강_Vocabulary와_Special_Tokens.md)
- **다음 강:** [32강. Language Model과 Next Token Prediction](32강_Language_Model과_Next_Token_Prediction.md)

<!-- /LECTURE_NAV -->
