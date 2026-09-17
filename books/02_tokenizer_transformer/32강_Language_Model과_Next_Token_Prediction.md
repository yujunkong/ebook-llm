# 32강. Language Model과 Next Token Prediction
## 이번 강에서 배우는 내용

- Language Model(언어 모델)의 정의
- Next Token Prediction(다음 토큰 예측)이 조건부 확률 $P(x_{t+1}\mid x_{\le t})$로 쓰이는 방식
- Autoregressive(자기회귀) 생성 루프
- 학습 시 서열을 한 칸씩 밀어 타깃을 만드는 법
- Teacher Forcing의 개념(상세 Loss는 34강)
- GPT 파이프라인에서 “이해/생성”이 같은 목표에서 나오는 이유

## 왜 중요한가?
표면적으로 LLM은 질문에 답하고, 코드를 짜고, 요약을 한다.  
훈련 목표의 핵심은 놀랍도록 단일한 경우가 많다.

> 지금까지의 토큰을 보고, **다음 토큰**을 맞춰라.

이 한 문장이 Pretraining의 뼈대이다.  
목표가 흐리면 Softmax·Loss·Decoding 전략이 각각 따로 노는 지식 조각이 된다.  
목표가 맑으면 이후 강의가 한 줄로 연결된다.

```text
context tokens
  → model → logits over V
    → Softmax → P(next)
      → Cross Entropy with true next
        → 파라미터 업데이트
```

## 선수 개념
- Token / Vocabulary (27·30강)
- Embedding 서열 `[T, d]` (31강)
- 확률의 초등 성질: $0 \le p \le 1$, $\sum p = 1$
- 조건부 확률 $P(A\mid B)$의 말뜻: “B를 알 때 A의 확률”
- Softmax가 “점수를 확률로” 바꾼다는 감각 (33강에서 계산)

Transformer 내부(Attention)는 아직 블랙박스로 두어도 된다.  
오늘은 **입출력 계약**에 집중한다.

## 핵심 개념
### 3.1 Language Model (언어 모델)

**Language Model(랭귀지 모델, 언어 모델)**은 토큰 서열의 **확률 분포**를 모형화하는 모델이다.

서열 $x_1, x_2, \ldots, x_T$에 대해, 이상적으로는 결합 확률

$$

P(x_1, x_2, \ldots, x_T)

$$

를 부여한다.

왜 필요한가?

- “자연스러운 문장”에 높은 확률
- “이상한 문장”에 낮은 확률
- 생성: 높은 확률의 다음 토큰을 반복 샘플링

현대 LLM의 주류는 아래의 **자기회귀 분해**를 쓴다.

### 3.2 Chain Rule Decomposition

확률의 연쇄 법칙:

$$

P(x_1,\ldots,x_T)
=
P(x_1)\,P(x_2\mid x_1)\,P(x_3\mid x_1,x_2)\cdots
P(x_T\mid x_1,\ldots,x_{T-1})

$$

짧게:

$$

P(x_{1:T})
=
\prod_{t=1}^{T} P(x_t \mid x_{<t})

$$

여기서 $x_{<t} = (x_1,\ldots,x_{t-1})$, $t=1$일 때 조건은 빈 문맥(또는 BOS).

Language Modeling 학습은 각 항

$$

P(x_t \mid x_{<t})

$$

을 잘 맞추도록 파라미터를 조정하는 일에 가깝다.

### 3.3 Next Token Prediction (다음 토큰 예측)

**Next Token Prediction(넥스트 토큰 프리딕션, 다음 토큰 예측)**은 주어진 문맥 $x_{\le t}$ (또는 $x_{<t}$)에서 **바로 다음 토큰**의 분포를 예측하는 과제이다.

표기 (강의마다 경계가 1칸 차이 날 수 있음):

$$

P_\theta(x_{t+1} \mid x_{\le t})

$$

예:

```text
문맥:  "오늘 날씨가"
다음:  "좋다" 에 높은 확률
      "의자" 에 낮은 확률
```

GPT식 Causal LM은 이 예측을 모든 위치에서 수행한다.

### 3.4 Autoregressive Language Model

**Autoregressive(오토리그레시브, 자기회귀) Language Model**은 이미 생성된 토큰을 다음 예측의 조건으로 **다시 넣는** 모델이다.

생성 루프:

```text
1. 프롬프트 토큰열 x_1..x_t 준비
2. P(x_{t+1} | x_1..x_t) 계산
3. 다음 토큰을 샘플/선택
4. 서열 끝에 붙이고 t ← t+1
5. EOS 또는 길이 제한까지 2~4 반복
```

“자기회귀”인 이유: 출력이 다음 단계의 입력 일부가 된다.

비유: 한 줄씩 글을 쓰며, 방금 쓴 줄을 다시 읽고 다음 줄을 쓰는 작가.

### 3.5 Teacher Forcing (티처 포싱) 미리보기

**Teacher Forcing(티처 포싱)**은 학습 시, 모델이 이전 스텝에서 틀린 예측을 했다 해도 **정답 토큰을 다음 입력으로 넣어** 주는 학습 방식이다.

학습 데이터에 정답 서열이 있을 때:

```text
정답: [BOS, A, B, C, EOS]

위치0 입력 문맥: [BOS]      타깃: A
위치1 입력 문맥: [BOS, A]   타깃: B   ← A는 모델 출력이 아니라 정답
위치2 입력 문맥: [BOS, A,B] 타깃: C
...
```

장점: 병렬로 모든 위치의 Loss를 한 번에 계산하기 쉽다. (Transformer + causal mask)

단점/이슈: 학습 때는 정답 문맥을 보다가, 추론 때는 자기 예측 문맥을 보게 되어 불일치가 생길 수 있다. (**Exposure Bias** — 이름만 기억)

상세한 Loss 계산은 34강 Cross Entropy에서 이어진다.

### 3.6 Context / Conditioning

문맥이 길수록 조건이 풍부해진다.

$$

P(\text{bank} \mid \text{river})
\quad\text{vs}\quad
P(\text{bank} \mid \text{money deposit})

$$

같은 다음 단어 후보라도 문맥에 따라 확률이 달라진다.  
Embedding만으로는 이 차이가 없고, **문맥을 섞는 모델(Transformer)**이 필요하다.

## 직관적으로 이해하기
빈칸 채우기 시험과 같다.

```text
나는 ___ 를 먹었다.
```

후보: `사과`, `어제`, `푸름` …

언어 모델은 매 순간 빈칸 시험을 본다.  
다만 빈칸이 항상 “맨 끝”에 있고, 채운 답이 다음 문장의 일부가 된다.

사전학습은 인터넷 규모 텍스트에서 이 빈칸 시험을 수없이 반복하는 것과 같다.  
그 부산물로 문법·사실·추론의 흔적이 통계 속에 남는다.  
“이해”처럼 보이는 행동도, 메커니즘 수준에서는 **다음 토큰 분포의 정확도**에서 출발한다.

## 수학적으로 이해하기
모델 파라미터 $\theta$에 대해

$$

P_\theta(x_t \mid x_{<t})
=
\text{Softmax}\big(f_\theta(x_{<t})\big)_{x_t}

$$

여기서 $f_\theta(x_{<t}) \in \mathbb{R}^{|V|}$는 **logit 벡터** (33강).

학습 목표(음의 로그가능도, NLL):

$$

L
=
-\sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})

$$

평균을 내면 Cross Entropy와 연결된다 (34강).

생성 시 탐욕(greedy):

$$

\hat{x}_{t+1} = \arg\max_{v} P_\theta(v \mid x_{\le t})

$$

샘플링:

$$

\hat{x}_{t+1} \sim P_\theta(\cdot \mid x_{\le t})

$$

(온도·top-k 등은 33강·후반 생성 강의)

## 작은 숫자로 직접 보기
초소형 vocab:

```text
0: <bos>
1: I
2: love
3: cats
4: dogs
5: <eos>
```

학습 문장:

```text
I love cats
→ ids: [0, 1, 2, 3, 5]
```

Teacher Forcing용 입·출력 쌍:

```text
입력 ids (context):  [0, 1, 2, 3]
타깃 ids (next):     [1, 2, 3, 5]
```

각 위치에서 모델은 $|V|=6$개 점수를 낸다.  
예를 들어 위치 t=1 (문맥 `[0,1]` 즉 `BOS I`)에서 이상적 분포:

```text
P(love)=0.70
P(cats)=0.05
P(dogs)=0.10
P(I)=0.05
P(<eos>)=0.05
P(<bos>)=0.05
```

정답이 `love`(id=2)이면, 학습은 $-\log 0.70$을 줄이는 방향이다.

생성 예 (탐욕):

```text
start: [0]
→ next=1 (I)
→ [0,1]
→ next=2 (love)
→ [0,1,2]
→ next=3 (cats)
→ [0,1,2,3]
→ next=5 (<eos>)  stop
```

## 코드로 구현하기 — 타깃 shift
```python
# ntp_shift.py
import torch

def make_causal_lm_batch(token_ids: torch.Tensor):
    """
    token_ids: [B, T]  (이미 BOS/EOS 포함 가능)
    returns:
      input_ids: [B, T-1]
      labels:    [B, T-1]
    """
    input_ids = token_ids[:, :-1]
    labels = token_ids[:, 1:]
    return input_ids, labels

if __name__ == "__main__":
    # batch 2개 문장 (패딩 없음 가정)
    batch = torch.tensor(
        [
            [0, 1, 2, 3, 5],  # BOS I love cats EOS
            [0, 1, 2, 4, 5],  # BOS I love dogs EOS
        ],
        dtype=torch.long,
    )
    x, y = make_causal_lm_batch(batch)
    print("input :\n", x)
    print("labels:\n", y)
```

출력 개념:

```text
input:
 tensor([[0, 1, 2, 3],
         [0, 1, 2, 4]])
labels:
 tensor([[1, 2, 3, 5],
         [1, 2, 4, 5]])
```

모든 위치를 한 번에 학습할 수 있는 형태이다.

## 코드로 구현하기 — 초소형 Autoregressive 루프
모델 본체는 “logit을 내는 함수”로 추상화한다.

```python
# tiny_generate.py
from typing import List

import torch
import torch.nn.functional as F

def greedy_generate(
    logits_fn,
    prompt_ids: List[int],
    eos_id: int,
    max_new_tokens: int = 20,
) -> List[int]:
    """
    logits_fn(ids: LongTensor[1, T]) -> FloatTensor[1, T, V]
    마지막 위치의 logit으로 다음 토큰을 고른다.
    """
    ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        x = torch.tensor([ids], dtype=torch.long)
        logits = logits_fn(x)          # [1, T, V]
        next_logits = logits[0, -1]    # [V]
        next_id = int(torch.argmax(next_logits).item())
        ids.append(next_id)
        if next_id == eos_id:
            break
    return ids

def sample_generate(
    logits_fn,
    prompt_ids: List[int],
    eos_id: int,
    temperature: float = 1.0,
    max_new_tokens: int = 20,
) -> List[int]:
    ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        x = torch.tensor([ids], dtype=torch.long)
        logits = logits_fn(x)[0, -1] / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        next_id = int(torch.multinomial(probs, num_samples=1).item())
        ids.append(next_id)
        if next_id == eos_id:
            break
    return ids

# 데모: "가짜 모델" — 단순히 다음 정답을 살짝 선호
class FakeBigram:
    def __init__(self, vocab_size: int):
        self.V = vocab_size
        # 전이: 0→1→2→3→5, 그 외는 균등에 가깝게
        self.table = {
            (0,): 1,
            (0, 1): 2,
            (0, 1, 2): 3,
            (0, 1, 2, 3): 5,
        }

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        logits = torch.zeros(B, T, self.V)
        for b in range(B):
            for t in range(T):
                ctx = tuple(int(v) for v in x[b, : t + 1].tolist())
                # 가장 긴 매칭 키
                prefer = None
                for k, v in self.table.items():
                    if ctx[-len(k) :] == k:
                        prefer = v
                if prefer is not None:
                    logits[b, t, prefer] = 5.0  # 높은 logit
                else:
                    logits[b, t] = 0.0
        return logits

if __name__ == "__main__":
    model = FakeBigram(vocab_size=6)
    out = greedy_generate(model, prompt_ids=[0], eos_id=5)
    print("generated ids:", out)
```

진짜 Transformer가 아니어도, **API 계약**(ids in → logits out → next id)은 LLM과 같다.

## LLM에서는 어디에 사용될까?
### 9.1 Pretraining

대규모 웹/책/코드 텍스트를 토큰 스트림으로 만들고,

```text
P(next token | previous tokens)
```

을 최소화( NLL )한다. 지시문이 없어도 “다음 토큰”만으로 일반 능력이 생긴다.

### 9.2 SFT / Chat

지시-응답 데이터도 결국 토큰 서열이다.

```text
[system...][user...][assistant 정답...]
```

Loss는 보통 **assistant 구간**에만 걸어, 모델이 답변을 생성하도록 맞춘다.  
형식은 special token 템플릿(30강)으로 고정한다.

### 9.3 Inference

```text
prompt → encode → 반복:
  forward(캐시 포함 가능)
  → 마지막 위치 Softmax
  → decode 전략(greedy/sample/beam)
  → append
→ detokenize
```

KV Cache 등은 5권 주제이나, 루프 구조는 오늘과 동일하다.

## “이해”와 Next Token Prediction
질문:

> 다음 단어만 맞추는데 어떻게 추론을 하나?

짧은 답:

- 긴 문맥에서 다음 토큰을 맞추려면 **사실·관계·형식**을 내부 상태에 압축하는 편이 유리하다.
- 그 압축이 외부에서 “이해”처럼 보인다.
- 한계도 같은 목표에서 온다: 확률적으로 그럴듯하지만 틀린 토큰(환각)이 샘플링될 수 있다.

이 책의 입장은 신비화를 걷어내는 것이다.  
메커니즘은 Next Token Prediction이고, 능력은 그 위의 창발·정렬·도구사용으로 확장된다.

## 실습
### 실습 1. Shift 타깃 만들기

문장 하나를 tokenize한 뒤 `input_ids`/`labels`를 만들고, 각 위치의 (문맥 → 정답 토큰)을 표로 쓰시오.

### 실습 2. 탐욕 vs 샘플링

온도 $T=0.2$와 $T=1.5$에서 짧은 프롬프트 생성을 비교하고, 다양성 차이를 관찰하시오. (33강 예고)

### 실습 3. 조건부 확률 문장

다음을 확률 기호로 쓰시오.

```text
"서울" 다음에 "의"가 올 확률
"대한민국의 수도는" 다음에 "서울"이 올 확률
```

### 실습 4. Teacher Forcing 한 줄 정의

학습과 추론에서 “다음에 넣는 토큰의 출처”가 어떻게 다른지 두 문장으로 쓰시오.

## 자주 하는 실수
1. **Language Model = 챗봇**으로만 이해  
   챗은 LM + 템플릿 + (대개) 추가 학습의 결과이다.

2. **타깃 shift를 한 칸 반대로 함**  
   입·출력이 어긋나면 Loss가 의미 없어진다.

3. **모든 위치에 같은 문맥만 있다고 착각**  
   Causal mask 때문에 위치 t는 과거만 본다.

4. **생성 품질만 보고 학습 목표를 잊음**  
   디코딩 전략과 학습 Loss는 별개 레버이다.

5. **프롬프트 토큰에도 무조건 Loss**  
   SFT에서는 마스크 설계가 중요하다.

## 핵심 요약
- Language Model은 토큰 서열의 확률을 모형화한다.
- 자기회귀 분해로 $P(x_t\mid x_{<t})$를 학습·생성에 사용한다.
- Next Token Prediction이 GPT식 LLM의 중심 목표이다.
- 학습은 보통 Teacher Forcing + 병렬 타깃 shift로 수행한다.
- 추론은 예측 토큰을 다시 조건에 넣는 autoregressive 루프이다.
- Softmax(33)·Cross Entropy(34)가 이 목표를 숫자로 만든다.

## 용어 사전
| 용어 | 의미 |
|---|---|
| Language Model | 토큰 서열 확률 모델 |
| Next Token Prediction | 다음 토큰 분포 예측 과제 |
| Autoregressive | 출력을 다음 입력 조건으로 재사용 |
| Teacher Forcing | 학습 시 정답 토큰을 다음 입력으로 제공 |
| Exposure Bias | 학습/추론 문맥 불일치 문제 |
| NLL | 음의 로그가능도 |
| Greedy Decoding | 최대 확률 토큰만 선택 |
| Sampling | 분포에서 확률적으로 선택 |

## 연습문제
### 문제 1 (개념)

자기회귀 언어 모델에서 결합 확률 $P(x_{1:T})$를 조건부 확률의 곱으로 쓰시오.

### 문제 2 (절차)

프롬프트 `[BOS, I, love]`에서 탐욕 생성 한 스텝이 하는 일을 순서대로 쓰시오.

### 문제 3 (학습)

서열 `[0,1,2,3]`의 Teacher Forcing 타깃 shift 결과를 쓰시오.

### 문제 4 (구분)

학습 시 Teacher Forcing과 추론 시 autoregressive 입력의 차이를 설명하시오.

### 문제 5 (LLM 연결)

Pretraining과 챗 SFT가 둘 다 Next Token Prediction일 수 있다면, 무엇이 다른가?

---

## 정답 및 해설
### 문제 1

$P(x_{1:T})=\prod_{t=1}^{T} P(x_t\mid x_{<t})$

### 문제 2

모델 forward → 마지막 위치의 vocab 분포 계산 → argmax로 다음 id 선택 → 서열에 append

### 문제 3

입력 `[0,1,2]` / 타깃 `[1,2,3]` (맨 끝 예측까지 포함하면 입력 `[0,1,2]` 타깃 `[1,2,3]` 또는 전체 길이 설계에 따라 `[0,1,2,3]`→`[1,2,3,EOS]` 형태. 핵심은 한 칸 shift.)

더 명확히: 입력 `[0,1,2]`, 타깃 `[1,2,3]`

### 문제 4

학습은 정답 과거 토큰을 조건으로 넣고, 추론은 모델이 방금 샘플한 토큰을 조건으로 넣는다.

### 문제 5

데이터 분포·포맷(문서 스트림 vs 지시-응답 템플릿)과 Loss 마스크(어떤 토큰을 예측 대상으로 삼는지)가 다르다. 목표의 수학 형태는 같은 가족이다.

## 다음 강의와 연결
이번 강의에서 “다음 토큰 확률”이라는 목표를 고정했다.

다음 **제33강. Softmax와 Logit**에서는, 모델이 내는 점수(logit)를 확률로 바꾸는 Softmax를 **손으로 계산**하고, Temperature 미리보기와 수치 안정성을 다룬다.  
34강 Cross Entropy Loss로 바로 이어진다.

이전 강의: **제31강. Embedding — 토큰을 벡터로**  
다음 강의: **제33강. Softmax와 Logit**

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [31강. Embedding — 토큰을 벡터로](31강_Embedding_토큰을_벡터로.md)
- **다음 강:** [33강. Softmax와 Logit](33강_Softmax와_Logit.md)

<!-- /LECTURE_NAV -->
