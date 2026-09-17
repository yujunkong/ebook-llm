# 1강. LLM을 밑바닥부터 배운다는 것
## 이번 강에서 배우는 내용

- AI, Machine Learning, Deep Learning, LLM이 어떻게 다른지
- 이 책이 “밑바닥부터”라고 말하는 이유가 무엇인지
- Python부터 vLLM·DGX Spark까지 이어지는 한 줄의 학습 경로
- 각 권이 전체 여정에서 어떤 역할을 하는지

## 왜 중요한가?
LLM을 처음 접하면 용어가 한꺼번에 쏟아진다.

Transformer, Attention, Token, Embedding, Pretraining, SFT, RLHF, KV Cache, vLLM…

각각을 따로 외우면 금방 잊힌다. 반대로, “지금 배우는 개념이 나중에 어디에 붙는지”를 먼저 알면 이후 강의가 훨씬 빠르게 이해된다.

이번 강의의 목적은 지식의 양보다 **학습의 좌표**를 잡는 것이다.

## 선수 개념
아직은 수학이나 PyTorch를 몰라도 된다. 다만 다음 태도는 필요하다.

1. **실행한다** — 코드를 읽기만 하지 않고 직접 돌려 본다.
2. **계산한다** — 수식을 보기만 하지 않고 작은 숫자로 손으로 확인한다.
3. **연결한다** — 기초 개념이 LLM의 어느 부품인지 계속 묻는다.

이 세 가지가 이 책의 학습 습관이다.

## 핵심 개념
### 3.1 AI란 무엇인가

**AI(Artificial Intelligence, 인공지능)**는 사람이 하던 판단, 분류, 생성, 계획 같은 작업을 컴퓨터 시스템이 수행하도록 만드는 기술 분야이다.

예를 들어 다음이 모두 AI의 결과물일 수 있다.

- 사진을 보고 “고양이”라고 분류하기
- 문장을 보고 감정을 추정하기
- 질문에 답하는 글을 생성하기
- 바둑에서 다음 수를 고르기

중요한 점은 AI가 하나의 알고리즘 이름이 아니라 **목표를 가리키는 큰 우산**이라는 것이다.

### 3.2 Machine Learning

**Machine Learning(머신러닝, 기계학습)**은 규칙을 사람이 일일이 작성하지 않고, **데이터로부터 규칙을 학습**하는 방법이다.

전통적인 프로그램은 대략 이렇게 동작한다.

```text
입력 + 사람이 쓴 규칙 → 출력
```

머신러닝은 이렇게 동작한다.

```text
입력 + 정답(또는 신호) → 학습 → 모델
모델 + 새 입력 → 출력
```

예를 들어 “스팸 메일 분류”를 생각해 보자.

- 규칙 기반: “무료”, “당첨” 같은 단어를 사람이 목록으로 관리한다.
- 머신러닝: 수많은 메일과 스팸/정상 라벨을 주고, 모델이 패턴을 스스로 찾게 한다.

LLM도 결국 이 큰 줄기 위에 있다. 다만 다루는 데이터가 텍스트이고, 모델이 매우 크며, 목표가 “다음 토큰 예측”과 그 위의 여러 학습 단계로 확장된다.

### 3.3 Deep Learning

**Deep Learning(딥러닝)**은 Machine Learning의 한 갈래로, **Neural Network(신경망)**를 여러 층으로 쌓아 복잡한 패턴을 학습하는 방법이다.

여기서 **Deep(깊다)**는 네트워크의 층(layer)이 많다는 뜻에 가깝다.

왜 층을 쌓는가?

- 앞쪽 층은 단순한 특징을 잡는다.
- 뒤쪽 층은 그 특징을 조합해 더 추상적인 표현을 만든다.

이미지에서는 “선 → 모서리 → 눈/귀 → 고양이”처럼 계층이 생길 수 있다.  
언어에서는 “글자/토큰 → 구 → 문장 의미 → 맥락”처럼 계층적 표현이 학습된다.

이 책의 후반에서 다루는 Transformer와 GPT도 Deep Learning의 구체적 구조이다.

### 3.4 LLM

**LLM(Large Language Model, 대규모 언어 모델)**은 대규모 텍스트 데이터로 학습되어, 자연어를 이해하고 생성하는 데 특화된 큰 언어 모델이다.

핵심 아이디어는 생각보다 단순하다.

> 지금까지의 토큰들을 보고, **다음에 올 토큰을 예측**한다.

이 목표를 **Next Token Prediction(다음 토큰 예측)**이라고 부른다.

예를 들어 입력이 다음과 같다고 하자.

```text
오늘은 날씨가
```

모델은 가능한 다음 토큰들에 점수를 매긴다.

```text
좋다    높은 점수
맑다    높은 점수
의자    낮은 점수
```

학습이 잘 되면 “자연스러운 이어짐”에 높은 점수를 주고, 이를 반복해 문장·문서·코드를 생성한다.

수식으로 쓰면, 지금까지의 토큰 서열을 $x_1,x_2,\ldots,x_t$라 할 때 모델은 다음 토큰의 조건부 분포

$$
P(x_{t+1}\mid x_1,\ldots,x_t)
$$

를 근사합니다. 점수를 logits $\mathbf{z}\in\mathbb{R}^{V}$로 만들고, Softmax로 확률로 바꿉니다.

$$
P(x_{t+1}=k\mid x_{\le t}) = \frac{e^{z_k}}{\sum_{j=1}^{V} e^{z_j}}
$$

학습은 대개 정답 다음 토큰 $y$에 대한 음의 로그 확률(크로스 엔트로피)을 줄이는 일입니다.

$$
L = -\log P(y\mid x_{\le t})
$$

1권에서는 Softmax·CE를 본격 구현하기 전에, **“점수 → 선택 → 반복”** 감각과 Tensor·미분·학습 루프를 먼저 익힙니다.

### 미니 예제 — 세 토큰 점수만으로 보기

어휘가 $\{$좋다, 맑다, 의자$\}$이고 점수가 $(2.0,\,1.0,\,0.0)$이면, Softmax 없이도 “가장 큰 점수”를 고르는 greedy 선택은

$$
\hat{y} = \arg\max_k z_k = \text{좋다}
$$

입니다. 확률로 바꾸면(손으로 대략)

$$
e^{2}\approx 7.39,\ e^{1}\approx 2.72,\ e^{0}=1
\quad\Rightarrow\quad
P(\text{좋다})\approx \frac{7.39}{11.11}\approx 0.67
$$

처럼 “높다/낮다”가 숫자로 드러납니다. 상세 계산은 이후 Loss·Softmax 강의에서 다룹니다.

하지만 실제 서비스용 LLM은 이 한 단계로 끝나지 않는다. 대개 다음 순서를 거친다.

```text
Pretraining
  → SFT(Instruction Tuning)
    → Preference Learning / RLHF / DPO / GRPO ...
      → Inference / Serving 최적화
```

이 책의 3권~5권이 바로 이 후반 여정이다.

## 직관적으로 이해하기
LLM을 처음 보면 “생각하는 기계”처럼 느껴질 수 있다. 더 정확한 직관은 다음이다.

**LLM은 매우 큰 다음 단어 맞히기 기계이다.**

다만 그 “맞히기”가 다음으로 확장된다.

1. 문법적으로 자연스러운 문장
2. 질문에 맞는 답
3. 코드 작성
4. 여러 단계 추론
5. 도구를 호출하는 에이전트 행동

확장의 기반은 여전히 **표현(Embedding)**, **문맥 참조(Attention)**, **대규모 학습(Pretraining/Post-Training)**, **빠른 실행(Inference)**이다.

따라서 이 책은 마법처럼 보이는 능력을, 다시 부품으로 분해한다.

## 이 책이 말하는 “밑바닥부터”의 의미
“밑바닥부터”는 두 가지를 동시에 의미한다.

### 5.1 라이브러리 사용법만 배우지 않는다

예를 들어 `loss.backward()`를 호출하는 법만 알면, 학습이 실패했을 때 원인을 찾기 어렵다.

이 책은 가능하면 다음 순서를 밟는다.

```text
직관
 → 용어
 → 수학
 → 작은 숫자 계산
 → Python / NumPy
 → PyTorch
 → 실제 LLM에서의 위치
 → 실습
```

모든 개념에 모든 단계를 억지로 넣지는 않는다. 다만 **핵심 개념은 건너뛰지 않는다.**

### 5.2 처음부터 거대한 모델을 만들지 않는다

처음에는 작은 숫자, 작은 네트워크, 작은 Transformer로 원리를 확인한다.  
원리가 손에 익으면 규모를 키운다.

이 순서가 중요한 이유:

- 작은 예제는 손으로 검증할 수 있다.
- 버그를 “느낌”이 아니라 계산으로 잡을 수 있다.
- 나중에 큰 모델을 다룰 때, 각 모듈이 무엇을 하는지 이미 알고 있다.

## 전체 학습 지도
이 책의 최종 경로는 다음과 같다.

```text
Python
→ 수학
→ Tensor
→ Neural Network
→ Backpropagation
→ PyTorch
→ Tokenizer
→ Embedding
→ Attention
→ Transformer
→ GPT
→ Pretraining
→ SFT
→ Preference Learning
→ RLHF
→ PPO
→ DPO
→ GRPO
→ RLVR
→ Reasoning
→ LLM Inference
→ vLLM
→ GLM
→ DGX Spark
→ LLM 최적화
```

한 줄로 압축하면 이렇다.

> **모델을 이해하고, 만들고, 학습시키고, 정렬하고, 빠르게 실행한다.**

## 5권의 역할
### 📗 1권 (1~26강) — 기초 체력

Python, NumPy, 선형대수, 미분, Gradient, Loss, Backpropagation, PyTorch를 다룬다.

목표: “학습이 어떻게 일어나는지”를 직접 계산하고 구현할 수 있다.

### 📘 2권 (27~54강) — LLM의 심장

Tokenizer, Embedding, Attention, Multi-Head Attention, RoPE, Transformer Block을 다룬다.

목표: 텍스트가 토큰이 되고, 토큰이 서로 참조하며, 다음 토큰을 예측하는 구조를 직접 만든다.

### 📙 3권 (55~78강) — 학습의 현실

GPT 구현, 생성 전략, Pretraining, Checkpoint, SFT, LoRA/QLoRA를 다룬다.

목표: 작은 GPT를 학습시키고, 지시 따르기 능력을 붙일 수 있다.

### 📕 4권 (79~98강) — 정렬과 추론 학습

강화학습 기초, Reward Model, RLHF, PPO, DPO, GRPO, RLVR, Reasoning Training을 다룬다.

목표: “더 나은 답”을 고르도록 모델을 조정하는 방법을 이해한다.

### 📔 5권 (99~120강) — 실행과 최적화

Prefill/Decode, KV Cache, Continuous Batching, vLLM, PagedAttention, GLM, MoE, DGX Spark, Tensor Parallel, NCCL을 다룬다.

목표: 학습된 모델을 실제로 서빙하고, GPU 관점에서 성능을 해석한다.

## 작은 숫자로 맛보기 — “다음 토큰” 직관
아직 Softmax를 자세히 배우지 않았다. 그래도 “점수 → 확률” 감각만 확인하자.

모델이 세 후보 토큰에 다음 점수를 줬다고 가정한다.

| 토큰 | 점수(logit) |
|---|---:|
| 좋다 | 2.0 |
| 맑다 | 1.0 |
| 의자 | 0.0 |

점수가 높을수록 선택될 가능성이 커진다.  
나중에 배우는 Softmax는 이 점수를 합이 1인 확률로 바꾼다.

지금은 다음만 기억하면 충분하다.

- LLM의 기본 출력은 “정답 하나”가 아니라 **후보 토큰별 점수**이다.
- 생성은 그 점수(또는 확률)를 이용해 다음 토큰을 고르는 과정의 반복이다.

## 코드로 맛보기 — 학습 여정의 최소 스케치
아래 코드는 동작을 위한 완전한 모델이 아니다.  
“앞으로 배우게 될 부품의 이름”을 파이썬으로 나열한 스케치이다.

```python
# roadmap_sketch.py
# 1강의 목적: 완성 구현이 아니라, 앞으로 만날 부품의 이름을 익힌다.

def tokenize(text: str) -> list[str]:
    """텍스트를 토큰 리스트로 나눈다. (2권에서 본격 구현)"""
    return text.lower().replace(".", "").split()

def predict_next_token(context_tokens: list[str]) -> str:
    """
    아주 단순한 규칙 기반 예시.
    실제 LLM은 Embedding + Attention + MLP로 점수를 계산한다.
    """
    if not context_tokens:
        return "안녕하세요"

    last = context_tokens[-1]
    toy_memory = {
        "날씨가": "좋다",
        "오늘은": "날씨가",
        "딥러닝을": "배운다",
    }
    return toy_memory.get(last, "...")

def generate(prompt: str, steps: int = 3) -> str:
    tokens = tokenize(prompt)
    for _ in range(steps):
        next_token = predict_next_token(tokens)
        tokens.append(next_token)
    return " ".join(tokens)

if __name__ == "__main__":
    print(generate("오늘은 날씨가"))
```

실행 결과 예:

```text
오늘은 날씨가 좋다 ... ...
```

이 코드에서 중요한 것은 출력이 똑똑한가보다, 다음 구조이다.

```text
텍스트 → 토큰 → (문맥을 보고) 다음 토큰 선택 → 반복
```

2권 이후에는 `toy_memory` 자리를 **학습된 Parameter**가 대체한다.

## LLM을 확률 함수로 보기

거시적으로 LLM은 **이산 토큰 열**을 받아 **다음 토큰 분포**를 내는 함수입니다.

$$
f_\theta : \{1,\ldots,V\}^{T} \to \Delta^{V-1}
$$

기호의 의미는 다음과 같습니다.

- $V$: 어휘(vocabulary) 크기. 토큰 ID는 $1,\ldots,V$ (구현에 따라 $0,\ldots,V-1$)
- $T$: 문맥 길이(토큰 개수)
- $\theta$: 학습 가능한 파라미터 전체(수억~수천억 개일 수 있음)
- $\Delta^{V-1}$: 확률 심플렉스. 성분 $\ge 0$이고 합이 $1$인 벡터의 집합

$$
\Delta^{V-1}
=
\left\{
p\in\mathbb{R}^{V}
\;\middle|\;
p_k \ge 0,\ 
\sum_{k=1}^{V} p_k = 1
\right\}
$$

이후 강의는 $f_\theta$ 안의 Embedding·행렬곱·Attention·정규화를 하나씩 엽니다. 1강의 목표는 “완벽한 공식”이 아니라, **나중에 꽂힐 좌표**를 잡는 것입니다.

### Next Token Prediction을 수식으로

문맥 $x_{1:t}=(x_1,\ldots,x_t)$가 주어졌을 때, 모델은 다음 토큰 $x_{t+1}$의 조건부 분포를 냅니다.

$$
p_\theta(x_{t+1}\mid x_{1:t})
=
\mathrm{softmax}\!\left(z_t\right)_{x_{t+1}}
$$

여기서 $z_t\in\mathbb{R}^{V}$는 위치 $t$의 **logit** 벡터입니다. Softmax는

$$
\mathrm{softmax}(z)_k
=
\frac{e^{z_k}}{\sum_{j=1}^{V} e^{z_j}}
$$

입니다. 생성(inference)은 이 분포에서 토큰을 고르는 일의 반복입니다.

$$
x_{t+1}
\sim
p_\theta(\cdot\mid x_{1:t})
\quad\text{또는}\quad
x_{t+1}
=
\arg\max_k\, p_\theta(k\mid x_{1:t})
$$

`toy_memory` 예제는 학습된 $p_\theta$ 대신 **사전 정의된 이어짐 표**를 쓴 축소판입니다. 2권 이후에는 그 표가 신경망으로 바뀝니다.

### 한 문장의 결합 확률

길이 $T$인 토큰 열 $x_{1:T}$의 모델 확률은 연쇄 법칙(chain rule)으로 분해됩니다.

$$
p_\theta(x_{1:T})
=
\prod_{t=1}^{T}
p_\theta(x_t\mid x_{1:t-1})
$$

($t=1$일 때는 빈 문맥 또는 BOS 토큰을 가정합니다.)

학습에서는 보통 **음의 로그 우도(NLL)** 를 줄입니다.

$$
L(\theta)
=
-\sum_{t=1}^{T}
\log p_\theta(x_t\mid x_{1:t-1})
$$

이 $L$이 Cross-Entropy Loss의 언어 모델 버전입니다. 1권 후반에서 Softmax·CE를, 2~3권에서 Transformer 안의 $z_t$를 만듭니다.

### 파라미터와 “밑바닥”의 관계

“밑바닥부터”란 다음을 **한 줄로 잇는다**는 뜻입니다.

```text
토큰 ID → 벡터 → 행렬곱/Attention → logit → Softmax → Loss → Gradient → 갱신
```

수식으로만 압축하면:

$$
x \xrightarrow{\mathrm{Emb}} h
\xrightarrow{f_{\mathrm{block}}} z
\xrightarrow{\mathrm{softmax}} p
\xrightarrow{\mathrm{CE}} L
\xrightarrow{\nabla} \Delta\theta
$$

지금은 각 화살표를 “이름만” 알면 충분합니다. 화살표마다 전용 강의가 있습니다.

### 미니 손계산 — 3토큰 어휘

어휘를 `{A,B,C}` ($V=3$)라 두고, 어떤 문맥에서 logit이

$$
z = [2.0,\ 1.0,\ 0.0]
$$

이라고 합시다. Softmax 분모는

$$
e^{2}+e^{1}+e^{0} \approx 7.389 + 2.718 + 1 = 11.107
$$

이므로

$$
p \approx
\left[
\frac{7.389}{11.107},\ 
\frac{2.718}{11.107},\ 
\frac{1}{11.107}
\right]
\approx
[0.665,\ 0.245,\ 0.090]
$$

다음 토큰으로 `A`를 고르면 NLL 항은 $-\log 0.665 \approx 0.408$입니다.  
이 책의 모든 “학습”은 결국 이런 항을 **데이터 전체에 대해 평균내어** $\theta$로 미분하는 일입니다.

```python
# 01강 감각용: Softmax를 아직 라이브러리 없이 맛보기
import math

z = [2.0, 1.0, 0.0]
ez = [math.exp(v) for v in z]
Z = sum(ez)
p = [v / Z for v in ez]
nll_A = -math.log(p[0])
print([round(x, 3) for x in p], round(nll_A, 3))
```

### 커리큘럼을 식에 대응시키기

| 조각 | 대략적 위치 | 이 책 |
|---|---|---|
| Python·배열·미분 | $x$, $\nabla$를 다루는 언어 | 1권 |
| Tokenizer | 텍스트 $\to$ ID | 2권 초반 |
| Attention / Transformer | $f_{\mathrm{block}}$ | 2권 |
| Pretrain / SFT | $L(\theta)$ 최소화 | 3권 |
| RLHF 등 | 선호에 맞는 목표로 $\theta$ 조정 | 4권 |
| Serving | $f_\theta$를 빠르게 평가 | 5권 |

> **핵심**
>
> 1강을 끝내면 “LLM = 조건부 토큰 분포 $p_\theta$”라는 한 문장을 자기 언어로 말할 수 있어야 합니다.

## LLM에서는 어디에 사용될까?
실제 LLM 제품 하나를 분해하면 대략 다음 층이 보인다.

| 층 | 하는 일 | 이 책의 위치 |
|---|---|---|
| 데이터·토큰화 | 텍스트를 Token ID로 변환 | 2권 |
| 모델 구조 | Embedding, Attention, MLP, Transformer | 2~3권 |
| 사전학습 | 대규모 말뭉치로 일반 능력 학습 | 3권 |
| 지도 미세조정 | 지시-응답 형식으로 맞춤 | 3권 |
| 선호·강화학습 | 더 나은 답을 고르도록 조정 | 4권 |
| 추론·서빙 | KV Cache, Batching, 병렬화로 빠르게 실행 | 5권 |

따라서 “ChatGPT 같은 서비스를 이해한다”는 말은, 위 층을 모두 어느 정도 이해한다는 뜻에 가깝다.

## 실습
### 실습 1 — 학습 지도 손으로 다시 쓰기

**목표:** 전체 경로를 자기 언어로 복원한다.

1. 이 강의의 학습 경로를 보지 말고, 기억나는 대로 종이나 메모장에 적어 본다.
2. 빠진 단계를 커리큘럼과 비교한다.
3. 각 단계 옆에 “내가 지금 아는 것 / 모르는 것”을 한 줄씩 적는다.

**예상 결과:** 모르는 단계가 많아도 괜찮다. 좌표가 생기면 이후 학습 속도가 달라진다.

### 실습 2 — 일상 문장을 토큰처럼 나눠 보기

**목표:** Token 감각을 만든다.

문장:

```text
오늘 저녁에 딥러닝을 공부한다.
```

1. 공백 기준으로 나눠 본다.
2. 조사(에, 을)를 붙인 경우와 분리한 경우를 비교한다.
3. “어떤 단위로 자르는가”에 따라 학습이 달라질 수 있음을 메모한다.

이 문제는 2권 Tokenizer 강의에서 더 정확히 다시 다룬다.

### 실습 3 — roadmap_sketch.py 수정

**목표:** Next Token Prediction 루프를 체감한다.

1. `toy_memory`에 자신만의 이어짐을 3개 추가한다.
2. `steps`를 5로 바꿔 실행한다.
3. 왜 금방 이상한 문장이 되는지도 짧게 적어 본다.

**추가 도전:** `predict_next_token`이 여러 후보 중 하나를 고르도록 바꿔 본다. (아직 확률 계산은 필요 없다.)

## 자주 하는 실수
1. **처음부터 큰 모델을 만지려고만 한다**  
   원리가 없으면 설정 파일만 바꾸다 끝나기 쉽다. 작은 예제로 계산 감각을 먼저 만든다.

2. **용어를 정의 없이 수집한다**  
   Attention, Gradient, LoRA를 단어 카드처럼만 모으면 연결이 없다. 이 책은 “정의 → 이유 → 예제 → 코드 → LLM 연결”을 반복한다.

3. **1권을 건너뛰고 Transformer로 바로 간다**  
   당장은 빨라 보여도, Backpropagation과 Tensor를 모르면 이후 학습 실패를 해석하기 어렵다.

4. **실행하지 않고 읽기만 한다**  
   이 책은 실전형이다. 코드는 실행하고, 숫자를 바꾸고, 깨지는 장면을 직접 봐야 한다.

## 수식 한 장으로 보는 1권의 목표

이 책이 궁극적으로 다루는 학습 한 스텝은 다음으로 압축됩니다.

$$
\theta \leftarrow \theta - \eta \nabla_\theta L(\theta;\,\text{batch})
$$

- $\theta$: 모델 파라미터
- $\eta$: 학습률
- $L$: 배치에서 계산한 Loss
- $\nabla_\theta L$: Gradient

1권의 Python·NumPy·미분·PyTorch는 모두 이 한 줄을 **안정적으로 실행·디버깅**하기 위한 준비입니다. 2권 이후에는 $L$이 다음 토큰 CE가 되고, $\theta$ 안에 Attention이 들어갑니다.

## 핵심 요약
- AI는 큰 목표, Machine Learning은 데이터로 규칙을 배우는 방법, Deep Learning은 깊은 신경망 기반 접근, LLM은 대규모 언어 모델이다.
- LLM의 기본 엔진은 Next Token Prediction이다.
- “밑바닥부터”는 사용법만이 아니라, 수학·계산·구현·LLM 연결을 함께  Mil다는 뜻이다.
- 이 책은 5권 120강으로, 기초 → 구조 → 학습 → 정렬 → 서빙 최적화를 한 줄로 잇는다.
- 지금 단계의 목표는 완벽한 이해가 아니라, **이후 강의가 꽂힐 지도**를 갖는 것이다.

## 용어 사전
| 용어 | 의미 |
|---|---|
| AI (Artificial Intelligence) | 사람처럼 보이는 지적 작업을 시스템이 수행하게 하는 기술 분야 |
| Machine Learning | 데이터로부터 규칙을 학습하는 방법 |
| Deep Learning | 여러 층의 Neural Network로 복잡한 표현을 학습하는 방법 |
| Neural Network | 입력에 가중치를 적용해 출력을 계산하는 계층적 모델 |
| LLM (Large Language Model) | 대규모 텍스트로 학습한 언어 모델 |
| Token | 텍스트를 나눈 최소 처리 단위 |
| Next Token Prediction | 지금까지의 토큰을 보고 다음 토큰을 예측하는 학습·생성 목표 |
| Parameter | 모델이 학습으로 조정하는 숫자들 |
| Pretraining | 대규모 데이터로 일반 능력을 먼저 학습하는 단계 |
| SFT | 지시-응답 데이터로 모델을 맞추는 지도 미세조정 |
| Inference | 학습된 모델로 실제 출력을 계산하는 과정 |

## 연습문제
### 문제 1 (개념)

AI, Machine Learning, Deep Learning, LLM의 포함 관계를 한 문장으로 설명하시오.

### 문제 2 (개념)

LLM의 가장 기본적인 학습 목표인 Next Token Prediction을 자신의 문장으로 설명하시오.

### 문제 3 (응용)

“번역 앱”을 만든다고 할 때, 이 책의 5권 중 어느 어디에 해당하는 지식이 필요한지 두 가지 이상 고르고 이유를 쓰시오.

### 문제 4 (코드)

`roadmap_sketch.py`에서 `tokenize`과 `predict_next_token`의 역할을 각각 한 줄로 쓰시오.

### 문제 5 (연결)

Gradient Descent는 아직 배우지 않았다. 그런데도 1권에서 반드시 배워야 하는 이유를, LLM 학습 관점에서 예측해 보시오.

---

## 정답 및 해설
### 문제 1

LLM ⊂ Deep Learning ⊂ Machine Learning ⊂ AI  
더 풀어 쓰면, LLM은 딥러닝 기반 언어 모델이고, 딥러닝은 머신러닝의 한 방법이며, 머신러닝은 AI를 구현하는 주요 접근이다.

### 문제 2

주어진 이전 토큰들을 조건으로, 다음에 올 토큰의 점수(또는 확률)를 예측하는 것이다. 생성이란 이 예측을 반복하는 과정이다.

### 문제 3 예시 답

- 2권: 문장을 토큰화하고 Transformer로 문맥을 처리해야 한다.
- 3권: 번역 데이터로 Pretraining/SFT 성격의 학습이 필요하다.
- 5권: 실제 서비스라면 지연시간·처리량을 위한 Inference 최적화가 필요하다.

(1권의 Tensor/Loss 기초, 4권의 선호 학습도 품질 개선에 연결될 수 있다.)

### 문제 4

- `tokenize`: 문자열을 모델이 다룰 토큰 단위로 나눈다.
- `predict_next_token`: 현재까지의 토큰을 보고 다음 토큰을 고른다.

### 문제 5

LLM 학습은 결국 Loss를 낮추도록 Parameter를 업데이트하는 과정이다. Gradient Descent는 그 업데이트 방향을 정하는 기본 원리다. Transformer나 GPT를 코드로만 따라 쳐도, 학습이 안 될 때 원인을 보려면 이 개념이 필요하다.

## 다음 강의와 연결
이번 강의에서 지도를 펼쳤다.

다음 **제2강. Python 환경 준비와 첫 프로그램**에서는, 이 긴 여정의 첫 도구인 Python을 실제로 실행할 환경을 만든다.  
에디터, 가상환경, 패키지 설치, 첫 스크립트 실행까지 다루며, 이후 NumPy·PyTorch로 이어질 실습의 출발점을 고정한다.

> 지도는 생겼다. 이제 연필을 잡자.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** 없음 (시리즈 시작)
- **다음 강:** [2강. Python 환경 준비와 첫 프로그램](02강_Python_환경_준비와_첫_프로그램.md)

<!-- /LECTURE_NAV -->
