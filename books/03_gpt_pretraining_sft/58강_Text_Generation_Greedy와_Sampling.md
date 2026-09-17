# 58강. Text Generation — Greedy와 Sampling
## 이번 강에서 배우는 내용

- Generation(텍스트 생성)이 forward와 어떻게 다른지 설명한다
- Greedy decoding(탐욕 선택)과 Multinomial sampling(다항 분포 샘플링)을 구분한다
- `context → logits → 규칙 → append → 반복` 루프를 구현한다
- EOS·`max_new_tokens`·`block_size` 슬라이딩을 처리한다
- 같은 모델에서 규칙만 바꿔 결과가 달라짐을 관찰한다
- 제59강의 Temperature / Top-k / Top-p가 끼어들 지점을 표시한다

## 왜 중요한가?
Loss가 내려가도 **생성이 없으면** 언어 모델인지 확인하기 어렵다. 반대로 생성만 보고 Loss를 무시하면 과적합·커닝을 놓친다.

```text
학습: 정답 다음 토큰의 로그확률 ↑
생성: 분포에서 토큰을 골라 문장을 이어 붙임
```

둘은 같은 `logits`를 쓰지만 **의사결정 규칙**이 다르다. 규칙을 잘못 짜면:

- 항상 같은 문장만 나옴 (과도한 greedy)
- 말도 안 되는 토큰이 잦음 (과도한 난수)
- 컨텍스트가 `block_size`를 넘어 크래시

제67강에서 PPL과 생성 품질이 다르다고 강조한다. 그 차이의 입구가 오늘이다.

## 선수 개념
- Softmax와 확률 분포 (제33강)
- Causal LM forward (제56강)
- Teacher forcing vs 자기회귀 피드백 (제57강)
- `model.eval()`, `torch.no_grad()` (추론 모드)
- 다항 분포의 직관: 확률 무게만큼 제비뽑기

아직 Temperature·nucleus는 제59강이다. 오늘은 **argmax**와 **원분포 샘플링**만 다룬다.

## 핵심 개념
### 3.1 Text Generation이란?

**Text Generation(텍스트 생성)**은 시작 토큰 서열(프롬프트)이 주어졌을 때, 모델이 정의한 조건부 분포에 따라 토큰을 반복적으로 이어 붙여 더 긴 서열을 만드는 과정이다.

한 스텝:

$$

x_{t}\sim \pi\big(\cdot \mid P_\theta(\cdot\mid x_{<t})\big)

$$

여기서 $\pi$가 **디코딩 정책(decoding policy)**이다.  
Greedy는 $\pi=\arg\max$, Sampling은 $\pi=$그 분포(또는 변형 분포)에서의 샘플.

### 3.2 생성 루프의 공통 골격

```text
idx = prompt_ids                    # [B, T0]
for step in 1..max_new_tokens:
    idx_cond = idx[:, -block_size:] # 길이 제한
    logits = model(idx_cond)        # [B, T, V]
    logits_last = logits[:, -1, :]  # [B, V]
    next_id = decode_rule(logits_last)
    idx = cat(idx, next_id)
    if all EOS: break
return idx
```

학습 때와 달리 **한 토큰씩** 늘린다. (일부 시스템에서는 speculative decoding 등으로 가속하지만, 개념은 동일하다.)

### 3.3 Greedy Decoding

**Greedy decoding(탐욕 디코딩)**은 매 스텝에서 확률이 가장 큰 토큰을 고른다.

$$

x_t=\arg\max_v P_\theta(v\mid x_{<t})=\arg\max_v z_{t,v}

$$

(마지막 등식은 Softmax가 순서를 보존하므로 logit argmax와 동일.)

특징:

- 결정적(동일 입력·모델이면 동일 출력, dropout 제외)
- 구현이 단순
- 안전한 “최빈 경로”에 가깝지만, 반복·단조로움이 생기기 쉬움
- 전역 최적 서열을 보장하지 않음 (빔 서치 동기)

### 3.4 Multinomial Sampling

**Multinomial sampling(다항 샘플링)**은 Softmax 확률을 그대로 사용해 제비뽑기한다.

$$

x_t\sim \mathrm{Categorical}\big(\mathrm{softmax}(\mathbf{z}_t)\big)

$$

특징:

- 비결정적 — 시드에 따라 다른 문장
- 창의성·다양성에 유리할 수 있음
- 꼬리 확률의 이상한 토큰이 가끔 뽑힘 (제59강에서 꼬리 절단)

이 책에서 “sampling”이라고만 하면 기본적으로 이 원분포 샘플을 가리킨다. Temperature 등은 분포를 **변형한 뒤** 샘플하는 확장이다.

### 3.5 Greedy vs Sampling 한눈에

| | Greedy | Multinomial sampling |
|---|---|---|
| 선택 | $\arg\max$ | 확률적 샘플 |
| 무작위성 | 없음(보통) | 있음 |
| 전형적 증상 | 반복·안전 | 다양·간헐적 붕괴 |
| 코드 | `argmax` | `multinomial` / `categorical` |

## 직관적으로 이해하기
비가 올 확률 예보:

```text
맑음 0.10 / 흐림 0.25 / 비 0.65
```

- Greedy: 항상 “비”
- Sampling: 65%로 “비”, 가끔 “흐림”, 드물게 “맑음”

언어 모델도 매 글자(토큰)마다 이런 예보를 한다.  
소설을 쓸 때는 가끔 낮은 확률 가지가 필요할 수 있고, 코드 자동완성에서는 greedy가 더 낫기도 하다. **정답 디코더는 과제가 정한다.**

## 수학적으로 이해하기
프롬프트 $x_{1:t_0}$ 이후 길이 $K$ 생성의 경로 확률:

$$

P_\theta(x_{t_0+1:t_0+K}\mid x_{1:t_0})
=\prod_{k=1}^{K}P_\theta(x_{t_0+k}\mid x_{<t_0+k})

$$

Greedy는 매 항에서 최대 항을 고르므로, 경로 전체의 전역 최대를 보장하지 않는다.  
예: 첫 토큰에서 약간 낮은 확률을 고르면 이후에 훨씬 높은 확률 경로가 열릴 수 있다. (빔 서치·탐색 계열의 동기. 이 책은 초반에 깊이 들어가지 않는다.)

Sampling의 기댓값은 모델 분포 자체를 따른다. Temperature로 변형하면 기댓 문체·엔트로피가 달라진다(제59강).

## 작은 숫자 예제
$V=4$, 마지막 logit:

$$

\mathbf{z}=(1.0,\ 2.0,\ 0.0,\ 0.5)

$$

Softmax(근사):

```text
e^z ≈ (2.72, 7.39, 1.00, 1.65), 합 ≈ 12.76
p   ≈ (0.213, 0.579, 0.078, 0.129)
```

- Greedy → 인덱스 `1` (확률 0.579)
- Sampling → `1`이 자주, `0`/`3`이 가끔, `2`는 드물게

같은 프롬프트로 5번 샘플하면 서로 다른 이어짐이 나오는 것이 정상이다.

## 코드로 구현하기
### 7.1 Greedy

```python
import torch

@torch.no_grad()
def generate_greedy(model, idx, max_new_tokens, eos_id=None):
    """
    model: GPT-like, forward(idx)->logits [B,T,V]
    idx: [B, T]
    """
    model.eval()
    block_size = model.config.block_size
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B,1]
        idx = torch.cat([idx, next_id], dim=1)
        if eos_id is not None:
            if (next_id.squeeze(-1) == eos_id).all():
                break
    return idx
```

### 7.2 Multinomial sampling

```python
@torch.no_grad()
def generate_sample(model, idx, max_new_tokens, eos_id=None):
    model.eval()
    block_size = model.config.block_size
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        probs = torch.softmax(logits[:, -1, :], dim=-1)  # [B,V]
        next_id = torch.multinomial(probs, num_samples=1)  # [B,1]
        idx = torch.cat([idx, next_id], dim=1)
        if eos_id is not None and (next_id.squeeze(-1) == eos_id).all():
            break
    return idx
```

`torch.distributions.Categorical(probs=probs).sample()`도 동일 계열이다.

### 7.3 배치 중 일부만 EOS

위 `.all()`은 배치 전원이 EOS일 때 중단한다. 실무에서는 **이미 끝난 시퀀스만 패딩/마스크**하고 루프는 `max_new_tokens`까지 도는 편이 단순하다.

```python
finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
# ...
next_id = next_id.masked_fill(finished.unsqueeze(-1), eos_id)
finished = finished | (next_id.squeeze(-1) == eos_id)
if finished.all():
    break
```

### 7.4 프롬프트 텐서 만들기

```python
def encode_prompt(tokenizer, text, device):
    ids = tokenizer.encode(text)  # list[int]
    return torch.tensor([ids], dtype=torch.long, device=device)
```

디코드:

```python
def decode_ids(tokenizer, idx):
    # idx: [1, T]
    return tokenizer.decode(idx[0].tolist())
```

토크나이저 API는 제61강 파이프라인과 맞춘다.

## PyTorch로 붙이는 최소 데모
제56강 `GPT`가 있다고 가정한 스케치:

```python
cfg = GPTConfig(vocab_size=100, block_size=32, n_layer=2, n_head=4, n_embd=64)
model = GPT(cfg).eval()

prompt = torch.randint(1, 100, (1, 5))  # 가짜 프롬프트
out_g = generate_greedy(model, prompt.clone(), max_new_tokens=10)
out_s = generate_sample(model, prompt.clone(), max_new_tokens=10)
print(out_g)
print(out_s)
```

학습 전에는 출력이 의미 없다. 배선 확인용이다.  
Mini Pretraining(제68강) 이후에는 같은 함수로 “흉내 문장”을 본다.

## 수식 보강 — Greedy · Temperature · Top-k

logits $\mathbf{z}\in\mathbb{R}^{V}$에서

$$
\text{Greedy: }\hat{y}=\arg\max_k z_k
$$

Temperature $\tau>0$:

$$
p_k = \frac{e^{z_k/\tau}}{\sum_j e^{z_j/\tau}}
$$

$\tau\to 0$이면 greedy에 가깝고, $\tau$가 크면 분포가 평평해집니다.

Top-$k$는 확률 상위 $k$개만 남기고 재정규화합니다. Top-$p$(nucleus)는 누적확률 $\ge p$가 되는 최소 집합을 남깁니다.

## 수식·정량 보강 — 디코딩 정책

Greedy: $x_t=\arg\max_v z_{t,v}$.

Sampling: $x_t\sim\mathrm{Categorical}(\mathrm{softmax}(z_t))$.

Temperature(제59강 예고): $p=\mathrm{softmax}(z/\tau)$.

### 워크드 예

$z=(3,2,0.5,-1)$, $\mathrm{softmax}\approx(0.644,0.237,0.053,0.012)$.

| 정책 | 결과 |
|---|---|
| Greedy | 항상 0 |
| Sample | 0 자주 |
| Top-$k=2$ 후 | $p'=(0.731,0.269,0,0)$ |

### 경로 확률이 보여주는 비최적성

두 스텝 장난감에서 첫 토큰 mode가 전체 곱 최대를 보장하지 않을 수 있음 → 빔 서치 동기.

엔트로피 $H(p)=-\sum p\log p$가 큰 위치에서 sampling 다양성↑.

경로:

$$
P(x_{t_0+1:t_0+K}\mid x_{1:t_0})=\prod_{k}P(x_{t_0+k}\mid x_{<t_0+k})
$$


## 워크드 예제 — 루프·시드·EOS

의사코드 한 바퀴:

1. `idx_cond = idx[:, -L:]` ($L=\mathrm{block\_size}$)
2. `logits = model(idx_cond)` → `[B,T',V]`
3. `z = logits[:, -1, :]`
4. greedy: `next = z.argmax(-1)` / sample: `Categorical(softmax(z))`
5. `idx = cat(idx, next)`
6. `next==EOS`면 해당 배치 종료

### 같은 $p$에서 5번 샘플

$p=(0.05,0.80,0.15)$이면 기댓값적으로 id1이 4번 안팎, id2가 가끔.  
`torch.manual_seed(0)`으로 재현 가능한지 확인하는 것이 디버깅 기본이다.

### Greedy 반복 병

모드만 고르면 $P(\text{same}|\text{same})$가 큰 토큰(마침표·줄바꿈·특수)에서 루프가 생기기 쉽다.  
Sampling·temperature·top-p는 그 **병의 출구**이지, 모델 가중치를 바꾸지는 않는다.


## 추가 연습 — 정책 비교표 채우기

$z=(0,3,1)$, $p=\mathrm{softmax}(z)\approx(0.042,0.865,0.093)$.

| trial | greedy | sample(시드마다) |
|---|---|---|
| 1 | 1 | ? |
| 2 | 1 | ? |
| 3 | 1 | ? |

Greedy 열은 모두 1. Sample 열은 1이 많되 0/2가 간헐.

`max_new_tokens=5`, EOS 없음 → 길이 정확히 +5.  
EOS가 중간이 나오면 조기 종료 설계를 검증.

## LLM에서는 어디에 사용될까?
제품 챗봇은 드물게 순수 greedy만 쓴다. 보통:

- 기본: Temperature + Top-p (제59강)
- 코드/정확한 답: 낮은 temperature 또는 greedy에 가깝게
- 창작: 높은 temperature / 넓은 nucleus
- 제약 디코딩: 문법·JSON 스키마 등으로 허용 토큰만 남김

서빙 엔진(vLLM 등, 5권)은 KV cache로 `idx_cond` 전체를 매번 재계산하지 않는다.  
개념 학습 단계에서는 **캐시 없는 재forward**로도 충분하다. 느린 것이 정상이다.

## 실습
### 실습 1 — 규칙 전환

같은 체크포인트·같은 프롬프트로 greedy 1회, sampling 5회를 비교하고 차이를 세 줄로 기록한다.

### 실습 2 — EOS

가짜 `eos_id`를 넣고, 모델이 EOS를 자주 내게 만들도록 라벨에 EOS를 넣은 장난감 학습(제57강 TinyLM) 후 조기 종료를 확인한다.

### 실습 3 — block_size

`max_new_tokens`를 크게 잡고, `idx_cond = idx[:, -block_size:]`를 제거해 오류를 재현한 뒤 복구한다.

### 실습 4 — 마지막 위치만

실수로 `logits.argmax`를 전체 `[B,T,V]`에 적용하면 어떤 shape/의미가 되는지 실험하고, 왜 `[:, -1, :]`인지 주석으로 남긴다.

### 실습 5 — 시드

```python
torch.manual_seed(0)
```

전후로 sampling 재현성을 확인한다. GPU 비결정 연산이 있으면 완전 재현이 깨질 수 있음을 메모한다.

## 자주 하는 실수
1. **학습 모드로 생성**  
   Dropout이 켜져 결과가 흔들린다. `eval()` 필수.

2. **gradient 추적 유지**  
   긴 생성에서 메모리 폭발. `torch.no_grad()` 사용.

3. **프롬프트까지 다시 샘플**  
   이미 있는 토큰을 덮어쓰지 않는다. append만.

4. **Softmax 없이 multinomial**  
   `multinomial`은 비음수 무게를 기대한다. logit 생입력은 부호 때문에 실패하거나 왜곡된다.

5. **배치 차원 누락**  
   `[T]`만 넘기면 Embedding이 잘못 해석한다. `[1,T]`로 유지.

6. **EOS를 디코더만의 문제로 착각**  
   데이터에 EOS가 거의 없으면 모델이 끝내는 법을 못 배운다(제60~61강 연결).


## 수식 카드 — Decoding

$$
x_t^{\mathrm{greedy}}=\arg\max_v z_{t,v},\quad
x_t^{\mathrm{sample}}\sim\mathrm{Categorical}(\mathrm{softmax}(z_t))
$$

$$
P(x_{t_0+1:t_0+K}\mid x_{1:t_0})=\prod_k P(x_{t_0+k}\mid x_{<t_0+k})
$$


## 연결 복습 — logits에서 문장으로

학습은 $\mathbb{E}[-\log p(x_t\mid x_{<t})]$를 줄인다.  
생성은 같은 $p$에서 $\pi$로 샘플/argmax한다.

제59강에서 $\pi$를 temperature·top-k·top-p로 바꾼다.  
오늘은 $\pi\in\{\arg\max,\ \mathrm{Cat}(p)\}$만.

실습 한 줄: 동일 체크포인트에서 greedy 1회 vs sample 5회 문장 길이·반복을 비교 기록.

## 핵심 요약
- 생성은 같은 GPT forward의 마지막 logit에 **디코딩 규칙**을 적용한 루프다
- Greedy는 argmax, Sampling은 Categorical 제비뽑기다
- `block_size` 슬라이스와 EOS/`max_new_tokens`가 루프의 안전장치다
- 결정성·다양성은 규칙 선택의 트레이드오프다
- Temperature·Top-k/p는 이 루프의 `decode_rule`을 치환·확장한다

## 용어 사전
| 용어 | 의미 |
|---|---|
| Decoding policy | logits→토큰 선택 규칙 |
| Greedy decoding | 매 스텝 최대 확률 토큰 |
| Multinomial sampling | Softmax 확률로 샘플 |
| Autoregressive generation | 출력을 다시 입력에 붙이며 진행 |
| Prompt | 생성을 조건 지우는 시작 서열 |
| EOS | 종료 토큰 |
| KV cache | (미리보기) 과거 키·값 재사용 가속 |

## 연습문제
### 문제 1 (개념)

생성 한 스텝의 입출력과, 학습 forward와의 공통점·차이점을 쓰시오.

### 문제 2 (규칙)

$\mathbf{p}=(0.05,0.80,0.15)$에서 greedy 선택과 sampling의 차이를 설명하시오.

### 문제 3 (코드)

왜 `logits[:, -1, :]`만 사용하는가?

### 문제 4 (디버깅)

생성 중 `ValueError: sequence length exceeds block_size`가 난다. 원인은?

### 문제 5 (연결)

제59강에서 Temperature를 적용한다면 Softmax **앞**과 **뒤** 중 어디에 끼우는가? (미리 답)

---

## 정답 및 해설
### 문제 1

공통: 같은 가중치로 logits 계산.  
차이: 학습은 전 위치 CE·정답 피드백, 생성은 마지막 위치만 골라 append.

### 문제 2

Greedy는 항상 인덱스 1. Sampling은 80%로 1, 15%로 2, 5%로 0.

### 문제 3

다음에 붙일 토큰의 분포는 “현재까지 컨텍스트의 끝” 위치에 해당한다. 과거 위치 logit은 이미 확정된 토큰용이다.

### 문제 4

잘린 컨텍스트 없이 전체 `idx`를 모델에 넣어 $T>block_size`가 됨.

### 문제 5

Temperature는 보통 Softmax **앞** logit을 $T$로 나눈다: $\mathrm{softmax}(\mathbf{z}/T)$.

## 다음 강의와 연결
원분포 샘플은 꼬리가 두껍다. 다음 강의는 그 분포를 **날카롭게·안전하게** 만드는 손잡이다.

**제59강. Temperature, Top-K, Top-P**에서는 온도, 상위 $k$ 절단, nucleus(top-p) 샘플링을 손계산·코드·함정까지 다룬다.

이전 강의: [제57강. Causal LM Training 목표](./57강_Causal_LM_Training_목표.md)  
다음 강의: [제59강. Temperature, Top-K, Top-P](./59강_Temperature_TopK_TopP.md)

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [57강. Causal LM Training 목표](57강_Causal_LM_Training_목표.md)
- **다음 강:** [59강. Temperature, Top-K, Top-P](59강_Temperature_TopK_TopP.md)

<!-- /LECTURE_NAV -->
