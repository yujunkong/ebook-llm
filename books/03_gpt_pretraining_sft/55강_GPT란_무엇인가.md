# 3권. GPT Pretraining과 SFT

## 제55강. GPT란 무엇인가

### 1. 이번 강의에서 배울 것

2권에서 Causal Language Model(인과 언어 모델)의 배선도를 그렸다. 3권의 첫 강의에서는 그 배선도에 **GPT라는 이름**을 붙이고, 이 책이 말하는 GPT의 범위를 고정한다.

이 강의를 마치면 다음을 말할 수 있어야 한다.

- GPT가 제품 이름이 아니라 **아키텍처·학습 패러다임의 계열명**으로 쓰이는 이유
- Decoder-only Causal LM이 GPT의 핵심 골격인 이유
- Pretraining(사전학습)과 이후 단계(SFT 등)가 역할상 어떻게 갈리는지
- 이 책이 구현할 “작은 GPT”가 무엇이고 무엇이 아닌지
- 제56강에서 코드를 칠 때 어디가 “GPT 본체”인지

### 2. 왜 이것을 배우는가

“GPT”라는 단어는 일상에서 여러 의미로 쓰인다.

```text
일상 대화:  “GPT한테 물어봐”  →  특정 채팅 서비스/제품
연구·구현: “GPT-style model” → Decoder-only Causal LM 계열
이 책:     “GPT”             → 우리가 밑바닥부터 조립하는 그 계열
```

의미가 섞이면 학습이 흔들린다. 어떤 사람은 API 사용법을 배우고, 어떤 사람은 논문 연표를 외우고, 어떤 사람은 가중치 파일을 다운로드하는 것을 “GPT를 안다”고 부른다.

이 책은 **구조를 만들고, 목표 함수로 학습하고, 텍스트를 생성**하는 쪽이다. 그래서 3권을 열기 전에 “GPT란 무엇인가”를 **구현 가능한 정의**로 못 박는다.

제54강에서 정리한 파이프라인:

```text
Tokenizer → Embed → Block×N → lm_head → logits
```

이것이 곧 GPT 계열의 뼈대다. 이번 강의는 그 뼈대에 역사·이름·학습 단계의 좌표를 붙인다.

### 3. 먼저 알아야 할 개념

제54강 체크리스트를 전제로 한다.

1. **Next Token Prediction** — 다음 토큰 예측 (제32강)
2. **Causal Mask** — 미래를 보지 못하게 하는 마스크 (제40강)
3. **Transformer Block** — Attention + FFN + Norm/Residual (제46강)
4. **Decoder-only** — Encoder 없이 Decoder 스택만 쓰는 구조 (제47강)
5. **Causal LM 전체 골격** — embed → blocks → lm_head (제48강)
6. **Cross Entropy Loss** — 분류(토큰 선택) 목적함수 (제34강)

부족하면 해당 강을 짧게 재독한 뒤 이어간다.

### 4. 핵심 개념 — GPT의 이름과 범위

#### 4.1 GPT라는 약어

**GPT(Generative Pre-trained Transformer)**는 대략 다음을 합친 이름이다.

| 단어 | 의미 | 이 책에서의 위치 |
|---|---|---|
| Generative | 토큰을 이어 붙여 텍스트를 **생성** | 제58~59강 |
| Pre-trained | 대량 텍스트로 **먼저** 학습 | 제60강 이후 Pretraining |
| Transformer | Attention 기반 신경망 | 2권 전체 |

이름을 글자 그대로 외울 필요는 없다. 중요한 것은 세 축이 **한 모델**에 동시에 있다는 점이다.

- Transformer만 있으면 Encoder나 번역 모델일 수도 있다.
- Pre-trained만 있으면 분류기일 수도 있다.
- Generative만 있으면 RNN 언어 모델일 수도 있다.

GPT 계열은 이 셋을 **Decoder-only Causal LM** 위에 올린다.

#### 4.2 이 책이 말하는 GPT

이 책에서 GPT는 **특정 회사의 제품·챗봇·API 이름**이 아니다.

다음을 의미한다.

> Decoder-only Transformer로 인과적(next-token) 언어 모델을 만들고,  
> 대량 텍스트로 Pretraining한 뒤, (필요하면) 지시·대화 데이터로 맞추는 계열.

제품명·버전명(예: 특정 상용 모델)은 역사적 배경으로만 언급한다. 구현 목표는 **같은 계열의 작은 모델**이다.

#### 4.3 Decoder-only Causal LM

**Decoder-only**는 Encoder 스택 없이, 자기회귀(autoregressive) 생성에 맞는 Decoder형 블록만 쌓는 구조를 뜻한다.

**Causal Language Model(CLM, 인과 언어 모델)**은 시점 $t$에서 $x_{<t}$만 보고 $x_t$를 예측한다.

$$
P(x_1,\ldots,x_T)=\prod_{t=1}^{T} P(x_t\mid x_{<t})
$$

제48강에서 본 배선도가 그대로 GPT의 본체다.

```text
token ids [B, T]
  → token embedding (+ position)
    → Transformer Block × N   (Causal Self-Attention)
      → final LayerNorm
        → lm_head
          → logits [B, T, V]
```

Encoder-Decoder(번역·요약에 흔함)와 비교하면:

| | Encoder-Decoder | Decoder-only (GPT 계열) |
|---|---|---|
| 입력 인코딩 | 별도 Encoder | 프롬프트도 같은 스택에 넣음 |
| Attention | Cross-Attn 포함 | (기본) Causal Self-Attn만 |
| 대표 목표 | seq2seq | next-token LM |
| 이 책 3권 | 다루지 않음(개요만 2권) | **주 경로** |

### 5. 직관적으로 이해하기

GPT를 “똑똑한 검색엔진”이나 “지식 DB”로 생각하면 구현이 막힌다. 더 정확한 직관은 다음과 같다.

```text
아주 큰 “다음 글자(토큰) 맞히기” 기계
```

문장 “오늘 날씨가” 다음에 올 토큰의 **확률 분포**를 만들고, 그 분포에서 하나를 고르면 생성이 한 걸음 나아간다. 그 걸음을 반복하면 문단이 된다.

학습은 “정답 다음 토큰의 로그확률을 키우는 것”이다(제57강).  
생성은 “분포에서 고르는 규칙”이다(제58~59강).  
데이터는 “그 분포를 현실 텍스트에 맞추는 연료”다(제60강).

제품 챗봇은 이 위에 **지시 따르기·안전·도구 호출** 같은 층을 더 올린 결과다. 3권 후반 SFT가 그 층의 입구다. 지금은 본체(Causal LM)에 집중한다.

### 6. 역사·계보를 어떻게 읽을 것인가

연표를 암기하는 강의가 아니다. **설계 선택이 왜 굳어졌는지**만 고른다.

#### 6.1 언어 모델 → Transformer → GPT 계열

대략의 흐름(단순화):

```text
n-gram / 고전 LM
  → RNN/LSTM 언어 모델
    → Transformer (Attention is All You Need)
      → Decoder-only + 대량 Pretraining  (GPT 계열)
        → 지시 튜닝 / RLHF 등 정렬 단계
```

핵심 전환점:

1. **Attention**으로 장거리 의존을 직접 연결 (2권)
2. **Causal mask**로 생성과 학습 목표를 맞춤 (제40강)
3. **규모 있는 unsupervised Pretraining**으로 범용 표현·생성 능력을 먼저 얻음
4. 이후 **지도·선호 데이터**로 쓰임새를 맞춤 (3권 후반~4권)

#### 6.2 “GPT-1 / GPT-2 / …”를 어떻게 다룰까

버전 번호는 **스케일·데이터·세부 트릭의 세대**를 가리키는 표지에 가깝다. 이 책은 각 세대의 파라미터 수·벤치마크 점수를 사실처럼 나열하지 않는다. 대신 공통 골격을 유지한다.

세대가 바뀌어도 보통 남는 것:

- Decoder-only
- Next-token 목표
- Block 반복 스택
- Softmax 위 어휘 분포로 생성

세대마다 달라질 수 있는 것:

- Norm 종류(LayerNorm / RMSNorm)
- 위치 인코딩(절대 / RoPE 등)
- Attention 변형(MHA / GQA 등)
- 토크나이저·데이터 혼합·학습 스케줄

제53강에서 본 “최신 변형 개요”는 **옵션 메뉴**다. 3권 초반은 **고전적 GPT-style 골격**으로 구현한다.

### 7. Pretraining이라는 단계

#### 7.1 Pretraining(사전학습)의 정의

**Pretraining**은 특정  Downstream 과제(감정 분류, QA 포맷 등)에 맞추기 **전에**, 대량의 일반 텍스트에서 언어 모델 목표로 가중치를 먼저 학습하는 단계다.

비유:

```text
악기 연습(음계·리듬)  = Pretraining
특정 곡 연주 연습     = SFT / 과제 적응
무대 매너·청중 맞춤   = (나중에) 정렬·RL 등
```

이 비유는 완벽하지 않다. 다만 “처음부터 챗봇 UI를 만들지 않는다”는 순서를 고정하는 데 충분하다.

#### 7.2 왜 먼저 일반 텍스트인가

특정 과제 데이터만으로 처음부터 학습하면:

- 데이터가 상대적으로 작고
- 문장 구조·세계 지식·어휘 분포를 덜 보고
- 생성 유창성이 부족하기 쉽다

일반 텍스트 Pretraining은 **다음 토큰을 맞히는 압력**만으로도 문법·사실 조각·문체·코드 패턴 등을 흡수한다(완전하지 않으며, 환각도 여기서 자란다).

#### 7.3 이 책의 Pretraining 범위

교육용 Mini GPT Pretraining은 다음을 목표로 한다.

- 작은 vocab / 작은 context
- 공개·재현 가능한 텍스트 샘플
- Loss가 내려가고, greedy/sample 생성이 말이 되는 수준

상용 규모(수조 토큰, 수천 GPU)를 **숫자로 흉내 내어 사실처럼 적지 않는다**. 규모감은 개념으로만 두고, 실습은 작은 루프로 증명한다.

### 8. GPT 이후의 층 — 미리 보는 지도

3권 전체를 한눈에 두면 현재 위치가 보인다.

```text
제55~56강  구조: GPT란 무엇 / 아키텍처 구현
제57강      목표: Causal LM Loss
제58~59강  생성: Greedy, Sampling, Temperature, Top-k/p
제60~61강  데이터: Corpus, Tokenization Pipeline, Packing
제62~68강  학습 루프·스케줄·체크포인트·Mini Pretraining 프로젝트
제69~76강  SFT(지도 미세조정)로 지시 따르기
제77강      Pretraining vs SFT 역할 정리
```

지금 강의는 **입구의 정의**다. 코드를 많이 치지 않아도 된다. 대신 “제품 GPT”와 “우리가 만드는 GPT”를 혼동하지 않게 한다.

### 9. 작은 사고 실험

다음 문장을 보고, GPT 계열이 내부적으로 하는 일을 한 줄로 바꿔 보자.

입력 토큰: `["오늘", "은", "비가"]`

모델이 하는 일(학습·추론 공통 forward):

```text
각 위치에서 vocab 전체에 대한 logit 벡터를 만든다.
마지막 위치의 Softmax ≈ P(다음 토큰 | "오늘","은","비가")
```

생성 한 스텝(제58강에서 구현):

```text
분포에서 "온다"를 고름
→ 컨텍스트에 append
→ 다시 forward
→ ...
```

“비가 온다는 사실을 검색한다”가 아니라, **조건부 토큰 분포를 계산한다**가 정확한 문장이다. 사실처럼 보이는 출력은 그 분포가 데이터에 잘 맞았을 때의 **부작용**에 가깝다.

### 10. 코드로 “GPT”를 가리키는 위치

아직 전체 구현은 제56강이다. 여기서는 **이름표**만 붙인다.

```python
class GPT(nn.Module):
    """이 책의 GPT: Decoder-only Causal LM."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        # token embedding, position, blocks, ln_f, lm_head ...

    def forward(self, idx):
        # idx: [B, T] -> logits: [B, T, V]
        ...
        return logits
```

클래스 이름이 `GPT`여도 API 키가 필요하지 않다.  
`forward`가 logits를 내고, Loss가 next-token CE를 쓰면 이 책의 GPT다.

Weight tying, dropout 위치, bias 유무 같은 세부 선택은 제56강에서 고정한다.

### 11. 실제 LLM에서는 어떻게 불리는가

실무·논문에서 자주 보는 표현:

| 표현 | 보통 의미 |
|---|---|
| GPT-style / GPT-like | Decoder-only Causal LM 골격 |
| Autoregressive LM | 자기회귀 언어 모델 (같은 계열) |
| Foundation model | 대규모 Pretraining된 기반 모델(과제 전) |
| Base model | 지시 튜닝 전 체크포인트 |
| Chat / Instruct 모델 | SFT 등으로 맞춘 변형 |

이 책은 초반에 **Base에 해당하는 작은 Causal LM**을 만든다. Chat 동작은 SFT 이후에 논의한다.

### 12. 실습

#### 실습 1 — 정의 문장 쓰기

빈칸을 채운 뒤, 소리 내어 읽어 본다.

> 이 책의 GPT는 (　　) 제품이 아니라, (　　) Transformer 위에서 (　　) 예측으로 학습·생성하는 모델 계열이다.

#### 실습 2 — 파이프라인 위치 표시

아래 단계 중 “GPT 아키텍처 본체”에 해당하는 구간에 밑줄을 친다.

```text
원시 텍스트 → 정제 → Tokenizer → ids → Embed → Blocks → lm_head → 생성 규칙
```

#### 실습 3 — 오해 교정

다음 문장이 왜 부정확한지 한 줄로 고친다.

1. “GPT는 Encoder로 문장을 이해하고 Decoder로 답한다.”
2. “Pretraining은 사람이 만든 Q&A만으로 한다.”
3. “Temperature는 학습 Loss의 일부다.”

### 13. 자주 하는 실수

1. **제품과 아키텍처를 동일시**  
   특정 앱 화면·구독 상품을 GPT의 정의로 두면, 구현 목표가 사라진다.

2. **Encoder-Decoder를 기본으로 착각**  
   번역 Transformer와 GPT 계열을 같은 그림으로 그리면 Causal mask·프롬프트 처리가 꼬인다.

3. **“지식을 저장하는 DB”로만 이해**  
   가중치는 분포 추정의 파라미터다. 검색·RAG는 다른 층이다.

4. **SFT/챗 정렬을 Pretraining과 혼동**  
   지시 데이터만으로 “GPT를 학습한다”고 말하면 단계가 뒤집힌다.

5. **규모 숫자를 사실처럼 암기**  
   토큰 수·파라미터 수는 출처·시점에 따라 다르다. 이 책은 원리를 우선한다.

### 14. 핵심 정리

- GPT는 이 책에서 **Decoder-only Causal LM + (대규모) Pretraining** 계열을 가리킨다.
- 본체 배선은 2권의 Causal LM과 동일하다: embed → Block×N → lm_head.
- Generative / Pre-trained / Transformer는 각각 생성·사전학습·구조를 가리킨다.
- Pretraining은 일반 텍스트에서 next-token 목표로 먼저 학습하는 단계다.
- 제품 챗봇은 그 위의 정렬·인터페이스 층이며, 3권 후반 SFT에서 입구를 연다.

### 15. 핵심 용어

| 용어 | 의미 |
|---|---|
| GPT | Generative Pre-trained Transformer. 이 책에선 아키텍처 계열명 |
| Decoder-only | Encoder 없이 Causal 블록 스택만 사용 |
| Causal LM | $P(x_t\mid x_{<t})$를 모델링하는 언어 모델 |
| Pretraining | 일반 텍스트로 먼저 학습하는 단계 |
| Autoregressive | 이전 출력을 다음 입력에 붙여 진행하는 방식 |
| Base model | 지시 튜닝 전 LM 체크포인트 |
| Foundation model | 광범위 Pretraining된 기반 모델(용어는 맥락 의존) |

### 16. 복습 문제

#### 문제 1 (정의)

이 책이 “GPT”라고 부를 때 포함하는 세 요소를 쓰시오.

#### 문제 2 (구조)

Decoder-only와 Encoder-Decoder의 차이를 GPT 관점에서 두 문장으로 쓰시오.

#### 문제 3 (단계)

Pretraining과 SFT의 역할 차이를 “데이터 종류” 기준으로 구분하시오.

#### 문제 4 (연결)

제48강의 Causal LM 파이프라인을 한 줄로 쓰고, 그것이 GPT 본체임을 표시하시오.

#### 문제 5 (오해)

“GPT는 항상 진실을 말한다”가 학습 목표와 어떻게 다른지 설명하시오.

---

### 정답 및 해설

#### 문제 1

Generative(생성), Pre-trained(사전학습), Transformer(구조). 구현상 Decoder-only Causal LM으로 합쳐진다.

#### 문제 2

Decoder-only는 프롬프트와 생성이 같은 Causal 스택을 쓴다. Encoder-Decoder는 입력 인코딩용 Encoder와 생성용 Decoder(+Cross-Attn)를 분리한다. GPT 계열은 전자.

#### 문제 3

Pretraining은 대량의 일반(또는 혼합) 텍스트에서 next-token을 학습한다. SFT는 지시·응답 형식의 지도 데이터로 행동을 맞춘다.

#### 문제 4

`ids → embed(+pos) → Block×N → (LN) → lm_head → logits`. 이 forward가 GPT 본체다.

#### 문제 5

학습 목표는 정답 토큰의 로그확률을 키우는 것이지, 사실 검증기가 아니다. 데이터에 있는 패턴을 따르면 그럴듯하지만 틀린 문장도 생성할 수 있다.

### 17. 다음 강의와 연결

이름을 고정했다. 다음은 **조립**이다.

**제56강. GPT 아키텍처 구현**에서는 2권의 Block·Embedding·Causal Mask를 가져와 `GPT` 클래스로 올린다. shape `[B, T, C]` → `[B, T, V]`를 코드로 고정하고, weight tying·final LN·생성용 `generate` 훅의 자리까지 마련한다.

이전 강의: [제54강. 2권 총정리 — GPT로 가는 길](../02_tokenizer_transformer/54강_2권_총정리_GPT로_가는_길.md)  
다음 강의: [제56강. GPT 아키텍처 구현](./56강_GPT_아키텍처_구현.md)
