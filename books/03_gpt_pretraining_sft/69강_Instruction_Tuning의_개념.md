# 제69강. Instruction Tuning의 개념

> **학습 목표**
> - Base LM과 Assistant(대화·지시 수행 모델)가 목표·데이터·평가에서 어떻게 다른지
> - Instruction following(지시 따르기)이 무엇을 의미하는지
> - SFT(Supervised Fine-Tuning)가 파이프라인에서 Pretraining과 RLHF 사이에 앉는 위치
> - Instruction Tuning이 “만능 정렬”이 아닌 이유와 한계
> - 제70강 데이터 형식·제71강 구현으로 넘어가기 전 지도

---
## 1. 왜 이것을 배우는가

Pretraining만 끝난 모델에게 이렇게 물어보면 자주 당황스럽다.

```text
User: 다음 문장을 한 줄로 요약해 줘. "....긴 글...."
Model: 다음 문장을 한 줄로 요약해 줘. "....긴 글...."
       요약이란 무엇인가? 요약의 역사는 ...
```

모델은 “인터넷/책에서 본 듯한 텍스트”를 **이어 쓰기**에는 강하지만, “사용자가 시킨 일만 하고 멈추기”에는 약하다. Base LM의 기본 본능은 **문서 완성**이다.

제품을 만들려면 본능을 바꿔야 한다.

```text
Pretraining          →  언어·세계 지식의 원재료
Instruction Tuning   →  “지시 → 응답” 형식에 맞추기  (이번 장)
RLHF / Preference    →  선호·안전·미묘한 정렬      (4권)
```

Instruction Tuning을 빼먹고 RLHF만 말하면, **지도 신호가 있는 쉬운 단계**를 건너뛴 설계가 된다.

## 2. 먼저 알아야 할 개념

- Causal LM / Next-token prediction (제57강, 2권 제32강)
- Mini GPT Pretraining 루프 (제68강)
- Fine-tuning 일반 개념: 이미 학습된 가중치를 다른 목표·데이터로 추가 학습
- Prompt: 모델에 넣는 입력 접두(문맥)

아직 깊게 들어가지 않는 것:

- LoRA / QLoRA 파라미터 효율 (제73·74강)
- Reward Model, PPO, DPO (4권)
- 특정 상용 모델의 “지시 성능 점수” 날조

## 3. 핵심 개념 설명

### 3.1 Base LM이란?

**Base LM(베이스 언어 모델)**은 주로 대량 텍스트의 next-token 목표로 Pretraining된 모델이다. 강점:

- 유창한 이어쓰기
- 광범위한 패턴·지식의 압축(데이터에 있을 때)
- 다양한 다운스트림의 **출발점**

약점(지시 관점):

- 질문이 와도 **질문 문체를 계속 생성**할 수 있음
- “답만 짧게” 같은 메타 지시를 무시하기 쉬움
- 대화 역할(user/assistant)이 데이터에 명시되지 않음
- 유해 요청에 대한 **거절 정책**이 없음(또는 불안정)

Base LM은 “똑똑한 자동완성”에 가깝다. “비서”가 아니다.

### 3.2 Assistant / Chat 모델이란?

**Assistant(어시스턴트)** 또는 Chat 모델은 사용자 요청에 대해 **도움이 되는 응답**을 내도록 추가 학습·정렬된 모델이다. 겉보기 인터페이스는 대개:

```text
system: (선택) 역할·규칙
user:   지시 / 질문
assistant: 응답
```

목표는 “그럴듯한 문서 계속”이 아니라 **과제를 수행하고 적절히 멈추는 것**이다.

### 3.3 Instruction Tuning이란?

**Instruction Tuning(인스트럭션 튜닝)**은 “(지시, 응답)” 또는 대화 형태의 데이터로 모델을 추가 학습하여, **지시 따르기(instruction following)** 능력을 키우는 과정이다.

좁은 의미에서는 지도 학습(SFT)만 가리키고, 넓은 의미에서는 이후 preference 단계까지 포함해 부르기도 한다. 이 책에서는 혼동을 줄이기 위해:

| 용어 | 이 책에서의 쓰임 |
|---|---|
| Instruction Tuning | 지시 데이터로 응답 형식을 가르치는 단계 (주로 SFT) |
| SFT | Supervised Fine-Tuning, 정답 응답에 CE/NLL로 학습 |
| RLHF 등 | 선호·보상 기반 추가 정렬 (4권) |

### 3.4 Instruction following

**Instruction following**은 모델이 사용자 지시의 **제약·형식·과제**를 존중하는 행동이다. 예:

- “세 줄로 요약”
- “JSON만 출력”
- “한국어로 답하고 코드는 Python”
- “모르면 모른다고 말함”

Pretraining PPL이 낮다고 해서 위 제약을 지키는 것은 아니다. (제67강: 생성 품질 ≠ PPL만)

### 3.5 SFT의 파이프라인 위치

전형적인 현대 LLM 파이프라인(개념도):

```text
1) Pretraining          (base LM)
2) SFT / Instruction Tuning
3) Preference optimization (RLHF, DPO, …)   ← 4권
4) (선택) 도구·추론·서빙 최적화              ← 5권 등
```

SFT의 역할:

1. **형식 전환**: 문서 완성 → 대화/지시-응답
2. **과제 분포 노출**: 요약, 번역, QA, 코딩 등 다양한 지시 패턴
3. **이후 단계의 초기 정책**: RLHF가 탐험할 “괜찮은 출발 정책” 제공

SFT가 없는 상태에서 preference만 적용하면, 모델이 아직도 “위키 문체 이어쓰기”일 수 있어 학습이 비효율·불안정해지기 쉽다.

## 4. 직관적으로 이해하기

비유:

| 단계 | 비유 |
|---|---|
| Pretraining | 도서관에서 책을 엄청 많이 읽음 |
| SFT | “질문하면 이렇게 답한다”는 **모범 답안집**으로 연습 |
| RLHF | 여러 답 중 **사람이 더 좋아하는 쪽**을 강화 |

모범 답안집만으로도 “시험 형식”에는 적응한다. 다만:

- 모범에 없는 미묘한 선호(톤, 안전, 장황함)는 남는다 → preference
- 모범이 틀리면 모델도 틀린 답을 **자신 있게** 흉내 낸다 → 데이터 품질

Instruction Tuning은 마법이 아니라 **분포를 바꾸는 학습**이다.

## 5. 수학적으로 이해하기 (목표만)

Pretraining 목표(복습):

$$

\mathcal{L}_{\mathrm{PT}}
=
-\sum_{t}\log p_\theta(x_t\mid x_{<t})

$$

시퀀스 $x$는 웹 문서·책 등 **일반 텍스트**다.

SFT에서는 샘플이 대개 $x=(\text{prompt}, \text{response})$ 형태다. 이상적인 목표는 응답 토큰에 집중한다.

$$

\mathcal{L}_{\mathrm{SFT}}
=
-\sum_{t\in\mathcal{T}_{\mathrm{resp}}}
\log p_\theta(x_t\mid x_{<t})

$$

$\mathcal{T}_{\mathrm{resp}}$는 assistant(응답) 구간 토큰 집합이다. 프롬프트 구간은 조건(문맥)으로만 쓰고 loss에서 빼는 것이 흔하다. (제70·71강에서 마스크로 구현)

직관:

- Pretraining: “모든 토큰이 타깃”
- SFT: “사용자가 이미 말한 부분은 맞출 필요 없고, **답할 부분**을 맞춘다”

## 6. 작은 예시로 보기

### 예 1 — Base LM의 이어쓰기

입력:

```text
Q: 대한민국의 수도는?
A:
```

Base LM이 학습 분포에 FAQ가 많다면 답을 이을 수도 있다. 하지만 입력이 조금만 달라도:

```text
사용자: 수도가 어디야? 한 단어로.
```

문서에 없는 말투면 **질문을 반복**하거나 장황한 백과 문체로 샐 수 있다.

### 예 2 — Instruction 데이터 한 건

```text
### Instruction:
한 단어로 답하라. 대한민국의 수도는?

### Response:
서울
```

또는 messages:

```json
[
  {"role": "user", "content": "한 단어로 답하라. 대한민국의 수도는?"},
  {"role": "assistant", "content": "서울"}
]
```

SFT는 이런 쌍을 많이 보여 주어, “지시 뒤에 응답이 온다”는 **조건부 분포**를 학습한다.

## 7. Instruction Tuning이 잘하는 것 / 못하는 것

### 잘하는 편

- 대화·QA·요약 등 **형식 전환**
- “단계별로”, “표로” 같은 **출력 포맷** 적응
- 데이터에 있는 과제 유형의 **모방**

### 한계

- 데이터에 없는 **새로운 추론 능력**을 마법처럼 창출하지 않음 (도움이 될 수는 있음)
- 잘못된 정답이 데이터에 있으면 **틀린 정렬**
- 안전·거부·미묘한 선호는 SFT만으로 불완전한 경우가 많음 → 4권
- Pretraining이 약하면 SFT로 “지식 구멍”을 다 메우기 어려움

## 8. 실제 LLM에서는 어떻게 사용하는가

공개·산업 파이프라인에서 흔히 관찰되는 패턴:

1. 대량 Pretraining으로 base 확보
2. 고품질 instruction/대화 데이터로 SFT
3. 사람 또는 AI가 만든 preference로 RLHF/DPO 등
4. 평가: 지시 준수·과제 성공·안전성 (PPL만 보지 않음)

데이터 출처 예(이름만 알고 넘어가도 됨): 공개 instruction 데이터셋, 인간 작성, 모델이 생성 후 필터링한 합성 데이터. **특정 벤치마크 점수나 “GPT-4를 이겼다”는 수치를 이 책이 만들어 내지 않는다.**

Hugging Face 등의 `Instruct` / `Chat` 체크포인트는 대개 base 위에 이런 단계가 올라간 결과물이다. 이름만 보고 base와 chat을 섞어 쓰면 프롬프트 형식이 깨진다. (제72강 Chat Template)

## 9. 실습

### 실습 1 — Base vs Instruct 행동 관찰 (개념)

가능하다면 같은 계열의 base와 instruct 체크포인트에 동일 프롬프트를 넣어 차이를 관찰한다. API가 없으면, 제68강 Mini GPT에 “지시문만” 넣었을 때와, 다음 강 형식의 모범 응답을 학습시킨 뒤의 차이를 **미니 실험으로** 비교할 계획을 적어 보라.

### 실습 2 — 지시 분해

다음 사용자 문장을 제약으로 분해하라.

```text
한국어로, 3개 불릿만, 코드 없이, 초등학생에게 설명하듯: CPU와 GPU 차이는?
```

제약 목록(언어, 개수, 금지, 청중)을 쓰면 “instruction following 평가 항목”이 된다.

### 실습 3 — 파이프라인 위치 암기

빈칸:

```text
Pretraining → (  ①  ) → (  ②  preference  )
```

① SFT/Instruction Tuning, ② RLHF 등.

## 10. 자주 하는 실수

1. **Base LM에게 ChatML만 넣고 instruct가 되길 기대**  
   템플릿은 형식이지 학습이 아니다. (학습 + 동일 템플릿이 필요)

2. **SFT = RLHF라고 생각**  
   SFT는 정답 모방, RLHF는 선호 최적화에 가깝다.

3. **PPL만으로 instruct 성공을 판정**  
   지시 준수는 별도 평가가 필요하다. (제67·75강)

4. **데이터가 곧 능력**  
   양보다 **형식 일관성·정답 품질·다양성**이 중요하다.

5. **Pretraining을 건너뛰고 작은 모델에 SFT만**  
   미니 실험은 가능하지만, “지식”과 “형식”을 혼동하지 말 것.

## 11. 핵심 정리

- Base LM은 문서 이어쓰기에 강하고, Assistant는 지시 수행에 맞춰 추가 학습된다.
- Instruction Tuning(주로 SFT)은 “(지시→응답)” 분포를 가르친다.
- SFT는 Pretraining 이후, RLHF 이전에 위치하는 **지도 미세조정**이다.
- 응답 토큰에 loss를 집중하는 것이 Pretraining 목표와의 핵심 차이다.
- SFT는 형식·과제 적응에 강력하지만, 선호·안전의 전부는 아니다.

## 12. 핵심 용어

| 용어 | 의미 |
|---|---|
| Base LM | Pretraining된 원천 언어 모델 |
| Assistant / Chat model | 지시·대화에 맞게 정렬된 모델 |
| Instruction Tuning | 지시 데이터로 따르기를 가르치는 단계 |
| Instruction following | 제약·과제를 존중하는 응답 행동 |
| SFT | Supervised Fine-Tuning, 정답 시퀀스에 대한 지도 학습 |
| Prompt | 모델 입력 접두(지시·문맥) |
| Response | 모델이 생성·학습하는 응답 구간 |
| RLHF | 인간 피드백 기반 강화학습 정렬 (4권) |

## 13. 연습 문제
### 문제 1 (개념)

Base LM과 Assistant의 차이를 “학습 목표” 관점에서 두 문장으로 쓰시오.

### 문제 2 (파이프라인)

SFT가 Pretraining과 RLHF 사이에 있어야 하는 이유를 한 가지 쓰시오.

### 문제 3 (목표)

$\mathcal{T}_{\mathrm{resp}}$에만 loss를 주는 이유를 설명하시오.

### 문제 4 (한계)

Instruction Tuning만으로 부족한 정렬 문제의 예를 하나 드시오.

### 문제 5 (연결)

제68강에서 만든 Mini GPT는 지금 파이프라인의 어느 단계 산출물인가? 다음 제70강에서 준비할 것은?

---

## 정답 및 해설

### 문제 1

Base LM은 일반 텍스트 next-token 예측이 목표다. Assistant는 사용자 지시에 대한 유용한 응답을 내도록 추가 학습된 모델이다.

### 문제 2

대화/지시 형식으로 정책을 먼저 맞춰 두어야, 이후 preference 학습이 의미 있는 응답 후보 위에서 작동하기 쉽기 때문이다.

### 문제 3

프롬프트는 이미 주어진 조건이므로 맞출 필요가 없고, 모델이 실제로 배워야 할 것은 응답 생성이기 때문이다. (구현은 제71강)

### 문제 4

예: 같은 정답이어도 톤·장황함·안전 거부에 대한 미묘한 인간 선호, 또는 데이터에 없는 정책적 거절.

### 문제 5

Pretraining(base) 단계의 미니 산출물이다. 제70강에서는 Instruction Dataset 형식(Alpaca식·messages)을 준비한다.

## 14. 다음 강의와 연결

개념상 “무엇을 왜 가르치는가”는 정리되었다. 다음은 **데이터 스키마**다.

다음 **제70강. Instruction Dataset 형식**에서는 Alpaca-like 필드, `system`/`user`/`assistant` messages, 그리고 프롬프트 마스킹의 미리보기를 다룬다.

> 지시 따르기를 말하려면, 먼저 “지시 데이터”가 어떤 모양인지 고정해야 한다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [68강. 프로젝트 — Mini GPT Pretraining](68강_프로젝트_Mini_GPT_Pretraining.md)
- **다음 강:** [70강. Instruction Dataset 형식](70강_Instruction_Dataset_형식.md)

<!-- /LECTURE_NAV -->
