# 94강. Reasoning Training
## 이번 강에서 배우는 내용

- Chain-of-thought 스타일 학습 신호가 무엇인지（프롬프트 트릭이 아니라 학습 신호로）
- 과정（process）보상과 결과（outcome）보상의 차이
- outcome-only RLVR이 왜 추론 흔적을 간접적으로 키우는지
- 선호 정렬（DPO/RLHF）과 reasoning RL이 같은 “정렬”이라도 신호의 종류가 다른 이유
- 제95강 미니 프로젝트로 넘어가기 전, 무엇을 장난감으로 줄일지

## 왜 중요한가?
SFT·선호 학습이 잘 되면 모델은 “그럴듯한 답”을 더 잘 낸다. 그런데 수학·코딩·논리처럼 **중간 단계가 틀리면 답이 틀리는** 과제에서는, 최종 문장만 예쁘게 만드는 신호가 부족하다.

```text
선호 정렬의 흔한 질문:  어느 답이 더 도움이 되는가?
추론 학습의 흔한 질문:  이 답이（검증 가능하게）맞는가? 과정이 타당한가?
```

둘 다 post-training이지만, **보상（또는 선호）이 어디서 오는지**가 다르다. 제84~91강의 인간/모델 선호와, 제93강의 검증기（verifier）는 같은 파이프라인 상자 안에 있어도 내용물이 다르다.

## Chain-of-thought를 “학습 신호”로 보기
### 2.1 추론 흔적이란

생성 시퀀스를 대략 다음처럼 나눈다.

$$

y = (z, a)

$$

- $z$: 중간 추론（scratchpad, 단계적 설명, 코드 초안 등）
- $a$: 최종 답（숫자, 선택지, 실행 결과, 짧은 결론）

프롬프트만으로 $z$를 유도하는 것과, **학습 목표가 $z$ 또는 $(z,a)$에 어떻게 걸리는지**는 별개다.

| 단계 | $z$의 역할 | 전형적인 신호 |
|---|---|---|
| 추론 프롬프팅 | 디코딩 시 유도 | 학습 없음 |
| SFT with CoT | 정답 풀이를 모방 | $(z^\star, a^\star)$에 대한 CE |
| Outcome RL | $z$는 자유 변수 | $a$만 검증 → 스칼라 보상 |
| Process RL | 단계마다 점수 | $z$의 부분 구간에 보상 |

### 2.2 SFT-CoT의 한계（역할만）

정답 풀이 데이터에 CE를 걸면 모델은 **그 스타일의 $z$** 를 흉내 낸다. 좋은 시작점이지만,

1. 정답 풀이 수집 비용이 크다
2. 틀린 중간 단계가 있어도 문체가 비슷하면 점수가 잘 나올 수 있다（평가가 느슨할 때）
3. 탐색（여러 $z$를 시도）을 직접 장려하지 않는다

그래서 많은 실무 흐름은 **SFT로 형식·기본 풀이 습관을 심은 뒤**, 검증 가능한 보상으로 RL（또는 유사 목표）을 얹는다. 이 문장은 **자주 보이는 설계 패턴**이지, 모든 논문이 동일한 순서를 썼다는 역사적 단정이 아니다（제97강）.

## Outcome reward vs Process reward
### 3.1 Outcome（결과）보상

최종 답 $a$만 본다.

$$

r_{\mathrm{out}}(x, y) = V(x, a),\quad y=(z,a)

$$

여기서 $V$는 규칙·유닛테스트·수식 동치 검사·컴파일러 등 **검증기**다. 제93강 RLVR의 핵심 형태다.

장점:

- 정의가 비교적 명확하다（맞다/틀리다, 또는 부분 점수）
- 인간 라벨러가 매 단계에 점수를 줄 필요가 없다
- $z$의 길이와 스타일을 모델이 스스로 탐색할 여지가 있다

한계:

- 보상이 **희소**하다. 긴 $z$ 전체가 하나의 스칼라에 묶인다
- 틀린 과정 + 운 좋은 정답, 또는 정답 복사 후 허위 설명이 남을 수 있다
- credit assignment가 어렵다（policy gradient의 분산↑）

### 3.2 Process（과정）보상

추론을 단계 $z_1,\ldots,z_K$로 나누고 단계 점수를 준다.

$$

r_{\mathrm{proc}}(x,y) = \sum_{k=1}^{K} r_k(x, z_{\le k})
\quad\text{또는}\quad
\text{단계별 advantage}

$$

$r_k$의 출처 예:

- 인간/모델이 “이 단계가 옳은가”를 판정
- 자동 검사（중간 식의 동치, 타입 체크, 부분 테스트）
- 학습된 process reward model（PRM）

장점:

- 중간 오류에 더 촘촘한 학습 신호
- 긴 CoT에서 credit assignment가 outcome-only보다 수월해질 **여지가** 있다

한계:

- 단계 분할·라벨 비용·판정자 편향
- PRM 자체가 reward hacking의 새 표적이 된다（제96강）
- “단계” 정의가 도메인마다 달라 파이프라인이 무겁다

### 3.3 한 장 비교

| | Outcome | Process |
|---|---|---|
| 보는 곳 | 최종 $a$ | 중간 $z_k$ |
| 신호 밀도 | 희소 | 상대적으로 밀집 |
| 자동화 | 검증기만 있으면 강함 | 단계 판정이 병목 |
| 대표 위험 | 희소 보상·허위 근거 | PRM/판정 해킹 |
| RLVR과의 관계 | 직접적 기본형 | 검증 가능한 과정이면 RLVR의 확장 |

실무에서는 **outcome을 기본 축**으로 두고, 도메인이 허용할 때만 process를 얹는 경우가 많다. 이것 역시 관찰된 경향이며, process가 항상 우월하다는 주장이 아니다.

## Outcome RL이 CoT를 “키우는” 메커니즘（직관）
검증기가 $a$만 본다면, 왜 $z$가 길어지거나 정교해질까?

정책 경사의 스케치（제82~83강）:

$$

\nabla_\theta\, J(\theta)
\;\propto\;
\mathbb{E}\big[
\,(r - b)\,
\nabla_\theta \log \pi_\theta(y\mid x)
\big]

$$

$y=(z,a)$ 전체의 로그확률에 같은 스칼라 $(r-b)$가 곱해진다. 정답을 자주 만드는 **궤적**（특정 길이·구조의 $z$ 포함）의 확률이 올라간다.

해석 포인트:

1. 모델이 “생각을 배운다”기보다, **정답률을 올리는 토큰 궤적이 강화**된다
2. $z$가 실제 계산에 쓰이지 않고 장식만 되어도, 우연히 $r=1$이면 그 장식이 살아남을 수 있다
3. 따라서 format 제약（답을 박스에 넣기, 최종 줄 규약）과 검증기 설계가 **추론의 질**을 좌우한다

```text
검증기 V가 엄격할수록 → 허위 CoT의 생존 공간 ↓
형식 규약이 명확할수록 → a 추출 오류 ↓
탐색 폭(그룹 샘플, temperature)이 있을수록 → 다양한 z 시도
```

GRPO류（제92강）처럼 **같은 프롬프트에 여러 샘플**을 뽑아 상대 비교하면, outcome 신호만으로도 분산을 줄이는 설계가 된다. reasoning 특화 “마법”이라기보다, **그룹 내 baseline**으로 credit을 안정화하는 쪽에 가깝다.

## 학습 신호의 스펙트럼（4권 좌표）
Reasoning training을 고립시키지 말고, 4권 신호 축 위에 올린다.

```text
[인간 선호 비교]
   Preference / RM / PPO / DPO
        │
        │  “더 나은 답” (주관·유용성·안전 포함)
        ▼
[검증 가능 보상]
   RLVR: V(x,a) 또는 V(x,z_k)
        │
        │  “맞다/틀리다” (또는 부분 점수)
        ▼
[혼합]
   선호 + 검증, 또는 안전 필터 + outcome RL
```

| 신호 | 대표 강의 | Reasoning에 주는 것 |
|---|---|---|
| SFT-CoT | （3권 SFT + 본강） | 풀이 형식의 초기 정책 |
| DPO/RLHF | 86~91 | 유용한 설명 문체·거부 정책 등 |
| Outcome RLVR | 93~94 | 정답률 지향 탐색 |
| Process reward | 94 | 단계 밀도 신호 |
| GRPO 등 그룹 상대 | 92 | 동일 문제 내 비교로 분산↓ |

**사실:** 최종 토큰 분포 $\pi_\theta(y\mid x)$를 바꾸는 최적화라는 점은 공통이다.  
**해석:** “reasoning 모델”이라는 제품 이름은 보통 **긴 $z$ + 검증 과제 + RL 단계**가 강조된 결과물을 가리키는 마케팅·분류 라벨에 가깝다.

## 데이터·프롬프트·검증기 설계 체크
Reasoning RL을 돌리기 전에 고정할 계약:

1. **문제 분포** — 수학만? 코드만? 도구 사용?
2. **답 추출 규칙** — 정규식, JSON 필드, 테스트 러너 stdout
3. **보상 스케일** — $\{0,1\}$, 부분 점수, 길이 페널티
4. **참조 정책** — SFT 초기값, KL（제89강）을 쓸지
5. **샘플 수** — 프롬프트당 $G$개（GRPO식）또는 PPO 롤아웃 길이
6. **과정 라벨** — 쓸 거면 단계 정의와 판정자

미니 스케일（제95강）에서는 3~4번을 극단적으로 단순화한다. 예: 덧셈/짝홀 판정 + `#### <answer>` 규약 + $r\in\{0,1\}$.

## 숫자로 보는 credit assignment（장난감）
문제: `2+3=?`. 두 궤적:

| 궤적 | $z$ | $a$ | $r$ |
|---|---|---|---|
| A | `2+3=5` | `5` | 1 |
| B | `구름이 예쁘다` | `5` | 1 |
| C | `2+3=6` | `6` | 0 |

Outcome만 보면 A와 B가 **같은 보상**이다. 한 번의 업데이트로는 B의 허위 근거도 같은 방향으로 밀릴 수 있다. 반복 학습과 다양한 문제에서 B가 평균적으로 불리해지길 기대하지만, **보장이 아니다**.

Process 신호가 “산술 단계가 맞는가”를 보면 B는 중간에서 감점된다. 이것이 process의 존재 이유이지, 자동으로 만능 해결책은 아니다.

## 구현 스케치（개념 코드）
대규모 프레임워크 전체가 아니라, **신호 위치**만 드러낸다.

```python
# reasoning_outcome_step.py — 개념 스케치 (완결 트레이너 아님)
"""Outcome reward on final answer; sequence logprob for policy loss."""

from __future__ import annotations

import re
from dataclasses import dataclass

ANS_RE = re.compile(r"####\s*(.+)\s*$", re.MULTILINE)

def extract_answer(text: str) -> str | None:
    m = ANS_RE.search(text)
    return m.group(1).strip() if m else None

def outcome_reward(prompt: str, completion: str, gold: str) -> float:
    pred = extract_answer(completion)
    if pred is None:
        return 0.0
    return 1.0 if pred == gold.strip() else 0.0

@dataclass
class Rollout:
    prompt: str
    completion: str
    logprob_sum: float  # Σ log π(y_t | ·) for tokens in completion
    reward: float

def reinforce_loss(rollouts: list[Rollout], baseline: float = 0.0) -> float:
    """Scalar surrogate: -E[(r - b) * logπ(y|x)]. Caller backprops through logprob_sum."""
    if not rollouts:
        return 0.0
    terms = [-(r.reward - baseline) * r.logprob_sum for r in rollouts]
    return sum(terms) / len(terms)
```

실제 학습에서는 `logprob_sum`을 토큰 로짓에서 미분 가능하게 연결하고, GRPO라면 같은 `prompt`의 그룹 평균을 baseline으로 쓴다. 제95강에서 더 작은 완결 파이프라인으로 내려간다.

Process 쪽 스케치:

```python
def process_rewards(steps: list[str], step_ok: list[bool]) -> list[float]:
    """Per-step +1/0; train with dense advantages if available."""
    assert len(steps) == len(step_ok)
    return [1.0 if ok else 0.0 for ok in step_ok]
```

## Reasoning Training에서 자주 하는 설계 선택
| 선택 | 흔히 쓰는 이유 | 주의 |
|---|---|---|
| 답 형식 강제 | 검증 자동화 | 형식만 맞추는 hacking |
| 길이 페널티/보너스 | 과도한 장문 억제 또는 최소 추론 유도 | 보상 재정의 → 새 해킹 |
| 다샘플@pass | 탐색·평가 | train/eval 프로토콜 혼동 |
| KL to SFT | 붕괴·언어 붕괴 완화 | 과도하면 탐색↓ |
| 도구 호출 | 계산 정확도↑ | 환경·샌드박스 비용 |

“추론이 늘었다”를 **토큰 길이만으로** 선언하지 않는다. 길이↑는 부산물일 수 있다. 가능하면 **검증 통과율**, 형식 준수율,（있다면）과정 오류율을 함께 본다. 구체 수치는 데이터·모델에 따라 달라지므로 이 책에서 임의로 박지 않는다.

## 제93강 RLVR과의 연결（한 줄 다리）
```text
RLVR:     검증 가능한 r = V(·) 로 정책을 갱신한다
본강:     그 r를 (z,a) 구조의 추론 생성에 어떻게 걸지 정리한다
다음(95): 선호 미니셋으로 DPO 또는 RM+REINFORCE를 손으로 닫는다
```

RLVR이 “보상 출처”의 이름이고, Reasoning Training은 **그 보상을 추론 궤적에 적용하는 설계 문제**다. 동치어처럼 쓰이기도 하나, 이 책에서는 위처럼 층을 나눈다.

## 수식 보강 — Reasoning 목표

긴 추론 자취 $z$와 답 $y$에 대해

$$
p(y\mid x)=\sum_z p(y,z\mid x)
$$

를 직접 쓰기 어려워, 샘플된 자취에 보상/검증 신호를 줍니다(RLVR 등).

## 수식으로 고정하기 — Outcome RL + KL
### 12.1 목표（개념）

프롬프트 $x$, 응답 $y=(z,a)$에 대해 검증기 $V$가 outcome 보상을 준다고 하자.

$$

r(x,y)=V(x,a)\in\{0,1\}
\quad\text{（또는 부분 점수）}

$$

참조 정책 $\pi_{\mathrm{ref}}$（보통 SFT）를 둔 KL-제약 목표:

$$

J(\theta)
=
\mathbb{E}_{x,\,y\sim\pi_\theta}
\big[r(x,y)\big]
-
\beta\,
\mathbb{E}_{x}
\big[
\mathrm{KL}\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x)\big)
\big]

$$

유효 보상으로 합치면（토큰/시퀀스 추정은 구현 의존）:

$$

R(x,y)=r(x,y)-\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}

$$

**사실:** 이 형태는 제86·89강의 RLHF 목표와 **같은 뼈대**이고, $r$의 출처만 선호 RM에서 검증기로 바뀐다.  
**해석:** Reasoning RL을 “완전히 다른 손실 우주”로 외우기보다, **신호 소켓만 교체된 정책 최적화**로 읽는다.

### 12.2 Policy gradient 한 줄

$$

\nabla_\theta J
\;\approx\;
\mathbb{E}\big[
(R-b)\,\nabla_\theta\log\pi_\theta(y\mid x)
\big]

$$

$b$는 baseline（배치 평균, 가치함수, 그룹 평균 등）. GRPO（제92강）는 같은 $x$의 그룹 $\{y^{(i)}\}_{i=1}^{G}$에서

$$

\hat{A}^{(i)}
=
\frac{r^{(i)}-\mathrm{mean}(r)}{\mathrm{std}(r)+\varepsilon}

$$

처럼 **상대 점수**로 $b$를 대체하는 설계다（정규화 세부는 구현마다 다름）.

### 12.3 Process 보상의 합

단계 $k=1..K$에 밀집 보상 $r_k$가 있으면, 단순 합

$$

R=\sum_{k=1}^{K}r_k
\quad\text{또는}\quad
R=\sum_{k}\gamma^{k}r_k

$$

로 return을 만들고, 토큰·단계에 Advantage를 배분한다. Outcome-only보다 식이 “더 똑똑해 보여도”, $r_k$ 자체가 틀리면 최적화가 그 틀린 판정을 증폭한다.

## 길이·형식과 보상의 결합（해킹 입구）
실무에서 자주 쓰는（그리고 자주 해킹되는）결합:

$$

r_{\mathrm{total}}
=
r_{\mathrm{correct}}
+\lambda_{\mathrm{fmt}}\,r_{\mathrm{format}}
-\lambda_{\mathrm{len}}\,|y|

$$

| 항 | 의도 | 부작용 후보 |
|---|---|---|
| $r_{\mathrm{correct}}$ | 정답 | 허위 $z$ + 운 좋은 $a$ |
| $r_{\mathrm{format}}$ | 파싱 가능 | 형식만 맞추고 내용은 빈약 |
| 길이 페널티 | 장문 억제 | 필요 추론까지 잘라 버림 |
| 길이 보너스 | 최소 추론 유도 | 군더더기 CoT |

**숫자 약속:** $\lambda$의 “최적값”을 이 책에 고정하지 않는다. 도메인·토크나이저·검증기에 따라 다시 고른다.

## 평가 프로토콜 — pass@k와 학습 신호 혼동 금지
검증 가능 과제에서 자주 쓰는 평가:

$$

\mathrm{pass@}k
=
\mathbb{P}(\text{적어도 1개가 통과}\mid k\text{ samples})

$$

학습 중 temperature·샘플 수와 평가 $k$를 섞어 쓰면, “모델이 똑똑해진 것”과 “더 많이 뽑아 운을 산 것”이 구분되지 않는다.

```text
학습:  그룹 G로 탐색·업데이트
평가:  고정 decode 설정 + 명시적 k
보고:  pass@1 과 pass@k 를 분리
```

제96강의 Goodhart와 같은 규율이다. 지표 정의를 리포트에 박제한다（제95·118강 습관）.

## LLM에서는 어디에 사용될까?
제품·연구에서 이 강의가 직접 닿는 지점:

1. **수학·코딩 어시스턴트** — 유닛테스트·심볼릭 검증으로 $r$를 줌
2. **도구 사용 에이전트** — 도구 실행 성공/실패가 outcome 신호
3. **장문 해설 제품** — 선호 정렬（문체）+ 검증（정답）혼합
4. **서빙 비용** — $z$가 길어지면 decode 토큰↑ → 5권 TTFT/TPOT/Throughput과 충돌（제98·107강）

```text
정렬 성공: 정답률↑, 평균 |z|↑
서빙 청구: 토큰당 비용·GPU시간↑
→ “더 잘 생각함”과 “더 비싸짐”이 동시에 올 수 있다
```

## 실습
### 실습 A — 신호 분류

다음 보상 정의를 outcome / process / preference 중 어디에 가까운지 고르시오.

1. 최종 숫자만 gold와 비교
2. 풀이 단계마다 인간 “OK/NG”
3. 두 전체 답 중 어느 쪽이 더 친절한가

### 실습 B — 허위 CoT 감사

제7절 표의 A/B/C에 대해, process 판정 규칙 한 줄을 스스로 적어 B만 감점되게 하시오.

### 실습 C — KL 사고실험

$r=1$인 궤적이 $\pi_{\mathrm{ref}}$에서 극단적으로 멀 때, $\beta$를 키우면 $R$이 어떻게 바뀌는지 문장으로 쓰시오.

### 실습 D — pass@k 설계

동일 모델에 대해 pass@1과 pass@8을 보고할 때, 디코드 설정을 어떻게 고정할지 체크리스트 5항목을 만드시오.

## 자주 하는 실수
1. CoT 프롬프트만 넣고 “reasoning training을 했다”고 말함  
2. Outcome 보상만으로 과정의 진실을 보장했다고 착각  
3. 길이↑를 능력↑의 증거로 사용  
4. pass@k와 학습 샘플 수를 같은 숫자로 혼용  
5. PRM을 쓰면서 PRM 해킹을 평가 축에 안 넣음  
6. 선호 정렬과 검증 RL을 한 손실로 뭉개서 디버깅  
7. 형식 보너스 가중을 과도하게 키워 빈 껍데기 답을 양산  

## 핵심 요약
- CoT는 프롬프트 기법만이 아니라, $y=(z,a)$에 걸리는 **학습 신호의 배치** 문제다.
- Outcome 보상은 최종 답 검증에 강하고 희소하며, Process 보상은 단계 밀도에 강하고 라벨·해킹 비용이 있다.
- Outcome RL은 정답을 만드는 궤적을 강화할 뿐, 허위 근거를 자동으로 제거하지 않는다.
- 선호 정렬과 reasoning RL은 같은 post-training 상자 안의 **다른 신호**다.
- 제95강은 선호 쪽 미니 파이프라인으로, 신호→손실→업데이트 루프를 코드로 닫는다.

## 용어 사전
| 용어 | 한 줄 의미 |
|---|---|
| Chain-of-thought ($z$) | 최종 답 앞의 중간 생성 |
| Outcome reward | 최종 답만으로 주는 보상 |
| Process reward | 중간 단계에 주는 보상 |
| Verifier $V$ | 규칙/테스트로 $r$를 계산하는 장치 |
| RLVR | 검증 가능 보상으로 하는 RL |
| Credit assignment | 어느 토큰·단계가 $r$에 기여했는지 배분 |
| GRPO（연결） | 그룹 샘플 상대 비교로 baseline을 잡는 흐름 |

## 연습문제
### 문제 1

$y=(z,a)$에서 outcome reward가 직접 보는 것은 $z$인가 $a$인가?

### 문제 2

같은 $r=1$인 두 궤적 중 하나가 허위 근거여도 outcome-only에서 동시에 강화될 수 있는 이유를 한 문장으로 쓰시오.

### 문제 3

Process reward의 대표적인 비용을 두 가지 쓰시오.

### 문제 4

RLVR과 Reasoning Training을 이 책이 나누는 기준을 한 줄로 쓰시오.

### 문제 5

제95강 미니 프로젝트에서 reasoning outcome RL 대신 **preference** 쪽을 먼저 실습하는 이유를, 신호의 관점에서 추측해 쓰시오.

---

## 정답 및 해설
### 문제 1

$a$（최종 답）.

### 문제 2

보상이 시퀀스 전체에 같은 스칼라로 곱해지므로, $a$만 맞으면 $z$의 질과 무관하게 로그확률이 같은 방향으로 갱신될 수 있다.

### 문제 3

예: 단계 분할·라벨 비용, PRM/판정자 해킹（또는 판정 편향）.

### 문제 4

RLVR은 검증 가능 보상이라는 신호 출처, Reasoning Training은 그 신호를 추론 궤적 $(z,a)$에 거는 설계.

### 문제 5

선호 쌍→DPO/RM은 검증기 없이도 닫히는 루프라, 4권 전반（84~91）과 연결해 “신호→손실”을 먼저 손에 익히기 좋다. Reasoning RL은 93~94의 확장으로 이어진다.

## 다음 강의와 연결
**제95강. 프로젝트 — Preference / RL 실습**에서 초소형 선호 데이터로 DPO 또는 RM+REINFORCE 미니 파이프라인을 끝까지 돌린다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [93강. RLVR과 Verifiable Reward](93강_RLVR과_Verifiable_Reward.md)
- **다음 강:** [95강. 프로젝트 — Preference / RL 실습](95강_프로젝트_Preference_RL_실습.md)

<!-- /LECTURE_NAV -->
