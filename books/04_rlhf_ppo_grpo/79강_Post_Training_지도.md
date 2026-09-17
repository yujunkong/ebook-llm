# 제79강. Post-Training 지도

> **학습 목표**
> - Post-Training이 SFT 다음에 오는 이유
> - SFT → Reward Model → RLHF/PPO → DPO/GRPO → RLVR의 위치
> - 각 단계가 무엇을 최적화하는지（목적함수의 종류）
> - “정렬(alignment)”이 한 알고리즘이 아니라 단계 묶음이라는 점
> - 제80강부터 쓰는 State / Action / Reward 기호가 어디에 꽂힐지

---
## 1. 왜 이것을 배우는가

커뮤니티·논문·제품 문서에는 이름이 섞여 나온다.

| 흔히 하는 말 | 실제로 가리킬 수 있는 것 |
|---|---|
| “RLHF 했다” | Reward Model만? PPO까지? DPO만? |
| “정렬했다” | SFT인지 Preference인지 |
| “reasoning 학습” | SFT 체인오브쏘트인지 RLVR인지 |

이름을 고정하지 않으면, 제76강 Mini SFT를 “인간 선호까지 맞췄다”고 오해하거나, DPO만 돌리고 “PPO를 이해했다”고 착각하게 된다.

지도의 목적은 암기가 아니라 **막힐 때 어느 소켓을 다시 볼지** 찾는 것이다.

```text
3권: 유창성 · 지시 형식 · 응답 CE
4권: 선호 · 보상 · 정책 업데이트 · (검증 가능) 보상
```

## 2. 먼저 알아야 할 개념

- **Pretraining(사전학습)**: 대규모 next-token으로 언어 분포를 학습（3권 55~68）
- **SFT(Supervised Fine-Tuning, 지도 미세조정)**: instruction–response에 대한 조건부 CE（3권 69~77）
- **Policy(정책) $\pi_\theta$**: 상태（프롬프트·부분 완성）에서 다음 토큰 분포를 내는 모델 — 지금은 “LLM 자체”로 생각해도 된다
- **Checkpoint**: SFT 직후 가중치. 4권의 거의 모든 파이프라인은 여기서 출발한다

제78강에서 남긴 빈칸을 다시 적는다.

> 단일（소수）reference CE는 “동등하게 좋은 여러 답”의 선호를 가리지 못한다.  
> 탐색·보상 루프가 없다. “무난한 평균 문체”로 붕괴하기 쉽다.

## 3. 핵심 개념 설명

### 3.1 Post-Training이란

**Post-Training(포스트 트레이닝)**은 Pretraining（그리고 보통 SFT）**이후**에, 모델의 행동 분포를 **선호·보상·검증 신호**로 다시 쓰는 단계 묶음이다.

넓은 의미에서는 SFT 자체도 post-training에 넣기도 한다.  
이 시리즈의 4권에서는 다음을 중심으로 말한다.

```text
SFT로 만든 π_SFT
  → Preference / Reward / RL / Direct Preference / Verifiable Reward
    → 더 정렬된 π_*
```

즉 **“이미 말을 할 줄 아는 정책”을 “무엇이 더 나은가”에 맞게 재조정**하는 층이다.

### 3.2 전체 파이프라인 지도

```text
[3권]
Corpus → Pretrain → SFT (π_SFT) → Eval

[4권 Post-Training]
π_SFT
  ├─(A) Preference Dataset 수집 · 정제          ← 제84강
  ├─(B) Reward Model (RM) 학습                  ← 제85강
  ├─(C) RLHF: RM 보상 + KL로 PPO 등 정책 학습   ← 제86~89강
  ├─(D) DPO: RM 없이 선호를 직접 정책에         ← 제90~91강
  ├─(E) GRPO: 그룹 상대 비교 기반 정책 최적화   ← 제92강
  └─(F) RLVR: 검증 가능 보상(정답·규칙)으로 RL  ← 제93~94강
         → 평가 · 부작용 점검 · 서빙으로 (5권)
```

실제 제품은 (C)만, (D)만, (C)+(F) 조합 등 **경로가 갈라진다**.  
지도는 “반드시 이 순서”가 아니라 **역할이 다른 부품들의 위치**다.

### 3.3 각 단계가 최적화하는 것

| 단계 | 입력 신호 | 최적화하는 것（한 줄） | 대표 실패 |
|---|---|---|---|
| SFT | (prompt, response) 쌍 | 응답 토큰의 CE ↓ | 선호 미반영, style collapse |
| Preference data | (prompt, $y_w$, $y_l$) | “좋은/나쁜 답” 라벨 자산 | 라벨 노이즈, 편향 |
| Reward Model | 선호 쌍 | $r_\phi(x,y)$가 선호를 점수화 | reward hacking의 씨앗 |
| RLHF/PPO | RM 점수 + KL | 기대 보상 ↑, 참조정책에서 너무 멀지 않게 | 불안정, over-optimize |
| DPO | 선호 쌍 | 선호 확률을 정책에 직접 | 분포 밖 일반화·과적합 |
| GRPO | 그룹 샘플 상대 비교 | 그룹 내 상대 우위로 정책 갱신 | 그룹 설계·분산 |
| RLVR | 규칙/정답 검증기 | 검증 가능 보상의 기대값 ↑ | 좁은 과제, gaming |

공통 뼈대는 3권과 같다.

```text
배치 → forward → (어떤) loss/objective → backward → optimizer.step
```

달라지는 것은 **데이터가 “정답 한 줄”이 아니라 “비교·점수·검증”**이라는 점이다.

### 3.4 Alignment(정렬)를 한 단어로 쓰지 않기

**Alignment(얼라인먼트, 정렬)**는 모델 출력을 인간·제품·안전 기준에 맞게 맞추려는 **목표 묶음**이다.  
알고리즘 이름이 아니다.

| 층 | 정렬에 기여하는 방식 |
|---|---|
| SFT | “이런 형식으로 답하라”는 지도 |
| Preference / RM | “이쪽이 더 낫다”는 비교 |
| RLHF / DPO / GRPO | 그 신호를 정책 파라미터로 전파 |
| RLVR | “맞다/틀리다”를 자동 검증 |

“정렬했다”고 말할 때는 **어느 층까지**인지 항상 덧붙인다.

## 4. 직관적으로 이해하기

### 4.1 요리 비유

- **Pretrain**: 재료·불·칼질을 배운다（세상 텍스트）
- **SFT**: 레시피대로 한 접시를 만든다（지시–응답）
- **Preference / RM**: 손님 시식 점수표
- **RLHF/PPO**: 점수표로 조리법을 조금씩 고친다（너무 이상해지지 않게）
- **DPO**: 점수표 없이 “이 접시 > 저 접시” 쌍으로 바로 조리법 수정
- **RLVR**: 타이머·온도계처럼 **객관 측정**이 되는 요리만 점수화

비유가 완벽한 대응은 아니다. 그러나 “SFT만으로 시식 선호까지 끝”이 아니라는 감각은 남는다.

### 4.2 왜 SFT 다음인가

SFT 정책 $\pi_{\mathrm{SFT}}$가 없으면:

1. Preference 비교의 **후보 응답 품질**이 낮다
2. Reward Model이 보는 $(x,y)$ 분포가 엉망이다
3. RL이 탐색할 **초기 정책**이 붕괴한다

그래서 실무 파이프라인은 거의 항상:

```text
강한 base →（종종）SFT → Post-Training
```

이다. 4권은 이 순서를 전제로 한다.

### 4.3 두 갈래: “보상 모델 경로” vs “직접 선호 경로”

```text
경로 C (고전 RLHF):
  Preference → RM → PPO(정책) + KL(참조=π_SFT)

경로 D (DPO 계열):
  Preference → （암묵적 보상）→ 정책 직접 업데이트
```

둘 다 “선호”를 쓰지만, **중간 점수 함수 $r_\phi$를 명시적으로 두느냐**가 갈린다.  
제86~91강에서 수식으로 다시 만난다.

## 5. 수학적으로 이해하기（지도 수준）

아직 유도하지 않는다. **기호가 어디에 쓰이는지**만 고정한다.

### 5.1 SFT（복습）

$$

L_{\mathrm{SFT}}(\theta)
=
-\mathbb{E}_{(x,y)\sim\mathcal{D}_{\mathrm{SFT}}}
\sum_{t\in y}\log\pi_\theta(y_t\mid x,y_{<t})

$$

### 5.2 Reward Model（미리보기）

선호 $y_w \succ y_l \mid x$ 가 주어질 때, 점수는 대략:

$$

P(y_w \succ y_l \mid x)
=
\sigma\!\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)

$$

$r_\phi$를 학습한 뒤, 정책은（이상적으로）

$$

\max_\theta\ \mathbb{E}_{x,\,y\sim\pi_\theta}\big[r_\phi(x,y)\big]
-\beta\,\mathrm{KL}\big(\pi_\theta\|\pi_{\mathrm{ref}}\big)

$$

같은 목적에 가깝게 움직인다（세부: 제86~89강）.

### 5.3 DPO（미리보기）

보상 모델을 따로 두지 않고, 선호 데이터로 정책 비율을 직접 맞춘다（제90강）.

### 5.4 RLVR（미리보기）

정답·단위 테스트·규칙 검증기 $v(x,y)\in\{0,1\}$ 또는 실수 보상이 있으면:

$$

\max_\theta\ \mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}\big[v(x,y)\big]

$$

“취향”이 아니라 **검증 가능(verifiable)** 신호다（제93강）.

## 6. 작은 숫자로 직접 보기

같은 질문 $x$에 답 두 개:

| 응답 | SFT CE（낮을수록 “정답에 가까움”） | 인간 선호 |
|---|---|---|
| $y_A$: 정확·친절·너무 김 | 0.40 | 승 |
| $y_B$: 짧고 무뚝뚝·핵심만 | 0.55 | 패 |

SFT만 보면 $y_A$의 CE가 더 낮아 “좋다”.  
그러나 제품 기준이 “간결함”이면 선호는 $y_A$가 아닐 수도 있다 — 표는 예시일 뿐이다.

요점: **CE와 선호는 같은 축이 아니다.**  
Post-Training은 후자（또는 검증 점수）를 학습 신호로 올린다.

또 다른 미니 표 — 단계별 “움직이는 것”:

| 단계 | 주로 업데이트되는 것 |
|---|---|
| SFT | $\pi_\theta$（전체 또는 LoRA） |
| RM | $r_\phi$（보통 별도 헤드/모델） |
| PPO | $\pi_\theta$（+ 종종 value head） |
| DPO | $\pi_\theta$（RM 없이） |

## 7. 코드로 스케치하기

아직 학습 루프를 완성하지 않는다. **단계가 다른 함수**임을 이름만으로 구분한다.

```python
# post_training_map.py
# 의도: 4권 파이프라인의 "소켓"만 보여 준다. 실제 loss는 이후 강의.

from dataclasses import dataclass
from typing import Callable, List, Tuple

@dataclass
class PreferencePair:
    prompt: str
    chosen: str   # y_w: 선호된 응답
    rejected: str # y_l: 거부된 응답

def sft_loss(logprobs_on_response) -> float:
    """지도: 응답 토큰 NLL. 3권과 동일 계열."""
    return -float(sum(logprobs_on_response) / max(len(logprobs_on_response), 1))

def reward_model_score(rm: Callable[[str, str], float], prompt: str, answer: str) -> float:
    """RM: (prompt, answer) → 스칼라 점수."""
    return float(rm(prompt, answer))

def rlhf_surrogate(reward: float, kl_to_ref: float, beta: float = 0.1) -> float:
    """
    RLHF가 추구하는 방향의 초단순 스케치.
    실제 PPO는 advantage, clip, value loss가 붙는다 (제87~88강).
    """
    return reward - beta * kl_to_ref

def dpo_direction(log_ratio_w: float, log_ratio_l: float) -> str:
    """
    DPO 직관: chosen의 (정책/참조) 로그비를 rejected보다 키운다.
    수식은 제90강.
    """
    return "increase_chosen_vs_rejected" if log_ratio_w > log_ratio_l else "need_update"

def verifiable_reward(checker: Callable[[str, str], bool], prompt: str, answer: str) -> float:
    """RLVR: 규칙/정답 검증 → 0/1 보상."""
    return 1.0 if checker(prompt, answer) else 0.0

def demo_pipeline():
    pairs: List[PreferencePair] = [
        PreferencePair("2+2?", "4", "5"),
        PreferencePair("수도는?", "서울", "부산"),
    ]
    # 가짜 RM: 정답 문자열이면 높은 점수
    rm = lambda p, a: 1.0 if a in {"4", "서울"} else -1.0
    checker = lambda p, a: a in {"4", "서울"}

    for pair in pairs:
        r_w = reward_model_score(rm, pair.prompt, pair.chosen)
        r_l = reward_model_score(rm, pair.prompt, pair.rejected)
        assert r_w > r_l
        v = verifiable_reward(checker, pair.prompt, pair.chosen)
        print(pair.prompt, "RM gap", r_w - r_l, "VR", v)

if __name__ == "__main__":
    demo_pipeline()
```

이 파일이 “학습기”는 아니다.  
**이름이 가리키는 책임이 다르다**는 것만 확인하면 된다.

## 8. 실제 LLM에서는 어떻게 사용하는가

산업·오픈 모델 문서에서 자주 보이는 패턴:

1. **Base** 사전학습 모델
2. **Instruct / Chat** = 주로 SFT（+ 약간의 preference）
3. **RLHF / DPO / Constitutional / RLAIF** 등 브랜드마다 다른 후처리
4. **Reasoning** 계열은 RLVR·장문 CoT SFT가 섞이기도 함

독자가 체크포인트 카드에서 볼 것:

- `*-base` vs `*-instruct` vs `*-rlhf` 명명
- 라이선스·데이터 공개 범위（선호 데이터는 비공개인 경우 많음）
- “DPO만”, “PPO+RM”, “GRPO” 등 **방법 표기**

4권 실습（제95강）은 미니 스케일로 경로를 하나 골라 끝까지 통과하는 것이 목표다.

## 9. 실습

### 실습 A — 파이프라인 빈칸

다음을 노트에 손으로 채운다.

```text
π_SFT → (1) ______ → (2) RM → (3) ______ → 배포
         ↘ (대안) (4) DPO/GRPO
         ↘ (과제형) (5) RLVR
```

### 실습 B — 제품 문장 해부

임의의 “우리 모델은 RLHF로 정렬되었습니다” 문장을 고른다.  
아래를 추정·조사한다（모르면 “불명”）.

- Preference 데이터 출처
- RM 사용 여부
- PPO vs DPO
- KL / 참조 정책 언급 여부

### 실습 C — SFT 한계 한 줄

제77~78강 내용을 한 문장으로 다시 쓴다.  
“CE만으로는 ______ 를 직접 최적화하지 못한다.”

## 10. 자주 하는 실수

1. **SFT = RLHF**로 부르는 것  
   → 지도 CE와 보상/선호 최적화는 다르다.

2. **DPO를 했는데 PPO를 이해했다고 착각**  
   → 목적·구현·분산 구조가 다르다. 지도는 공유하되 동일시하지 말 것.

3. **Reward Model을 “정답 분류기”로만 생각**  
   → 선호 근사기이며, 해킹·편향에 노출된다（제85·96강）.

4. **Post-Training이 Pretrain을 대체한다고 생각**  
   → 보통 **위에 얹는다**. 기반 분포가 약하면 선호 학습도 약하다.

5. **모든 과제에 RLVR**  
   → 검증기가 없는 주관적 품질（문체·공손）에는 선호/RM이 더 자연스럽다.

## 11. 핵심 정리

- Post-Training은 SFT 이후, **선호·보상·검증**으로 정책을 재조정하는 단계 묶음이다.
- 고전 경로: Preference → RM → RLHF/PPO（+KL）.
- 직접 경로: Preference → DPO（및 변형）.
- 그룹 상대: GRPO.
- 검증 가능 보상: RLVR.
- Alignment는 목표 묶음이고, 알고리즘 이름이 아니다.
- 다음 강의부터는 이 지도를 **강화학습 기호**로 다시 그린다.

## 12. 핵심 용어

| 용어 | 한 줄 의미 |
|---|---|
| Post-Training | SFT 이후 선호·보상·검증으로 정책을 재쓰는 단계들 |
| Alignment | 인간/제품/안전 기준에 맞추려는 목표 묶음 |
| Preference Dataset | 같은 프롬프트에 대한 선호/비선호 응답 쌍 |
| Reward Model (RM) | $(x,y)$ → 스칼라 점수로 선호를 근사하는 모델 |
| RLHF | RM（또는 유사 보상）으로 정책을 RL 최적화하는 흐름 |
| PPO | RLHF에서 자주 쓰는 안정적 policy gradient 계열 알고리즘 |
| DPO | RM 없이 선호를 정책에 직접 넣는 방법 |
| GRPO | 그룹 샘플의 상대 비교에 기댄 정책 최적화 |
| RLVR | Verifiable（규칙/정답）보상으로 하는 RL |
| $\pi_{\mathrm{ref}}$ | KL 제약의 참조 정책（보통 $\pi_{\mathrm{SFT}}$） |

## 13. 연습 문제
### 문제 1

Post-Training이 보통 SFT **다음**에 오는 이유를 두 가지 쓰시오.

### 문제 2

다음 중 “명시적 Reward Model”이 **필수가 아닌** 쪽에 가까운 것은?  
(a) 고전 RLHF/PPO 경로 (b) DPO 경로

### 문제 3

SFT loss와 RLHF가 최적화하는 신호의 차이를 한 문장으로 쓰시오.

### 문제 4

RLVR의 “V”가 가리키는 성질은 무엇인가? Preference와의 차이를 한 줄로.

### 문제 5

제78강 → 제79강 → 제80강으로 이어지는 한 줄을 완성하시오.

```text
SFT 한계 정리 → Post-Training (    ) → RL 기호 (State/Action/Reward)
```

---

## 정답 및 해설

### 문제 1

예: (1) 선호/보상을 줄 후보 응답의 품질·형식이 SFT 뒤에 안정적이다. (2) RL의 초기 정책으로 $\pi_{\mathrm{SFT}}$가 필요하다.

### 문제 2

(b) DPO 경로.

### 문제 3

SFT는（대개）단일/소수 응답에 대한 토큰 CE이고, RLHF는 보상（선호 근사）의 기댓값을 올리며 보통 KL로 참조정책에 묶는다.

### 문제 4

Verifiable — 규칙·정답 등으로 **자동 검증 가능한** 보상. Preference는 주관적/비교 라벨에 가깝다.

### 문제 5

`지도`（또는 좌표 / 파이프라인）.

## 14. 다음 강의와 연결

지도가 생겼다.  
다음 **제80강. 강화학습 기초 — State, Action, Reward**에서는 LLM 생성 한 줄을 MDP에 가깝게 올려 본다.

- State ≈ 프롬프트 + 지금까지 만든 토큰
- Action ≈ 다음 토큰
- Reward ≈ 스칼라 피드백（선호·RM·검증）

제81~83강에서 Policy / Value / Policy Gradient / Advantage를 쌓은 뒤,  
**제84강. Preference Dataset**에서 다시 데이터 층으로 돌아온다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [제78강. 3권 총정리 — Post-Training으로](../03_gpt_pretraining_sft/78강_3권_총정리_Post_Training으로.md)
- **다음 강:** [제80강. 강화학습 기초 — State, Action, Reward](80강_강화학습_기초_State_Action_Reward.md)

<!-- /LECTURE_NAV -->
