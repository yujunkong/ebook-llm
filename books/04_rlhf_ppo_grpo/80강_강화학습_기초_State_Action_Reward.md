# 제80강. 강화학습 기초 — State, Action, Reward

> **학습 목표**
> - State(상태), Action(행동), Reward(보상)의 정의
> - LLM에서 state ≈ prompt + partial completion, action ≈ next token
> - Episode / Trajectory / Return이 생성 한 번과 어떻게 대응하는지
> - 보상이 “매 토큰”이 아니라 “응답 끝”에 오는 경우가 많다는 점
> - 제81강 Policy·Value로 넘어갈 기호 $s_t, a_t, r_t$

---
## 1. 왜 이것을 배우는가

RLHF 논문을 열면 곧바로 다음이 나온다.

$$

\max_\theta\ \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta}\big[r(x,y)\big]

$$

이 한 줄은 “보상의 기댓값을 올려라”이다.  
그러나 구현·직관·분산 분석은 모두 **시간 순서로 펼친 상호작용** 위에서 이야기한다.

| 지도 학습 감각 | RL 감각 |
|---|---|
| $(x,y)$ 정답 쌍 | 환경과 주고받으며 점수 |
| 즉시 CE | 늦게 오는 보상 |
| 한 샘플 = 한 loss | 한 trajectory = 여러 결정 |

SFT만 알면 “토큰 CE”에 익숙하다.  
RLHF는 “토큰 결정의 연쇄가 만든 **결과물**에 점수”를 준다.  
State / Action / Reward를 고정하지 않으면 PPO 식의 $A_t$, KL 항, GRPO 그룹 비교가 공중에 뜬다.

## 2. 먼저 알아야 할 개념

- **Autoregressive generation(자기회귀 생성)**: $y_t \sim \pi(\cdot\mid x,y_{<t})$（3권 57~59）
- **Policy $\pi_\theta$**: 조건부 토큰 분포를 내는 모델（제81강에서 정식화）
- **스칼라(scalar)**: 하나의 실수. 보상은 보통 스칼라다
- 제79강의 RM / Preference / RLVR — “점수가 어디서 오는지”의 출처만 기억

선수 증명·벨만 최적성 정리까지는 필요하지 않다.

## 3. 핵심 개념 설명

### 3.1 강화학습이 푸는 문제（한 줄）

**Reinforcement Learning(강화학습)**은 에이전트가 환경과 상호작용하며, **누적 보상**을 크게 만드는 행동 규칙을 찾는 문제 설정이다.

구성 요소의 최소 세트:

```text
Agent(정책)  ↔  Environment
   ↑ action        ↓ state, reward
```

LLM 정렬에서는:

- Agent ≈ 언어 모델（정책）
- Environment ≈ “프롬프트를 주고, 토큰을 받아 이어 붙이고, 마지막에 점수를 주는” 장치
- 점수 출처 ≈ 인간·RM·검증기

### 3.2 State (상태)

**State(스테이트, 상태)** $s$는 에이전트가 결정을 내릴 때 보는 **상황의 요약**이다.

이상적으로는 **Markov 성질**을 원한다.

$$

P(s_{t+1}\mid s_t,a_t,s_{t-1},a_{t-1},\ldots)
=
P(s_{t+1}\mid s_t,a_t)

$$

즉 “미래는 현재 상태와 행동에만 의존한다”.  
LLM 토큰 생성에서는 지금까지의 토큰 열이 곧 상태이므로, 이 성질을 **문자열을 상태에 넣음으로써** 거의 만족시킨다.

**LLM에서의 state（실무 근사）:**

$$

s_t = (x,\, y_{<t})
=
(\text{prompt},\, \text{지금까지 생성한 토큰})

$$

예:

- $t=0$: $s_0 = ($“한국의 수도는?”, $)$ — 빈 완성
- $t=1$: $s_1 = ($“한국의 수도는?”, “서”$)$
- $t=2$: $s_2 = ($“한국의 수도는?”, “서울”$)$

프롬프트만 state라고 부르는 거친 근사도 논문에 있다.  
토큰 단위 결정까지 내려가면 **부분 완성(partial completion)** 을 상태에 넣는 편이 정확하다.

### 3.3 Action (행동)

**Action(액션, 행동)** $a_t$는 상태 $s_t$에서 에이전트가 고르는 결정이다.

**LLM에서의 action:**

$$

a_t = y_t \in \mathcal{V}

$$

$\mathcal{V}$는 vocabulary（어휘 집합）.  
한 스텝 = 다음 토큰 하나.

연속 제어（로봇 관절 각도）와 달리, LLM의 action space는 **이산·초대형**（ $|\mathcal{V}|$ 수만~십만）이다.  
이 점이 Policy Gradient 분산·탐색을 어렵게 만든다（제82~83강）.

특수 토큰（EOS）을 고르면 episode가 끝날 수 있다.

### 3.4 Reward (보상)

**Reward(리워드, 보상)** $r_t$는 행동 직후（또는 구간 후）환경이 주는 **스칼라 피드백**이다.

$$

r_t \in \mathbb{R}

$$

높을수록 “그 결정（들）이 좋았다”는 학습 신호다.

**LLM Post-Training에서 흔한 형태:**

1. **Terminal / sparse reward**  
   응답이 끝난 뒤에만 $r = r(x,y)$ （RM 점수, 선호, 정확도）
2. **Step reward（드묾）**  
   토큰마다 소보상 — 설계가 어렵고 해킹되기 쉬움
3. **Verifiable reward**  
   정답이면 1, 아니면 0（RLVR）

고전 RLHF는 대부분 (1)에 가깝다.  
그래서 “어느 토큰 덕에 점수가 올랐는지”가 모호하다 — Advantage·credit assignment가 중요해진다（제83강）.

### 3.5 Transition (전이)

행동을 하면 상태가 바뀐다.

$$

s_{t+1} = (x,\, y_{<t+1}) = (x,\, y_{<t}+a_t)

$$

결정 LM 환경에서 전이는 **결정적(deterministic)** 인 경우가 많다.  
토큰을 붙이면 문자열이 그대로 길어진다.  
불확실성은 주로 **정책의 샘플링**에 있다.

### 3.6 Episode, Trajectory, Return

**Episode(에피소드)**는 시작부터 종료까지의 한 판이다.  
LLM에서는 보통 “한 프롬프트에 대한 응답 하나 생성”.

**Trajectory(궤적)** $\tau$:

$$

\tau = (s_0,a_0,r_0,\,s_1,a_1,r_1,\,\ldots,\,s_T,a_T,r_T)

$$

토큰 길이 $T$인 응답이면 길이 $T$의 결정 연쇄다.

**Return(리턴, 반환)**은 누적 보상이다. 할인율 $\gamma\in[0,1]$를 쓰면:

$$

G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}

$$

응답 끝에만 보상 $R$이 있고 중간 $r_t=0$이면, $\gamma=1$일 때 모든 $t$에 대해 $G_t = R$이 된다.  
“마지막 점수가 모든 토큰에 그대로 전파” — REINFORCE의 단순한 LLM 적용이 이렇게 보인다（제82강）.

## 4. 직관적으로 이해하기

### 4.1 미로 vs 문장

| | 미로 에이전트 | LLM |
|---|---|---|
| State | 칸 좌표 | 프롬프트+부분 문장 |
| Action | 상하좌우 | 다음 토큰 |
| Reward | 출구 +1, 함정 -1 | RM/정답/선호 점수 |
| 한 episode | 미로 탈출 시도 | 응답 하나 |

미로에서는 “왼쪽”이 즉시 벽인지 안다.  
문장에서는 “이 형용사”가 좋은지 **문장이 끝난 뒤**에야 점수가 온다.  
같은 RL 언어, **보상의 밀도**가 다르다.

### 4.2 대화로 한 스텝 펼치기

프롬프트: `2+2=`

| $t$ | $s_t$（요약） | $a_t$ | $r_t$ |
|---|---|---|---|
| 0 | `2+2=` | `4` | 0 |
| 1 | `2+2=4` | `EOS` | +1 （정답 검증） |

또는 오답:

| $t$ | $a_t$ | 끝 보상 |
|---|---|---|
| 0 | `5` |  |
| 1 | `EOS` | 0 |

에이전트는 $t=0$에서 이미 “운명”을 가른다.  
중간 보상이 없으면, 학습 알고리즘이 **어떤 토큰을 강화할지**를 통계로 추정해야 한다.

### 4.3 “환경”은 어디에 있나

코드로 보면 환경은 거창한 시뮬레이터가 아닐 수 있다.

```text
1) 데이터셋에서 prompt x 샘플
2) 모델이 y 샘플（토큰 루프）
3) r = RM(x,y) 또는 checker(x,y)
4) (x,y,r)를 버퍼에 저장 → 정책 업데이트
```

Gym의 `env.step`과 모양이 달라도, 기호 $s,a,r$의 역할은 같다.

## 5. 수학적으로 이해하기

### 5.1 MDP 튜플（라이트）

**MDP(Markov Decision Process, 마르코프 결정 과정)**는 대략 다음 튜플이다.

$$

(\mathcal{S},\,\mathcal{A},\,P,\,R,\,\gamma)

$$

- $\mathcal{S}$: 상태 공간
- $\mathcal{A}$: 행동 공간
- $P(s'\mid s,a)$: 전이
- $R(s,a)$ 또는 $R(s,a,s')$: 보상 함수
- $\gamma$: 할인율

LLM 생성 근사:

$$

\begin{aligned}
\mathcal{S} &\approx \mathcal{V}^{*}\times\mathcal{V}^{*}
\quad\text{（prompt, partial）}\\
\mathcal{A} &= \mathcal{V}\\
P &\approx \text{결정적 이어붙이기}\\
R &\approx \text{terminal } r(x,y)
\end{aligned}

$$

### 5.2 목적: 기대 반환

정책 $\pi$ 아래 기대 반환:

$$

J(\pi) = \mathbb{E}_{\tau\sim\pi}\big[G_0\big]

$$

terminal reward만 있으면 $G_0 = r(x,y)$이므로:

$$

J(\pi) = \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi(\cdot\mid x)}\big[r(x,y)\big]

$$

제79강 RLHF 목적의 **알맹이**가 이것이다.  
（KL 항은 “참조 정책에서 너무 멀지 말라”는 **정규화/제약**으로 나중에 붙는다.）

### 5.3 할인은 언제 쓰나

토큰 단위로 $\gamma<1$을 두면 앞쪽 토큰 보상이 상대적으로 커진다.  
LLM RLHF 구현에서는 terminal reward + $\gamma=1$ 또는 **토큰 평균 advantage**로 다루는 경우가 많다.  
세부 관례는 PPO 구현 강（제88강）에서 맞춘다.  
지금은 “$G_t$는 앞을 보는 누적 점수” 정도만 잡으면 된다.

## 6. 작은 숫자로 직접 계산하기

어휘를 극단적으로 줄인다.

$$

\mathcal{V}=\{\texttt{A},\,\texttt{B},\,\texttt{EOS}\}

$$

프롬프트 $x$는 고정. 보상:

$$

r(y)=
\begin{cases}
+1 & y=\texttt{A EOS}\\
0 & y=\texttt{B EOS}\\
-1 & \text{그 외（길이 초과 등）}
\end{cases}

$$

에피소드 예 1:

$$

s_0=x,\ a_0=\texttt{A},\ r_0=0,\ 
s_1=(x,\texttt{A}),\ a_1=\texttt{EOS},\ r_1=+1

$$

$\gamma=1$이면:

$$

G_0=0+1=1,\quad G_1=1

$$

에피소드 예 2:

$$

a_0=\texttt{B},\ a_1=\texttt{EOS},\ r_1=0
\Rightarrow G_0=G_1=0

$$

정책이 $\texttt{A}$를 더 자주 뽑게 바꾸면 $J(\pi)$가 오른다.  
이것이 “토큰 확률을 보상 방향으로 민다”는 RLHF의 초미시 모형이다.

숫자로 $J$ 근사:

| 정책이 A를 고를 확률 | $\mathbb{E}[r]$ （B면 0, 기타 무시） |
|---|---|
| 0.2 | 0.2 |
| 0.5 | 0.5 |
| 0.9 | 0.9 |

## 7. 코드로 구현하기

작은 이산 환경과 무작위 정책 롤아웃.

```python
# tiny_llm_mdp.py
"""LLM 생성 MDP의 초소형 시뮬레이터."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

VOCAB = ["A", "B", "EOS"]

@dataclass
class Step:
    state: Tuple[str, str]  # (prompt, partial)
    action: str
    reward: float
    next_state: Tuple[str, str]
    done: bool

def terminal_reward(prompt: str, completion: str) -> float:
    # 검증 가능 보상 예시: 정답이 "A"
    if completion == "A":
        return 1.0
    if completion == "B":
        return 0.0
    return -1.0

def step_env(prompt: str, partial: str, action: str, max_len: int = 2) -> Step:
    state = (prompt, partial)
    if action == "EOS" or len(partial) + 1 >= max_len:
        # EOS가 아니어도 길이 제한이면 강제 종료
        completion = partial if action == "EOS" else partial + action
        if action != "EOS" and len(partial) + 1 >= max_len:
            completion = partial + action
        if action == "EOS":
            completion = partial
        reward = terminal_reward(prompt, completion)
        next_partial = completion
        done = True
    else:
        next_partial = partial + action
        reward = 0.0
        done = False
    return Step(state, action, reward, (prompt, next_partial), done)

def random_policy(state: Tuple[str, str]) -> str:
    prompt, partial = state
    # 부분 완성이 있으면 EOS 비중을 높임
    if partial:
        return random.choices(VOCAB, weights=[0.2, 0.2, 0.6], k=1)[0]
    return random.choices(VOCAB, weights=[0.45, 0.45, 0.1], k=1)[0]

def rollout(prompt: str = "Q", max_len: int = 2) -> List[Step]:
    partial = ""
    steps: List[Step] = []
    for _ in range(max_len):
        action = random_policy((prompt, partial))
        tr = step_env(prompt, partial, action, max_len=max_len)
        steps.append(tr)
        partial = tr.next_state[1]
        if tr.done:
            break
    return steps

def returns_from_steps(steps: List[Step], gamma: float = 1.0) -> List[float]:
    """뒤에서부터 G_t 계산."""
    G = 0.0
    out: List[float] = []
    for st in reversed(steps):
        G = st.reward + gamma * G
        out.append(G)
    out.reverse()
    return out

if __name__ == "__main__":
    random.seed(0)
    total = 0.0
    n = 200
    for _ in range(n):
        steps = rollout()
        Gs = returns_from_steps(steps)
        total += Gs[0]
        # 한 번만 샘플 출력
    print("mean return ~", total / n)
    demo = rollout()
    print("demo steps:")
    for s, g in zip(demo, returns_from_steps(demo)):
        print(s, "G=", g)
```

실행하면 평균 반환과 한 trajectory의 $(s,a,r,G)$를 볼 수 있다.  
정책이 고정 난수이므로 학습은 없다 — **기호가 코드 필드와 1:1**인지만 확인한다.

## 8. PyTorch로 “토큰=행동”만 맛보기

실제 LLM 없이도, 로짓 → 샘플 → 로그확률을 action으로 취급할 수 있다.

```python
# action_from_logits.py
import torch
import torch.nn.functional as F

def sample_action(logits: torch.Tensor) -> tuple[int, torch.Tensor]:
    """
    logits: (V,)
    returns action index, log π(a|s)
    """
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()
    logp = dist.log_prob(action)
    return int(action.item()), logp

if __name__ == "__main__":
    torch.manual_seed(0)
    # 가짜 상태 인코딩 → 로짓 (V=3)
    logits = torch.tensor([2.0, 0.5, -1.0])
    a, logp = sample_action(logits)
    print("action", a, "log_prob", float(logp))
```

제82강 Policy Gradient는 이 `log_prob`에 보상（또는 Advantage）을 곱한다.

## 9. 실제 LLM에서는 어떻게 사용하는가

### 9.1 상태 표현

- 학습 코드: `input_ids`가 prompt+completion을 이어 붙인 텐서
- 마스크: prompt 구간은 보통 업데이트 신호에서 제외（SFT와 유사）
- KV cache는 **추론 가속**이지 MDP 정의의 일부가 아니다

### 9.2 행동 샘플링

RLHF 롤아웃에서는 temperature / top-p를 켜 **탐색**한다.  
Greedy만 쓰면 같은 응답만 나와 선호 비교·그라디언트 신호가 빈약해진다.

### 9.3 보상 타이밍

| 출처 | 언제 점수? |
|---|---|
| Reward Model | 보통 완성 $y$ 전체 |
| 인간 선호 | 쌍 비교 후 RM으로 증류되거나 오프라인 |
| 단위 테스트 / 정답 | 완성 후 검증기 |
| 안전 필터 | 완성 후（또는 중간）페널티 |

PPO 구현은 토큰마다 value head를 두더라도, **외부 보상은 terminal**인 경우가 많다.

### 9.4 다턴 대화

멀티턴이면 state에 대화 이력이 들어간다.  
형식은 chat template（3권 72강）이 규정한다.  
개념적으로는 여전히 “문자열 상태 + 다음 토큰 행동”이다.

## 10. 실습

### 실습 A — 기호 번역

다음 문장을 $s,a,r$로 번역해 보시오.

> 모델이 프롬프트 “요약해”를 보고 “짧게…”라고 쓰기 시작해, 문장이 끝난 뒤 RM이 0.7점을 주었다.

### 실습 B — sparse reward

길이 5 토큰, 보상은 마지막에만 $+2$일 때 $\gamma=1$이면 $G_0,\ldots,G_4$는?

### 실습 C — 코드 수정

`tiny_llm_mdp.py`에서 정답을 `"B"`로 바꾸고, `random_policy` 가중치를 수동으로 높여 평균 반환이 올라가는지 확인하시오.

### 실습 D — 지도와 연결

제79강 경로 중 RM을 쓰는 경로와 RLVR 경로에서, $r$의 **출처**만 각각 한 줄로 쓰시오.

## 11. 자주 하는 실수

1. **State = 프롬프트만**이라고 고정해 버리는 것  
   → 토큰 단위 결정에서는 partial completion이 상태에 포함된다.

2. **Reward = loss의 음수**로만 생각  
   → CE는 지도 신호. RL 보상은 선호/검증 등 **별 채널**일 수 있다.

3. **매 토큰 보상이 필수**라고 믿음  
   → LLM RLHF는 sparse/terminal이 흔하다.

4. **Action = 전체 응답 문자열**로만 정의  
   → 그렇게 정의하는 bandit 시각도 있으나, Policy Gradient·PPO 토큰 로그확률과는 해상도가 다르다. 이 시리즈 기본은 **토큰 action**.

5. **환경 난수와 정책 난수를 혼동**  
   → 문자열 이어붙이기는 결정적인 경우가 많고, 샘플링 난수는 정책 쪽이다.

6. **에피소드 = 학습 에폭**  
   → episode는 한 번 생성, epoch는 데이터 패스. 용어를 섞지 말 것.

## 12. 핵심 정리

- RL은 상태·행동·보상 상호작용으로 누적 점수를 키우는 문제 설정이다.
- LLM: $s_t=(x,y_{<t})$, $a_t=y_t$, 보상은 종종 terminal $r(x,y)$.
- Trajectory는 토큰 결정의 연쇄이고, Return $G_t$는 앞을 본 누적 보상이다.
- 전이는 대개 결정적 이어붙이기, 불확실성은 샘플링에 있다.
- 다음 강의에서 “행동을 고르는 규칙”과 “상태가 얼마나 좋은지”를 함수로 이름 붙인다.

## 13. 핵심 용어

| 용어 | 한 줄 의미 |
|---|---|
| State $s$ | 결정에 쓰는 상황 요약（prompt+partial） |
| Action $a$ | 다음 토큰 |
| Reward $r$ | 스칼라 피드백 |
| MDP | 상태·행동·전이·보상·할인으로 정의된 결정 과정 |
| Episode | 시작~종료 한 판（보통 응답 1개） |
| Trajectory $\tau$ | $(s_t,a_t,r_t)$ 시퀀스 |
| Return $G_t$ | $t$ 이후 누적（할인）보상 |
| Terminal / sparse reward | 끝에만 오는 보상 |
| Credit assignment | 늦게 온 점수를 어느 행동 탓으로 돌릴지 |

## 14. 연습 문제
### 문제 1

LLM 생성에서 state와 action의 표준 근사를 기호로 쓰시오.

### 문제 2

보상이 마지막에만 $R=3$이고 길이가 4, $\gamma=1$일 때 $G_0$과 $G_3$은?

### 문제 3

왜 LLM 환경의 전이를 “거의 결정적”이라 하는가?

### 문제 4

SFT의 정답 토큰 라벨과 RL의 reward가 다른 점을 한 문장으로.

### 문제 5

제79강의 Reward Model은 이 강의 기호로 어디에 해당하는가?

---

## 정답 및 해설

### 문제 1

$s_t=(x,y_{<t})$, $a_t=y_t$（어휘 위 이산）.

### 문제 2

$G_0=3$, $G_3=3$（중간 보상 0）.

### 문제 3

다음 상태는 선택한 토큰을 문자열에 붙인 결과로 정해지고, 환경 쪽 추가 난수가 보통 없기 때문이다.

### 문제 4

SFT는 “이 토큰이 정답”이라는 지도 라벨이고, RL reward는 완성（또는 구간）의 품질을 나타내는 스칼라로 토큰별 정답을 직접 지정하지 않을 수 있다.

### 문제 5

$r_\phi(x,y)$ — 에피소드（응답）에 대한 보상 함수 근사.

## 15. 다음 강의와 연결

기호 $s,a,r$가 생겼다.  
다음 **제81강. Policy와 Value Function**에서는:

- Policy $\pi(a\mid s)$ — “이 상태에서 토큰 확률”
- Value $V(s)$ — “이 상태가 얼마나 좋은가”
- Action-value $Q(s,a)$ — “이 상태에서 이 토큰을 고르면”

를 정의하고, LLM alignment 코드·수식에 어떻게 나타나는지 연결한다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [79강. Post-Training 지도](79강_Post_Training_지도.md)
- **다음 강:** [81강. Policy와 Value Function](81강_Policy와_Value_Function.md)

<!-- /LECTURE_NAV -->
