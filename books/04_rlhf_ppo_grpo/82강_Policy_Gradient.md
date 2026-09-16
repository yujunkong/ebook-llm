# 4권. RLHF · PPO · GRPO

## 제82강. Policy Gradient

### 1. 이번 강의에서 배울 것

제81강에서 정책 $\pi_\theta(a\mid s)$를 정의했다.  
이번 강의는 그 정책을 **보상의 기댓값이 커지도록** $\theta$로 미분하는 가장 기본 형태 — **Policy Gradient(정책 경사)** — 를 다룬다.

중심 이미지는 REINFORCE다.

$$
\nabla_\theta J(\theta)
\;\propto\;
\mathbb{E}\big[\nabla_\theta\log\pi_\theta(a\mid s)\, G\big]
$$

이 강의를 마치면 다음을 말할 수 있어야 한다.

- 목적 $J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[G_0]$의 의미
- $\nabla\log\pi\cdot G$（또는 $R$）직관: “잘한 행동의 로그확률을 키운다”
- 작은 이산 예제에서 그라디언트 방향 계산
- 코드로 REINFORCE 한 스텝
- LLM 토큰 로그확률에 같은 식을 붙이는 방법
- 분산이 크다는 한계 → 제83강 Advantage로 이어짐

PPO clip·GAE는 제87~88강. 오늘은 **뿌리**다.

### 2. 왜 이것을 배우는가

RLHF·PPO·GRPO·일부 RLVR 구현의 공통 뼈대는 다음과 같다.

```text
롤아웃으로 y 샘플
  → 보상/Advantage 계산
    → Σ ∇ log π_θ(y_t | …) · weight
      → optimizer.step
```

SFT는 weight 자리에 “정답이면 1”이 들어 있는 특수한 지도다.  
Policy Gradient는 weight 자리에 **경험적으로 얻은 점수**가 들어간다.

이 뿌리를 모르면 PPO의 surrogate loss가 “그냥 복잡한 CE”로만 보인다.

### 3. 먼저 알아야 할 개념

- Trajectory · Return $G_t$（제80강）
- $\pi_\theta$, $\log\pi$（제81강）
- Chain rule / autograd（1권）
- Softmax cross-entropy와의 관계: CE는 $-\log\pi(a^*)$

### 4. 핵심 개념 설명

#### 4.1 목적 함수

$$
J(\theta)
=
\mathbb{E}_{\tau\sim\pi_\theta}\big[G_0(\tau)\big]
=
\mathbb{E}_{\tau\sim\pi_\theta}\Big[\sum_{t=0}^{T}\gamma^t r_t\Big]
$$

LLM terminal reward $R=r(x,y)$만 있으면:

$$
J(\theta)=\mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot\mid x)}\big[R(x,y)\big]
$$

하고 싶은 일: $J$를 키우는 방향으로 $\theta$ 업데이트.

$$
\theta \leftarrow \theta + \eta\, \widehat{\nabla_\theta J}
$$

（경사 **상승**. 손실을 줄이는 딥러닝 습관과 부호가 반대일 수 있다. 구현은 $-J$를 loss로 두기도 한다.）

#### 4.2 왜 “그냥 $R$로 backprop”이 안 되나

$R$는 환경·RM이 준 스칼라로, $\theta$에 대한 **미분 가능 경로가 없는** 경우가 많다.  
샘플링 $y\sim\pi_\theta$의 이산 선택도 직선 미분이 어렵다.

Policy Gradient는 **점수 함수(score function) 항등**으로 우회한다.

$$
\nabla_\theta \pi_\theta(a\mid s)
=
\pi_\theta(a\mid s)\,
\nabla_\theta \log\pi_\theta(a\mid s)
$$

기댓값의 기울기를 “로그확률 기울기 × 스칼라” 기댓값으로 바꾼다.

#### 4.3 REINFORCE (가장 단순한 형태)

한 스텝 bandit（상태 하나, 행동 하나, 즉시 보상 $R$）:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{a\sim\pi_\theta}\big[
\nabla_\theta\log\pi_\theta(a)\, R
\big]
$$

알고리즘:

1. $a\sim\pi_\theta$ 샘플
2. 환경에서 $R$ 관측
3. $\theta$에 대해 $\log\pi_\theta(a)$를 미분해 $R$를 곱함
4. 상승 방향으로 한 스텝

직관:

| 결과 | 효과 |
|---|---|
| $R>0$ | 방금 뽑은 $a$의 확률 ↑ |
| $R<0$ | 방금 뽑은 $a$의 확률 ↓ |
| $|R|$ 큼 | 더 세게 밀고 당김 |

#### 4.4 궤적 버전

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_\tau\Big[
\sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)\, G_t
\Big]
$$

（유도 세부·인과 트릭에 따라 $G_t$ 대신 $\sum_{k\ge t}\gamma^{k-t}r_k$를 씀.）

LLM에서 흔히 쓰는 단순형（시퀀스 보상 $R$）:

$$
\widehat{\nabla J}
=
\sum_{t=1}^{|y|}
\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})
\cdot R
$$

모든 토큰에 **같은** $R$를 곱한다.  
단순하지만 분산이 크다 — “어느 토큰 덕분인지”를 구분하지 못함.

#### 4.5 SFT와의 한 줄 비교

SFT（정답 $y^*$）:

$$
\nabla_\theta L_{\mathrm{SFT}}
=
-\sum_t\nabla_\theta\log\pi_\theta(y^*_t\mid\ldots)
$$

REINFORCE（샘플 $y$, 보상 $R$）:

$$
\nabla_\theta(-J)
\sim
-\sum_t\nabla_\theta\log\pi_\theta(y_t\mid\ldots)\, R
$$

형태는 “가중 CE”에 가깝다.  
가중치가 데이터 라벨이 아니라 **롤아웃 점수**라는 점이 핵심이다.

### 5. 직관적으로 이해하기

#### 5.1 칭찬·꾸중 비유

모델이 문장을 말한다.  
선생님이 점수 $R$를 준다.

- 점수 좋음 → “방금 그 말투（토큰들）를 더 자주”
- 점수 나쁨 → “방금 그 말투를 덜”

선생님은 토큰마다 빨간펜을 안 친다.  
그래도 통계적으로 많은 시도를 평균하면, 좋은 궤적에 자주 나온 패턴의 확률이 오른다.

#### 5.2 로또 티켓

각 응답은 티켓, $R$는 당첨금.  
Policy Gradient는 **당첨된 티켓의 번호를 다음에 더 사게** 만든다.  
꽝 티켓은 줄인다.

문제는 티켓이 길고（토큰 많음）당첨금 분산이 크다는 것 → Advantage·baseline이 필요해진다.

#### 5.3 양수 보상만 있을 때의 함정

모든 $R\ge0$이면 “덜 나쁜” 샘플도 확률이 올라갈 수 있다.  
상대 비교·Advantage·그룹 내 정규화（GRPO）가 등장하는 이유 중 하나다.

### 6. 수학적으로 이해하기

#### 6.1 단스텝 유도 스케치

$$
J(\theta)=\sum_a \pi_\theta(a)\, R(a)
$$

（보상이 행동만의 함수인 bandit）

$$
\begin{aligned}
\nabla J
&=
\sum_a \nabla\pi_\theta(a)\, R(a)\\
&=
\sum_a \pi_\theta(a)\,\nabla\log\pi_\theta(a)\, R(a)\\
&=
\mathbb{E}_{a\sim\pi_\theta}\big[\nabla\log\pi_\theta(a)\, R(a)\big]
\end{aligned}
$$

#### 6.2 Baseline 불변（맛보기）

임의의 상태만의 함수 $b(s)$에 대해:

$$
\mathbb{E}_{a\sim\pi(\cdot\mid s)}\big[\nabla\log\pi(a\mid s)\, b(s)\big]=0
$$

이므로

$$
\nabla J
=
\mathbb{E}\big[\nabla\log\pi\,(R-b)\big]
$$

도 성립한다. $b=V(s)$로 두면 Advantage 쪽으로 간다（제83강）.

#### 6.3 Softmax 정책의 그라디언트

로짓 $z$, $\pi=\mathrm{softmax}(z)$일 때, 샘플 $a$에 대한 $\partial\log\pi(a)/\partial z$는  
“원-핫 $- \pi$” 형태다.  
즉 보상 가중 CE의 backward와 구현이 공유된다.

### 7. 작은 숫자로 직접 계산하기

행동 $\{L,R\}$, 보상 $R(L)=1$, $R(R)=0$.  
정책 $\pi(L)=p$, $\pi(R)=1-p$.  
파라미터를 $p$ 자체로 둔다（$0<p<1$）.

$$
J=p\cdot1+(1-p)\cdot0=p
$$

진짜 기울기: $\partial J/\partial p=1$.

REINFORCE 샘플 추정:

- $L$ 샘플: $\nabla_p\log\pi(L)=\partial_p\log p=1/p$, 곱 $R=1$ → 기여 $1/p$
- $R$ 샘플: $\nabla_p\log(1-p)=-1/(1-p)$, 곱 $0$ → 기여 $0$

$p=0.5$에서 $L$이 나오면 추정 기울기 $2$.  
기댓값: $P(L)\cdot(1/p)=p\cdot(1/p)=1$ — **불편(unbiased)**.

그러나 한 샘플은 $0$ 또는 $2$로 흔들린다.  
이것이 “맞지만 시끄러운” 추정기다.

토큰 두 개짜리 미니 응답:

| $y$ | $\log\pi$ 합（가정） | $R$ | 가중치 합 $\sum\log\pi\cdot R$ |
|---|---|---|---|
| `OK` | $\log0.4+\log0.5=-1.609$ | +1 | -1.609 |
| `NO` | $\log0.3+\log0.2=-2.813$ | 0 | 0 |

좋은 응답의 로그확률을（상승 방향으로）키우는 신호가 산다.

### 8. 코드로 구현하기 — 순수 Python bandit

```python
# reinforce_bandit.py
"""2-arm bandit에서 REINFORCE로 p(L)을 올리기."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class SoftmaxBernoulli:
    """logit_l 하나로 π(L)=sigmoid(logit_l)."""

    logit_l: float = 0.0

    @property
    def p_l(self) -> float:
        return 1.0 / (1.0 + math.exp(-self.logit_l))

    def sample(self) -> str:
        return "L" if random.random() < self.p_l else "R"

    def log_prob(self, a: str) -> float:
        p = self.p_l
        return math.log(p if a == "L" else (1.0 - p) + 1e-12)

    def grad_log_prob_wrt_logit(self, a: str) -> float:
        # d log π / d logit_l
        p = self.p_l
        if a == "L":
            return 1.0 - p  # sigmoid'(logit)=p(1-p), /p → 1-p
        return -p


def reward(a: str) -> float:
    return 1.0 if a == "L" else 0.0


def train(steps: int = 500, lr: float = 0.2, seed: int = 0):
    random.seed(seed)
    agent = SoftmaxBernoulli(logit_l=0.0)
    hist = []
    for t in range(steps):
        a = agent.sample()
        r = reward(a)
        g = agent.grad_log_prob_wrt_logit(a) * r
        agent.logit_l += lr * g  # gradient ascent
        hist.append(agent.p_l)
    return agent, hist


if __name__ == "__main__":
    agent, hist = train()
    print("final p(L)=", round(agent.p_l, 4))
    print("early", [round(x, 3) for x in hist[::50][:5]])
```

### 9. PyTorch로 REINFORCE

```python
# reinforce_torch.py
import torch
import torch.nn as nn
import torch.optim as optim


class Policy(nn.Module):
    def __init__(self, n_actions: int = 2):
        super().__init__()
        # 상태 없는 bandit: 로짓만 학습
        self.logits = nn.Parameter(torch.zeros(n_actions))

    def dist(self):
        return torch.distributions.Categorical(logits=self.logits)


def reinforce_step(policy: Policy, rewards_table: torch.Tensor, opt, baseline: float = 0.0):
    dist = policy.dist()
    action = dist.sample()
    r = rewards_table[action]
    # maximize E[R] ↔ minimize -(logπ)·(R-b)
    loss = -(dist.log_prob(action) * (r - baseline))
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(r), int(action)


if __name__ == "__main__":
    torch.manual_seed(0)
    rewards = torch.tensor([1.0, 0.0])  # action0 좋음
    policy = Policy(2)
    opt = optim.Adam(policy.parameters(), lr=0.1)
    avg = 0.0
    for t in range(1, 401):
        r, a = reinforce_step(policy, rewards, opt, baseline=0.5)
        avg += (r - avg) / t
        if t % 100 == 0:
            probs = torch.softmax(policy.logits.detach(), dim=0)
            print(t, "avgR", round(avg, 3), "probs", probs.tolist())
```

`baseline=0.5`를 넣으면 같은 식의 분산이 줄어든다.  
제83강에서 이를 $V(s)$로 일반화한다.

### 10. LLM 연결 — 토큰 루프

```python
# llm_reinforce_sketch.py
"""의사코드에 가까운 스케치. 실제 모델 forward는 생략."""

from typing import List, Callable
import torch


def sequence_logprobs(
    logprob_fn: Callable[[List[int]], List[torch.Tensor]],
    token_ids: List[int],
) -> torch.Tensor:
    """각 토큰의 log π(y_t | y_<t)를 이어 합산/스택."""
    logps = logprob_fn(token_ids)
    return torch.stack(logps)


def reinforce_loss_on_response(
    token_logprobs: torch.Tensor,
    reward: float,
) -> torch.Tensor:
    """
    token_logprobs: (T,)
    모든 토큰에 같은 reward를 곱하는 초보 LLM-REINFORCE.
    loss = - R * sum logπ  → 상승은 R>0일 때 logπ 증가
    """
    return -(reward * token_logprobs.sum())


# 예: 가짜 로그확률
if __name__ == "__main__":
    fake = torch.tensor([-0.5, -1.0, -0.2], requires_grad=True)
    loss = reinforce_loss_on_response(fake, reward=1.2)
    loss.backward()
    print("loss", float(loss), "grad", fake.grad)
```

실제 스택에서는:

1. `generate`로 샘플（no grad 또는 별도 롤아웃 모델）
2. 같은 토큰을 teacher force로 다시 forward해 logprob 수집
3. RM 점수 $R$（+ KL 페널티）를 곱해 loss
4. （PPO면）옛 정책 비율·clip

이 강의의 `reward * sum logπ`는 (3)의 가장 거친 형태다.

### 11. 실제 LLM에서는 어떻게 사용하는가

#### 11.1 RLHF

Reward Model이 $R=r_\phi(x,y)$를 주면, 정책은 Policy Gradient 계열（대개 PPO）로 $J$를 키운다.  
순수 REINFORCE만 쓰면 불안정해 **PPO·GRPO**로 갈아타는 경우가 많다.

#### 11.2 RLVR

$R\in\{0,1\}$（정답 여부）여도 식은 같다.  
정답 trajectory의 로그확률을 올린다.

#### 11.3 온폴리시 vs 오프폴리시

REINFORCE 기본형은 **지금 정책으로 샘플한** 데이터에 대한 온폴리시 추정이다.  
버퍼에 쌓아 여러 번 쓰면 비율 $\pi/\pi_{\mathrm{old}}$ 보정이 필요 → PPO.

#### 11.4 온도

롤아웃 temperature↑ → 탐색↑, 그라디언트 분산↑.  
학습용 logprob는 보통 temperature=1 로짓 기준이다.

### 12. 실습

#### 실습 A

$p=0.25$, $R(L)=1$, $R(R)=0$일 때 $L$ 샘플 하나의 REINFORCE 기울기（$\partial/\partial p$）는?

#### 실습 B

`reinforce_bandit.py`에서 `lr`을 크게 키우면 생기는 현상을 관찰하고 한 줄로 기록하시오.

#### 실습 C

모든 보상에 상수 $+10$을 더하면 이론적 $\nabla J$는? 실무에서 위험한 이유는?

#### 실습 D

제79강 KL 항이 Policy Gradient 목적에 붙으면 식이 어떻게 바뀌는지 **말로** 쓰시오（수식 완전 유도는 제89강）.

### 13. 자주 하는 실수

1. **loss에 $+\log\pi\cdot R$를 두고 descent**  
   → 부호 규약을 고정할 것. “maximize $R$”인지 확인.

2. **샘플은 temperature 0.8, logprob는 다른 분포**  
   → 온폴리시 가정이 깨진다.

3. **프롬프트 토큰까지 REINFORCE**  
   → 보통 completion 구간만.

4. **한 배치에 보상 스케일이 제각각**  
   → 학습이 폭주. 정규화·Advantage 필요.

5. **REINFORCE = PPO**  
   → PPO는 클리핑된 surrogate. REINFORCE는 조상.

6. **분산이 큰데 LR만 키움**  
   → 더 흔들린다. baseline부터.

### 14. 핵심 정리

- Policy Gradient는 $\mathbb{E}[R]$를 $\nabla\log\pi\cdot R$로 추정해 정책을 갱신한다.
- REINFORCE는 그 가장 단순한 몬테카를로 형태다.
- LLM에서는 $\sum_t\nabla\log\pi(y_t)\cdot R$가 기본 스케치다.
- 추정은 불편일 수 있으나 분산이 크다.
- 다음 강의에서 $R-V$ / $Q-V$로 분산을 줄이는 Advantage를 다룬다.

### 15. 핵심 용어

| 용어 | 한 줄 의미 |
|---|---|
| Policy Gradient | 정책 파라미터로 기대 반환의 경사 |
| REINFORCE | $\nabla\log\pi\cdot G$ 몬테카를로 추정 |
| Score function | $\nabla\log\pi$ 항등 트릭 |
| Gradient ascent | $J$를 키우는 방향 업데이트 |
| On-policy | 현재 정책 샘플로 추정 |
| Surrogate loss | 직접 $J$ 대신 쓰는 대체 목표（PPO） |
| Credit assignment | 점수 기여를 토큰에 배분하는 문제 |

### 16. 복습 문제

#### 문제 1

단스텝 bandit에서 $\nabla J=\mathbb{E}[\nabla\log\pi(a)\,R]$를 한 줄로 유도하는 핵심 항등은?

#### 문제 2

LLM 초보 REINFORCE에서 같은 $R$를 모든 토큰에 곱할 때의 장점·단점 하나씩.

#### 문제 3

SFT 그라디언트와 REINFORCE 그라디언트의 공통점·차이점.

#### 문제 4

$b(s)$ baseline을 빼도 기댓값 기울기가 남는 이유（스케치）.

#### 문제 5

다음 강의(제83강)가 다루는 기호의 이름은?

---

### 정답 및 해설

#### 문제 1

$\nabla\pi=\pi\nabla\log\pi$（score function identity）.

#### 문제 2

장점: 구현 단순, RM 한 번. 단점: 토큰별 기여 구분 약함·고분산.

#### 문제 3

공통: $\nabla\log\pi$ 형태. 차이: SFT는 정답 토큰·가중 1, REINFORCE는 샘플 토큰·가중 $R$.

#### 문제 4

$\mathbb{E}_{a\sim\pi}[\nabla\log\pi(a\mid s)b(s)]=b(s)\nabla\sum_a\pi=0$.

#### 문제 5

Advantage（우세 / $Q-V$ 또는 $G-V$）.

### 17. 다음 강의와 연결

Policy Gradient의 뼈대는 얻었다.  
다음 **제83강. Advantage**에서는

$$
A(s,a)=Q(s,a)-V(s)
\quad\text{또는}\quad
A_t \approx G_t - V(s_t)
$$

로 **평균 대비 초과 보상**을 정의해, 왜 PPO·GRPO가 이 신호를 쓰는지 고정한 뒤  
**제84강. Preference Dataset**으로 데이터 층에 착륙한다.
