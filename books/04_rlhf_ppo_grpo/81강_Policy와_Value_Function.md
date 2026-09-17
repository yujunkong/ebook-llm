# 제81강. Policy와 Value Function

> **학습 목표**
> - $\pi(a\mid s)$, $V^\pi(s)$, $Q^\pi(s,a)$의 정의
> - LLM에서 $\pi$가 곧 next-token softmax라는 점
> - $V$ / $Q$가 RLHF·PPO value head와 어떻게 연결되는지
> - “좋은 정책”과 “높은 가치”의 관계
> - 제82강 Policy Gradient로 넘어갈 준비（$\nabla\log\pi$）

---
## 1. 왜 이것을 배우는가

RLHF 구현 저장소를 열면 보통 두 머리가 보인다.

```text
backbone (Transformer)
  ├─ policy head:  vocab logits → π(a|s)
  └─ value head:   scalar V(s)   （PPO류）
```

SFT만 할 때는 policy head（LM head）만 있으면 된다.  
PPO는 **value**로 baseline·Advantage를 만들어 분산을 줄인다（제83강）.

기호 없이 “점수가 높은 쪽으로 확률을 올린다”만 반복하면,  
왜 KL을 $\pi$에 걸고 value loss를 따로 두는지가 설명되지 않는다.

## 2. 먼저 알아야 할 개념

- State / Action / Reward / Return $G_t$（제80강）
- Softmax · logit（2권 33강）
- Autoregressive LM: $p(y_t\mid y_{<t},x)$（3권）
- 기댓값 $\mathbb{E}[\cdot]$ — “분포로 평균”

## 3. 핵심 개념 설명

### 3.1 Policy (정책)

**Policy(폴리시, 정책)** $\pi$는 각 상태에서의 **행동 분포**다.

$$

\pi(a\mid s) = P(A_t=a\mid S_t=s)

$$

결정적 정책은 $a=\mu(s)$처럼 하나를 고르지만, LLM은 **확률적 정책**이 기본이다.  
같은 프롬프트에도 여러 응답이 가능해야 탐색·다양성이 생긴다.

**모수화:**

$$

\pi_\theta(a\mid s)

$$

$\theta$는 네트워크 가중치（또는 LoRA）.  
학습 = $\theta$를 바꿔 좋은 행동의 확률을 키우는 것.

### 3.2 LLM Policy의 정체

디코더 LM의 한 스텝:

$$

\mathbf{h}_t = \mathrm{LM}_\theta(x,y_{<t}),\quad
\mathbf{z}_t = W\mathbf{h}_t,\quad
\pi_\theta(a\mid s_t)=\mathrm{softmax}(\mathbf{z}_t)_a

$$

여기서 $s_t=(x,y_{<t})$, $a\in\mathcal{V}$이다.

즉 **3권까지 쓰던 next-token 분포가 곧 policy**다.  
새 물리가 생긴 것이 아니라, **부르는 이름과 목적함수**가 RL 쪽으로 바뀐다.

| 이름 | 같은 물건 |
|---|---|
| Causal LM 분포 | $\pi_\theta(y_t\mid x,y_{<t})$ |
| SFT가 키우는 것 | 정답 토큰의 $\log\pi$ |
| RLHF가 키우는 것 | 보상 높은 궤적의 $\pi$ |

### 3.3 Value Function $V^\pi(s)$

**State-value function(상태 가치 함수)** $V^\pi(s)$는  
정책 $\pi$를 따를 때 상태 $s$에서 시작하는 **기대 반환**이다.

$$

V^\pi(s) = \mathbb{E}_\pi\big[G_t \mid S_t=s\big]
=
\mathbb{E}_\pi\Big[\sum_{k\ge0}\gamma^k r_{t+k}\,\Big|\,S_t=s\Big]

$$

직관: “지금 이 상황에서, 앞으로 내 정책대로 살면 **평균적으로 몇 점**?”

- 높은 $V(s)$: 이미 유리한 상태（쉬운 문제, 좋은 partial）
- 낮은 $V(s)$: 불리한 상태

LLM에서 terminal reward만 있으면, 응답 초반의 $V(s_0)$는 “이 프롬프트에서 내 정책의 기대 RM 점수”에 가깝다.

### 3.4 Action-Value $Q^\pi(s,a)$

**Action-value function(행동 가치, Q함수)**는  
상태 $s$에서 **일단 행동 $a$를 한 뒤**, 이후는 $\pi$를 따를 때의 기대 반환이다.

$$

Q^\pi(s,a)
=
\mathbb{E}_\pi\big[G_t \mid S_t=s, A_t=a\big]

$$

직관: “지금 **이 토큰**을 쓰면 앞으로 평균 몇 점?”

관계:

$$

V^\pi(s) = \sum_a \pi(a\mid s)\, Q^\pi(s,a)
=
\mathbb{E}_{a\sim\pi(\cdot\mid s)}\big[Q^\pi(s,a)\big]

$$

즉 $V$는 $Q$를 정책으로 평균한 값이다.

### 3.5 Advantage 맛보기（제83강 예고）

$$

A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)

$$

“평균（$V$）보다 이 행동이 얼마나 낫/나쁘냐”.  
Policy Gradient는 종종 $Q$ 대신 $A$를 쓴다.

### 3.6 최적 정책 · 최적 가치（존재만）

$$

V^*(s)=\max_\pi V^\pi(s),\quad
\pi^*\in\arg\max_\pi V^\pi(s)

$$

실무 LLM RLHF는 “전 우주 최적 $\pi^*$”를 보장하지 않는다.  
참조 정책 근처에서 보상을 올리는 **지역적 개선**에 가깝다.

## 4. 직관적으로 이해하기

### 4.1 식당 메뉴

- State: 배고픔·예산·날씨
- Policy: 메뉴판에서 확률적으로 주문
- $Q$(상태, “라면”): 라면 시킨 뒤의 평균 만족
- $V$(상태): 평소 내 주문 습관의 평균 만족
- Advantage: 라면이 평소보다 얼마나 나은지

LLM: “평소 내 next-token 습관”이 $\pi$이고,  
RM이 매긴 점수의 기대가 $V$에 가깝다.

### 4.2 같은 상태, 다른 정책

프롬프트 “농담 해줘”.

| 정책 | 행동 경향 | $V(s_0)$（상상） |
|---|---|---|
| $\pi_{\mathrm{SFT}}$ | 안전한 아재 개그 | 0.4 |
| $\pi_{\mathrm{RLHF}}$ | 더 웃기지만 공격적이지 않게 | 0.7 |
| $\pi_{\mathrm{bad}}$ | 혐오 발언 | -2.0 （안전 RM） |

가치는 **정책에 종속**이다. $V^\pi$의 위첨표를 생략해도 머릿속에는 남겨 둔다.

### 4.3 Q가 토큰 선택에 주는 그림

부분 문장: `수도는 `

| $a$ | 직관적 $Q$ |
|---|---|
| `서울` | 높음（정답 과제） |
| `부산` | 낮음 |
| `사과` | 매우 낮음 |

정책 학습은 높은 $Q$ 쪽으로 $\pi(a\mid s)$를 민다.  
다만 진짜 $Q$ 테이블을 $|\mathcal{S}|\times|\mathcal{V}|$로 만들 수 없으므로,  
신경망·샘플 추정·Advantage로 우회한다.

## 5. 수학적으로 이해하기

### 5.1 벨만 기대 방정식（읽기용）

$$

Q^\pi(s,a)
=
\mathbb{E}\big[r_t + \gamma V^\pi(s_{t+1}) \mid s_t=s,a_t=a\big]

$$

$$

V^\pi(s)
=
\mathbb{E}_{a\sim\pi}\big[Q^\pi(s,a)\big]

$$

LLM terminal reward·결정적 전이에서는 단순해진다.  
길이 $T$, 끝 보상 $R$, $\gamma=1$, 중간 $r=0$이면:

$$

Q(s_t,a_t) = \mathbb{E}[R\mid s_t,a_t],\quad
V(s_t)=\mathbb{E}[R\mid s_t]

$$

（이후 정책으로 완성한다는 조건 하에）

### 5.2 로그 정책과 학습

나중에 쓸 항등:

$$

\nabla_\theta \log\pi_\theta(a\mid s)
=
\frac{\nabla_\theta \pi_\theta(a\mid s)}{\pi_\theta(a\mid s)}

$$

Softmax 정책이면 이 기울기가 LM 학습의 기본 블록이다.  
SFT는 정답 $a^*$에 대해 $\log\pi$를 키우고,  
REINFORCE는 **샘플된** $a$의 $\log\pi$에 반환을 곱한다（제82강）.

### 5.3 Value의 근사

대형 상태 공간을 위해:

$$

V_\psi(s) \approx V^\pi(s)

$$

$\psi$는 value head 파라미터.  
PPO는 롤아웃 반환（또는 GAE）과 $V_\psi$의 오차를 줄이는 **value loss**를 같이 최소화한다.

$$

L_V(\psi) = \mathbb{E}_t\big[\big(V_\psi(s_t) - \hat{G}_t\big)^2\big]

$$

## 6. 작은 숫자로 직접 계산하기

어휘 $\{0,1\}$, 에피소드 길이 1（한 번 행동하고 보상）.

보상: $r(0)=0$, $r(1)=1$.  
상태 $s$ 하나뿐.

정책: $\pi(1\mid s)=p$, $\pi(0\mid s)=1-p$.

$$

\begin{aligned}
Q(s,1) &= 1,\quad Q(s,0)=0\\
V(s) &= p\cdot 1 + (1-p)\cdot 0 = p\\
A(s,1) &= 1-p,\quad A(s,0)=0-p=-p
\end{aligned}

$$

예: $p=0.3$이면 $V=0.3$, $A(s,1)=0.7$, $A(s,0)=-0.3$.

해석: 지금도 $1$을 고르면 평균보다 훨씬 이득 → 확률을 올려야 한다.

두 상태 미니 표（프롬프트 난이도）:

| $s$ | $\pi(정답)$ | $V(s)$ （정답=1, 오답=0） |
|---|---|---|
| 쉬운 문제 | 0.9 | 0.9 |
| 어려운 문제 | 0.2 | 0.2 |

같은 보상 스케일이어도 **상태 가치**가 다르다.  
Advantage는 이 차이를 빼서 “원래 쉬운데 잘한 것”과 “어려운데 잘한 것”을 공정히 비교하려는 방향으로 간다.

## 7. 코드로 구현하기

테이블형 미니 정책·가치.

```python
# tiny_policy_value.py
"""표 형태의 π, V, Q, A 계산."""

from __future__ import annotations

from typing import Dict, Tuple

Action = str
State = str

def expected_V(pi: Dict[Action, float], Q: Dict[Action, float]) -> float:
    return sum(pi[a] * Q[a] for a in pi)

def advantages(pi: Dict[Action, float], Q: Dict[Action, float]) -> Dict[Action, float]:
    v = expected_V(pi, Q)
    return {a: Q[a] - v for a in Q}

def softmax_policy(logits: Dict[Action, float]) -> Dict[Action, float]:
    import math
    m = max(logits.values())
    exps = {a: math.exp(z - m) for a, z in logits.items()}
    z = sum(exps.values())
    return {a: exps[a] / z for a in exps}

if __name__ == "__main__":
    # 한 상태 bandit
    Q = {"yes": 1.0, "no": 0.0}
    logits = {"yes": 0.0, "no": 0.0}  # 균등
    pi = softmax_policy(logits)
    print("pi", pi)
    print("V", expected_V(pi, Q))
    print("A", advantages(pi, Q))

    # 로짓을 올리면 정책이 바뀌고 V가 오른다
    logits2 = {"yes": 2.0, "no": 0.0}
    pi2 = softmax_policy(logits2)
    print("pi2", pi2, "V2", expected_V(pi2, Q))
```

## 8. PyTorch로 Policy · Value head 스케치

```python
# policy_value_heads.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyPV(nn.Module):
    """장난감: 상태 임베딩 → policy logits + value."""

    def __init__(self, n_states: int, n_actions: int, d: int = 16):
        super().__init__()
        self.emb = nn.Embedding(n_states, d)
        self.policy = nn.Linear(d, n_actions)
        self.value = nn.Linear(d, 1)

    def forward(self, state_idx: torch.Tensor):
        h = self.emb(state_idx)
        logits = self.policy(h)
        v = self.value(h).squeeze(-1)
        return logits, v

def policy_logp_and_entropy(logits: torch.Tensor, actions: torch.Tensor):
    logp_all = F.log_softmax(logits, dim=-1)
    logp = logp_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    ent = -(logp_all.exp() * logp_all).sum(-1)
    return logp, ent

if __name__ == "__main__":
    torch.manual_seed(0)
    net = TinyPV(n_states=5, n_actions=3)
    s = torch.tensor([1, 3, 4])
    logits, v = net(s)
    a = torch.tensor([0, 2, 1])
    logp, ent = policy_logp_and_entropy(logits, a)
    print("V", v.detach())
    print("logp", logp.detach())
    print("entropy", ent.detach())
```

실제 PPO는 여기에 GAE·clip·KL이 붙는다.  
오늘은 **머리가 둘**이라는 구조만 각인한다.

DPO 경로（제90강）는 별도 value head 없이 정책만 업데이트하는 경우가 많다.  
지도（제79강）에서 갈라진 이유이기도 하다.

## 9. 실제 LLM에서는 어떻게 사용하는가

### 9.1 Policy = LM head

Hugging Face `AutoModelForCausalLM`의 `lm_head`가 $\pi$다.  
RLHF 후에도 같은 헤드를 쓰며, 로짓 스케일·temperature는 **생성 시** 탐색용이다.

### 9.2 Value head

PPO 구현（TRL·OpenRLHF 등）은 종종:

- 마지막 히든 위에 선형층 1개 → 토큰（또는 시퀀스） value
- 또는 reward model과 별도

학습 신호:

- policy: Advantage 가중 로그확률（+ KL）
- value: 회귀

### 9.3 $Q$를 직접 안 두는 이유

$|\mathcal{V}|$가 커서 $Q(s,\cdot)$ 전체를 매 스텝 출력하기 부담된다.  
대신:

$$

\hat{A}_t \approx \hat{Q}_t - V(s_t)

$$

형태의 **스칼라 Advantage**만 정책 그라디언트에 곱한다.

### 9.4 SFT 정책과의 관계

$$

\pi_{\mathrm{ref}} = \pi_{\mathrm{SFT}}

$$

로 두고 KL$(\pi_\theta\|\pi_{\mathrm{ref}})$를 건다（제89강）.  
Value는 “참조보다 얼마나 좋은가”가 아니라 “현재 정책의 기대 반환”을 추적한다.  
둘을 혼동하지 말 것.

### 9.5 Alignment에서의 역할 요약

| 함수 | Alignment에서 |
|---|---|
| $\pi$ | 실제로 사용자에게 답하는 분포 |
| $r$ / RM | 선호 근사 신호 |
| $V$ | PPO 안정화·baseline |
| $Q$/$A$ | “어느 토큰을 강화할지”의 방향 |

## 10. 실습

### 실습 A

$p=\pi(1\mid s)=0.8$, $Q(s,1)=1$, $Q(s,0)=0$일 때 $V,A(s,1),A(s,0)$을 구하시오.

### 실습 B

제80강 terminal reward 설정에서, 응답이 끝나기 전 중간 상태의 $V(s_t)$가 의미하는 바를 한 문장으로.

### 실습 C

`TinyPV`에서 value head를 제거하고도 SFT·DPO가 가능한 이유를 제79강 경로와 연결해 설명하시오.

### 실습 D

다음 중 올바른 것을 고르시오.  
(a) $V^\pi$는 정책과 무관하다  
(b) $V^\pi$는 정책 $\pi$에 의존한다

## 11. 자주 하는 실수

1. **Policy와 Value를 같은 출력이라 생각**  
   → 하나는 분포, 하나는 스칼라（또는 $|\mathcal{A}|$차원 Q）.

2. **$V(s)$를 보상 $r$와 동일시**  
   → $V$는 **기대 누적**. sparse이면 미래 $R$의 조건부 기댓값.

3. **$Q$테이블을 LLM에 꼭 필요하다고 믿음**  
   → 실무는 $V$+샘플로 Advantage.

4. **$\pi_{\mathrm{SFT}}$를 value라고 부름**  
   → 참조 정책이지 가치 함수가 아니다.

5. **결정적 argmax 정책만 떠올림**  
   → 학습·탐색에는 softmax 샘플링이 핵심이다.

6. **높은 entropy = 항상 좋음**  
   → 탐색에는 도움, 과도하면 보상·안전이 무너질 수 있다.

## 12. 핵심 정리

- Policy $\pi(a\mid s)$는 상태에서의 행동 분포이고, LLM에서는 next-token softmax다.
- $V^\pi(s)$는 정책 하 기대 반환, $Q^\pi(s,a)$는 특정 행동을 강제했을 때의 기대 반환이다.
- $V=\mathbb{E}_{a\sim\pi}[Q]$, Advantage는 $Q-V$로 제83강에서 본격 사용한다.
- PPO류는 policy head + value head, DPO류는 주로 policy만.
- 다음 강의는 $\pi$의 파라미터를 **보상 방향으로** 미는 Policy Gradient다.

## 13. 핵심 용어

| 용어 | 한 줄 의미 |
|---|---|
| Policy $\pi(a\mid s)$ | 상태 조건 행동 분포 |
| Stochastic policy | 확률적으로 행동 샘플 |
| $V^\pi(s)$ | 상태 가치（기대 반환） |
| $Q^\pi(s,a)$ | 행동 가치 |
| Value head | $V$를 근사하는 출력층 |
| Softmax policy | 로짓→확률 정책 |
| $\pi_{\mathrm{ref}}$ | KL용 참조 정책（보통 SFT） |
| Entropy | 정책 불확실성·탐색 지표 |

## 14. 연습 문제
### 문제 1

LLM에서 $\pi_\theta(a\mid s_t)$를 수식으로 쓰시오（softmax·로짓）.

### 문제 2

$V^\pi(s)$와 $Q^\pi(s,a)$의 정의를 기댓값으로 쓰시오.

### 문제 3

$V(s)=\sum_a\pi(a\mid s)Q(s,a)$가 뜻하는 바를 한 줄로.

### 문제 4

PPO에 value head가 필요한 이유（분산/baseline）를 예고 수준으로 쓰시오.

### 문제 5

제80강 → 제81강 → 제82강 연결 문장을 완성하시오.

```text
(s,a,r) → (π, V, Q) → (    Gradient)
```

---

## 정답 및 해설

### 문제 1

$\pi_\theta(a\mid s_t)=\mathrm{softmax}(W\,h_\theta(s_t))_a$（표기는 동등하면 인정）.

### 문제 2

$V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s]$,  
$Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a]$.

### 문제 3

상태에서 정책이 고를 행동에 대해 Q를 평균하면 상태 가치가 된다.

### 문제 4

반환/보상의 그라디언트 분산을 줄이기 위해 baseline으로 $V$를 빼 Advantage를 만들기 위함（제83강）.

### 문제 5

`Policy`.

## 15. 다음 강의와 연결

정책과 가치를 이름으로 불렀다.  
다음 **제82강. Policy Gradient**에서는 목적 $J(\theta)=\mathbb{E}[R]$를 $\theta$로 미분해,

$$

\nabla_\theta J \approx \mathbb{E}\big[\nabla_\theta\log\pi_\theta(a\mid s)\, G\big]

$$

형태（REINFORCE）를 직관·수식·작은 숫자·코드로 고정한다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [제80강. 강화학습 기초 — State, Action, Reward](80강_강화학습_기초_State_Action_Reward.md)
- **다음 강:** [제82강. Policy Gradient](82강_Policy_Gradient.md)

<!-- /LECTURE_NAV -->
