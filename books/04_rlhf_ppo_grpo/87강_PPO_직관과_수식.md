# 87강. PPO 직관과 수식
## 이번 강에서 배우는 내용

- Surrogate objective와 확률비 $r_t(\theta)=\pi_\theta/\pi_{\mathrm{old}}$의 의미
- Clipping이 过大 업데이트를 어떻게 자르는지
- Advantage $A_t$가 PPO 손실에서 하는 일
- GAE(Generalized Advantage Estimation)의 고수준 위치
- 작은 숫자로 clip 전후 손실을 직접 계산하기

## 왜 중요한가?
순수 Policy Gradient(제82강)는 이론적으로 가능하지만, 스텝이 크면 정책이 한순간에 망가질 수 있다.

```text
한 번의 큰 업데이트
→ π가 이상해짐
→ 다음 롤아웃 품질↓
→ 추정 Advantage도 엉망
→ 회복 어려움
```

PPO의 핵심 아이디어는 단순하다.

> **이전 정책에서 너무 멀리 가지 마라. 가깝게, 여러 번, 안정적으로.**

LLM RLHF에서 PPO가 자주 거론되는 이유는 “만능”이라서가 아니라, **구현 가능성과 안정성 사이의 실무 타협**으로 널리 쓰여 왔기 때문이다.  
**사실:** 최근에는 DPO·GRPO 등 대안도 활발하다(제90·92강).  
**설명:** 이 강의는 PPO 자체를 이해시키는 것이 목적이지, “유일한 SOTA 알고리즘”을 주장하지 않는다.

## 선수 개념
- Policy Gradient: $\nabla J \approx \mathbb{E}[\nabla\log\pi\cdot A]$ (제82강)
- Advantage $A$ (제83강)
- On-policy 롤아웃 (제86강)
- Importance sampling 비율의 직관: 옛 분포 샘플로 새 분포 기댓 보정
- Clip / hinge: 값을 구간 안으로 자르기

아직 깊게 들어가지 않는 것:

- 완전한 LLM 엔지니어링(패딩, pack, KV) — 제88강에서 단순화본
- KL 항의 다양한 추정식 비교 — 제89강
- GRPO의 그룹 상대 베이스라인 — 제92강

## 핵심 개념
### 3.1 PPO란?

**PPO**는 정책 업데이트 폭을 제한하면서 surrogate 목표를 최적화하는 on-policy 알고리즘 계열이다. LLM 맥락에서 흔히 말하는 PPO는 **clipped surrogate objective**를 가리킨다.

구성요소:

1. 롤아웃으로 $\pi_{\mathrm{old}}$ 샘플·로그확률·보상 수집
2. Advantage 추정
3. clip이 들어간 정책 손실로 $\theta$ 업데이트 (여러 epoch)
4. (선택) Value 손실 + Entropy 보너스

### 3.2 확률비 (Ratio)

토큰(또는 시점) $t$에서:

\[
\rho_t(\theta)
=
\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
\]

문헌에서 $r_t(\theta)$로도 쓴다. 이 책은 보상 $r$와 헷갈리지 않게 **$\rho_t$** 를 기본으로 쓰고, 필요 시 “ratio”라고 부른다.

해석:

| $\rho_t$ | 의미 |
|---|---|
| $=1$ | 새 정책이 old와 동일 확률 |
| $>1$ | 새 정책이 그 행동을 더 좋아함 |
| $<1$ | 새 정책이 그 행동을 덜 좋아함 |

LLM에서는 $a_t=y_t$, $s_t=(x,y_{<t})$.

### 3.3 Surrogate objective (클립 전)

Advantage가 있을 때, (비클립) surrogate:

\[
L^{\mathrm{CPI}}(\theta)
=
\mathbb{E}_t\big[\rho_t(\theta)\,A_t\big]
\]

직관:

- $A_t>0$ (평균보다 좋았음) → $\rho$를 키워 그 행동을 더 자주
- $A_t<0$ → $\rho$를 줄여 그 행동을 덜

이는 importance-weighted policy gradient와 같은 계열의 목표이다.

문제: $\rho$가 폭주하면 한 스텝에 정책이 멀리 간다.

### 3.4 Clipped surrogate

PPO clip:

\[
L^{\mathrm{CLIP}}(\theta)
=
\mathbb{E}_t
\Big[
\min\big(
\rho_t(\theta)\,A_t,\;
\mathrm{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\,A_t
\big)
\Big]
\]

$\epsilon$은 보통 작은 양수(예: 0.1~0.2 개념). **구체 최적값은 과제 의존**이며 여기서 절대 추천치를 진리처럼 고정하지 않는다.

$\mathrm{clip}(\rho,1-\epsilon,1+\epsilon)$는 $\rho$를 $[1-\epsilon,1+\epsilon]$로 자른다.

### 3.5 왜 min인가?

목표는 **최대화**한다. $\min$은 “낙관적 증가를 불신”하는 장치다.

사례 분석:

**Case A — $A>0$** (좋은 행동)

- $\rho A$를 키우고 싶음
- 하지만 $\rho>1+\epsilon$로 과하게 키워도, clip된 항은 $(1+\epsilon)A$에서 멈춤
- $\min$이 더 작은 쪽을 고르므로 **과대 증가의 이득을 인정하지 않음**

**Case B — $A<0$** (나쁜 행동)

- $\rho A$를 작게(더 음수) 만들고 싶지 않음… 아니, 나쁜 행동의 확률을 줄이면 $\rho<1$, $A<0$이면 $\rho A$는 양수로 커질 수 있음
- clip이 $\rho$가 너무 작아지는 쪽의 과도한 이득도 제한
- 결과적으로 **한 방향으로의 극단 업데이트 억제**

한 줄:

> Clip은 “좋아 보이는 방향으로 한없이 밀지 못하게” 하는 trust-region 근사다.

### 3.6 전체 손실(실무 형태)

최대화 $L^{\mathrm{CLIP}}$ 대신 최소화로 구현할 때가 많다.

\[
\mathcal{L}_{\mathrm{policy}}
=
-\mathbb{E}_t\big[
\min(\rho_t A_t,\;\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t)
\big]
\]

가치 함수:

\[
\mathcal{L}_{V}
=
\mathbb{E}_t\big[(V_\psi(s_t)-\hat{R}_t)^2\big]
\]

엔트로피 보너스(탐험):

\[
\mathcal{L}_H
=
-\mathbb{E}_t\big[\mathcal{H}(\pi_\theta(\cdot\mid s_t))\big]
\]

총합(기호는 구현마다 계수 이름만 다름):

\[
\mathcal{L}
=
\mathcal{L}_{\mathrm{policy}}
+c_v\mathcal{L}_V
+c_H\mathcal{L}_H
\]

LLM RLHF에서는 여기에 **KL reward/penalty**가 보상 쪽에 들어가거나 손실에 추가된다(제86·89강).

### 3.7 Advantage와 GAE (고수준)

제83강 Advantage:

\[
A_t = Q(s_t,a_t) - V(s_t)
\]

실제로는 $Q$를 모르고 샘플 return으로 추정한다.  
**GAE(Generalized Advantage Estimation)** 는 TD 잔차의 지수 가중 합으로, bias-variance를 $\lambda$로 조절한다.

\[
\delta_t = R_t + \gamma V(s_{t+1}) - V(s_t)
\]

\[
\hat{A}_t^{\mathrm{GAE}(\gamma,\lambda)}
=
\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}
\]

LLM 응답 단위 보상에서는 단순화 버전이 흔하다.

```text
많은 LLM RLHF 구현:
- 응답 끝에 outcome reward
- 토큰마다 같은 Advantage를 쓰거나
- 단순 baseline (예: 배치 평균)만 사용
```

이 강의에서는 GAE의 **존재 이유**(분산↓, 신용할당)만 붙잡고, 완전한 시계열 유도는 생략해도 된다. 제88강 토이 코드는 더 단순한 Advantage를 쓴다.

### 3.8 Old policy와 여러 epoch

PPO는 한 번 모은 롤아웃으로 **여러 gradient step**을 수행한다.  
단, $\rho$가 clip 범위를 너무 자주 벗어나면 그 배치의 효용이 끝난다.

```text
rollout (π_old)
  └─ for epoch in 1..K:
        for minibatch:
           update θ
```

$\pi_{\mathrm{old}}$의 logprob는 롤아웃 때 저장해 두고 고정한다.

### 3.9 ε의 직관

| $\epsilon$ | 효과 |
|---|---|
| 너무 작음 | 업데이트 거의 없음 (과보수) |
| 너무 큼 | clip이 헐거워져 붕괴 위험↑ |

ε는 “한 업데이트에서 확률비가 허용하는 **상대 변화 폭**”이다. 절대 학습률과 별개 축이다.

## 직관적으로 이해하기
등산 로프:

```text
Policy gradient = 급하게 정상으로 뛰어가기
PPO clip = 로프 길이 제한
로프보다 멀리 가려는 스텝은 인정하지 않음
```

또는 편집 거리:

```text
π_old 초고
π_θ   수정본
ratio = 각 토큰 선택 확률의 상대 변화
clip = "한 문단에서 너무 많이 고치지 말 것"
```

좋은 문장(A>0)은 조금 더 자주, 나쁜 문장(A<0)은 조금 덜.  
“조금”을 강제하는 장치가 clip이다.

## 수학적으로 이해하기
### 5.1 비클립에서 클립으로

비클립 목표 $L=\mathbb{E}[\rho A]$의 $\theta$ 경사는, $\rho=\pi_\theta/\pi_{\mathrm{old}}$이므로 $\log\pi_\theta$ 경사에 $\rho A$가 가중된 형태가 된다.  
$\pi_{\mathrm{old}}$ 샘플을 재사용할수록 $\rho$가 1에서 멀어져 추정이 나빠져 **신뢰 구간**이 필요하다. Clip은 그 soft 제약이다.

### 5.2 구간별 동작 ($A>0$)

$\rho$에 대해 $f(\rho)=\min(\rho A,\;\mathrm{clip}(\rho)A)$, $A>0$.

- $\rho < 1+\epsilon$: 보통 $\rho A$가 유효 (증가 장려)
- $\rho > 1+\epsilon$: $\min$이 $(1+\epsilon)A$를 선택 → $\rho$를 더 키워도 목표값 고정 → 그 샘플에 대한 추가 이득 0

### 5.3 구간별 동작 ($A<0$)

$A<0$이면 부등식 방향이 뒤집혀, $\rho$가 $1-\epsilon$보다 너무 작아지는 쪽의 추가 이득이 잘린다.

### 5.4 Value clip (참고)

정책뿐 아니라 value에도 clip을 거는 변형이 있다. 필수는 아니며, 제88강 토이 구현에서는 단순 MSE만 쓸 수 있다.

### 5.5 LLM 시퀀스로의 확장

응답 $y=(y_1,\ldots,y_T)$에 대해 토큰 평균:

\[
L^{\mathrm{CLIP}}
=
\mathbb{E}_{x,y}
\frac{1}{T}
\sum_{t=1}^{T}
\min\big(\rho_t A_t,\;\mathrm{clip}(\rho_t)A_t\big)
\]

Outcome reward만 있으면 $A_t=\hat{A}(x,y)$로 토큰에 방송(broadcast)하는 단순화가 흔하다.

## 작은 숫자로 직접 계산하기
설정: $\epsilon=0.2$, 따라서 clip 구간 $[0.8, 1.2]$.

### 예제 1 — 좋은 행동, 과한 비율

$A=+2.0$, $\rho=1.5$ (이미 50% 증가)

\[
\rho A=1.5\times 2=3.0
\]

\[
\mathrm{clip}(\rho)A=1.2\times 2=2.4
\]

\[
\min(3.0, 2.4)=2.4
\]

비클립이면 3.0을 목표에 반영하지만, PPO는 2.4까지만 인정.

### 예제 2 — 좋은 행동, 온건한 비율

$A=+2.0$, $\rho=1.1$

\[
\rho A=2.2,\quad \mathrm{clip}A=2.2,\quad \min=2.2
\]

clip 미발동.

### 예제 3 — 나쁜 행동

$A=-1.0$, $\rho=0.5$

\[
\rho A=-0.5
\]

\[
\mathrm{clip}(\rho)A=0.8\times(-1)=-0.8
\]

\[
\min(-0.5,-0.8)=-0.8
\]

최대화 관점에서 $\min$이 더 비관적(낮은) 값을 선택 → 과도한 확률 감소로 얻는 이득을 제한하는 쪽으로 동작한다.

### 예제 4 — 배치 평균

세 토큰:

| t | $\rho$ | $A$ | unclipped $\rho A$ | clipped term | min |
|---|---|---|---|---|---|
| 1 | 1.1 | +1.0 | 1.10 | 1.10 | 1.10 |
| 2 | 1.5 | +1.0 | 1.50 | 1.20 | 1.20 |
| 3 | 0.7 | -1.0 | -0.70 | -0.80 | -0.80 |

평균 surrogate $=(1.10+1.20-0.80)/3=0.50$  
비클립 평균 $=(1.10+1.50-0.70)/3=0.633$

Clip이 목표값을 낮춰 **과신을 제거**했다.

손실로 바꿀 때 $\mathcal{L}=-0.50$ (최소화 프레임).

## 코드로 구현하기 — NumPy clip 손실
```python
import numpy as np

def ppo_clip_surrogate(ratio, adv, eps=0.2):
    """Maximize this (or minimize the negative)."""
    unclipped = ratio * adv
    clipped = np.clip(ratio, 1.0 - eps, 1.0 + eps) * adv
    return np.minimum(unclipped, clipped)

def ppo_policy_loss(ratio, adv, eps=0.2):
    return -ppo_clip_surrogate(ratio, adv, eps).mean()

ratio = np.array([1.1, 1.5, 0.7])
adv = np.array([1.0, 1.0, -1.0])
print(ppo_clip_surrogate(ratio, adv))
print(ppo_policy_loss(ratio, adv))
```

기대 surrogate: `[1.1, 1.2, -0.8]`, loss: `-0.5`의 부호 반대 → `+0.5`가 아니라 평균의 음수이므로 `-0.5`.

## PyTorch로 구현하기 — 손실 함수
```python
import torch

def ppo_clip_loss(logp, logp_old, adv, eps=0.2):
    """
    logp, logp_old, adv: [B, T] or [N]
    returns scalar loss (minimize)
    """
    ratio = torch.exp(logp - logp_old)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * adv
    surrogate = torch.min(unclipped, clipped)
    return -surrogate.mean()

def value_loss(values, returns):
    return torch.mean((values - returns) ** 2)

def entropy_from_logits(logits):
    # logits: [..., V]
    p = torch.softmax(logits, dim=-1)
    logp = torch.log_softmax(logits, dim=-1)
    return -(p * logp).sum(dim=-1).mean()
```

비율 계산 안정성:

```text
ratio = exp(logp - logp_old)   # 권장
ratio = exp(logp)/exp(logp_old) # 비추천 (오버플로)
```

## 수식 보강 — PPO clip 한 줄

$$
L^{\mathrm{CLIP}}=\mathbb{E}\big[\min(r_t A_t,\ \mathrm{clip}(r_t,1-\varepsilon,1+\varepsilon)A_t)\big]
$$

$r_t=\pi_\theta/\pi_{\mathrm{old}}$입니다.

## LLM에서는 어디에 사용될까?
RLHF-PPO 스택에서의 위치:

```text
rollout tokens + logp_old + rewards
        │
        ▼
  Advantage / GAE (단순 또는 정교)
        │
        ▼
  L_CLIP (+ L_V + entropy)  ← 이번 강의
        │
        ▼
  optimizer step on π_θ (and V)
```

실무 관찰 지표:

- `approx_kl` ≈ $\mathbb{E}[\log\pi_\theta-\log\pi_{\mathrm{old}}]$
- `clip_fraction`: $\rho$가 clip 바깥인 비율
- `explained_variance` of value
- reward mean / length

`clip_fraction`이 항상 높으면 ε·lr·epoch가 공격적일 수 있다.

변형:

- PPO-clip + adaptive KL (구 OpenAI 스타일 논의)
- 토큰 레벨 vs 시퀀스 레벨 정규화
- reward whitening

제92강 GRPO는 critic 없이 그룹 내 상대 점수로 Advantage를 대체하는 흐름과 대비된다.

## 실습
### 실습 A — 손계산

$\epsilon=0.1$, $A=3$, $\rho=1.25$일 때 unclipped, clipped, min 값을 구하시오.

### 실습 B — 부호

$A=-2$, $\rho=1.3$, $\epsilon=0.2$에서 min 값을 구하고, “정책이 이 행동을 더 좋아하게 만드는 방향이 이득인지”를 문장으로 설명하시오.

### 실습 C — NumPy

제8절 코드를 실행하고, `eps=0.05`로 줄였을 때 surrogate 평균이 어떻게 변하는지 기록하시오.

### 실습 D — 개념 연결

다음을 한 줄 매핑하시오.

```text
제82강 Policy Gradient →
제83강 Advantage →
제87강 clip →
```

### 실습 E — 다이어그램

제86강 파이프라인 그림에 $L^{\mathrm{CLIP}}$ 상자를 어디에 그릴지 표시하시오.

## 자주 하는 실수
1. **ratio를 $\pi_{\mathrm{old}}/\pi_\theta$로 뒤집음**
2. **Advantage 정규화 잊고 스케일 폭주**
3. **clip 없이 여러 epoch** → 사실상 큰 off-policy 점프
4. **$A$의 방송 차원 불일치** (브로드캐스트 버그)
5. **최대화/최소화 부호 혼선** (`-min` 잊음)
6. **ε를 학습률과 동일시**
7. **GAE를 쓰면서 $\gamma,\lambda$를 보상 스케일과 무관하게 복붙**
8. **clip_fraction=0만 보고 성공 선언** (그냥 업데이트가 죽은 것일 수도)

## 핵심 요약
- PPO clipped objective는 $\rho A$와 clip된 $\rho A$의 최소를 취해 过大 업데이트를 막는다.
- $\rho=\pi_\theta/\pi_{\mathrm{old}}$는 정책 변화의 국소 측정이다.
- Advantage 부호가 “밀지/당길지”를 결정하고, clip이 “얼마나”를 제한한다.
- GAE는 Advantage 추정의 분산-편향 조절 도구(고수준).
- 구현은 제88강에서 토이 LM/밴딧으로 연결한다.

## 용어 사전
| 용어 | 의미 |
|---|---|
| PPO | Proximal Policy Optimization |
| Ratio $\rho$ | $\pi_\theta/\pi_{\mathrm{old}}$ |
| Surrogate objective | 직접 $J$ 대신 쓰는 대리 목표 |
| Clipping | 비율을 $[1-\epsilon,1+\epsilon]$로 제한 |
| $\epsilon$ | clip 폭 |
| Advantage | 평균 대비 초과 보상 |
| GAE | 일반화 Advantage 추정 |
| Clip fraction | clip이 활성화된 샘플 비율 |
| Entropy bonus | 탐험을 위한 엔트로피 항 |
| Old policy | 롤아웃 시점 정책 |

## 연습문제
### 문제 1（수식）

$\rho_t(\theta)$의 정의를 쓰시오.

### 문제 2（직관）

$A>0$이고 $\rho>1+\epsilon$일 때, clip이 막는 것은 무엇인가?

### 문제 3（계산）

$\epsilon=0.2$, $\rho=1.5$, $A=2$일 때 $L^{\mathrm{CLIP}}$의 샘플 항 $\min(\cdot)$ 값은?

### 문제 4（구현）

`exp(logp - logp_old)`를 쓰는 이유는?

### 문제 5（연결）

제88강 구현에서 $\pi_{\mathrm{old}}$의 logprob는 언제 저장해야 하는가?

### 문제 6（GAE）

GAE의 $\lambda$를 1에 가깝게 하면 보통 무엇이 커지는가? (편향/분산 관점, 고수준)

### 문제 7（부호）

최소화 프레임에서 policy loss 앞에 음수가 붙는 이유는?

---

## 정답 및 해설
### 문제 1

$\rho_t(\theta)=\pi_\theta(a_t\mid s_t)/\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)$.

### 문제 2

좋은 행동의 확률을 clip 허용 범위 너머로 **과하게 키워 얻는 대리 이득**을 막는다.

### 문제 3

$\min(3.0, 1.2\times2)=\min(3.0,2.4)=2.4$.

### 문제 4

로그 공간에서 빼면 수치가 안정적이고 오버플로를 줄인다.

### 문제 5

롤아웃(생성) 시점, 즉 업데이트 전에 old 정책 분포로 저장.

### 문제 6

일반적으로 분산이 커지고(몬테카를로에 가까워지고) 편향은 줄어드는 쪽으로 간다. (고수준 진술)

### 문제 7

원래 surrogate는 최대화 대상이므로, 최소화 옵티마이저에 넣으려면 부호를 뒤집는다.

## 다음 강의와 연결
수식을 손으로 계산할 수 있게 되었다. 남은 것은 **루프로 돌리는 일**이다.

다음 **제88강. PPO 구현**에서는 토이 언어모델 또는 밴딧형 설정에서 단순화한 PPO 루프를 구현하고, 무엇이 full LLM PPO에서 빠졌는지 명시한다. 그다음 **제89강. KL Divergence의 역할**에서 RLHF 보상 안의 KL 닻을 본격적으로 다룬다.

> PPO는 “더 좋게”만이 아니라 “너무 멀리 가지 말고 더 좋게”를 수식으로 적은 알고리즘이다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [86강. RLHF 전체 구조](86강_RLHF_전체_구조.md)
- **다음 강:** [88강. PPO 구현](88강_PPO_구현.md)

<!-- /LECTURE_NAV -->
