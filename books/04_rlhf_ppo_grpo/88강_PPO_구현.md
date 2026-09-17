# 88강. PPO 구현
## 이번 강에서 배우는 내용

- 밴딧형·미니 LM형 설정에서 롤아웃 → Advantage → clip loss → 업데이트를 코딩하기
- `logp`, `logp_old`, `ratio`, `eps`가 코드 변수로 어떻게 대응되는지 보이기
- 단순화 목록을 명시하고, full LLM PPO와 구분하기
- 학습 로그(reward, clip fraction, approx KL)를 읽어 건강 상태를 판단하기
- 제89강 KL 항을 어디에 꽂을지 예고하기

## 왜 중요한가?
수식만 외우면 다음에서 무너진다.

```text
logprob 슬라이싱 한 칸 빗나감
→ ratio 폭주
→ clip_fraction=1
→ 정책 붕괴
```

구현으로 “비율이 1 근처에서 움직이는지”를 눈으로 봐야 한다. 또한 RLHF 전체(제86강)에서 PPO 상자가 실제로 어떤 텐서를 소비하는지 고정해야 제89·90강으로 넘어갈 수 있다.

## 선수 개념
- PPO clip 수식 (제87강)
- Advantage 기초 (제83강)
- RLHF 루프 (제86강)
- PyTorch `Categorical` / `log_softmax`
- 오토그라드: `logp`가 현재 그래프에 연결되어야 함

이번 구현에서 **의도적으로 빼는 것**은 §10·§12에서 목록화한다.

## 핵심 개념
### 3.1 두 가지 토이 설정

**설정 A — Contextual Bandit (추천)**

```text
상태 s (이산)
행동 a ∈ {0..A-1}
즉시 보상 R(s,a)
에피소드 길이 1
```

PPO 비율이 한 스텝이라 디버깅이 쉽다.

**설정 B — Tiny autoregressive LM**

```text
짧은 프롬프트 x
짧은 응답 y (고정 길이 T)
보상 = 장난감 규칙 (예: 특정 토큰 포함)
```

LLM RLHF에 한 걸음 더 가깝다.

이 강의는 A로 원리를 固め고, B로 시퀀스 확장을 보여 준다.

### 3.2 알고리즘 뼈대

```text
for iter = 1..N:
  # 1) Rollout with π_θ, store logp_old
  # 2) Compute rewards / advantages
  # 3) for epoch = 1..K:
  #      recompute logp with current π_θ
  #      PPO clip loss + (optional) value loss
  #      optimizer.step()
  # 4) Log metrics
```

핵심 불변조건:

1. `logp_old`는 롤아웃 때 고정
2. `logp`는 업데이트마다 재계산
3. Advantage는 (기본) 롤아웃 배치 기준으로 고정하거나, value 학습과 함께 재추정

### 3.3 Full LLM PPO와의 차이 (미리 보기)

| 항목 | 이번 토이 | Full LLM PPO |
|---|---|---|
| 모델 | MLP / tiny GPT | 수천~수백억 파라미터급 가능 |
| 보상 | 규칙/테이블 | RM + KL (제85·89강) |
| Advantage | 배치 평균 baseline | GAE / value head 등 |
| 생성 | 고정 길이·단순 샘플 | 온도·top-p·EOS·패딩 |
| 분산 | 단일 프로세스 | 다중 GPU 롤아웃 |
| 안정화 | 최소 | whitening, sched, LoRA… |

토이가 가르치는 것: **ratio·clip·update 계약**.  
토이가 가르치지 않는 것: 인프라·스케일·제품 정렬 품질.

## 직관적으로 이해하기
공장 CCTV:

```text
오늘 근무조 = π_old  (롤아웃)
작업 영상 저장 = (s,a,logp_old,R)
저녁 회의 = PPO epochs
규칙: 어제 매뉴얼 대비 너무 다른 지시 금지 (clip)
```

밴딧 버전은 “한 동작짜리 영상”, LM 버전은 “여러 토큰짜리 영상”이다. 편집 규칙은 같다.

## 수학적으로 이해하기 — 코드 대응
$$
\rho=\exp(\log\pi_\theta-\log\pi_{\mathrm{old}})
$$

```python
ratio = torch.exp(logp - logp_old)
```

$$
L=-\mathrm{mean}\min(\rho A,\mathrm{clip}(\rho)A)
$$

```python
loss = -torch.min(ratio * adv, clipped_ratio * adv).mean()
```

밴딧에서 $T=1$이므로 시퀀스 평균이 곧 샘플 손실이다.  
LM에서는 토큰 차원 mean을 한 번 더 취한다.

## 작은 숫자로 직접 계산하기
토이 밴딧 한 샘플:

```text
logp_old = -1.0
logp     = -0.5
A        = +1.5
eps      = 0.2
```

$$
\rho=e^{(-0.5)-(-1.0)}=e^{0.5}\approx 1.6487
$$

$$
\mathrm{clip}(\rho)=1.2
$$

$$
\min(1.6487\times1.5,\;1.2\times1.5)=\min(2.473,1.8)=1.8
$$

$$
\mathcal{L}=-1.8
$$

같은 상황에서 `eps=0.5`면 clip 상한 1.5,

$$
\min(2.473,1.5\times1.5)=\min(2.473,2.25)=2.25
$$

→ 더 큰 surrogate(덜 보수적).

## 코드로 구현하기 — NumPy 밴딧 1스텝
```python
import numpy as np

def softmax(logits):
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()

def ppo_bandit_step(logits, action, logp_old, adv, eps=0.2, lr=0.05):
    """logits: [A], one sample update (교육용)."""
    pi = softmax(logits)
    logp = np.log(pi[action] + 1e-12)
    ratio = np.exp(logp - logp_old)
    unclipped = ratio * adv
    clipped = np.clip(ratio, 1 - eps, 1 + eps) * adv
    surrogate = min(unclipped, clipped)
    # 정책 경사 근사: ∂logπ(a)/∂logits
    grad_logp = -pi
    grad_logp[action] += 1.0
    # maximize surrogate ≈ ascent on logp * d_surrogate/d_logp
    # 단순화: A>0이고 ratio in range면 logp 증가 방향
    d_surr_d_logp = 0.0
    if adv > 0 and ratio < 1 + eps:
        d_surr_d_logp = ratio * adv  # ρA w.r.t logp via ρ
    elif adv < 0 and ratio > 1 - eps:
        d_surr_d_logp = ratio * adv
    # 실제 autograd가 더 정확; 여기선 개념 스케치
    logits = logits + lr * d_surr_d_logp * grad_logp
    return logits, ratio, surrogate
```

NumPy 수동 미분은 오류 여지가 크므로, **본 구현은 PyTorch**로 간다.

## PyTorch로 구현하기
### 8.1 Contextual Bandit PPO

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class BanditPolicy(nn.Module):
    def __init__(self, n_states=5, n_actions=3, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Embedding(n_states, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, states):
        return self.net(states)  # logits [B, A]

def make_reward_table(n_states=5, n_actions=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    # 각 상태마다 하나의 최적 행동
    table = torch.zeros(n_states, n_actions)
    optima = torch.randint(0, n_actions, (n_states,), generator=g)
    for s in range(n_states):
        table[s, optima[s]] = 1.0
        # 나머지에 작은 노이즈 보상
        for a in range(n_actions):
            if a != optima[s].item():
                table[s, a] = 0.1 * torch.rand((), generator=g)
    return table, optima

@torch.no_grad()
def rollout_bandit(policy, reward_table, batch_size=64):
    n_states = reward_table.size(0)
    states = torch.randint(0, n_states, (batch_size,))
    logits = policy(states)
    dist = Categorical(logits=logits)
    actions = dist.sample()
    logp_old = dist.log_prob(actions)
    rewards = reward_table[states, actions]
    return states, actions, logp_old, rewards

def compute_advantages(rewards):
    # 단순 baseline: 배치 평균
    return rewards - rewards.mean()

def ppo_update_bandit(policy, optimizer, batch, eps=0.2, epochs=4):
    states, actions, logp_old, rewards = batch
    adv = compute_advantages(rewards)
    # 선택: adv 표준화
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    metrics = []
    for _ in range(epochs):
        logits = policy(states)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(actions)
        ratio = torch.exp(logp - logp_old.detach())
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
        loss = -torch.min(unclipped, clipped).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            approx_kl = (logp_old - logp).mean()
            clip_frac = (
                ((ratio < 1 - eps) | (ratio > 1 + eps)).float().mean()
            )
        metrics.append(
            {
                "loss": float(loss.item()),
                "reward_mean": float(rewards.mean().item()),
                "approx_kl": float(approx_kl.item()),
                "clip_frac": float(clip_frac.item()),
            }
        )
    return metrics

def train_bandit_ppo(iters=80, seed=0):
    torch.manual_seed(seed)
    reward_table, optima = make_reward_table()
    policy = BanditPolicy()
    opt = torch.optim.Adam(policy.parameters(), lr=1e-2)

    history = []
    for it in range(iters):
        batch = rollout_bandit(policy, reward_table)
        metrics = ppo_update_bandit(policy, opt, batch)
        # 탐욕 정확도
        with torch.no_grad():
            states = torch.arange(reward_table.size(0))
            greedy = policy(states).argmax(dim=-1)
            acc = (greedy == optima).float().mean().item()
        history.append({"iter": it, "acc": acc, **metrics[-1]})
    return history, policy, optima
```

건강 신호:

```text
acc ↑ (장난감 최적 행동 맞춤)
reward_mean ↑
approx_kl 작은 양수 근처
clip_frac 이 가끔 있으나 항상 1은 아님
```

### 8.2 Tiny LM PPO (시퀀스)

장난감 보상: 응답에 토큰 `1`이 많이 포함될수록 보상↑ (의미 없는 규칙 — 원리 검증용).

```python
class TinyLM(nn.Module):
    def __init__(self, vocab=11, d=32, n_layers=1):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.rnn = nn.GRU(d, d, n_layers, batch_first=True)
        self.head = nn.Linear(d, vocab)

    def forward(self, tokens):
        # tokens: [B, T]
        x = self.emb(tokens)
        h, _ = self.rnn(x)
        return self.head(h)  # [B, T, V]

@torch.no_grad()
def generate(policy, prompt_ids, resp_len=4):
    """prompt_ids: [B, P]. returns full ids, resp logp sum/stack."""
    B, P = prompt_ids.shape
    ids = prompt_ids
    logps = []
    for _ in range(resp_len):
        logits = policy(ids)[:, -1, :]
        dist = Categorical(logits=logits)
        nxt = dist.sample()
        logps.append(dist.log_prob(nxt))
        ids = torch.cat([ids, nxt.unsqueeze(1)], dim=1)
    logp_tokens = torch.stack(logps, dim=1)  # [B, resp_len]
    return ids, logp_tokens

def toy_reward(resp_tokens):
    # resp_tokens: [B, T]
    return (resp_tokens == 1).float().mean(dim=1)

def ppo_update_lm(policy, optimizer, prompts, resp_len=4, eps=0.2, epochs=3):
    with torch.no_grad():
        full_ids, logp_old = generate(policy, prompts, resp_len)
    resp = full_ids[:, -resp_len:]
    rewards = toy_reward(resp)
    adv = rewards - rewards.mean()
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    # 토큰에 방송
    adv_tokens = adv.unsqueeze(1).expand_as(logp_old)

    last = None
    for _ in range(epochs):
        # 재평가: 응답 구간의 logπ
        logits = policy(full_ids[:, :-1])  # next-token logits
        # 응답 토큰 위치: 마지막 resp_len개 예측
        resp_logits = logits[:, -resp_len:, :]
        logp_all = F.log_softmax(resp_logits, dim=-1)
        logp = logp_all.gather(2, resp.unsqueeze(-1)).squeeze(-1)

        ratio = torch.exp(logp - logp_old)
        unclipped = ratio * adv_tokens
        clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * adv_tokens
        loss = -torch.min(unclipped, clipped).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            approx_kl = (logp_old - logp).mean()
            clip_frac = ((ratio - 1.0).abs() > eps).float().mean()
        last = {
            "loss": float(loss.item()),
            "reward": float(rewards.mean().item()),
            "approx_kl": float(approx_kl.item()),
            "clip_frac": float(clip_frac.item()),
        }
    return last
```

주의: `generate` 안의 `logp_old`는 `no_grad`로 저장한다. 업데이트 루프의 `logp`만 그래프에 연결한다.

### 8.3 KL 페널티 자리(미리 꽂을 위치)

제89강 내용의 훅:

```python
def shaped_reward(rm_score, logp_theta, logp_ref, beta):
    # 시퀀스 합 KL 근사
    kl = (logp_theta - logp_ref).sum(dim=-1)
    return rm_score - beta * kl
```

토이 LM에서는 `π_ref`를 초기 스냅샷으로 freeze해 두고, `toy_reward` 대신 `shaped_reward`를 넣을 수 있다. 자세한 해석은 제89강.

### 8.4 로그 출력 예

```python
def demo():
    hist, _, _ = train_bandit_ppo(iters=50)
    for row in hist[::10]:
        print(
            f"iter={row['iter']:3d} acc={row['acc']:.2f} "
            f"R={row['reward_mean']:.3f} kl={row['approx_kl']:.4f} "
            f"clip={row['clip_frac']:.2f}"
        )
```

실행 결과는 시드·환경에 따라 달라진다. **특정 숫자를 SOTA처럼 인용하지 말 것.**

## 수식 보강 — PPO clip 목적

확률비 $r_t(\theta)=\pi_\theta(a_t\mid s_t)/\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)$에 대해

$$
L^{\mathrm{CLIP}}(\theta)=\mathbb{E}_t\left[\min\big(r_t(\theta)A_t,\ \mathrm{clip}(r_t(\theta),1-\varepsilon,1+\varepsilon)A_t\big)\right]
$$

$A_t$는 advantage, $\varepsilon$는 클립 폭입니다. 정책이 한 번에 너무 크게 바뀌지 않게 막는 장치입니다.

## 정량 스케치 — clip 메트릭

$$

L^{\mathrm{CLIP}}=\mathbb{E}_t\Big[\min\big(\rho_t\hat A_t,\ \mathrm{clip}(\rho_t,1-\varepsilon,1+\varepsilon)\hat A_t\big)\Big]
$$

$$

\rho_t=\exp(\log\pi_\theta-\log\pi_{\mathrm{old}})
$$

$$

\mathrm{clip\_frac}=\mathbb{E}[\mathbf{1}(|\rho-1|>\varepsilon)]
$$

예: $A=1$, $\varepsilon=0.2$, $\rho=1.8$ → $\min(1.8,1.2)=1.2$ (보수화).

$A=-1$, $\rho=0.5$ → $\min(-0.5,-0.8)=-0.8$ (과억제 스텝 절단).

Approx KL 감각: $\widehat{\mathrm{KL}}\approx\mathbb{E}[\log\pi_\theta-\log\pi_{\mathrm{old}}]$.

롤아웃 예산 $N\approx B\cdot L_{\mathrm{gen}}$, 재사용 epoch $K$↑면 clip_frac↑ 경향.


## LLM에서는 어디에 사용될까?
Full stack 대응표:

```text
토이 BanditPolicy.forward
   ↔ LLM logits = CausalLM(...)

rollout_bandit / generate
   ↔ vLLM/HF generate + token logprobs

toy_reward / reward_table
   ↔ Reward Model score (+ safety)

compute_advantages (mean baseline)
   ↔ GAE + Value head

ppo_update_*
   ↔ PPO trainer (DeepSpeed/FSDP, LoRA…)

(없음)
   ↔ reference KL, packing, EOS mask, whitening
```

실무 체크:

1. Prompt 토큰에 policy loss를 주지 않기 (응답 구간만)
2. Pad 토큰 mask
3. `logp_old`와 tokenizer 경계 일치
4. 보상 정규화
5. ref / RM freeze

제95강 프로젝트에서 preference/RL 실습으로 확장한다.

## 실습
### 실습 A — 밴딧 PPO 실행

`train_bandit_ppo`를 실행하고 `acc` 곡선을 기록하라.  
`eps=0.05`와 `eps=0.5`를 비교해 clip_frac 차이를 쓰라.

### 실습 B — 버그 주입

`logp_old.detach()`를 제거하거나, `ratio = exp(logp_old - logp)`로 뒤집어 증상을 관찰하라.

### 실습 C — Tiny LM

`toy_reward`를 “토큰 2의 비율”로 바꾸고, 평균 보상이 올라가는지 확인하라.

### 실습 D — 단순화 목록 작성

이 구현이 full LLM PPO 대비 빠진 항목 8개를 체크리스트로 쓰라.

### 실습 E — RLHF 연결

제86강 다이어그램의 `θ ← PPO update` 상자에, 이번 함수 이름(`ppo_update_lm`)을 기입하라.

## 자주 하는 실수
1. **old logprob를 매 epoch 재샘플**  
   on-policy 계약 파괴.

2. **생성 그래프에 autograd를 남김**  
   메모리·의미 모두 꼬임. 롤아웃은 no_grad가 기본.

3. **프롬프트 토큰까지 PPO loss**  
   질문 문장을 “행동”으로 학습.

4. **mask 없이 패딩 포함**  
   ratio 오염.

5. **Advantage 차원 방송 실수**  
   silent broadcasting bug.

6. **lr·epochs 과다 + eps 과대**  
   KL 폭주.

7. **토이 보상을 언어 품질로 해석**  
   규칙 해킹을 “지능”으로 오해.

8. **ref KL 없이 RM만 장시간**  
   hacking (제89강으로 이어짐).

## 단순화 vs Full LLM PPO (명시)
**포함한 것**

- ratio + clip loss
- rollout 시 logp_old 저장
- 다 epoch 재사용
- 기본 메트릭(kl, clip_frac, reward)

**뺀 것 / 단순화한 것**

- 거대한 Transformer·병렬 롤아웃
- 학습된 RM (규칙 보상으로 대체)
- GAE의 전체 재귀 구현
- Value clip, reward whitening
- Adaptive KL / β 스케줄
- EOS·padding·chat template
- 정책·value 분리 최적화 트릭
- 분산 학습·수치 정밀도

> 토이 성공 ≠ 챗봇 정렬 성공.  
> 토이 성공 = PPO 계약 이해.

## 핵심 요약
- PPO 구현의 심장은 `exp(logp-logp_old)`와 clip된 surrogate다.
- 롤아웃과 업데이트를 분리하고, old 로그확률을 고정한다.
- 밴딧으로 디버깅한 뒤 시퀀스 LM으로 확장한다.
- 메트릭으로 붕괴를 감시한다.
- Full LLM PPO는 이 골격 위에 RM·KL·인프라가 얹힌다.
- 다음은 KL Divergence의 역할(제89강)이다.

## 용어 사전
| 용어 | 의미 |
|---|---|
| Rollout | 현재 정책으로 궤적 수집 |
| `logp_old` | 롤아웃 시점 로그확률 |
| Clip fraction | 비율이 clip 밖인 비율 |
| Approx KL | $\mathbb{E}[\log\pi_{\mathrm{old}}-\log\pi_\theta]$ 근사 |
| Baseline | Advantage용 평균/가치 빼기 |
| Outcome reward | 시퀀스 끝 스칼라 보상 |
| Tiny LM | 교육용 소형 자기회귀 모델 |
| Grad clip | 그래디언트 노름 제한 |
| Freeze | 파라미터 고정 |
| Simplification | 교육용 생략 목록 |

## 연습문제
### 문제 1（코드）

`ratio = torch.exp(logp - logp_old.detach())`에서 `detach`가 old 쪽에 필요한 이유는?

### 문제 2（계산）

$\log\pi-\log\pi_{\mathrm{old}}=0$이면 $\rho$와 unclipped 항 $\rho A$는?

### 문제 3（구분）

토이 `toy_reward`와 제85강 RM의 공통점·차이점 한 가지씩.

### 문제 4（메트릭）

`clip_frac`이 매 스텝 1.0에 가깝다면 무엇을 의심하는가?

### 문제 5（연결）

제89강에서 이 루프의 `rewards` 자리에 추가로 빼게 될 전형적인 항은?

### 문제 6（구조）

PPO epoch 루프 안에서 응답을 **다시 sample**하면 안 되는 이유를 쓰시오.

### 문제 7（단순화）

GAE를 빼고 배치 평균 baseline만 쓰면 잃는 것(고수준)은?

---

## 정답 및 해설
### 문제 1

old 확률은 고정 타깃이어야 하며, 그래프에 연결되면 old까지 미분되어 비율의 의미가 붕괴한다.

### 문제 2

$\rho=1$, unclipped 항은 $A$ 그대로.

### 문제 3

공통: 응답에 스칼라 보상을 줌. 차이: RM은 선호 데이터로 학습된 모델, `toy_reward`는 고정 규칙.

### 문제 4

학습률·ε·epoch가 과도하거나 logprob 정렬 버그로 비율이 항상 바깥에 있는 상황.

### 문제 5

reference 대비 KL 페널티($\beta\mathrm{KL}$).

### 문제 6

`logp_old`와 행동이 같은 궤적에 묶여 있어야 clip 비율이 정의된다. 다시 샘플하면 짝이 깨진다.

### 문제 7

시간(토큰)축 신용 할당의 정교함·분산 조절 수단을 잃는다(고수준).

## 다음 강의와 연결
제83강 Advantage → 제84·85강 데이터·RM → 제86강 구조 → 제87강 수식 → **이번 구현**으로 RLHF-PPO의 세로축이 한 번 관통했다.

다음 **제89강. KL Divergence의 역할**에서는 $\pi_{\mathrm{ref}}$와의 KL이 보상·손실에 어떻게 들어가고, 왜 없으면 정책이 붕괴·해킹으로 흐르는지 다룬다. 그 위에 제90강 DPO가 “명시 RM+PPO” 없이도 preference를 쓰는 길을 연다.

제86강의 다이어그램을 다시 펼쳐 보라. 비어 있던 update 상자가 이제 코드로 채워졌다.

> 먼저 밴딧에서 ratio를 길들이고, 그다음 토큰 시퀀스로 확장하라. 스케일은 나중에 오면 된다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [87강. PPO 직관과 수식](87강_PPO_직관과_수식.md)
- **다음 강:** [89강. KL Divergence의 역할](89강_KL_Divergence의_역할.md)

<!-- /LECTURE_NAV -->
