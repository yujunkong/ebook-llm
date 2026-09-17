# 86강. RLHF 전체 구조
## 이번 강에서 배우는 내용

- SFT 정책 → 샘플링 → RM 점수 → RL 업데이트의 흐름
- Reference model($\pi_{\mathrm{ref}}$)이 왜 필요한지
- Actor(policy) · Reward · (선택) Value / Critic의 역할 분담
- 텍스트로 그린 end-to-end 다이어그램을 읽고 재현하기
- 제87강 PPO가 이 구조의 “업데이트 엔진”임을 위치지하기

## 왜 중요한가?
RLHF를 “PPO 돌리면 되는 것”으로만 이해하면 실패한다. 실제로는 여러 모델·데이터·목표가 맞물린 **시스템**이다.

```text
데이터가 나쁘면   → RM이 나쁘고 → 정책이 나빠진다
KL이 없으면      → 정책이 붕괴·빙빙 맴돈다
SFT가 약하면     → 샘플이 쓰레기라 선호가 무의미
PPO만 튜닝하면   → 증상만 만지작
```

전체를 보지 않으면 디버깅 좌표를 잃는다. 이 강의의 목표는 좌표를 주는 것이다.

```text
3권:  언어 모델이 지시를 따르게 한다 (SFT)
4권:  그 위에 인간 선호로 정책을 미세 조정한다 (RLHF)
```

## 선수 개념
- SFT 정책 $\pi_{\mathrm{SFT}}$ (3권)
- Preference · RM $r_\phi$ (제84·85강)
- Policy / Reward / Advantage (제80~83강)
- Sampling: temperature, top-p (3권 제58·59강)
- KL divergence 직관: 두 분포가 얼마나 다른가 (상세는 제89강)

아직 깊게 들어가지 않는 것:

- Clipped surrogate 유도 (제87강)
- 토큰 단위 logprob 구현 디테일 (제88강)
- DPO가 RL 루프를 우회하는 방식 (제90강)

## 핵심 개념
### 3.1 RLHF란?

**RLHF**는 인간(또는 그에 준하는) 피드백으로 학습된 보상 신호를 사용해, 강화학습으로 언어 모델 정책을 최적화하는 절차의 총칭이다.

좁은 의미의 고전적 3단계:

```text
Stage 1: Supervised Fine-Tuning (SFT)
Stage 2: Reward Model 학습 (Preference → r_φ)
Stage 3: RL 정책 최적화 (보통 PPO) + KL to reference
```

이 책은 Stage 1을 3권에서, Stage 2를 제84·85강에서, Stage 3의 구조·엔진을 제86~88강에서 다룬다.

### 3.2 한 장 다이어그램 (텍스트)

```text
┌─────────────────────────────────────────────────────────┐
│                    RLHF Pipeline                        │
│                                                         │
│  [1] SFT                                                 │
│   Instruction data ──► π_SFT                            │
│                         │                               │
│                         ▼                               │
│  [2] Preference                                         │
│   prompts X ──► sample y's ──► human prefs ──► D        │
│                                           │             │
│                                           ▼             │
│  [3] Reward Model                                       │
│   D ──► train r_φ(x,y)                                  │
│                                                         │
│  [4] RL loop (반복)                                      │
│   x ~ prompts                                           │
│   y ~ π_θ(·|x)          ◄── 학습 중인 policy            │
│   r = r_φ(x,y)                                          │
│   bonus = -β KL( π_θ || π_ref )                         │
│   R = r + bonus                                         │
│   θ ← PPO update(θ; R, π_old)                           │
│                                                         │
│  π_ref 는 보통 π_SFT 를 freeze                           │
└─────────────────────────────────────────────────────────┘
```

읽기 순서:

1. 아래(데이터)에서 위를 만들지 말고, **SFT → Preference → RM → RL** 순으로 쌓는다
2. RL 루프 안에서는 **샘플 → 점수 → 업데이트**가 돈다
3. $\pi_{\mathrm{ref}}$는 RL 중에 보통 **고정**

### 3.3 등장인물(모델) 정리

| 이름 | 기호 | 학습? | 역할 |
|---|---|---|---|
| SFT / Reference | $\pi_{\mathrm{ref}}$ (흔히 $\pi_{\mathrm{SFT}}$) | RL 중 freeze | 출발점·KL 앵커 |
| Policy / Actor | $\pi_\theta$ | 학습 | 응답 생성 |
| Reward Model | $r_\phi$ | RL 전 학습, RL 중 freeze가 흔함 | 스칼라 보상 |
| Value / Critic | $V_\psi$ | PPO에서 학습 | baseline / Advantage |
| (선택) Old policy | $\pi_{\theta_{\mathrm{old}}}$ | 롤아웃 스냅샷 | PPO ratio |

초보가 헷갈리는 지점:

```text
π_ref ≠ r_φ
π_ref 는 "언어분포 앵커"
r_φ 는 "선호 점수기"
```

### 3.4 Stage 1 — SFT policy

RLHF의 출발점은 대개 **이미 지시를 따르는 정책**이다.

```text
π_θ ← π_SFT
π_ref ← copy(π_SFT); freeze
```

SFT가 약하면:

- 샘플이 비문법·비협조 → 선호 라벨 비용↑
- RM이 “덜 나쁜 것”만 학습
- RL이 형식부터 다시 싸워야 함

**설명:** RLHF는 SFT의 대체재가 아니라 **refinement 층**에 가깝다.

### 3.5 Stage 2 — 샘플과 선호, RM

제84·85강의 요약:

```text
x ~ 프롬프트 분포
y1,y2,... ~ π_SFT (또는 혼합)
사람: yw ≻ yl
r_φ ← minimize BT loss
```

이 단계가 끝나면 “환경의 reward 함수”가 근사된다. LLM에는 게임 점수판이 없으므로 **RM이 환경 역할**을 한다.

### 3.6 Stage 3 — RL 업데이트의 목표

개념적 목표(자주 쓰는 형태):

\[
\max_\theta \;
\mathbb{E}_{x\sim\mathcal{D},\, y\sim\pi_\theta(\cdot\mid x)}
\big[r_\phi(x,y)\big]
\;-\;
\beta\,
\mathbb{E}_{x}
\big[
\mathrm{KL}\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x)\big)
\big]
\]

해석:

- 첫째 항: RM이 좋아하는 응답을 많이
- 둘째 항: reference에서 너무 멀어지지 않게

$\beta$가 크면 보수적(안전·안정), 작으면 공격적(보상 추격·붕괴 위험).

유효 보상으로 합치기도 한다:

\[
R(x,y)=r_\phi(x,y)-\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
\]

(구현·추정 방식은 제88·89강에서 구체화.)

### 3.7 Reference model이 필요한 이유

Reference가 없으면 정책은 RM의 허점을 찾아 **보상만 올리는 이상한 문장**으로 도망칠 수 있다.

```text
예 (정성):
RM이 특정 인사말에 고점
→ KL 제약 없으면 모든 답이 그 인사말로 시작
→ 유창성·다양성 붕괴
```

Reference는:

1. **언어능력 보존**의 닻
2. **최적화 안정화**
3. 과도한 distribution shift 완화

제89강에서 KL을 본격적으로 다룬다. 여기서는 파이프라인상 위치를 고정한다.

### 3.8 롤아웃(Rollout)이란?

RL 한 사이클의 데이터 수집:

```text
1. 프롬프트 배치 선택
2. 현재 정책으로 응답 생성 (토큰 시퀀스)
3. 각 토큰의 log π_θ_old 저장
4. RM으로 r(x,y) 계산
5. (PPO) Advantage / return 추정
6. 여러 epoch 동안 정책·가치 업데이트
```

LLM에서 “에피소드”는 대개 **한 번의 응답 생성**이다. 상태·행동 대응:

| RL 용어 | LLM 대응 |
|---|---|
| State | 지금까지의 토큰 접두 |
| Action | 다음 토큰 |
| Trajectory | 전체 응답 토큰열 |
| Terminal reward | RM 점수(응답 단위) |

### 3.9 On-policy 감각

PPO는 기본적으로 **현재 정책이 만든 샘플**로 업데이트한다(근사 on-policy).  
따라서 정책이 바뀌면 Periodically 다시 샘플링해야 한다.

```text
offline preference만으로 끝내는 방법: DPO 등 (제90강)
online sampling이 필요한 방법: PPO RLHF (이번 구조)
```

### 3.10 평가가 파이프라인에 끼는 위치

학습만 돌리면 “reward↑”에 속는다. 병행 평가:

```text
- 자동: RM score, KL, 길이, perplexity vs ref
- 쌍비교: 새 정책 vs SFT win-rate (사람 또는 심사 모델)
- 안전: 거절률·유해 유발 테스트
- 과제: 벤치마크 (도메인별; 숫자 날조 금지)
```

**사실:** 벤치마크 점수 향상량은 설정마다 다르다.  
**설명:** 이 책은 허구 SOTA를 만들지 않고, **무엇을 재는지**만 명시한다.

## 직관적으로 이해하기
요리사 비유를 한 단계 확장한다.

```text
SFT:     레시피 학교 (기본기)
Preference: 손님 블라인드 테스트
RM:      손님 취향을 점수표로 압축한 심사위원
π_ref:   학교 졸업 시점의 요리사 (기본 맛 기억)
RL:      점수표를 보며 매일 메뉴 조정
KL:      “학교 때 배우지 않은 이상한 요리”로 도망가지 않기
```

공장 라인 비유:

```text
원재료(prompts)
 → 생성기(policy)
 → 검사기(RM)
 → 피드백으로 생성기 나사 조임(PPO)
 → 단, 안전 규격(reference KL) 준수
```

## 수학적으로 이해하기
### 5.1 정책으로서의 LM

\[
\pi_\theta(y\mid x)=\prod_{t=1}^{|y|}\pi_\theta(y_t\mid x,y_{<t})
\]

로그 확률:

\[
\log\pi_\theta(y\mid x)=\sum_t \log\pi_\theta(y_t\mid x,y_{<t})
\]

PPO는 이 토큰 로그확률의 비율을 사용한다(제87강).

### 5.2 기대 보상 목표

\[
J(\theta)=\mathbb{E}_{x,y\sim\pi_\theta}[r_\phi(x,y)]
\]

그대로 올리면 KL 폭발 위험 → 페널티 포함:

\[
J_\beta(\theta)=
\mathbb{E}[r_\phi(x,y)]
-\beta\,\mathbb{E}_x\big[\mathrm{KL}(\pi_\theta\|\pi_{\mathrm{ref}})\big]
\]

### 5.3 왜 “한 방 경사”로 안 끝내는가?

$y$가 이산·길고, 보상이 응답 끝에만 있으면 분산이 크다.  
그래서 Policy Gradient + baseline/Advantage + trust region(PPO clip) 조합이 실무 기본값이 되었다. 제82·83·87강의 연결이다.

### 5.4 정보 흐름 (수식 스케치)

롤아웃 샘플 $(x,y)$에 대해:

\[
\begin{aligned}
r &\leftarrow r_\phi(x,y)\\
A &\leftarrow \mathrm{Advantage}(r, V_\psi,\ldots)\\
\theta &\leftarrow \arg\max_\theta\,
\mathbb{E}\big[\mathrm{PPO\text{-}clip}(\theta; A, \pi_{\theta_{\mathrm{old}}})\big]
\end{aligned}
\]

Value는

\[
\min_\psi \mathbb{E}\big[(V_\psi(x)-R)^2\big]
\]

형태로 같이 학습하는 구현이 많다.

## 작은 숫자로 직접 계산하기
교육용 미니 배치. 프롬프트 1개, 후보 응답 3개.

| $y$ | $r_\phi$ | $\log\pi_\theta$ | $\log\pi_{\mathrm{ref}}$ | $\mathrm{KL\,term}=\log\pi_\theta-\log\pi_{\mathrm{ref}}$ | $R=r-\beta\cdot(\cdot)$, $\beta=0.1$ |
|---|---|---|---|---|---|
| A | 2.0 | -10.0 | -10.5 | 0.5 | $2.0-0.05=1.95$ |
| B | 2.5 | -8.0 | -12.0 | 4.0 | $2.5-0.40=2.10$ |
| C | 0.5 | -11.0 | -11.0 | 0.0 | $0.5-0.00=0.50$ |

관찰:

- B는 RM 점수가 가장 높지만 ref에서 많이 벗어남 → KL이 깎음
- 그래도 $R_B$가 가장 큼 → 정책이 B 쪽으로 조금 이동할 유인
- $\beta$를 1.0으로 키우면 $R_B=2.5-4.0=-1.5$로 급락 → A가 유리

**숫자 예는 교육용**이다. 실제 토큰 로그확률 스케일은 모델·길이 의존적이다.

β 민감도 한 줄:

```text
β↑ → ref에 붙음 (보수)
β↓ → RM 추격 (공격)
```

## 코드로 구현하기 — 파이프라인 스케치
완성 PPO는 제88강. 여기서는 **구조가 보이게** 의사코드 수준으로 고정한다.

```python
def rlhf_pipeline_sketch():
    # --- Stage 1 ---
    pi_sft = load_sft_checkpoint()
    pi_ref = freeze(copy(pi_sft))
    pi_theta = copy(pi_sft)  # trainable policy

    # --- Stage 2 ---
    prefs = load_preference_dataset()  # from lecture 84
    rm = train_reward_model(prefs)     # lecture 85
    freeze(rm)

    # --- Stage 3 ---
    value = init_value_head(pi_theta)
    for iteration in range(num_iters):
        prompts = sample_prompts(batch_size)
        rollouts = []
        for x in prompts:
            y, logp_old = generate_with_logprobs(pi_theta, x)
            r = rm.score(x, y)
            kl = compute_approx_kl(pi_theta, pi_ref, x, y)
            R = r - beta * kl
            rollouts.append((x, y, logp_old, R))

        advantages = compute_advantages(rollouts, value)  # lec 83/87
        for _ in range(ppo_epochs):
            ppo_update(pi_theta, value, rollouts, advantages)  # lec 88

    return pi_theta
```

스테이지 경계를 assert로 문서화:

```python
assert is_frozen(pi_ref)
assert is_frozen(rm)
assert is_trainable(pi_theta)
```

## PyTorch로 구현하기 — 모듈 경계
실제 코드베이스에서 파일을 나눈다면:

```text
sft_policy.py      # π_θ, generate
reference.py       # frozen π_ref, kl
reward_model.py    # r_φ
rollout.py         # sample + store logprobs
ppo.py             # loss + update
train_rlhf.py      # orchestration
```

핵심 인터페이스:

```python
class Policy(nn.Module):
    def forward(self, input_ids, attention_mask):
        """logits [B,T,V]"""

    def generate(self, prompt_ids, **gen_kwargs):
        """responses + optional logprobs"""

class RewardModel(nn.Module):
    def score(self, input_ids, attention_mask) -> torch.Tensor:
        """scalar per sequence [B]"""

def approx_kl(logp_theta, logp_ref):
    # per-token mean or sum; 정의 일관성 유지가 중요
    return (logp_theta - logp_ref).sum(dim=-1)
```

오케스트레이션에서 실수하기 쉬운 점:

1. ref에 gradient가 흐름 → `torch.no_grad()` / `requires_grad=False`
2. RM에 gradient가 흐름 → RL 단계에서는 freeze가 기본
3. prompt 길이·response 길이를 나눠 logprob 슬라이싱

## LLM에서는 어디에 사용될까?
연구·제품에서 보이는 변형:

| 변형 | 요지 |
|---|---|
| Classic RLHF | SFT → RM → PPO |
| RLAIF | 선호 라벨을 AI가 생성 |
| Best-of-N only | RL 없이 RM으로 샘플 선택 |
| DPO 계열 | 명시 RM+PPO 대신 직접 preference |
| Online RLHF | 배포 중 피드백을 지속 반영 |

자원 관점(정성):

```text
비용이 큰 부분:
- 인간 주석
- 긴 응답 롤아웃 (생성)
- 정책·ref·RM을 동시에 올리는 메모리
```

메모리 절약 패턴:

- ref는 LoRA 없이 freeze, policy만 LoRA
- RM 별도 GPU
- 응답 길이 상한

운영 위험:

1. **Reward hacking**
2. **Mode collapse** (다양성↓)
3. **회귀:** 일부 과제 win-rate↓
4. **과한 KL:** 사실상 SFT와 동일
5. **데이터 드리프트:** 옛 RM + 새 사용자 분포

제96강 Alignment 한계에서 부작용을 더 다룬다.

## 실습
### 실습 A — 다이어그램 재구성

이 강의의 텍스트 다이어그램을 보지 말고, 빈칸을 채우시오.

```text
Instruction data → (    )
prompts → sample → (    ) → D
D → (    )
RL: y~π → (    ) → R → (    ) update
anchor: (    )
```

### 실습 B — 역할 카드

팀 4명에게 역할(`π_θ`, `π_ref`, `r_φ`, `V`)을 나누고, 각자가 “나는 학습되는가/입출력은?”을 30초 발표.

### 실습 C — β 사고실험

제7절 표에서 $\beta=0, 0.1, 1.0$일 때 각 $y$의 $R$ 순위를 다시 매기시오.

### 실습 D — 실패 모드 매핑

다음 증상을 파이프라인 어느 단계부터 의심할지 고르시오.

1. 모든 답이 동일 문구로 시작
2. pairwise RM accuracy가 50% 근처
3. KL=0이고 reward도 SFT와 동일
4. 생성은 유창한데 지시 무시

### 실습 E — 체크리스트 작성

학습 시작 전 고정 체크리스트 8항목을 만드시오. 예:

```text
[ ] ref freeze 확인
[ ] RM freeze 확인
...
```

## 자주 하는 실수
1. **SFT를 건너뛰고 base에 PPO**  
   탐색 공간이 너무 넓다.

2. **π_ref를 학습 정책과 공유해 업데이트**  
   KL 닻이 함께 흘러가 제약이 사라짐.

3. **RM을 RL 중에 같이 학습(의도 없이)**  
   목표가 움직이는 과녁이 됨. (연구적 온라인 RM은 별도 설계)

4. **보상만 모니터**  
   KL·길이·win-rate를 안 보면 hacking을 놓침.

5. **평가 프롬프트 = 학습 프롬프트 복붙**  
   과적합 win-rate.

6. **Stage 순서를 뒤섞어 설명**  
   “PPO로 preference를 직접…” 같은 용어 혼선.

7. **Value head 없음 + baseline 없음**  
   그래디언트 분산 폭발(제83강 복습).

8. **다이어그램 없이 하이퍼파라미터만 튜닝**  
   어느 모듈 문제인지 모름.

## 핵심 요약
- RLHF는 SFT → Preference/RM → RL(+KL)의 **시스템**이다.
- 학습 중 정책 $\pi_\theta$만 움직이는 것이 기본이고, $\pi_{\mathrm{ref}}$와 $r_\phi$는 닻·점수기 역할로 freeze되는 경우가 많다.
- 루프는 sample → score → update다.
- 목표는 보상 최대화와 reference 근접의 균형이다.
- PPO는 그 목표를 안정적으로 쫓는 업데이트 엔진이다.
- 다음 강의에서 clip ratio의 직관과 수식을 푼다.

## 용어 사전
| 용어 | 의미 |
|---|---|
| RLHF | 인간 피드백 기반 강화학습 정렬 |
| Policy / Actor | 응답을 생성하는 $\pi_\theta$ |
| Reference model | KL 앵커 $\pi_{\mathrm{ref}}$ |
| Reward Model | $r_\phi$ 점수기 |
| Critic / Value | Advantage용 가치 함수 |
| Rollout | 정책으로 궤적(응답) 수집 |
| KL penalty | reference 이탈 비용 |
| β | KL 가중 계수 |
| On-policy | 현재 정책 샘플로 학습 |
| Win-rate | 쌍비교 승률 평가 |

## 연습문제
### 문제 1（순서）

RLHF 고전 3단계 순서를 쓰시오.

### 문제 2（역할）

$\pi_{\mathrm{ref}}$와 $r_\phi$의 차이를 한 문장으로.

### 문제 3（수식）

$R=r-\beta\mathrm{KL}$에서 $\beta$를 키우면 정책이 어디로 끌리는가?

### 문제 4（다이어그램）

RL 루프에서 RM은 보통 매 iteration마다 재학습되는가? (예/아니오 + 이유)

### 문제 5（연결）

제87강 PPO가 이 구조의 어느 화살표를 구체화하는가?

### 문제 6（실패）

보상은 오르는데 사용자 평가가 떨어질 때 의심할 현상 이름은?

### 문제 7（LLM 대응）

에피소드 종료 시 보상이 RM 스칼라 하나로 온다면, 토큰 행동의 신용 할당을 돕는 개념은? (제83강)

---

## 정답 및 해설
### 문제 1

SFT → Reward Model(선호 학습) → RL(PPO 등) 최적화.

### 문제 2

$\pi_{\mathrm{ref}}$는 언어분포 앵커(정책), $r_\phi$는 응답 선호 점수기다.

### 문제 3

reference 정책 분포에 더 가깝게(보수적으로) 끌린다.

### 문제 4

아니오. 고전 파이프라인에서 RM은 Stage 2 후 freeze하고 점수로만 쓰는 경우가 많다.

### 문제 5

$R$과 롤아웃 로그확률을 받아 $\theta$를 갱신하는 **update** 화살표.

### 문제 6

Reward hacking (또는 보상 오용/과적합).

### 문제 7

Advantage (또는 value baseline / GAE).

## 다음 강의와 연결
지도가 생겼으니, 이제 엔진을 연다.

다음 **제87강. PPO 직관과 수식**에서는 clipped surrogate objective, 비율 $\pi_\theta/\pi_{\mathrm{old}}$, 왜 clip이 필요한지, GAE의 위치(고수준)를 수식과 작은 숫자로 다룬다. 제88강에서 그 수식을 토이 루프로 구현한다.

제85강의 RM 점수와 제83강의 Advantage가, 제87강의 PPO 손실 안에서 만난다.

> RLHF는 단일 손실이 아니라, 닻(reference)·점수(RM)·정책(π)·업데이트(PPO)의 협주이다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [85강. Reward Model 구현](85강_Reward_Model_구현.md)
- **다음 강:** [87강. PPO 직관과 수식](87강_PPO_직관과_수식.md)

<!-- /LECTURE_NAV -->
