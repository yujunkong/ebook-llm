# 92강. GRPO
## 이번 강에서 배우는 내용

- 그룹 샘플링과 group-relative advantage의 직관
- PPO와의 공통점·차이점(설명 수준)
- 왜 reasoning / verifiable reward 설정과 잘 맞는지
- 수식 스케치와 작은 숫자 예
- “논문·구현마다 세부가 다르다”는 읽는 법

## 왜 중요한가?
최근 reasoning 모델 학습 이야기에서 PPO 대신 GRPO/유사 알고리즘 이름이 자주 등장한다. 배경은 대략 세 가지다.

1. **규칙·검증기 보상**(정답 일치, 유닛테스트)이 상대적으로 잘 정의됨 → 93강 RLVR
2. 같은 문제에 여러 rollout을 뽑기 쉬움 → 그룹 통계가 의미 있음
3. Critic LLM을 하나 더 두지 않아 **메모리 압박**을 줄이려는 실무 동기

```text
프롬프트 x
  → y1, y2, …, yG  ~ π_old
  → 각 yi에 보상 ri (RM 또는 verifier)
  → 그룹 내 정규화 → Âi
  → 정책 비율/클리핑으로 π 업데이트 (+ KL 등)
```

92강은 “이름 암기”가 아니라 **상대 비교로 baseline을 대체한다**는 설계 축을 잡는 데 있다.

## 선수 개념
- Policy gradient / Advantage (82~83강)
- PPO clip ratio $\rho=\pi_\theta/\pi_{\mathrm{old}}$ (87~88강)
- KL to reference (89강)
- 샘플링으로 응답 생성 (3권 58~59강)
- DPO의 “상대 비교” 감각 (90강) — 다만 데이터 출처가 다름

## 핵심 개념
### 3.1 그룹이란?

고정된 프롬프트 $x$에 대해 정책(보통 $\pi_{\mathrm{old}}$)에서 $G$개 응답을 독립 샘플한다.

$$

y_1,\ldots,y_G\sim\pi_{\mathrm{old}}(\cdot\mid x)

$$

각 응답에 스칼라 보상 $r_i=r(x,y_i)$를 붙인다. 보상은 RM일 수도, **검증 가능한 규칙**일 수도 있다.

### 3.2 Group-relative advantage

아이디어의 핵은 baseline을 value 네트워크 대신 **같은 그룹의 통계**로 쓰는 것이다. 대표적 설명용 형태:

$$

\hat A_i=\frac{r_i-\mathrm{mean}(r_{1:G})}{\mathrm{std}(r_{1:G})+\varepsilon}

$$

또는 평균만 빼는 형태:

$$

\hat A_i=r_i-\mathrm{mean}(r_{1:G})

$$

해석:

- 그룹 평균보다 좋은 응답 → 양의 advantage → 확률 ↑
- 평균보다 나쁜 응답 → 음의 advantage → 확률 ↓
- **절대 점수**가 아니라 **동료 대비 순위/편차**가 학습 신호

$\mathrm{std}$ 정규화는 스케일이 다른 프롬프트 간 학습률 체감을 맞추려는 장치다. 쓰는 구현/안 쓰는 구현이 모두 있다.

### 3.3 정책 업데이트 (PPO-유사)

많은 GRPO 설명은 PPO식 clipped surrogate를 유지한다.

$$

\rho_i(\theta)=\frac{\pi_\theta(y_i\mid x)}{\pi_{\mathrm{old}}(y_i\mid x)}

$$

$$

\mathcal{L}_{\mathrm{clip}}=-\frac{1}{G}\sum_{i=1}^{G}
\min\big(
\rho_i\hat A_i,\;
\mathrm{clip}(\rho_i,1-\epsilon,1+\epsilon)\hat A_i
\big)

$$

여기에 KL 페널티를 더하는 형태가 흔히 함께 소개된다.

$$

\mathcal{L}=\mathcal{L}_{\mathrm{clip}}+\beta\,\widehat{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})

$$

**설명용 골격**이다. 토큰 단위 합·평균, advantage broadcast, KL 추정 위치는 구현마다 다르다.

### 3.4 Critic이 빠지면 무엇이 좋은가 / 위험한가

이득:

- 학습 대상 파라미터가 actor 쪽에 집중
- value head 학습 실패(부정확한 baseline) 위험이 사라짐
- 구현 표면이 “샘플 → 점수 → 상대화 → PPO-like step”으로 단순

대가:

- 그룹 크기 $G$가 작으면 baseline 분산이 큼
- 보상이 전부 같으면($r_i\equiv c$) 신호가 0
- 프롬프트마다 난이도·보상 스케일이 다르면 정규화 선택이 민감
- online 샘플링 비용은 그대로(또는 $G$배) 든다

## PPO와의 차이 — 표로 고정
| 축 | PPO (전형적 LLM-RLHF) | GRPO (설명용) |
|---|---|---|
| Baseline | Learned value $V_\psi$ | 그룹 내 상대 통계 |
| 샘플 | 프롬프트당 1개 또는 소수 | 프롬프트당 $G$개 묶음 |
| Advantage | GAE 등 | group-relative $r$ |
| Clip / ratio | 있음 | 유사하게 유지하는 서술 많음 |
| Critic 메모리 | 필요 | 없음(또는 최소화) |
| 데이터 | online | online (그룹) |
| 선호 쌍 | 직접 쓰지 않음 | 쓰지 않음(보상이 신호) |

DPO와의 차이도 한 줄로:

- DPO: **오프라인 고정 쌍**의 로그비
- GRPO: **온라인 그룹 샘플**의 상대 보상

## 직관적으로 이해하기
### 5.1 “반에서 몇 등인가”

시험 원점수 80점이 좋은지는 반 평균에 달려 있다. GRPO는 매 프롬프트를 **작은 학급**으로 본다. 같은 학급에서 상대적으로 잘한 답안만 밀어 올린다.

이 때문에 **검증기 보상**과 궁합이 좋다. 정답/오답이 섞인 그룹이 나오면, 맞춘 쪽이 양의 $\hat A$, 틀린 쪽이 음이 되어 “풀이 경로”를 강화하기 쉽다(93~94강).

### 5.2 왜 value 대신 그룹인가?

Value는 “이 프롬프트의 기대 보상”을 학습해야 한다. LLM에서는 프롬프트 공간이 넓어 $V$ 학습이 어렵다. 그룹 평균은 **그 프롬프트, 지금 정책**에 대한 국소 Monte Carlo baseline이다. 편법은 아니고, control variate를 **같은 $x$의 동시 샘플**로 쓰자는 선택이다.

### 5.3 Mode collapse와의 관계

그룹 안 다양성이 죽으면 $r_i$가 비슷해져 신호가 약해진다. 반대로 보상 해킹 패턴이 그룹을 지배하면 그 패턴이 상대 우위로 강화된다. KL(89강)·샘플 온도·보상 설계가 여전히 필요하다.

## 작은 숫자 예제
$G=4$, 보상 $r=(1.0,\;0.0,\;1.0,\;0.0)$.

평균 $\bar r=0.5$. 표준편차:

$$

\mathrm{std}=\sqrt{\frac{(0.5)^2\times4}{4}}=0.5

$$

(모집단/표본 정의는 구현마다 다름. 여기선 설명용으로 $\sqrt{\mathbb{E}[(r-\bar r)^2]}$.)

$$

\hat A=\frac{r-0.5}{0.5}=(1,\,-1,\,1,\,-1)

$$

정책이 아직 $\rho_i=1$이면 clip 안쪽 surrogate는 $-\mathrm{mean}(\hat A)=0$이지만, **그래디언트**는 양의 $A$를 가진 $y$의 가능도를 올리고 음의 $A$를 내린다.

다른 예: 전부 정답 $r=(1,1,1,1)$ → $\hat A=\mathbf{0}$ → 이 배치에서는 정책 갱신 신호가 없다. “이미 다 맞음”을 value가 알려주는 대신, **그룹이 알려준다**.

해킹 예: $r=(5,5,5,0)$이고 5점이 모두 “이모지 스팸”이면, 스팸이 상대 승자가 된다. GRPO는 보상 함수의 철학을 바꾸지 않는다. **상대화는 분산을 줄이지, 목표를 교정하지 않는다.**

## 알고리즘 스케치 (의사코드)
```text
for update = 1 … U:
  프롬프트 배치 X를 뽑는다
  for x in X:
    y[1..G] ~ π_old(·|x)
    r[i] = reward(x, y[i])          # RM 또는 verifier
    A[i] = normalize(r[i]; r[1..G]) # mean/std 등
  π_θ 를 clipped objective (+ KL)로 여러 epoch 업데이트
  주기적으로 π_old ← π_θ
```

교육용 PyTorch 스케치(손실 핵심만):

```python
def group_advantages(rewards, eps=1e-8):
    # rewards: [B, G]
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True)
    return (rewards - mean) / (std + eps)

def grpo_like_loss(logp_theta, logp_old, advantages, clip_eps=0.2):
    # 시퀀스 합 로그확률 기준의 단순화 버전
    ratio = torch.exp(logp_theta - logp_old)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -torch.min(unclipped, clipped).mean()
```

이 코드는 **개념 검증용**이다. 토큰 마스크, importance sampling 세부, KL 항, 엔트로피 보너스 등은 의도적으로 생략했다.

## 연구·실무 읽는 법 (사실 고지)
다음을 **설명**으로 읽고, 만고불변 명세로 외우지 않는다.

1. 이름 “GRPO”가 가리키는 수식 디테일은 출처에 따라 다르다.
2. 그룹 정규화에 mean-only / z-score / rank 변환 등 변형이 가능하다.
3. PPO clip ε, $G$, β, 샘플 온도는 과제(수학·코드·대화)마다 다시 튜닝한다.
4. “Critic 없음 = 항상 더 좋음”은 아니다. 보상 밀도·분산에 의존한다.
5. 벤치마크 승리는 데이터·검증기·스케줄과 얽혀 있어, 알고리즘 이름만으로 재현되지 않는다.

커리큘럼상 위치는 **PPO를 이해한 뒤의 대안축**이다. 92강을 읽기 위해 87~89강이 먼저인 이유가 여기 있다.

## DPO·PPO·GRPO 한 장 지도
```text
선호 쌍이 이미 있다
  → DPO / 변형 (offline)

보상 모델 + online, value 사용
  → PPO RLHF

보상(특히 verifiable) + online, 그룹 상대 baseline
  → GRPO 계열
```

실무팀은 종종 이들을 하이브리드한다. 예: SFT → DPO → (verifier) GRPO. 순서는 조직·과제마다 다르다.

## 실패 모드와 대응
| 실패 | 증상 | 대응 방향 |
|---|---|---|
| 보상 전부 동일 | $\hat A\approx 0$ | 커리큘럼·온도·부분 점수 |
| $G$ 과소 | baseline 잡음 | $G$ 증가(비용↑) |
| 해킹 패턴 지배 | 상대 승자가 편법 | verifier/RM 수정, KL↑ |
| clip 과다 | 업데이트 정체 | ε·lr 재조정 |
| 길이 폭발 | 장문만 고득점 | 길이 페널티·형식 보상 |

## RLOO 등과 한 줄 비교 (설명)
Leave-One-Out baseline 등 **같은 프롬프트의 다른 샘플로 baseline을 만드는** 기법들이 GRPO와 아이디어가 겹친다.

```text
공통: 동시 샘플로 control variate
차이: 정규화 식, clip 유무, 토큰/시퀀스 단위, KL 배치
```

이름을 외우기보다 “critic 대신 **Monte Carlo 동료 샘플**”이라는 문장을 공유한다. 세부 승패는 과제·구현에 달렸다.

## 메모리·계산 스케치
프롬프트 배치 $B$, 그룹 $G$, 평균 길이 $L$이면 롤아웃 토큰 수는 대략 $B\cdot G\cdot L$이다.

- 생성 비용 ∝ $G$
- 정책 forward(학습)도 샘플 수에 비례
- Critic이 없어도 **샘플 자체가** 비싸다

따라서 GRPO는 “free lunch”가 아니라 **메모리(value) vs 샘플(G)** 의 교환이다. 추론 가속(5권 vLLM 등)이 있으면 그룹 확대가 현실적이 된다.

## 토큰 단위로 펼치기
시퀀스 advantage $\hat A_i$를 모든 토큰에 방송(broadcast)하는 단순화가 흔하다.

$$

\hat A_{i,t}=\hat A_i \quad(t=1\ldots|y_i|)

$$

과정 보상이나 토큰 가치가 없을 때의 기본값이다.  
중간 토큰의 기여를 더 정교히 나누는 연구는 별도 축이다(process reward, 94강 힌트).

## 두 번째 숫자 예 — 표준편차 민감도
$r=(1.0, 0.9, 0.1, 0.0)$, $\bar r=0.5$.

편차: $+0.5,+0.4,-0.4,-0.5$.  
대략 $\mathrm{std}\approx 0.45$라면 z-score는 $(1.11, 0.89, -0.89, -1.11)$ 근처.

mean-only면 $(0.5, 0.4, -0.4, -0.5)$.  
스케일만 다르고 **순위 신호**는 같다. 학습률·β와 결합될 때 체감 스텝 크기가 달라지므로, 정규화 선택을 바꾸면 lr도 다시 본다.

## FAQ
**Q. GRPO는 PPO를 대체하나?**  
A. 일부 설정(특히 verifiable)에서 대안으로 쓰인다. 전 과제 만능 대체는 아니다.

**Q. 그룹 안에 반드시 정답과 오답이 섞여야 하나?**  
A. 섞일 때 신호가 가장 또렷하다. 항상 보장되진 않으므로 문제 난이도 설계가 중요하다.

**Q. DPO 다음에 GRPO를 꼭 해야 하나?**  
A. 필수는 아니다. 데이터가 선호 쌍 중심이면 DPO로 끝날 수 있고, 검증기가 있으면 GRPO가 자연스럽다.

**Q. 이 책 수식과 오픈소스 코드가 다르면?**  
A. 코드·원논문이 우선이다. 강의는 직관을 고정한다.

## 16b. 롤아웃·업데이트 분리

실무 스케치에서 중요한 리듬:

```text
1) π_old 로 G개 생성 (추론 모드, KV cache 활용 가능)
2) verifier/RM 으로 r 계산 (CPU/샌드박스)
3) π_θ 로 logprob 재평가 (학습 모드)
4) clip loss + KL
5) 수 epoch 후 π_old ← π_θ
```

(3)에서 “생성 때 쓴 로그확률”을 재사용하지 않고 **현재 θ로 다시 계산**하는 구현이 많다. 세부 캐싱은 라이브러리마다 다르다.

## 16c. 엔트로피와 탐험

그룹이 다양해야 상대 신호가 산다. 온도·top-p·엔트로피 보너스는 탐험 노브다.

$$

\mathcal{L}_{\mathrm{ent}}=-\eta\,\mathbb{E}\big[H(\pi_\theta(\cdot\mid x,y_{<t}))\big]

$$

$\eta$가 크면 다양성↑, 너무 크면 정답 집중↓. verifiable 설정에서는 “형식은 유지하되 풀이 경로는 다양”이 목표가 되기도 한다. 수치 처방전은 과제마다 다시 쓴다.

## 16d. 미니 실험 설계 (95강 예고)

작은 모델로 GRPO를 체감하려면:

1. 산술 10문제, verifier=exact match
2. $G=4$, 온도 0.8
3. 200 업데이트 후 pass@1 전후 비교
4. 로그: mean r, frac_nonzero_A, KL, unique answers

이 실험의 목적은 SOTA가 아니라 **상대 advantage가 0이 아닌 배치 비율**을 눈으로 확인하는 것이다.

## 16e. 제90·91강 DPO와의 역할 분담 복습

| 질문 | DPO | GRPO |
|---|---|---|
| 데이터가 선호 쌍뿐인가? | 적합 | 부적합(보상 필요) |
| 검증기가 있는가? | 쓸 수 있으나 필수는 아님 | 매우 잘 맞음 |
| 새 응답 탐색 | 약함 | 강함(online) |
| Critic | 없음 | 없음(전형 서술) |
| 구현 표면 | SFT-like | RL-like |

둘을 적대적으로 두지 않는다. 파이프라인에서 **연속 구간**으로 쓰는 팀이 많다.

## 16f. 한 장 요약 다이어그램

```text
                 ┌── y1 ─ r1 ─┐
x ─ sample G ───┼── y2 ─ r2 ─┼── normalize → Â ── clip(ρ,Â) ── θ
                 └── yG ─ rG ─┘              └─ + β KL(π||π_ref)
```

기억 문장:

> PPO가 “가치로 기댓값을 빼는” 법이라면, GRPO는 “동료 샘플로 기댓값을 빼는” 법에 가깝다.

## 16g. 구현 체크리스트 (설명용)

1. 샘플은 $\pi_{\mathrm{old}}$에서, 학습 로그확률은 현재 $\pi_\theta$에서 계산하는가?
2. advantage 정규화가 프롬프트 그룹 단위인가, 배치 전체인가?
3. KL의 상대는 $\pi_{\mathrm{ref}}$인가 $\pi_{\mathrm{old}}$인가? (둘 다 쓸 수 있으나 의미가 다름)
4. 보상이 상수인 그룹을 스킵하거나 로깅하는가?
5. 생성 길이와 형식 실패율을 별도 모니터하는가?

이 다섯을 통과하면 “이름만 GRPO”가 아니라 **상대 학습 루프**를 돌리는 상태에 가깝다.

## 16h. 닫기 전 자가 질문

1. 이 업데이트의 baseline은 value인가, 그룹 통계인가?
2. $G$를 반으로 줄이면 어떤 신호가 먼저 죽는가?
3. 보상 함수를 바꿨을 때 GRPO 식 자체는 바뀌는가, 안 바뀌는가?
4. KL 항이 없으면 장기적으로 무엇이 위험한가?
5. DPO 데이터만 있는 팀에 GRPO를 권할 조건은 무엇인가?

## 수식 보강 — GRPO 그룹 상대

같은 프롬프트에 응답 그룹 $\{y_i\}_{i=1}^G$를 샘플링하고, 그룹 내 상대 점수(또는 advantage)로 정책을 업데이트합니다. 개념적으로

$$
A_i = R(y_i) - \frac{1}{G}\sum_{j=1}^G R(y_j)
$$

처럼 **그룹 평균 대비**로 정규화한 뒤 policy gradient/PPO류 업데이트를 적용합니다. 절대 보상 스케일보다 상대 순위가 중요해집니다.

## 정량 스케치 — 그룹 상대 심화

$$

\hat A_i=\frac{r_i-\mu_x}{\sigma_x+\varepsilon},\quad
\rho_i=\frac{\pi_\theta(y_i\mid x)}{\pi_{\mathrm{old}}(y_i\mid x)}
$$

$$

L=-\frac1G\sum_i\min(\rho_i\hat A_i,\mathrm{clip}(\rho_i)\hat A_i)+\beta\,\widehat{\mathrm{KL}}
$$

이진 보상: $\mu=S/G$, $\sigma=\sqrt{\mu(1-\mu)}$.  
$S\in\{0,G\}$면 신호 0; $S\approx G/2$에서 상대 신호 최대.

LOO: $b_i=\frac1{G-1}\sum_{j\neq i}r_j$, $\hat A_i=r_i-b_i$.

비용 $\propto G\cdot(T_{\mathrm{gen}}+T_{\mathrm{reward}}+T_{\mathrm{logprob}})$.

손계산: $G=2$, $r=(1,0)$, mean-only $\hat A=(0.5,-0.5)$, $\rho=(1.5,0.7)$, $\varepsilon=0.2$  
→ $\min$ 항 $(0.6,-0.4)$, 평균 surrogate 감각 $0.1$.

다양성 프록시 $u=\#\{\mathrm{unique\ ex}(y_i)\}/G$.


## 그룹 크기 $G$ 선택 가이드（설명）

| $G$ | 이득 | 비용 |
|---:|---|---|
| 2 | 최소 상대 비교 | baseline 잡음 큼 |
| 4~8 | 혼합 그룹 빈도↑ (이진 $r$) | 샘플·verify 배수 |
| 16+ | 통계 안정 | 처리량·지연 압박 |

verifiable 이진 보상에서는 $1-p^G-(1-p)^G$가 충분히 커지도록 난이도·$G$·온도를 **같이** 본다.

### 로그 필수 항목

```text
mean_r, std_r, frac_zero_A, frac_mixed_groups,
clip_frac, approx_kl, unique_answers, format_rate
```

`frac_zero_A≈1`이면 학습이 멈춘 것과 같다.


## LLM에서는 어디에 사용될까?

이번 92강에서 배운 개념은 이후 Transformer · GPT · 서빙 강의에서 반복해서 등장합니다. 각 수식·코드 블록을 “실제 모델의 어느 단계인가”와 연결해 다시 읽어 보세요.

## 핵심 요약
- GRPO 계열의 핵심은 **같은 프롬프트의 그룹 샘플**로 상대 advantage를 만드는 것이다.
- Critic을 제거/축소하는 대신, 그룹 통계가 baseline 역할을 한다.
- 업데이트 골격은 PPO식 ratio·clip과 닮은 설명이 많다.
- 보상 설계가 나쁘면 상대 우위로 해킹이 강화된다.
- 세부 식은 논문·구현 의존이다. 이 강의는 직관·비교축을 고정한다.
- 비용의 중심은 value가 아니라 **그룹 샘플링**일 수 있다.

## 용어 사전
| 용어 | 한 줄 의미 |
|---|---|
| GRPO | 그룹 상대 신호로 정책을 갱신하는 계열 명칭 |
| Group size $G$ | 프롬프트당 샘플 수 |
| Group-relative advantage | 그룹 mean/std 등으로 만든 $\hat A$ |
| $\pi_{\mathrm{old}}$ | 샘플링에 쓰는 롤아웃 정책 |
| Critic-free | 학습 value net을 쓰지 않음 |
| Verifier reward | 규칙·테스트로 매기는 보상(93강) |
| Broadcast advantage | 시퀀스 $\hat A$를 토큰에 복사 |
| Leave-one-out baseline | 자기 제외 동료 평균으로 baseline |

## 연습문제
### 문제 1（계산）

$r=(2,0,2,0)$. mean-only advantage $r-\bar r$를 구하시오.

### 문제 2（개념）

$G=2$이고 두 보상이 항상 같으면 학습 신호는 어떻게 되는가?

### 문제 3（비교）

PPO의 GAE advantage와 GRPO의 group-relative advantage의 **기준점(baseline)** 차이는?

### 문제 4（설계）

수학 문제에서 보상이 $\{0,1\}$뿐일 때 그룹 크기 $G$를 키우면 어떤 이득이 기대되는가?

### 문제 5（연결）

제93강 RLVR의 “verifiable reward”가 GRPO와 만나면, RM 없이도 online 상대 학습이 가능한 이유를 한 문장으로 쓰시오.

### 문제 6（비용）

$B=8$, $G=16$일 때 프롬프트당이 아니라 **배치 전체** 샘플 수는?

### 문제 7（정규화）

mean-only에서 z-score로 바꾸면 순위는 유지되는데 왜 lr을 다시 만져야 할 수 있는가?

---

## 정답 및 해설
### 문제 1

$\bar r=1$, $\hat A=(1,-1,1,-1)$.

### 문제 2

상대 편차가 0에 가까워져 advantage≈0, 해당 그룹에서 정책 갱신 신호가 사라진다.

### 문제 3

GAE는 (학습된) value가 예측한 기댓값을 기준으로 하고, group-relative는 **동시 샘플된 동료 응답들의 보상 통계**를 기준으로 한다.

### 문제 4

정답·오답이 같은 그룹에 섞일 확률이 커져, 상대 신호가 더 자주 살아난다(비용은 샘플 $G$배).

### 문제 5

검증기가 $r_i$를 직접 주면 RM이 없어도 그룹 내 상대 advantage를 만들 수 있어, GRPO식 online 업데이트가 성립한다.

### 문제 6

$8\times 16=128$개 응답.

### 문제 7

advantage 스케일이 바뀌어 같은 lr에서도 실효 스텝 크기가 달라지기 때문이다.

## 다음 강의와 연결
그룹 상대 학습의 빈 칸은 “그 $r_i$를 어디서 얻느냐”다.  
다음 **제93강. RLVR과 Verifiable Reward**에서는 유닛테스트·수학 체커처럼 **자동으로 참/거짓을 판정할 수 있는 보상**과, 그 위에서 돌아가는 RL(및 reasoning 모델) 연결을 다룬다.

> 상대화할 점수가 깨끗할수록, 그룹 학습의 이빨이 살아난다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [91강. DPO 구현](91강_DPO_구현.md)
- **다음 강:** [93강. RLVR과 Verifiable Reward](93강_RLVR과_Verifiable_Reward.md)

<!-- /LECTURE_NAV -->
