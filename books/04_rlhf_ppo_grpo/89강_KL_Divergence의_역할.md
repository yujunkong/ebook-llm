# 89강. KL Divergence의 역할
## 이번 강에서 배우는 내용

- $\mathrm{KL}(\pi\Vert\pi_{\mathrm{ref}})$가 무엇을 재는지, 왜 방향이 중요한지
- RLHF에서 $\pi_{\mathrm{ref}}$를 보통 SFT 모델로 두는 이유
- 실무에서 자주 쓰는 형태: $r_{\mathrm{total}} = r - \beta\,\mathrm{KL}$
- $\beta$가 크면/작으면 정책이 어떻게 움직이는지
- Mode collapse·보상 해킹과 KL의 관계
- PPO 구현(88강)에서 KL이 어디에 꽂히는지

## 왜 중요한가?
RLHF의 겉모습은 “보상을 올려라”다. 속모습은 “보상을 올리되, **참조 정책에서 너무 멀어지지 마라**”다.

```text
SFT 정책 π_ref
  → (RL) 보상 r로 π를 움직임
    → 멀어질수록 언어가 붕괴·반복·해킹 위험 ↑
      → KL 항이 “멀어짐”에 비용을 붙임
```

KL을 빼 먹으면 다음이 자주 보인다.

1. 같은 구절을 반복해 보상 점수를 뽑는 **mode collapse**
2. Reward Model이 좋아하는 **허위 패턴**만 증폭
3. SFT가 지키던 형식·금지어·대화 톤이 급격히 무너짐

반대로 KL만 과도하면 정책이 **거의 안 움직인다**. 정렬의 이득이 사라진다. 이번 강의의 핵심은 “KL이 무엇인지”보다 **브레이크와 액셀의 균형**이다.

## 선수 개념
- Policy $\pi_\theta(y\mid x)$ (81~82강)
- Advantage / PPO clip (83, 87~88강)
- Reward Model $r_\phi(x,y)$ (85강)
- SFT 체크포인트가 “좋은 시작점”이라는 감각 (3권 71~77강)
- 로그 확률 $\log\pi(y\mid x)$, 시퀀스 합 $\sum_t\log\pi(y_t\mid\ldots)$

아직 깊게 들어가지 않는 것:

- DPO가 KL 제약을 **닫힌 형태**로 흡수하는 유도 (90강)
- GRPO의 그룹 상대 이득 (92강)

## 핵심 개념
### 3.1 KL Divergence 정의

이산 분포 $p,q$에 대해

$$

\mathrm{KL}(p\Vert q)=\sum_i p_i\log\frac{p_i}{q_i}

$$

성질을 세 줄로 고정한다.

1. $\mathrm{KL}(p\Vert q)\ge 0$, 등호는 $p=q$일 때만
2. **대칭이 아니다**: $\mathrm{KL}(p\Vert q)\neq\mathrm{KL}(q\Vert p)$ 일반
3. $p$가 질량을 둔 곳에서 $q$가 작으면 KL이 **크게** 오른다

정책 맥락에서는 $p=\pi_\theta(\cdot\mid x)$, $q=\pi_{\mathrm{ref}}(\cdot\mid x)$로 둔다.

$$

\mathrm{KL}\big(\pi_\theta(\cdot\mid x)\Vert\pi_{\mathrm{ref}}(\cdot\mid x)\big)
=\sum_y \pi_\theta(y\mid x)\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}

$$

직관: “지금 정책이 **자주 내는** 응답에서, 참조 정책보다 **얼마나 더/덜 확신하는가**”의 기댓값이다.

### 3.2 왜 $\mathrm{KL}(\pi\Vert\pi_{\mathrm{ref}})$인가

문헌·구현에 따라 $\mathrm{KL}(\pi_{\mathrm{ref}}\Vert\pi)$를 쓰는 변형도 있다. 이 책의 RLHF 기본선은 **forward KL**에 해당하는 $\mathrm{KL}(\pi\Vert\pi_{\mathrm{ref}})$ 계열을 기준으로 한다.

이유(직관):

- 학습 중 샘플은 $\pi$에서 뽑힌다.
- “내가 실제로 쓰는 질량”이 참조에서 멀어지는 비용을 재고 싶다.
- $\pi$가 참조가 거의 안 내는 이상한 $y$에 질량을 올리면 $\log(\pi/\pi_{\mathrm{ref}})$가 커져 벌점이 커진다.

완벽히 한 방향만 “정답”은 아니다. **어떤 KL을 근사·추정하는지**는 논문·라이브러리마다 다르다. 구현을 읽을 때는 기호만 보지 말고 **샘플이 어디서 나오는지**를 본다.

### 3.3 참조 정책 $\pi_{\mathrm{ref}}$

보통 $\pi_{\mathrm{ref}}$는 **SFT 직후 모델**을 고정(freeze)한 것이다.

```text
Pretrain → SFT → π_ref (고정 복사본)
                ↘ π_θ (RL로 학습)
```

왜 SFT인가?

1. 이미 지시 추종·형식·기본 안전이 어느 정도 들어가 있다.
2. Reward만으로 “처음부터 언어를 다시 배우게” 하지 않는다.
3. KL이 **언어 붕괴 방지용 닻** 역할을 한다.

$\pi_{\mathrm{ref}}$를 매 스텝 갱신하면 닻이 함께 움직여 KL 의미가 바뀐다. 초보 구현에서는 **고정**이 기본이다.

### 3.4 보상 변형: $r - \beta\,\mathrm{KL}$

RLHF에서 자주 쓰는 형태는 다음과 같다.

$$

r_{\mathrm{total}}(x,y)=r_\phi(x,y)-\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}

$$

기댓값으로 보면

$$

\mathbb{E}_{y\sim\pi_\theta}\big[r_{\mathrm{total}}\big]
=\mathbb{E}[r_\phi]-\beta\,\mathrm{KL}(\pi_\theta\Vert\pi_{\mathrm{ref}})

$$

즉 **보상 최대화 + KL 최소화**를 한 스칼라로 합친 것이다.

- $\beta$: KL 가중치. 클수록 $\pi_{\mathrm{ref}}$에 붙는다.
- 토큰 단위로 $\sum_t\big(\log\pi_\theta-\log\pi_{\mathrm{ref}}\big)$를 쓰는 구현이 흔하다.
- PPO의 value/advantage 계산에도 이 $r_{\mathrm{total}}$을 넣을 수 있다(88강 연결).

다른 형태도 있다.

| 형태 | 요지 |
|---|---|
| 보상에서 KL 차감 | 위 식. RM 점수에 직접 페널티 |
| 손실에 KL 가산 | $\mathcal{L}=-\mathbb{E}[A]+\beta\,\widehat{\mathrm{KL}}$ |
| 적응적 $\beta$ | 목표 KL에 맞춰 $\beta$를 조절(구현마다 상이) |

“항상 같은 한 줄 식”이 산업 표준은 아니다. **의도는 같고 배치만 다르다**고 기억한다.

### 3.5 토큰 단위 추정

전체 응답 공간 $\sum_y$는 불가능하다. 실무는 샘플 하나로 근사한다.

$$

\widehat{\mathrm{KL}}(x,y)=\log\pi_\theta(y\mid x)-\log\pi_{\mathrm{ref}}(y\mid x)

$$

$y\sim\pi_\theta$이면 이것은 $\mathrm{KL}(\pi_\theta\Vert\pi_{\mathrm{ref}})$의 **단측 추정**에 가깝다(편향·분산은 생략).

시퀀스 로그확률:

$$

\log\pi(y\mid x)=\sum_{t=1}^{|y|}\log\pi(y_t\mid x,y_{<t})

$$

패딩·프롬프트 토큰은 마스크로 제외한다.

## 직관적으로 이해하기
### 4.1 “멀리 가지 마라”의 그림

```text
π_ref ────────●────────── (안전·형식·언어)
               \
                \  β가 작음: 멀리 탐험, 보상↑ / 붕괴 위험↑
                 \
                  ● π_θ
               /
              /  β가 큼: 거의 제자리, 안전 / 정렬 이득↓
```

KL은 거리의 완벽한 미터기는 아니지만, “참조와 로그비”라는 **실용적 끈**이다.

### 4.2 Mode collapse와의 관계

Reward Model이 “감사 인사 + 이모지 폭주”에 높은 점수를 준다고 하자. KL이 없으면 정책이 그 패턴으로 **확률 질량을 몰아** 다양성이 죽는다. 이것이 mode collapse의 한 모습이다.

KL 항은 “참조가 그런 응답에 낮은 확률을 주던 습관”을 기억하게 해, **한 모드로만 쏠리는 속도**를 늦춘다. 만능 치료제는 아니다. $\beta$가 너무 작거나 RM이 심하게 편향되면 여전히 붕괴한다.

### 4.3 보상 해킹(Reward Hacking)

모델이 RM의 허점을 찾아 점수만 올리는 현상이다. KL은 해킹을 **완전히 막지 못한다**. 다만 “참조 분포 밖으로 급가속”하는 경로를 비싸게 만들어, 해킹이 **덜 극단적**이 되게 돕는다. 근본 대응은 데이터·RM·평가 다양화다(96강에서 한계를 다시 본다).

## 작은 숫자 예제
응답이 단 세 개뿐인 장난감 분포를 둔다. $x$는 고정.

| 응답 $y$ | $\pi_{\mathrm{ref}}(y)$ | $\pi_\theta(y)$ |
|---|---:|---:|
| A (정중) | 0.50 | 0.40 |
| B (보통) | 0.30 | 0.20 |
| C (아첨·반복) | 0.20 | 0.40 |

### 5.1 KL 계산

$$

\begin{align}
\mathrm{KL}
&=0.40\log\frac{0.40}{0.50}+0.20\log\frac{0.20}{0.30}+0.40\log\frac{0.40}{0.20}\\
&=0.40\log 0.8+0.20\log(2/3)+0.40\log 2
\end{align}

$$

근사값($\ln$):

- $\ln 0.8\approx -0.2231$ → $0.40\times(-0.2231)\approx -0.0893$
- $\ln(2/3)\approx -0.4055$ → $0.20\times(-0.4055)\approx -0.0811$
- $\ln 2\approx 0.6931$ → $0.40\times 0.6931\approx 0.2772$

합: $\mathrm{KL}\approx 0.1068$ nat.

해석: C에 질량을 올린 대가가 KL에 드러난다.

### 5.2 보상과 $r_{\mathrm{total}}$

RM 점수(가상): $r(A)=1.0$, $r(B)=0.5$, $r(C)=2.0$ (C가 해킹에 유리).

$\beta=0.5$일 때 샘플 $y=C$의 토큰-합 로그비:

$$

\log\frac{\pi_\theta(C)}{\pi_{\mathrm{ref}}(C)}=\log\frac{0.40}{0.20}=\log 2\approx 0.693

$$

$$

r_{\mathrm{total}}(C)=2.0-0.5\times 0.693\approx 1.653

$$

샘플 $y=A$:

$$

\log\frac{0.40}{0.50}=\log 0.8\approx -0.223,\quad
r_{\mathrm{total}}(A)=1.0-0.5\times(-0.223)\approx 1.112

$$

표면상 C의 $r$는 훨씬 큰데, KL 페널티 후 격차가 줄었다. $\beta=2.0$이면

$$

r_{\mathrm{total}}(C)=2.0-2.0\times 0.693\approx 0.614,\quad
r_{\mathrm{total}}(A)=1.0-2.0\times(-0.223)\approx 1.446

$$

이제 **정중 A가 총보상이 더 크다**. $\beta$가 “해킹 모드”를 눌러 앉히는 장면이다.

### 5.3 $\beta$ 스weep 감각

같은 $\pi_\theta$에서 $\mathbb{E}[r]-\beta\,\mathrm{KL}$을 보면($\mathbb{E}[r]=0.4\cdot1+0.2\cdot0.5+0.4\cdot2=1.3$, $\mathrm{KL}\approx0.107$):

| $\beta$ | $\mathbb{E}[r]-\beta\mathrm{KL}$ | 감각 |
|---:|---:|---|
| 0 | 1.300 | 보상만 추격 |
| 0.5 | 1.247 | 약한 끈 |
| 2 | 1.086 | 강한 끈 |
| 10 | 0.230 | 거의 움직이지 말라는 압력 |

최적 $\pi$ 자체도 $\beta$에 따라 바뀌므로, 위 표는 “현재 분포에서의 점수”일 뿐이다. 그래도 **$\beta$ 튜닝의 방향**은 이 숫자로 충분히 느껴진다.

## PPO와의 연결 (88강 복습)
PPO 업데이트의 골격:

$$

\mathcal{L}_{\mathrm{PPO}}=-\mathbb{E}\big[\mathrm{clip}(\rho_t,\hat A_t)\big]+\ldots

$$

여기에 KL이 들어가는 대표 경로 두 가지.

1. **Advantage 쪽**: $r_{\mathrm{total}}=r-\beta(\log\pi_\theta-\log\pi_{\mathrm{ref}})$로 보상을 만든 뒤 GAE
2. **손실 쪽**: 정책 손실에 $\beta\,\widehat{\mathrm{KL}}$을 더하거나, 타깃 KL을 넘기면 early stop

실무 라이브러리는 (1)(2)를 섞기도 한다. 읽을 때 체크리스트:

- `ref_model` forward는 `torch.no_grad()`인가?
- KL은 시퀀스 합인가, 토큰 평균인가?
- $\beta$는 고정인가, 적응형인가?

## 구현 스케치 (개념 코드)
```python
import torch
import torch.nn.functional as F

def sequence_logprobs(logits, labels, mask):
    """logits: [B, T, V], labels/mask: [B, T]"""
    logp = F.log_softmax(logits, dim=-1)
    token_lp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (token_lp * mask).sum(dim=-1)  # [B]

def kl_penalty(logp_pi, logp_ref, beta):
    # 샘플 y ~ π 기준의 대략적 KL 기여
    return beta * (logp_pi - logp_ref)

def total_reward(r_rm, logp_pi, logp_ref, beta):
    return r_rm - kl_penalty(logp_pi, logp_ref, beta)
```

주의:

- `logp_ref`는 참조 모델에서 계산하고 그래디언트를 끊는다.
- `logp_pi`는 정책 모델에서 오며, PPO에서는 **old policy 비율**과 함께 쓰인다(88강).
- 길이 정규화($/|y|$)를 할지 여부는 팀·논문마다 다르다. 길이가 긴 응답이 KL 합에서 불리해질 수 있다.

## 흔히 하는 실수
1. **KL 방향을 뒤집고도 같은 식이라고 착각**  
   $\log(\pi/\pi_{\mathrm{ref}})$와 $\log(\pi_{\mathrm{ref}}/\pi)$는 부호·의미가 다르다.

2. **프롬프트 토큰까지 KL에 포함**  
   보통 응답 구간만 본다. 마스크를 확인한다.

3. **$\beta=0$으로 장시간 학습**  
   보상은 오르는데 문장이 붕괴한다. 로그에 $\widehat{\mathrm{KL}}$을 항상 찍는다.

4. **참조 모델을 정책과 공유 가중치로 업데이트**  
   닻이 함께 움직여 “KL≈0”이 되어 버린다.

5. **KL만으로 안전을 해결했다고 믿기**  
   KL은 분포 근접이지, 유해성 필터가 아니다.

## LLM 학습에서의 위치
```text
84 Preference data
  → 85 Reward Model
    → 86 RLHF 파이프라인
      → 87~88 PPO
        → ★ 89 KL 브레이크
          → 90 DPO (KL 제약을 손실에 내장)
```

89강은 “명시적 RL + 명시적 KL”의 정점이다. 90강 DPO는 같은 정신을 **선호 손실 하나**로 접는다.

## Forward KL과 Reverse KL — 한 단계 더
교육용으로 $\mathrm{KL}(\pi\Vert\pi_{\mathrm{ref}})$를 기본선으로 두었다. 대칭이 아니므로 방향을 바꾸면 **벌점의 성격**이 달라진다.

$$

\mathrm{KL}(\pi\Vert\pi_{\mathrm{ref}})
=\mathbb{E}_{y\sim\pi}\Big[\log\pi(y)-\log\pi_{\mathrm{ref}}(y)\Big]

$$

$$

\mathrm{KL}(\pi_{\mathrm{ref}}\Vert\pi)
=\mathbb{E}_{y\sim\pi_{\mathrm{ref}}}\Big[\log\pi_{\mathrm{ref}}(y)-\log\pi(y)\Big]

$$

직관 대비:

| 방향 | 샘플이 나오는 곳 | 성격(거칠게) |
|---|---|---|
| $\mathrm{KL}(\pi\Vert\pi_{\mathrm{ref}})$ | $\pi$ | $\pi$가 참조에 없는 모드를 켜면 비싸다. mode-seeking 경향과 자주 묶여 설명됨 |
| $\mathrm{KL}(\pi_{\mathrm{ref}}\Vert\pi)$ | $\pi_{\mathrm{ref}}$ | 참조가 쓰던 질량을 $\pi$가 커버하지 못하면 비싸다. mass-covering 경향 |

RLHF 샘플링이 $\pi$에서 이뤄지므로, **샘플 로그비** $\log\pi-\log\pi_{\mathrm{ref}}$로 forward 쪽을 근사하는 구현이 자연스럽다.  
일부 논문·블로그는 reverse/Jensen–Shannon 등을 논의한다. 읽을 때 “어느 기댓값인가”만 확인하면 혼동이 줄어든다.

작은 숫자로 방향 차이를 본다. $\pi_{\mathrm{ref}}=(0.9,0.1)$, $\pi=(0.5,0.5)$.

$$

\mathrm{KL}(\pi\Vert\pi_{\mathrm{ref}})
=0.5\ln\frac{0.5}{0.9}+0.5\ln\frac{0.5}{0.1}
\approx 0.5(-0.588)+0.5(1.609)\approx 0.511

$$

$$

\mathrm{KL}(\pi_{\mathrm{ref}}\Vert\pi)
=0.9\ln\frac{0.9}{0.5}+0.1\ln\frac{0.1}{0.5}
\approx 0.9(0.588)+0.1(-1.609)\approx 0.368

$$

같은 두 분포인데 값이 다르다. **기호만 보고 숫자를 재사용하면 안 된다.**

## Trust Region 감각과 PPO clip의 역할 분담
고전 TRPO는 “한 스텝의 KL 반지름” 안에서 정책을 움직인다. PPO는 그 아이디어를 **clip / 페널티**로 단순화한 실용 알고리즘이다(87~88강).

LLM RLHF에서 기준점이 **두 개** 생긴다.

```text
π_ref  (SFT 닻)     … 장기·누적 이탈 제한  ← 이번 강의의 β KL
π_old  (롤아웃 정책) … 단기·한 업데이트 제한 ← PPO clip
```

비유:

- $\pi_{\mathrm{old}}$ clip: “오늘 등산, 한 발의 보폭”
- $\pi_{\mathrm{ref}}$ KL: “베이스캠프에서 너무 멀리 떨어지지 말 것”

둘 중 하나만 있으면 구멍이 난다.

- clip만: 매 스텝은 작아도 수백 스텝 후 SFT에서 멀리 갈 수 있다.
- ref KL만: 한 배치에서 ratio가 폭발해 학습이 불안정할 수 있다.

로그에는 보통 다음을 함께 남긴다.

1. $\widehat{\mathrm{KL}}(\pi\Vert\pi_{\mathrm{ref}})$ 배치 평균
2. $\widehat{\mathrm{KL}}(\pi\Vert\pi_{\mathrm{old}})$ 또는 clip hit ratio
3. reward / length / unique token ratio

## 적응적 $\beta$ — 아이디어만
고정 $\beta$ 대신 **목표 KL** $\kappa$를 두고 $\beta$를 조절하는 컨트롤러가 있다.

```text
매 N 스텝:
  측정 k̂ = mean KL(π || π_ref)
  k̂ > κ  → β를 키운다 (더 세게 붙잡기)
  k̂ < κ  → β를 줄인다 (더 탐험)
```

장점: 학습 중 보상 스케일이 바뀌어도 “얼마나 멀어질지”를 어느 정도 고정할 수 있다.  
단점: 컨트롤러 자체 하이퍼파라미터·진동. 초보 파이프라인은 **고정 β + 강한 로깅**으로 충분하다.

이 책에서는 구현 필수 과제로 두지 않는다. “그런 노브가 있다” 정도만 기억한다.

## 시퀀스 합 vs 토큰 평균
같은 응답이라도 정의가 둘로 갈린다.

$$

\mathrm{KL}_{\mathrm{sum}}=\sum_t\big(\log\pi_t-\log\pi_{\mathrm{ref},t}\big)

$$

$$

\mathrm{KL}_{\mathrm{avg}}=\frac{1}{|y|}\sum_t\big(\log\pi_t-\log\pi_{\mathrm{ref},t}\big)

$$

| 선택 | 경향 |
|---|---|
| sum | 긴 응답이 KL 페널티를 더 크게 받음 |
| avg | 길이 편향이 줄지만, “총 이탈 질량” 해석이 약해짐 |

RM 점수가 길이 상관을 이미 가지면, KL 정의와 **이중으로** 길이를 처벌/보상할 수 있다. 실험 일지에 “KL = sum or mean?”을 한 줄로 적어 둔다.

## Mode collapse를 수치로 감시하기
보상만 보면 붕괴를 늦게 발견한다. 최소한의 모니터:

| 지표 | 붕괴 시 전형 패턴 |
|---|---|
| Self-BLEU / unique bigram 비율 | 다양성 ↓ |
| 평균 응답 길이 | 비정상적 단문 또는 무한 반복 |
| 상위 1 템플릿 점유율 | 특정 오프닝이 지배 |
| $\widehat{\mathrm{KL}}$ | 급상승 또는(잘못 구현 시) 항상 ≈0 |
| RM 점수 | 오르는데 사람 평가↓ → 해킹 의심 |

“KL을 넣었으니 안전하다”가 아니라, **KL 곡선과 다양성 곡선을 같이 본다**가 실무 문장이다.

## FAQ
**Q. KL이 0이면 완벽한가?**  
A. $\pi=\pi_{\mathrm{ref}}$라는 뜻이다. 정렬 이득도 0에 가깝다. 목표는 0이 아니라 **예산 안의 이탈**이다.

**Q. 토큰 KL과 시퀀스 KL 중 무엇을 로그에 찍나?**  
A. 학습에 쓰는 정의와 **동일한 정의**를 찍는다. 다른 정의를 섞으면 튜닝이 불가능해진다.

**Q. 참조를 EMA로 천천히 따라가게 하면?**  
A. 연구·실무에 변형이 있다. 초보 과정에서는 고정 참조로 원리를 먼저 익힌다. EMA는 닻의 의미를 바꾼다.

**Q. DPO를 쓰면 이 강의는 쓸모없는가?**  
A. 아니다. DPO의 $\beta\log(\pi/\pi_{\mathrm{ref}})$가 바로 이 KL-제약 최적성의 재매개다. 89강은 90강의 해석 열쇠다.

## LLM에서는 어디에 사용될까?

이번 89강에서 배운 개념은 이후 Transformer · GPT · 서빙 강의에서 반복해서 등장합니다. 각 수식·코드 블록을 “실제 모델의 어느 단계인가”와 연결해 다시 읽어 보세요.

## 핵심 요약
- $\mathrm{KL}(\pi\Vert\pi_{\mathrm{ref}})$는 정책이 참조에서 얼마나 멀어졌는지의 (비대칭) 척도다.
- RLHF에서 $\pi_{\mathrm{ref}}$는 대개 고정 SFT 모델이다.
- $r_{\mathrm{total}}=r-\beta\,\mathrm{KL}$은 보상 추격과 분포 보존을 한 목표로 묶는다.
- $\beta$가 작으면 탐험·해킹·붕괴 위험이, 크면 정렬 정체가 온다.
- Mode collapse와 보상 해킹을 KL이 완화할 수 있지만, 데이터·RM 품질을 대체하지는 못한다.
- 구현·논문마다 KL 배치(보상 vs 손실)와 추정 방식이 다르다. **의도**를 기준으로 읽는다.

## 용어 사전
| 용어 | 한 줄 의미 |
|---|---|
| KL Divergence | $\sum p\log(p/q)$; 분포 간 비대칭 “거리” |
| $\pi_{\mathrm{ref}}$ | 참조 정책. 보통 고정 SFT |
| $\beta$ | KL 가중치(온도·끈의 세기) |
| $r_{\mathrm{total}}$ | RM 보상에서 KL 페널티를 뺀 총보상 |
| Mode collapse | 확률 질량이 소수 패턴으로 붕괴 |
| Reward hacking | RM 허점을 노려 점수만 올리는 행동 |
| Forward KL | $\mathrm{KL}(\pi\Vert\pi_{\mathrm{ref}})$ 방향 |
| Reverse KL | $\mathrm{KL}(\pi_{\mathrm{ref}}\Vert\pi)$ 방향 |
| Adaptive KL | 목표 KL에 맞춰 $\beta$를 조절하는 기법 |
| Trust region | 한 스텝 정책 이동을 KL 반경 안으로 묶는 생각 |

## 연습문제
### 문제 1（계산）

$\pi=(0.7,0.3)$, $\pi_{\mathrm{ref}}=(0.5,0.5)$일 때 $\mathrm{KL}(\pi\Vert\pi_{\mathrm{ref}})$를 $\ln$으로 계산하시오(소수 셋째 자리 근사).

### 문제 2（개념）

왜 RLHF에서 참조 모델을 **매 배치 업데이트하지 않고** 고정하는가?

### 문제 3（숫자）

$r=1.5$, $\log\pi_\theta-\log\pi_{\mathrm{ref}}=0.4$, $\beta=1.0$일 때 $r_{\mathrm{total}}$은?

### 문제 4（설계）

학습 로그에서 reward는 꾸준히 오르는데 응답이 같은 문장을 반복한다. $\beta$와 모니터링 관점에서 무엇을 의심하는가?

### 문제 5（연결）

제88강 PPO clip과 이번 KL 페널티는 둘 다 “급격한 정책 변화”를 막는다. 역할이 어떻게 다른가? 제90강으로 넘어가면 KL이 식에서 **사라지는 것처럼 보이는** 이유는 무엇이라고 예상하는가?

### 문제 6（방향）

$\pi=(0.5,0.5)$, $\pi_{\mathrm{ref}}=(0.9,0.1)$일 때 forward/reverse KL이 같은지 다른지 말하고, 실무 근사 $\mathbb{E}_{y\sim\pi}[\log\pi-\log\pi_{\mathrm{ref}}]$가 어느 쪽에 가까운지 고르시오.

### 문제 7（정의）

시퀀스 길이 100과 20인 두 응답이 토큰당 로그비가 같다면, $\mathrm{KL}_{\mathrm{sum}}$과 $\mathrm{KL}_{\mathrm{avg}}$ 중 어느 쪽이 길이에 더 민감한가?

---

## 정답 및 해설
### 문제 1

$$

0.7\ln\frac{0.7}{0.5}+0.3\ln\frac{0.3}{0.5}
=0.7\ln 1.4+0.3\ln 0.6
\approx 0.7\cdot 0.3365+0.3\cdot(-0.5108)
\approx 0.2356-0.1532
\approx 0.082

$$

### 문제 2

참조가 함께 움직이면 “멀어짐”의 기준점이 따라와 KL 제약이 약화·왜곡된다. SFT 분포라는 **고정 닻**이 필요하기 때문이다.

### 문제 3

$r_{\mathrm{total}}=1.5-1.0\times 0.4=1.1$.

### 문제 4

$\beta$가 너무 작거나 KL 추정이 빠졌을 가능성을 의심한다. $\widehat{\mathrm{KL}}$, 유니크 n-gram, 평균 길이도 함께 본다.

### 문제 5

PPO clip은 **이전 정책 $\pi_{\mathrm{old}}$** 대비 한 업데이트의 비율을 제한한다. KL 페널티는 **SFT 참조 $\pi_{\mathrm{ref}}$** 로부터의 누적 이탈을 제한한다. 시간 척도와 기준점이 다르다.  
90강 DPO는 최적 정책의 KL-제약 보상을 **선호 확률 모델로 재매개**하여, 별도 RM+RL 루프 없이 같은 정신을 손실에 넣는다. KL이 “개념적으로 사라지는” 것이 아니라 **식 안으로 흡수**된다.

### 문제 6

다르다(§11 수치 참고). 샘플이 $\pi$에서 나오면 forward 쪽 근사에 가깝다.

### 문제 7

$\mathrm{KL}_{\mathrm{sum}}$이 길이에 비례해 커지므로 더 민감하다. $\mathrm{KL}_{\mathrm{avg}}$는 같다.

## 다음 강의와 연결
이제 “명시적 보상 + 명시적 KL”의 축을 이해했다.  
다음 **제90강. DPO — Preference를 직접 학습하기**에서는 Reward Model과 PPO 루프를 우회하고, **chosen/rejected 쌍의 로그확률만으로** 정책을 직접 갱신하는 길을 유도한다. $\beta$는 그곳에서도 중심 상수로 다시 등장한다.

> KL은 RLHF의 브레이크다. 브레이크의 정신을 손실 함수 하나에 접어 넣는 것이 DPO다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [88강. PPO 구현](88강_PPO_구현.md)
- **다음 강:** [90강. DPO — Preference를 직접 학습하기](90강_DPO_Preference를_직접_학습하기.md)

<!-- /LECTURE_NAV -->
