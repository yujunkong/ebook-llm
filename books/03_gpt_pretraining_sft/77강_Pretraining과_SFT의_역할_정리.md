# 77강. Pretraining과 SFT의 역할 정리
## 이번 강에서 배우는 내용

- Pretraining = 일반적 next-token 세계모형 학습
- SFT = 지시·대화 형식에 맞춘 조건부 응답 정렬（지도）
- 각 단계가 할 수 있는 것 / 할 수 없는 것
- “데이터가 부족하면 SFT만 더” 같은 잘못된 처방이 왜 실패하는지
- 제78강·4권 Preference/RLHF로 넘어가는 빈칸이 무엇인지

## 왜 중요한가?
실무·커뮤니티 문장에는 단계가 섞여 있다.

| 흔히 하는 말 | 섞인 단계 |
|---|---|
| “모델을 학습시켰다” | Pretrain / SFT / RL 중 무엇? |
| “정렬했다” | SFT인지 Preference인지 |
| “도메인 적응” | 추가 pretrain인지 SFT인지 |

이름을 고정하지 않으면, 제76강에서 만든 미니 SFT를 “일반 지능을 얻었다”고 오해하거나, 반대로 Pretrain만으로 “지시에 잘 따른다”고 기대하게 된다.

## 한 줄 정의
$$

\begin{aligned}
\textbf{Pretraining:}&\quad
\max_\theta\ \mathbb{E}_{x\sim\mathcal{D}_{\mathrm{web}}}\sum_t \log p_\theta(x_t\mid x_{<t}) \\[4pt]
\textbf{SFT:}&\quad
\max_\theta\ \mathbb{E}_{(c,y)\sim\mathcal{D}_{\mathrm{inst}}}\sum_{t\in y} \log p_\theta(y_t\mid c, y_{<t})
\end{aligned}

$$

여기서 $c$는 instruction+input(+system) 등 **조건**, $y$는 응답이다. SFT도 수학적으로는 next-token CE지만, **데이터 분포와 마스크**가 다르다.

```text
Pretrain:  세상의 텍스트를 이어 쓴다
SFT:       주어진 지시에 대한 “바람직한 답”을 이어 쓴다
```

## 무엇을 최적화하는가
| | Pretraining | SFT |
|---|---|---|
| 데이터 | 대규모 일반 코퍼스 | 상대적으로 작은 instruction 쌍 |
| 목표 | 광범위 분포 추정 | 조건부 응답 분포 추정 |
| Loss 범위 | 거의 전 토큰 | 보통 response만 |
| 성공 신호 | PPL·생성 유창성 | 지시 준수·형식·유용성 |
| 대표 위험 | 노이즈·편향 흡수 | 과적합·style collapse·누수 |

둘 다 “크로스 엔트로피를 내린다”는 **사실**이 같다.  
무엇이 좋은 텍스트인지에 대한 **인간 선호도 비교**는 아직 직접 넣지 않는다는 **역할 한계**가 있다（그 빈칸이 4권）.

## Pretraining이 할 수 있는 것 / 없는 것
### 4.1 할 수 있는 것

1. **언어 유창성** — 문법·콜로케이션·장문 이어쓰기
2. **세계지식의 통계적 흔적** — 코퍼스에 반복된 사실·패턴
3. **다양한 스타일 모사** — 문체·장르의 이어쓰기
4. **다운스트림의 좋은 초기화** — SFT·PEFT의 출발점

제68강 Mini GPT는 규모만 작을 뿐, 같은 역할의 축소판이다.

### 4.2 할 수 없는 것（또는 매우 약함）

1. **안정적인 지시 준수** — “JSON만”, “세 줄만”을 기본 동작으로 보장하지 않음
2. **대화 역할 고정** — user/assistant 경계를 데이터 없이 학습하지 않음
3. **선호 정렬** — “두 답 중 어느 쪽이 더 나은가”를 직접 최적화하지 않음
4. **최신·사내 지식 주입** — 코퍼스에 없으면 모름（검색/툴은 별 계층）

**설명:** 프롬프트만으로도 어느 정도 지시를 따르는 경우가 있으나, 그것은 사전학습 분포에 **지시문 비슷한 패턴이 섞여 있기 때문**이지, SFT와 동일한 목적함수를 푼 결과가 아니다.

## SFT가 할 수 있는 것 / 없는 것
### 5.1 할 수 있는 것

1. **지시→응답 형식**을 기본 동작으로 만듦
2. Chat template·특수 토큰으로 **대화 인터페이스**를 고정
3. 도메인 톤·절차·출력 스키마를 **지도**로 주입
4. LoRA/QLoRA로 **저비용 적응**（제73~74강）

제76강 before/after에서 기대한 변화가 바로 이 축이다.

### 5.2 할 수 없는 것（또는 부족）

1. **비교 선호** — “A보다 B가 낫다”는 쌍 데이터가 없음（→ DPO/RLHF）
2. **보상 신호를 통한 탐색** — 시도·피드백 루프 없음（→ PPO/GRPO 등）
3. **사전학습이 심지 않은 지식의 마법 생성** — 없는 사실은 SFT로도 안 생김
4. **단일 정답 CE의 주관성 한계** — 동등한 좋은 답이 여러 개일 때 한 라벨에 묶임

**사실:** SFT는 대개 하나의（또는 소수의） reference 응답에 대한 CE다.  
**설명:** 그래서 “무난한 평균 답”으로 style collapse하기 쉽고, 미묘한 선호 차이는 Preference 단계가 담당한다.

## 잘못된 처방전과 올바른 좌표
| 증상 | 잘못된 처방 | 더 나은 좌표 |
|---|---|---|
| 지시를 무시함 | “더 길게 Pretrain” | SFT 데이터·마스크·템플릿 |
| 전문 용어를 모름 | “SFT epoch만 증가” | 도메인 코퍼스 추가 pretrain / RAG |
| 두 답 중 선호 불명 | “SFT 정답 하나 더” | Preference / Reward（4권） |
| VRAM 부족 | “포기” | LoRA/QLoRA·배치·길이 조절 |
| 벤치만 높음 | “더 학습” | 누수 검사·private harness（75강） |

## 파이프라인 위에서의 위치
```text
[2권] Causal LM 부품
   → [3권 Pretrain] 일반 next-token 능력
      → [3권 SFT] 지시 추종·대화 형식
         → [4권 Post-Training]
              Preference / Reward / PPO / DPO / GRPO ...
            → [5권] Inference · Serving
```

3권이 닫히는 지점은 “챗봇 완성”이 아니라 **지도식 정렬의 끝**이다. 그 다음에야 **선호·보상·정책 최적화**가 붙는다.

## 능력 이양 표（압축）
| 능력 | Pretrain | SFT | Post-Training(미리보기) |
|---|---|---|---|
| 유창한 이어쓰기 | ● | ○ | ○ |
| 지시 형식 준수 | △ | ● | ● |
| 단일 모범답안 모사 | △ | ● | ○ |
| 상대적 선호 반영 | ✕ | ✕ | ● |
| 보상 최대화 탐색 | ✕ | ✕ | ● |
| 저비용 도메인 적응 | △ | ●(LoRA) | ● |

범례: ● 주역할 / ○ 유지·보조 / △ 부분적 / ✕ 비목표

## 제76강 경험을 이론에 매핑
| 프로젝트에서 본 것 | 역할 해석 |
|---|---|
| Pretrain 모델의 장황·무시 | 지시 분포 미정렬 |
| Mask 후 형식 학습 | SFT의 핵심 신호 |
| Epoch 과다 시 암기 | SFT 일반화 한계 |
| Harness 점수 | Pretrain PPL과 다른 성공 정의 |

미니 스케일에서도 **역할 차이**는 재현된다. 규모가 커져도 논리적 경계는 같다.

## 실무 의사결정 트리（짧은）
```text
문제가 “말을 이어 쓰지 못함 / PPL 붕괴”인가?
  └─ Yes → 아키텍처·pretrain·데이터 품질
  └─ No
      문제가 “지시를 안 따름 / 형식 붕괴”인가?
        └─ Yes → SFT 데이터·템플릿·mask·PEFT
        └─ No
            문제가 “어느 답이 더 나은지 / 미묘한 톤·안전”인가?
              └─ Yes → 4권 Preference / RL
              └─ No → 평가·디코딩·툴/RAG 등 다른 층
```

## 수식 보강 — Pretrain vs SFT

Pretrain: $\mathcal{D}_{\mathrm{web}}$ 위 무조건부 LM.

SFT: $\mathcal{D}_{\mathrm{instr}}$ 위 조건부 응답 LM.

$$
p_{\mathrm{pt}}(x)\ \text{ vs }\ p_{\mathrm{sft}}(y\mid c)
$$

같은 CE 뼈대, 다른 데이터 분포입니다.


## 수학적으로 이해하기 — 목적함수 좌표

Pretraining과 SFT는 둘 다 Negative Log-Likelihood(NLL) 계열이지만, 기댓값의 **측도(measure)** 가 다릅니다.

$$

\begin{aligned}
L_{\mathrm{PT}}(\theta)
&= \mathbb{E}_{x\sim\mathcal{D}_{\mathrm{PT}}}
\Big[-\sum_{t=1}^{|x|}\log p_\theta(x_t\mid x_{<t})\Big] \\[6pt]
L_{\mathrm{SFT}}(\theta)
&= \mathbb{E}_{(c,y)\sim\mathcal{D}_{\mathrm{SFT}}}
\Big[-\sum_{t\in\mathcal{R}(c,y)}\log p_\theta(y_t\mid c,y_{<t})\Big]
\end{aligned}

$$

여기서 $\mathcal{R}(c,y)$는 응답(assistant) 토큰 위치 집합입니다. 마스크를 $m_t\in\{0,1\}$로 쓰면

$$

L_{\mathrm{SFT}}
= \mathbb{E}\Big[-\sum_t m_t\log p_\theta(x_t\mid x_{<t})\Big],
\quad
m_t=\mathbf{1}[t\in\mathcal{R}]

$$

이 됩니다. Pretrain은 대개 $m_t=1$（패딩 제외）이고, SFT는 $m_t$가 응답에만 1입니다.

### 분포 관점 한 줄

- Pretrain: $p_\theta$를 웹·도서 등 **광역 텍스트 분포**에 맞춥니다.
- SFT: $p_\theta(\cdot\mid c)$를 **지시 조건 아래의 모범 응답 분포**에 맞춥니다.

같은 Softmax·같은 CE라도, 맞추는 대상 분포가 다르면 **역할이 갈라집니다**.

## 작은 숫자 스케치 — 마스크가 신호를 바꾸는 방식

길이 8 토큰의 미니 시퀀스를 가정합니다.

```text
ids:   [SYS] [U1] [U2] [U3] [A1] [A2] [A3] [EOS]
mask:    0     0    0    0    1    1    1    1
```

각 위치의 순간 NLL이 모두 $1.0$ nat라고 하면（설명용）:

- Pretrain식（전 구간）: $L = 8\times 1.0 = 8.0$
- SFT식（응답 4칸）: $L = 4\times 1.0 = 4.0$

평균을 “마스크된 토큰 수”로 나누면 둘 다 $1.0$이지만, **기울기가 흐르는 위치**가 다릅니다. 사용자 구간을 맞추는 방향으로는 SFT 기울기가 거의 없습니다（의도）.

토큰 평균 vs 샘플 평균을 섞으면 로그가 왜곡됩니다. 교육용으로는

$$

L_{\mathrm{tok}} = \frac{\sum_t m_t(-\log p_t)}{\sum_t m_t}

$$

처럼 **마스크 가중 토큰 평균**을 하나의 규약으로 고정하는 편이 안전합니다.

## 직관 카드 — “세계모형”과 “대본 배우”

```text
Pretraining
  = 많은 책을 읽어 “다음에 올 말”을 맞히는 연습
  → 유창성·지식의 통계적 흔적

SFT
  = 특정 대본(지시→응답)을 반복 연습
  → 형식·톤·준수 습관

Preference / RL (4권)
  = “어느 대사가 더 좋은가”를 비교·보상으로 다듬기
```

미니 실험（제68·76강）에서도 이 비유는 유효합니다. 규모만 작을 뿐 **목적함수 좌표**는 같습니다.

## 능력 이양 — 정량 스케치（가짜 벤치 금지）

숫자 벤치를 주장하지 않습니다. 대신 **신호의 종류**만 표로 고정합니다.

| 관찰 신호 | Pretrain에서 의미 | SFT에서 의미 |
|---|---|---|
| Train CE ↓ | 코퍼스 암기·적합 진행 | 응답 패턴 적합 진행 |
| Val CE ↓ | 일반 텍스트 일반화 | （약함）형식 일반화의 거친 대리 |
| PPL | 불확실성 지표（67강） | 지시 품질 대리로 부적절하기 쉬움 |
| Harness 준수율 | 거의 무관 | 주력 성공 지표（75강） |
| Before/After 생성 | 유창성 샘플 | 지시 추종 변화의 증거 |

**사실:** 같은 CE 숫자라도 단계마다 해석이 다릅니다.  
**설명:** 지표를 단계에 맞게 고르는 것이 “역할을 섞지 않는” 실무의 첫걸음입니다.

## 잘못된 처방 — 워크드 의사결정

### 사례 A. “전문 용어를 모른다”

```text
증상: 의료/법률 용어를 엉터리로 이어 씀
잘못된 처방: SFT epoch만 5→20
이유: SFT는 없는 지식을 창조하지 못함
좌표:
  1) 도메인 코퍼스 추가 pretrain（또는 지속 pretrain）
  2) RAG/툴로 근거 문서 연결
  3) 그 다음 짧은 도메인 SFT로 형식 고정
```

### 사례 B. “JSON만 출력하라”를 무시

```text
증상: 설명문을 앞에 붙임
잘못된 처방: 더 긴 pretrain
좌표: 템플릿 일치（72）+ response mask（71）+ 형식 문항 harness（75）
```

### 사례 C. “두 답 중 어느 쪽이 더 친절한가”

```text
증상: 둘 다 형식은 맞지만 톤 선호가 갈림
잘못된 처방: 정답 하나를 더 넣어 SFT
좌표: Preference / DPO / RLHF（4권）— CE 단일 라벨의 한계
```

## 수식 보강 — KL로 보는 “너무 먼 SFT”

참조 정책 $\pi_{\mathrm{ref}}$（보통 SFT 직전 또는 SFT 자체）에서 너무 멀어지면 유창성이 깨질 수 있습니다. 4권 RLHF는 흔히

$$

\max_\pi\ \mathbb{E}[r] - \beta\,\mathrm{KL}(\pi\|\pi_{\mathrm{ref}})

$$

형태의 제약을 둡니다. 오늘 강의에서 기억할 한 줄은 다음뿐입니다.

> SFT는 $\pi_{\mathrm{ref}}$에 가까운 **지도 초기 정책**을 만들고, Post-Training은 그 근처에서 선호를 밀어 올립니다.

## 연결 복습 — 2권·3권 소켓

| 소켓 | 강의 | Pretrain/SFT에서 하는 일 |
|---|---|---|
| Causal LM 분해 | 32, 48, 57 | 공통 next-token 뼈대 |
| Packing | 61 | Pretrain 효율 |
| Checkpoint | 65 | 단계 이관（PT→SFT） |
| PPL | 67 | Pretrain 모니터 |
| Chat template | 72 | SFT 입출력 계약 |
| LoRA | 73~74 | SFT 비용 축소 |
| Harness | 75 | SFT 성공 정의 |

## 자가 점검（합니다체 체크）

- [ ] Pretrain/SFT 목적함수를 마스크 기호로 다시 쓸 수 있습니다.
- [ ] “둘 다 CE”와 “역할이 같다”를 혼동하지 않습니다.
- [ ] 지식 부족 vs 형식 불이행 vs 선호 모호를 처방으로 구분합니다.
- [ ] 제76강 before/after를 역할 표에 매핑할 수 있습니다.
- [ ] 4권으로 넘길 빈칸이 “비교 선호/보상”임을 한 문장으로 말합니다.

## 부록 A. 한 페이지 공식 카드

$$

\begin{aligned}
&\textbf{PT:}&&
L=-\sum_t \log p_\theta(x_t\mid x_{<t}) \\
&\textbf{SFT:}&&
L=-\sum_{t\in\mathcal{R}}\log p_\theta(y_t\mid c,y_{<t}) \\
&\textbf{LoRA:}&&
W=W_0+\tfrac{\alpha}{r}BA \\
&\textbf{빈칸(4권):}&&
\text{preference / reward / policy opt.}
\end{aligned}

$$

## 부록 B. “SFT만으로 충분?” 인터뷰 질문

1. 성공을 어떤 **관측 가능 지표**로 정의하는가?
2. 그 지표는 Pretrain PPL과 독립인가?
3. 동등하게 좋은 답이 여러 개인가? → Yes면 Preference 후보
4. 지식이 코퍼스에 있는가? → No면 검색/추가 pretrain 후보
5. 학습·추론 템플릿이 동일한가?

예/아니오로만 답해도, 단계 혼동을 상당수 줄일 수 있습니다.

## 부록 C. 미니 수치 — 응답 비율과 유효 신호

시퀀스 길이 $T=512$, 응답 길이 기대값 $\mathbb{E}[|\mathcal{R}|]=64$이면 응답 비율은

$$

\rho = \frac{64}{512} = 0.125

$$

입니다. 전 구간 CE를 쓰면 신호의 약 $87.5\%$가 프롬프트 쪽에 흐를 수 있습니다（대략）. 이것이 response-only mask를 쓰는 **구조적 이유**입니다. 실제 $\rho$는 데이터마다 다르므로, 로거에

```text
mean_response_tokens / mean_seq_tokens
```

를 남기는 습관을 권합니다.


<!-- enrich-block-77 -->
## Pretraining vs SFT — 목적함수 대비

Pretraining (CLM):

$$
L_{\mathrm{PT}} = -\mathbb{E}_{x\sim\mathcal{D}_{\mathrm{web}}}\sum_t \log p_\theta(x_t\mid x_{<t})
$$

SFT:

$$
L_{\mathrm{SFT}} = -\mathbb{E}_{(q,a)}\sum_{t\in a}\log p_\theta(x_t\mid x_{<t})
$$

차이는 **데이터 분포**와 **손실을 주는 토큰 범위**(응답 구간 마스킹)입니다.

### 분포 이동을 한 줄로

$$
p_{\mathrm{PT}}(x) \;\rightarrow\; p_{\mathrm{SFT}}(x\mid \text{instruction style})
$$

### 언제 어떤 손실인가?

| 단계 | 목표 | 전형 손실 |
|---|---|---|
| PT | 언어 통계 | next-token CE |
| SFT | 지시 따르기 | response-mask CE |
| RM/RL | 선호 정렬 | BT / PPO / DPO |


## 부록 D. 목적함수 차원 맞추기

배치 안에서 샘플마다 응답 길이가 다르면, 샘플 평균과 토큰 평균이 어긋납니다.

$$

L_{\mathrm{sample}}=\frac{1}{B}\sum_{b=1}^{B} L^{(b)},
\qquad
L_{\mathrm{token}}=\frac{\sum_{b,t} m_{b,t}(-\log p_{b,t})}{\sum_{b,t} m_{b,t}}
$$

짧은 응답만 많은 데이터에서는 $L_{\mathrm{sample}}$이 낙관적으로 보일 수 있습니다. 로거에는 둘 다, 또는 토큰 평균+평균 응답 길이를 남깁니다.

## 부록 E. “정렬” 단어 사용 규칙

| 말 | 이 시리즈에서의 권장 의미 |
|---|---|
| SFT / Instruction Tuning | 지도식 지시 추종 |
| Preference alignment | 선호 쌍 기반 정렬（4권） |
| RLHF | 보상+RL 기반 정렬（4권） |
| “정렬했다”（모호） | **금지에 가깝게 피하고** 단계를 명시 |

혼동은 기술 문제가 아니라 **용어 위생** 문제입니다.


<!-- enrich-extra-77 -->
## 실습 — 응답 구간만 손실

```python
# SFT: prompt 토큰은 loss mask=0
import torch
import torch.nn.functional as F

# labels: prompt=-100, response=token id
logits = torch.randn(2, 8, 50)   # (B,T,V)
labels = torch.tensor([
    [-100, -100, -100, 3, 4, 5, 6, 7],
    [-100, -100, 9, 10, 11, 12, 13, 14],
])
loss = F.cross_entropy(
    logits.reshape(-1, 50),
    labels.reshape(-1),
    ignore_index=-100,
)
print(loss.item())
```

### 수식으로 쓰면

$$
L_{\mathrm{SFT}}=-\sum_{t:\,m_t=1}\log p_\theta(x_t\mid x_{<t})
$$

$m_t$가 응답 마스크입니다. Pretraining은 보통 $m_t=1$ for all $t$.


<!-- enrich-pass-1f64 -->
## 역할 좌표 — 한 장 더

Pretraining은 **세계/언어 통계**를, SFT는 **지시-응답 프로토콜**을 맡습니다. 같은 CE라도 기대하는 분포가 다릅니다.

$$
\theta_{\mathrm{PT}}
=\arg\min_\theta
\mathbb{E}_{x\sim\mathcal{D}_{\mathrm{PT}}}
\big[-\sum_t\log p_\theta(x_t\mid x_{<t})\big]
$$

$$
\theta_{\mathrm{SFT}}
=\arg\min_\theta
\mathbb{E}_{(c,r)\sim\mathcal{D}_{\mathrm{SFT}}}
\big[-\sum_{t\in r}\log p_\theta(x_t\mid x_{<t})\big]
$$

### 능력 이양（정성）

| 능력 | PT | SFT |
|---|---|---|
| 유창성·지식 스케치 | 주력 | 보존·소폭 조정 |
| 형식 준수 | 약함 | 주력 |
| 거절/안전 경계 | 거의 없음 | 데이터에 의존 |
| 선호 미묘함 | 없음 | 부족 → 4권 |

## 구현 스케치 — “어느 손실을 쓰나” 스위치

```python
def loss_for_stage(stage, logits, labels, prompt_mask=None):
    if stage == "pt":
        return token_ce(logits, labels)  # 거의 모든 토큰
    if stage == "sft":
        labels = labels.masked_fill(prompt_mask.bool(), -100)
        return token_ce(logits, labels)
    raise ValueError(stage)
```

## 실패 모드 — 역할 혼동

| 잘못된 처방 | 왜 위험한가 | 올바른 좌표 |
|---|---|---|
| SFT만으로 지식 주입 | 환각·암기 | PT 데이터/검색 |
| PT만으로 챗봇 | 지시 무시 | SFT/템플릿 |
| Loss↓=정렬 완료 | 선호 미반영 | 4권 preference |
| 벤치 만점=일반화 | 누수 가능 | private harness |

## 실습 코드 — 마스크 비율로 단계 식별

```python
def stage_fingerprint(mask_ratio):
    if mask_ratio > 0.9:
        return "likely_pretrain_or_leak"
    if 0.1 <= mask_ratio <= 0.7:
        return "likely_sft"
    return "check_pipeline"
```

절대 임계값은 데이터에 따라 다릅니다. **상대 비교**로 쓰세요.

## 수식 보강 — 너무 먼 SFT（KL 스케치）

참조 정책 $\pi_{\mathrm{ref}}$（보통 PT 또는 SFT 초기）에 대해

$$
\mathrm{KL}(\pi_\theta\|\pi_{\mathrm{ref}})
$$

가 과도하면 유창성이 손상될 수 있습니다. 4권의 KL 페널티·DPO가 이 자리를 채웁니다. 지금은 “SFT도 분포를 옮긴다”는 좌표만 기억하세요.

$$
\pi_{\mathrm{PT}}\;\xrightarrow{\mathrm{SFT}}\;\pi_{\mathrm{SFT}}\;\xrightarrow{\mathrm{pref}}\;\pi_{\mathrm{aligned}}
$$

## LLM에서는 어디에 사용될까?

이번 77강에서 배운 개념은 이후 Transformer · GPT · 서빙 강의에서 반복해서 등장합니다. 각 수식·코드 블록을 “실제 모델의 어느 단계인가”와 연결해 다시 읽어 보세요.

## 핵심 요약
- Pretraining은 일반 next-token으로 **유창성과 기초 지식 분포**를 만든다.
- SFT는 지시 조건 아래 **모범 응답**을 지도 학습한다.
- SFT는 선호 비교·보상 탐색을 대체하지 않는다.
- 증상별로 Pretrain / SFT / Post-Training 처방을 섞지 않는다.
- 다음 총정리（78강）에서 3권 전체를 Post-Training 입구에 연결한다.

## 용어 사전
| 용어 | 한 줄 의미 |
|---|---|
| Pretraining | 대규모 코퍼스 next-token 학습 |
| SFT | Instruction 쌍에 대한 지도 미세조정 |
| Alignment（좁은 의미） | 인간 의도에 맞게 동작 조정（단계 혼용 주의） |
| PEFT | LoRA 등 파라미터 효율 미세조정 |
| Post-Training | SFT 이후 Preference/RL 계열 단계 총칭 |

## 연습문제
### 문제 1（정의）

Pretrain과 SFT의 목적함수가 “둘 다 CE”인데도 역할이 갈라지는 이유를 데이터·마스크 관점에서 쓰시오.

### 문제 2（한계）

SFT만으로 해결하기 어려운 요구사항 한 가지를 쓰고, 4권의 어느 방향과 연결되는지 말하시오.

### 문제 3（처방）

도메인 용어를 모르는데 SFT epoch만 늘리는 것이 약한 이유를 쓰시오.

### 문제 4（표）

지시 형식 준수의 주역할 단계에 ●를 어디에 두어야 하는가?

### 문제 5（연결）

제78강 총정리가 “챗봇 완성 선언”이 되면 안 되는 이유를 한 줄로 쓰시오.

---

## 정답 및 해설
### 문제 1

최적화하는 텍스트 분포가 다르고（일반 코퍼스 vs 지시-응답）, SFT는 보통 응답 구간에만 loss를 걸어 학습 신호가 다르다.

### 문제 2

예: “두 응답 중 선호 반영” → Preference / DPO / RLHF. 또는 보상 기반 탐색 → PPO/GRPO.

### 문제 3

SFT는 없는 지식을 창조하지 못하고, 좁은 응답 패턴 과적합만 키울 수 있다. 도메인 코퍼스·검색 등이 필요하다.

### 문제 4

SFT（●）. Pretrain은 △ 수준.

### 문제 5

3권은 지도식 정렬까지이며, 선호·RL Post-Training과 서빙이 남아 있기 때문이다.

## 다음 강의와 연결
역할 경계가 선명해졌다.  
**제78강. 3권 총정리 — Post-Training으로**에서는 55~77강을 한 지도에 다시 올리고, 체크리스트를 만든 뒤 **제79강. Post-Training 지도**로 넘긴다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [76강. 프로젝트 — Mini GPT + SFT](76강_프로젝트_Mini_GPT_SFT.md)
- **다음 강:** [78강. 3권 총정리 — Post-Training으로](78강_3권_총정리_Post_Training으로.md)

<!-- /LECTURE_NAV -->
