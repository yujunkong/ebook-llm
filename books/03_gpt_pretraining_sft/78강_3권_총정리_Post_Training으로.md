# 78강. 3권 총정리 — Post-Training으로
## 이번 강에서 배우는 내용

- GPT Pretraining → 생성·최적화 → SFT·PEFT로 이어진 한 줄 파이프라인
- Mini GPT Pretrain（68）과 Mini GPT SFT（76）이 증명한 범위
- 스스로 점검할 체크리스트
- 제79강. Post-Training 지도로 넘어가기 전 준비물
- Preference / RLHF / DPO / GRPO가 왜 SFT 다음에 오는지

## 왜 중요한가?
3권은 속도가 빠르다. GPT 정의에서 시작해 학습 인프라, Mini Pretrain, Instruction Tuning, LoRA/QLoRA, 평가, Mini SFT까지 왔다. 각 강의를 통과해도 머리속에서는 조각으로 남을 수 있다.

총정리의 목적은 복습 시험이 아니라 **좌표 재정렬**이다.

```text
조각 지식
  → 하나의 Pretrain→SFT 파이프라인
    → 4권 Post-Training이 꽂힐 소켓 확인
```

## 3권이 답하려 한 질문
한 문장으로 압축하면 다음과 같다.

> Decoder-only GPT를 대규모 next-token으로 사전학습하고, 지시 데이터로 지도 미세조정하여, 평가 가능한 “지시 추종 모델”까지 가져갈 수 있는가?

2권이 “부품과 미니 Causal LM”이었다면, 3권은 **규모 있는 학습 루프와 지도 정렬**을 장착한 단계다.

## 개념 지도 — 3권 → Post-Training 경로
### 3.1 전체 파이프라인

```text
텍스트 코퍼스
  → Tokenizer · Packing
    → GPT (Causal LM) Pretraining
      → Checkpoint
        → （선택）LoRA/QLoRA 준비
          → Instruction Dataset + Chat Template
            → SFT (response-only CE)
              → Eval harness / before-after
                → ★ 여기까지가 3권
                  → Preference · Reward · PPO/DPO/GRPO (4권)
```

학습 루프의 공통 뼈대는 1~2권과 같다.

```text
배치 → forward → loss → backward → optimizer.step → (가끔) generate/eval
```

달라지는 것은 **데이터·마스크·성공 지표·움직이는 파라미터 집합**이다.

### 3.2 강의별로 꽂히는 위치

| 3권 주제 | 대표 강 | 파이프라인 위치 |
|---|---|---|
| GPT 정의·아키텍처 | 55~56 | 모델 본체 |
| Causal LM 목표·생성 | 57~59 | Pretrain 목표·디코딩 |
| 데이터·패킹 | 60~61 | Pretrain 입구 |
| Training loop·Optim·AMP | 62~64 | 학습 엔진 |
| Checkpoint·Val·PPL | 65~67 | 측정·재개 |
| Mini GPT Pretrain | 68 | 사전학습 관문 프로젝트 |
| Instruction Tuning 개념·형식 | 69~70 | SFT 데이터 |
| SFT 구현·Chat template | 71~72 | 지도 미세조정 |
| LoRA · QLoRA | 73~74 | PEFT·메모리 |
| SFT 평가·실패 | 75 | harness·회귀 |
| Mini GPT + SFT | 76 | SFT 관문 프로젝트 |
| Pretrain vs SFT 역할 | 77 | 경계 이론 |
| 총정리 | 78 | Post-Training 입구 |

표를 외울 필요는 없다. **막힐 때마다 어느 소켓을 다시 볼지** 찾는 용도다.

### 3.3 수식 한 줄로 다시 쓰기

Pretrain:

$$

L_{\mathrm{PT}} = -\sum_t \log p_\theta(x_t\mid x_{<t})

$$

SFT（응답 집합 $\mathcal{R}$）:

$$

L_{\mathrm{SFT}} = -\sum_{t\in\mathcal{R}} \log p_\theta(y_t\mid c, y_{<t})

$$

LoRA:

$$

W = W_0 + \frac{\alpha}{r}BA,\quad W_0\ \mathrm{frozen}

$$

아직 등장하지 않은 빈칸（4권 미리보기）:

$$

\text{Preference / RL:}
\quad
\text{“어떤 응답이 더 나은가”를 직접 신호로}

$$

### 3.4 코드 한 줄로 다시 쓰기

Pretrain:

```python
logits = model(x)
loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
```

SFT:

```python
logits = model(input_ids)
loss = F.cross_entropy(
    logits[:, :-1].reshape(-1, V),
    labels[:, 1:].reshape(-1),
    ignore_index=-100,  # prompt masked
)
```

생성 비교:

```python
before = generate(pretrained, prompt)
after = generate(sft_model, prompt)
```

## 두 프로젝트가 남긴 증거
| 프로젝트 | 증명한 것 | 증명한 것이 아닌 것 |
|---|---|---|
| 68 Mini Pretrain | loss↓, 패턴 흉내 생성 | 제품 지식·지시 추종 |
| 76 Mini SFT | mask·형식·before/after | 선호 정렬·보상 최적화 |

올바른 독자 자세:

> 미니 GPT로 말을 이어 쓰고, 미니 SFT로 지시에 형식적으로 답할 수 있게 되었으니, 이제 **무엇이 더 나은 답인지**를 학습하는 층으로 넘어갈 준비가 되었다.

## Post-Training으로의 다리
### 5.1 SFT의 빈칸

제77강에서 고정한 한계를 다시 적는다.

1. 단일（소수）reference CE → 동등한 좋은 답의 선호를 못 가림
2. 탐색·보상 루프 없음
3. “무난한 평균 문체”로 붕괴하기 쉬움

이 빈칸을 메우는 단계 묶음을 이 시리즈에서는 **Post-Training**이라 부른다.

### 5.2 4권에서 만날 이름（지도만）

| 이름 | 한 줄（미리보기） |
|---|---|
| Preference dataset | 같은 질문에 좋은 답/나쁜 답 쌍 |
| Reward Model | 선호를 점수로 근사 |
| RLHF | 보상 最大化 +（흔히）KL로 참조정책 근접 |
| PPO | 안정적 policy gradient 계열 최적화 |
| DPO | 보상 모델 없이 선호를 직접 정책에 |
| GRPO | 그룹 상대 비교 기반 정책 최적화 흐름 |

세부 수식·구현은 **제79강 이후**다. 지금 필요한 것은 위치 감각뿐이다.

```text
SFT 정책 π_SFT
  → 선호 데이터 / 보상
    → π_* （더 정렬된 정책）
```

### 5.3 제79강이 하는 일

**제79강. Post-Training 지도**는 4권의 입구다. 강화학습 기호（state/action/reward）로 깊게 들어가기 전에, Pretrain·SFT·Preference·RL 계열이 **한 지도에서 어디에 앉는지**를 먼저 고정한다.

## 능력 체크리스트
3권을 닫기 전, 가능하면 직접 확인한다.

### 6.1 개념

- [ ] GPT = Decoder-only Causal LM +（보통）대규모 pretrain 계보를 설명할 수 있다
- [ ] Greedy / temperature / top-k / top-p 차이를 말한다
- [ ] Packing·checkpoint·AMP·grad accumulation의 존재 이유를 말한다
- [ ] PPL과 “지시 품질”이 다른 축임을 안다
- [ ] SFT response mask를 수식·코드로 쓴다
- [ ] LoRA $\Delta W\approx BA$, QLoRA = 양자화 베이스+LoRA를 구분한다
- [ ] Overfitting / style collapse / leakage를 구분한다
- [ ] Pretrain vs SFT 할 수 있는/없는 일을 표로 말한다

### 6.2 실습

- [ ] Mini Pretrain 학습 로그·샘플 생성을 남겨 두었다
- [ ] Mini SFT before/after 아티팩트가 있다
- [ ] 초미니 harness 점수를 한 번이라도 돌려 보았다
- [ ] （선택）LoRA 적용·merge를 한 번 실험했다

하나라도 비면 해당 강으로 짧게 돌아간다. 전부 채울 때까지 4권을 금지하라는 뜻은 아니다. **빈칸을 인지한 채** 넘어가면 된다.

## 추천 재학습 경로（막힐 때）
| 증상 | 돌아갈 강 |
|---|---|
| GPT 구조가 흐리다 | 55~56, （2권）48 |
| 생성이 이상하다 | 58~59 |
| 학습이 불안정/OOM | 62~64, 74 |
| PPL만 보고 착각한다 | 66~67, 75 |
| 지시 데이터가 헷갈린다 | 69~72 |
| 마스크 버그 | 71, 76 |
| PEFT·메모리가 헷갈린다 | 73~74 |
| 역할이 섞인다 | 77 |

## 수식 보강 — Post-training으로

다음 단계는 선호/보상:

$$
\max_\pi\ \mathbb{E}[r]-\beta\mathrm{KL}(\pi\|\pi_{\mathrm{ref}})
$$

또는 DPO류로 보상 모델 없이 선호를 직접 학습합니다.


<!-- enrich-block-78 -->
## 3권 총정리 수식 카드

학습률 스케줄 예 (cosine):

$$
\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max}-\eta_{\min})\bigl(1+\cos(\pi t/T)\bigr)
$$

Gradient accumulation:

$$
g = \frac{1}{K}\sum_{k=1}^{K}\nabla L_k,\qquad
\theta \leftarrow \theta - \eta\, g
$$

Mixed precision에서 손실 스케일:

$$
L_{\mathrm{scaled}} = s\cdot L,\qquad
\nabla_\theta L = \frac{1}{s}\nabla_\theta L_{\mathrm{scaled}}
$$

LoRA 갱신:

$$
W' = W + \frac{\alpha}{r}BA
$$

다음 4권은 선호·보상·정책 경사로 넘어갑니다.


## 수학적으로 이해하기 — 3권 공식 지도

한 장에 모으면 다음과 같습니다.

$$

\begin{aligned}
&\textbf{Pretrain}&&
L_{\mathrm{PT}}=-\sum_t\log p_\theta(x_t\mid x_{<t})\\
&\textbf{PPL}&&
\mathrm{PPL}=\exp(L_{\mathrm{tok}})\\
&\textbf{SFT}&&
L_{\mathrm{SFT}}=-\sum_{t\in\mathcal{R}}\log p_\theta(y_t\mid c,y_{<t})\\
&\textbf{LoRA}&&
W=W_0+\frac{\alpha}{r}BA\\
&\textbf{QLoRA}&&
W_0\ \text{in 4-bit},\ \Delta W=\tfrac{\alpha}{r}BA\ \text{in higher prec.}
\end{aligned}

$$

생성（복습）:

$$

x_{t+1}\sim \mathrm{Categorical}\big(\mathrm{softmax}(z_t/\tau)\ \text{with top-}k/\text{top-}p\big)

$$

아직 비어 있는 4권 자리:

$$

\text{Preference:}\quad
(x,y_w,y_l)\ \mapsto\ \pi_\theta\ \text{가 } y_w\succ y_l\ \text{쪽으로}

$$

세부 손실（DPO/PPO 등）은 제79강 이후에서 전개합니다. 지금은 **자리가 비어 있다**는 좌표만 기억합니다.

## 작은 숫자 스케치 — 체크포인트 이관

Pretrain 체크포인트 크기를 아주 거칠게 스케치합니다（교육용）.

$$

\mathrm{size}
\approx
4\cdot\#\theta
\quad(\mathrm{float32\ weights\ only})

$$

$\#\theta=10^8$이면 약 $400$MB입니다. Optimizer（Adam）를 포함하면 모멘트 때문에 **약 2~3배**로 불어날 수 있습니다（구현·dtype에 따라 다름）.  
SFT로 이관할 때는 보통 **가중치（+config）**만 가져가고, Adam 모멘트는 새로 시작합니다. LoRA면 저장량은 $r$에 비례해 훨씬 작아집니다.

```text
ckpt_pt/model.pt  →  load weights
                  →  （선택）attach LoRA
                  →  SFT optim 새로 생성
```

## 직관적으로 이해하기 — 파이프라인 극장

```text
1막 Pretrain:  대본 없이 세상 이야기를 이어 씀
2막 SFT:       지시에 답하는 대본을 연습
3막 Post-Train: 어느 대사가 더 좋은지 심사·재연습（4권）
앙코르 Serving: 빠르게 무대에 올림（5권）
```

3권 총정리는 2막 커튼콜입니다. 박수는 치되, **공연이 끝났다고 말하지 않습니다**.

## Packing·마스크·평가가 한 줄에 만나는 곳

| 단계 | 효율 장치 | 품질 장치 |
|---|---|---|
| Pretrain | packing（61）, AMP（64） | val/PPL（66~67） |
| SFT | LoRA（73）, 짧은 길이 | harness（75）, mask（71） |
| 이관 | checkpoint（65） | before/after（76） |

효율만 보면 품질이 안 보이고, 품질만 보면 비용에 무너집니다. 3권은 둘을 **같은 루프**에 넣는 연습입니다.

## 학습 불안정 미니 맵（디버깅 좌표）

별도 “디버깅 전용 강”이 없어도, 증상→소켓은 고정할 수 있습니다.

| 증상 | 먼저 볼 강 |
|---|---|---|
| Loss NaN / 폭주 | 62~64（lr, AMP, accum） |
| Resume 후 거동 이상 | 65（optim/scaler） |
| Val↓인데 생성 붕괴 | 67, 58~59 |
| 지시 무시 | 71~72, 75 |
| OOM | 64, 74 |
| 점수만 높음 | 75 leakage |

## 부록 A. 3권 한 페이지 체크（인쇄용）

- [ ] GPT = Decoder-only Causal LM 계보를 설명한다
- [ ] $L_{\mathrm{PT}}$와 $L_{\mathrm{SFT}}$를 마스크로 구분한다
- [ ] Packing 효율 $\eta$ 감각（61）을 말한다
- [ ] Checkpoint에 optim을 넣는 이유를 말한다
- [ ] $\mathrm{PPL}=\exp(L)$을 쓴다
- [ ] Chat template 학습/추론 일치를 말한다
- [ ] LoRA $BA$와 QLoRA 양자화를 구분한다
- [ ] Harness와 PPL을 혼동하지 않는다
- [ ] Mini Pretrain·Mini SFT 아티팩트가 있다
- [ ] 4권 빈칸이 Preference/RL임을 안다

## 부록 B. 수식→프로젝트 매핑

| 수식 | 프로젝트 증거 |
|---|---|
| $L_{\mathrm{PT}}$ | 68강 loss 곡선 |
| generate | 68강 샘플 txt |
| $L_{\mathrm{SFT}}$+mask | 76강 ignore_index |
| before/after | 76강 비교 로그 |
| harness | 75~76 점수표 |

## 부록 C. Post-Training으로 넘기는 문장（암기 금지, 이해용）

> 3권은 $\pi_{\mathrm{SFT}}$라는 **지도 초기 정책**까지 만든다.  
> 4권은 “무엇이 더 나은 응답인가”를 데이터·보상·정책 최적화로 $\pi_{\mathrm{SFT}}$를 밀어 올린다.

## 부록 D. 워크드 — 역할이 섞인 문장 고치기

원문: “RLHF로 사전학습했다.”  
교정: “사전학습（next-token）후 SFT로 지시 추종을 심고, RLHF로 선호를 정렬했다.”

원문: “PPL이 낮아서 지시 따르기다.”  
교정: “PPL은 유창성·적합의 대리일 뿐, 지시 준수는 harness로 본다.”


<!-- enrich-extra-78 -->
## 실습 — Grad Accum 등가성

```python
# K micro-batch ≈ 큰 배치 (평균)
import torch
K = 4
params = torch.zeros(3, requires_grad=True)
opt = torch.optim.SGD([params], lr=0.1)
opt.zero_grad()
for k in range(K):
    x = torch.randn(3)
    loss = ((params - x) ** 2).mean() / K  # 평균 맞추기
    loss.backward()
opt.step()
print(params.detach())
```

$$
g=\frac{1}{K}\sum_{k=1}^K\nabla L_k
$$

마이크로 배치마다 `/K`를 빼먹으면 유효 학습률이 $K$배처럼 커집니다.

## Post-Training으로 넘길 체크

- PT 체크포인트 경로·토크나이저 버전 고정
- SFT 템플릿과 special token 문서화
- 평가셋은 학습 지시문과 분리

## LLM에서는 어디에 사용될까?

이번 78강에서 배운 개념은 이후 Transformer · GPT · 서빙 강의에서 반복해서 등장합니다. 각 수식·코드 블록을 “실제 모델의 어느 단계인가”와 연결해 다시 읽어 보세요.

## 핵심 요약
- 3권의 핵심은 GPT Pretraining과 SFT로 **지시 추종 가능한 정책의 지도적 초기값**을 만드는 것이다.
- 성공 지표는 Pretrain에서 PPL·유창성, SFT에서 준수·harness로 갈라진다.
- LoRA/QLoRA는 목적함수가 아니라 **효율적 적응 장치**다.
- Mini 프로젝트 두 개는 이론을 “돌아가게” 만든 관문이다.
- 다음은 제79강에서 Post-Training 좌표를 펴는 일이다.

## 용어 사전
| 용어 | 한 줄 의미 |
|---|---|
| GPT | Decoder-only Causal LM 계열의 대표 좌표 |
| Pretraining | 대규모 next-token 학습 |
| Sampling / Temperature | 생성 분포 조절 |
| Packing | 짧은 문서를 길게 이어 배치 효율↑ |
| AMP / Grad accumulation | 메모리·실효 배치 조절 |
| Perplexity | 토큰 불확실성 지표 |
| Instruction Tuning / SFT | 지시-응답 지도 미세조정 |
| Chat template | 역할·특수토큰 문자열 계약 |
| LoRA | 저랭크 어댑터 PEFT |
| QLoRA | 4-bit 베이스 + LoRA |
| Eval harness | 고정 문항·채점 회귀 장치 |
| Post-Training | SFT 이후 선호·RL 정렬 단계 |

## 연습문제
### 문제 1（통합）

다음 빈칸을 채우시오.

```text
Corpus → Pretrain → (    ) → SFT → Eval → (    권 Post-Training)
```

### 문제 2（손실）

SFT에서 prompt에 `ignore_index`를 쓰는 이유를 Pretrain loss와 대비해 쓰시오.

### 문제 3（PEFT）

LoRA와 QLoRA 중 “베이스 저장 비트”를 줄이는 쪽은?

### 문제 4（평가）

Train loss↓만으로 SFT 성공을 선언하면 안 되는 실패 모드 두 가지를 쓰시오.

### 문제 5（다리）

제79강으로 가기 전, 체크리스트에서 가장 약한 항목 하나와 재학습 강 번호를 지정하시오.

---

## 정답 및 해설
### 문제 1

`Checkpoint`（또는 Instruction data 준비）, `4`.

### 문제 2

Pretrain은 전 구간 next-token이 목표이고, SFT는 응답 구간 신호에 집중해 지시 추종을 학습하려는 것이 보통이다.

### 문제 3

QLoRA（양자화된 베이스）. LoRA는 주로 학습 파라미터 수를 줄인다.

### 문제 4

예: instruction overfitting, data leakage（또는 style collapse）.

### 문제 5

개인 답. 예: “QLoRA가 약하다 → 제74강”, “마스크가 약하다 → 제76강”.

## 다음 강의와 연결
3권의 문을 닫는다.

다음 책은 **4권. RLHF · PPO · GRPO**다.  
첫 강의 **제79강. Post-Training 지도**에서는, 지금까지 만든 SFT 정책 위에 Preference·Reward·RL·DPO·GRPO가 어떤 순서로 얹히는지를 **지도**로 먼저 펼친다.

Pretraining과 SFT는 그대로 가져간다.  
앞으로는 그 위에 **선호와 보상으로 정책을 다시 쓰는** 이야기를 올린다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [77강. Pretraining과 SFT의 역할 정리](77강_Pretraining과_SFT의_역할_정리.md)
- **다음 강:** [79강. Post-Training 지도](../04_rlhf_ppo_grpo/79강_Post_Training_지도.md)

<!-- /LECTURE_NAV -->
