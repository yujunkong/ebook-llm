# 74강. QLoRA
## 이번 강에서 배우는 내용

- 사실: 비트 수가 줄어들면 같은 개수의 가중치를 저장하는 용량이 줄어든다.
- 설명: 그 때문에 학습 시 “모델 가중치 메모리” 항이 작아져, 같은 GPU에서 더 큰 모델을 미세조정하기 쉬워진다（다른 항—optimizer, activation—은 별개）.
- 4-bit 베이스 + LoRA 구조의 블록 다이어그램을 그린다.
- NF4가 “정규분포에 맞춘 4-bit 코드북” 아이디어임을 고수준으로 설명한다.
- 학습과 추론에서 양자화가 다르게 쓰일 수 있음을 말한다.
- 가짜 VRAM 숫자를 단정하지 않고, 예시 스케일로만 감각을 잡는다.

## 왜 중요한가?
실무 대화에 자주 나오는 문장:

> “7B를 4-bit로 올려 LoRA 돌렸다.”

이 문장을 해체하지 않으면 다음을 혼동한다.

| 혼동 | 바로잡기 |
|---|---|
| “4-bit면 항상 4배 빠르다” | 속도는 커널·하드웨어 지원에 크게 좌우 |
| “LoRA와 양자화는 같은 것” | LoRA=어댑터, 양자화=가중치 표현 |
| “학습도 4-bit 그라디언트” | 보통 어댑터는 고정밀도로 학습 |

제75강의 실패 사례(과적합·누수)와 무관하게, **자원 제약**만으로도 QLoRA는 실험 진입 장벽을 낮춘다.

## 선수 개념
1. LoRA의 $\Delta W \approx BA$（제73강）
2. 부동소수점 비트와 메모리（대략: FP16은 2바이트/파라미터）
3. SFT 학습 루프에서 메모리를 잡아먹는 항목: **가중치, 그라디언트, optimizer state, activation**
4. “저장 용량”과 “연산 throughput”은 다른 축

## QLoRA 한 줄 정의
$$

\text{QLoRA} = \underbrace{\mathrm{Quantize}(W_0)}_{\text{동결된 저비트 베이스}} + \underbrace{\mathrm{LoRA}(A,B)}_{\text{고정밀도 어댑터 학습}}

$$

전형적인 설정:

- 베이스: 4-bit（예: NF4）로 로드, **학습하지 않음**
- LoRA: FP16/BF16（또는 FP32）파라미터로 학습
- 필요 시 일부 계층（예: norm）은 고정도 유지

Forward 개념:

```text
x  →  dequant(W0_q)로 행렬곱（또는 fused 커널）
   →  + LoRA 경로 (xA)B
   →  다음 층
```

핵심: **지식의 대부분을 담은 $W_0$는 싸게 보관**하고, **과제에 맞는 작은 방향만 $A,B$로 학습**한다.

## 비트가 줄어들면 왜 메모리가 내려가는가
### 4.1 사실 — 저장량

파라미터 하나가 차지하는 비트 수:

| 표현 | 비트/파라미터（대략） | 바이트/파라미터（대략） |
|---|---:|---:|
| FP32 | 32 | 4 |
| FP16 / BF16 | 16 | 2 |
| INT8 | 8 | 1 |
| 4-bit | 4 | 0.5 |

**사실:** 동일한 $N$개 파라미터를 4-bit로 저장하면, FP16 대비 저장 용량은 이론상 약 $1/4$ 수준이다（코드북·스케일 메타데이터는 별도 소량）.

**설명:** 그래서 “가중치 텐서를 GPU에 올리는 비용”이 줄어든다. 이것이 QLoRA가 노리는 1차 효과다.

### 4.2 설명용 스케일（단정 금지）

아래 숫자는 **특정 하드웨어 벤치마크가 아니라**, “자릿수 감각”을 위한 **설명용 예시**다.

> 예시: 파라미터가 약 $7\times 10^9$개인 모델을 FP16으로만 올리면 가중치만 대략 $14\,\mathrm{GB}$ 수준이다. 4-bit로 두면 가중치 저장은 대략 $3.5\,\mathrm{GB}$ 수준으로 내려갈 수 있다（메타데이터·정렬 오버헤드 제외한 거친 계산）.

주의:

- 학습 중에는 **activation**, **LoRA 그라디언트**, **Adam 상태** 등이 추가된다.
- 따라서 “VRAM이 정확히 X GB”라고 암기할 대상이 아니다.
- 배치 크기·시퀀스 길이·gradient checkpointing 여부에 따라 전체가 크게 요동한다.

### 4.3 메모리를 나누어 생각하기

| 항목 | Full FT (고정도) | QLoRA |
|---|---|---|
| 베이스 가중치 | 큼 | 4-bit로 축소 |
| 베이스 그라디언트 | 큼 | 보통 없음（동결） |
| Optimizer（베이스） | 매우 큼 | 없음 |
| LoRA 파라미터·상태 | — | 작음 |
| Activation | 큼 | 여전히 큼（길이·배치에 민감） |

**설명:** QLoRA의 체감 이득은 “베이스+옵티마이저” 항을 잘라내는 데서 온다. Activation이 지배적이면 체감이 덜할 수 있다.

### 4.4 메모리 항을 기호로

학습 중 대략:

$$

\mathrm{Mem}
\approx
\underbrace{M_{\mathrm{W}}}_{\text{가중치}}
+
\underbrace{M_{\mathrm{G}}}_{\text{그라디언트}}
+
\underbrace{M_{\mathrm{opt}}}_{\text{옵티마이저}}
+
\underbrace{M_{\mathrm{act}}}_{\text{activation}}

$$

| 항 | Full FT (FP16 감각) | QLoRA |
|---|---|---|
| $M_{\mathrm{W}}$ | $\approx 2N$ bytes | $\approx 0.5N$ (+메타) |
| $M_{\mathrm{G}}$ | $\sim M_{\mathrm{W}}$ 수준 | LoRA만 ($\ll$) |
| $M_{\mathrm{opt}}$ | Adam이면 $\sim 2\sim 8\times$ 가중치 감각 | LoRA만 |
| $M_{\mathrm{act}}$ | $O(B T \cdot \ldots)$ | **거의 그대로** |

$N$ = 파라미터 수. **설명용**이며 구현·오프로드·체크포인팅에 따라 달라진다.

AdamW가 파라미터당 모멘트 2개를 FP32로 두면, Full FT에서 $M_{\mathrm{opt}}$가 $M_{\mathrm{W}}$를 압도하는 경우가 많다. QLoRA는 이 항을 어댑터 크기로 축소한다.

## NF4 아이디어（고수준）
NF4(4-bit NormalFloat)는 “그냥 균등 격자”가 아니라, **정규분포에 가까운 가중치 분포**를 더 잘 표현하도록 양자화 레벨을 배치한 4-bit 형식이다.

직관만 잡자.

1. 학습된 LLM 가중치는 종종 0 근처에 몰린다.
2. 균등 구간으로 자르면, 밀도가 높은 중앙은 거칠고 꼬리는 낭비일 수 있다.
3. NF4는 **정규분포의 분위수에 맞춘 코드북**으로, 같은 4비트로 더 유용한 해상도를 노린다.

작은 숫자 비유（설명용）:

```text
균등 4-bit 느낌:  ....|....|....|....   (간격 일정)
NF4 느낌:         .|.|..|....|......    (0 근처 촘촘 — 개념도)
```

이 강의에서 코드북 표를 외울 필요는 없다. 기억할 문장:

> **사실:** NF4는 4비트로 가중치를 이산 레벨에 매핑하는 양자화 형식의 하나다.  
> **설명:** 정규분포 가정이 잘 맞을수록, 같은 비트로 재구성 오차를 줄이려는 설계다.

실무 라이브러리（예: `bitsandbytes` 계열）는 블록 단위 스케일(absmax 등)과 함께 NF4를 구현한다. 세부 커널은 버전·GPU에 따라 다르다.

### 균등 vs 분위수 — 장난감 1D

값 집합이 표준정규에 가깝다고 가정하고, 2-bit(4레벨)만으로 비교하는 **초소형** 비유:

- 균등: $[-2,2]$를 4등분 → 경계 $-2,-1,0,1,2$. 중앙($0$ 근처)과 꼬리가 같은 폭.
- 분위수: 표준정규의 25%, 50%, 75% 분위 근처에 경계를 더 촘촘히.

실제 NF4는 4-bit(16레벨)와 정규화·절대최댓값 스케일이 결합된다.  
블록 크기 $B_b$로 나눈 뒤:

$$

\hat{w}_i = s \cdot q_i,\quad
q_i\in\mathrm{Codebook}_{\mathrm{NF4}},\quad
s = \mathrm{scale}(w_{1:B_b})

$$

재구성 오차 $\|w-\hat{w}\|$를 줄이는 것이 목표(고수준).

## Double quantization 티저（선택）
QLoRA 논문/구현에는 **양자화 상수 자체를 다시 양자화**하는 double quantization 아이디어가 나온다.

- 1차: 가중치 → 4-bit
- 2차: 스케일/상수 메타데이터도 압축

초보 단계에서는 “메타데이터 오버헤드를 더 줄이려는 장치” 정도로만 알면 충분하다. 제5권 양자화 강의에서 더 깊게 다룬다.

## 학습 시 forward / backward 스케치
개념 단계:

```text
저장: W0_q (4-bit), A, B (고정도)

Forward:
  W0_hat = dequantize(W0_q)     # 연산에 필요한 정밀도로 일시 복원
  y = x @ W0_hat^T + scaling * (x @ A^T) @ B^T

Backward:
  ∂L/∂A, ∂L/∂B 만 갱신
  W0_q 는 동결
```

포인트:

1. **저장은 4-bit**, 연산 직전에 고정도로 풀어 쓰는 패턴이 흔하다（fused 커널이면 경계가 흐려질 수 있음）.
2. 그라디언트는 LoRA 쪽에 흐름 → 옵티마이저 상태도 작음.
3. 베이스를 4-bit로 둔 채 **정확한 full FT**를 하는 것은 QLoRA의 목표가 아니다.

### 수식 — QLoRA forward

동결된 양자화 가중치 $W_0^{(q)}$, 스케일 등 메타데이터로 복원한 $\widetilde{W}_0$, LoRA $A,B$, 스케일 $\alpha/r$:

$$

y = x\,\widetilde{W}_0^\top + \frac{\alpha}{r}(x A^\top)B^\top

$$

(행벡터 관례. 구현의 transpose와 맞출 것.)

그라디언트:

$$

\frac{\partial L}{\partial A},\ 
\frac{\partial L}{\partial B}
\quad\text{만 갱신},\qquad
\frac{\partial L}{\partial W_0^{(q)}}=0
$$

$\widetilde{W}_0$는 저장본이 아니라 **연산용 일시 텐서**인 경우가 많다.

### 손계산 — 메모리 비율만

$N=7\times 10^9$, FP16 가중치 $2N$ bytes vs 4-bit $0.5N$ bytes → 비율 $1/4$.  
LoRA를 $W_q,W_v$에 $r=8$, $C=4096$, $N_{\mathrm{layer}}=32$만 붙이면 학습 파라미터:

$$

32\cdot 2\cdot 8\cdot(4096+4096) = 4{,}194{,}304
$$

약 4M. Full $N$의 $\sim 0.06\%$ 수준.  
Optimizer가 이 4M에만 붙으면, Full FT의 Adam 상태와 자릿수가 다르다.

## 학습 vs 추론 — 역할을 섞지 말 것
| 단계 | 흔한 패턴 | 목적 |
|---|---|---|
| QLoRA **학습** | 4-bit 베이스 + LoRA 학습 | 메모리 안에서 SFT/도메인 적응 |
| **추론（고품질）** | 어댑터 merge 후 FP16/BF16 등 | 지연·품질 안정 |
| **추론（저메모리）** | 양자화 엔진으로 서빙 | 배포 비용 절감（5권） |

주의할 혼동:

- 학습 때 쓰던 4-bit 베이스를 **그대로** 제품 추론 경로라고 단정하지 않는다.
- merge한 뒤 다시 양자화할 수도 있고, 고정도로 서빙할 수도 있다.
- “QLoRA로 학습했으니 추론도 반드시 NF4”는 **사실이 아니다**.

### 병합 후 양자화 (선택 경로)

$$

W_{\mathrm{merged}} = \widetilde{W}_0 + \frac{\alpha}{r}BA
\quad\to\quad
\mathrm{Quantize}(W_{\mathrm{merged}})
$$

학습 때 쓰던 $W_0^{(q)}$와 **동일할 필요는 없다**. 배포 엔진·품질 요구에 따라 FP16 서빙도 흔하다.

## 최소 사용 스케치（개념）
아래는 API 암기가 아니라 **구성 요소 체크리스트**용 의사코드다.

```python
# 의사코드 — 실제 인자는 라이브러리 버전에 따라 다름
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "your-base-model",
    quantization_config=bnb_config,
    device_map="auto",
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 이후 SFT Trainer / 자체 루프로 response-only loss
```

체크:

- [ ] 베이스가 정말 양자화되어 로드되었는가
- [ ] trainable이 LoRA（+선택 norm）뿐인가
- [ ] chat template / loss mask가 제71~72강과 일치하는가

### 체크리스트를 수식으로 연결

trainable 파라미터 집합 $\Theta_{\mathrm{LoRA}}=\{A_\ell,B_\ell\}_\ell$에 대해서만

$$

\theta \leftarrow \theta - \eta\, \widehat{\nabla}_\theta L_{\mathrm{SFT}}
\quad(\theta\in\Theta_{\mathrm{LoRA}})
$$

베이스 $\Theta_{\mathrm{base}}$는 $\nabla=0$.  
`print_trainable_parameters()`가 보고하는 비율은 대략 $|\Theta_{\mathrm{LoRA}}|/(|\Theta_{\mathrm{base}}|+|\Theta_{\mathrm{LoRA}}|)$.

## 품질·안정성 트레이드오프
| 이득 | 대가·위험 |
|---|---|
| 큰 모델을 작은 VRAM에서 SFT 가능（설명） | 양자화 오차가 베이스 표현을 손상 |
| 실험 회전율↑ | 커널/버전 이슈, 재현성 관리 필요 |
| 어댑터만 저장 | merge·배포 파이프라인 설계 필요 |

실무 팁（감각）:

1. 같은 데이터로 **LoRA(고정도 베이스)** vs **QLoRA**를 소규모로 비교해 본다.
2. 평가 세트는 제75강처럼 **학습에 안 쓴 지시**로 고정한다.
3. loss만 보고 “성공”이라고 하지 않는다.

## 사실 / 설명 구분 요약
| 문장 | 유형 |
|---|---|
| 4-bit 저장은 FP16 대비 파라미터당 비트가 약 1/4이다 | **사실**（메타데이터 제외한 이상적 비율） |
| 그래서 같은 GPU에서 더 큰 베이스를 올려 SFT하기 쉬워진다 | **설명**（다른 메모리 항에 좌우） |
| NF4는 정규분포 분위수에 맞춘 4-bit 코드북 아이디어다 | **설명**（설계 동기）+ 형식 존재는 **사실** |
| QLoRA 학습 후 추론도 반드시 4-bit여야 한다 | **거짓** |
| VRAM이 항상 정확히 N GB로 줄어든다 | **단정 금지** — 예시 스케일만 |

## 수식 보강 — QLoRA와 양자화 오차 감각

QLoRA는 기본 가중치를 저비트(예: 4-bit)로 두고 LoRA만 고정밀로 학습합니다.

$$
W \approx \mathrm{Dequant}(W_{\mathrm{NF4}}) + BA
$$

양자화 오차를 $\mathcal{E}=W_0-\mathrm{Dequant}(W_{\mathrm{NF4}})$라 하면 실효 변환은

$$
h = W_0 x - \mathcal{E}x + BAx
$$

에 가깝습니다. LoRA가 $\mathcal{E}x$의 일부를 보상하도록 학습되는 셈입니다.

## 수식·정량 보강 — QLoRA 메모리

$$
\mathrm{Mem}\approx M_W+M_G+M_{\mathrm{opt}}+M_{\mathrm{act}}
$$

4-bit $M_W\approx0.5N$ bytes vs FP16 $2N$ → 이상적 $1/4$(메타 제외).

Forward:

$$
y=x\widetilde W_0^\top+\frac{\alpha}{r}(xA^\top)B^\top
$$

$\partial L/\partial W_0^{(q)}=0$, $\partial L/\partial A,\partial L/\partial B$만 학습.

손계산: $N=7\cdot10^9$ FP16≈14GB 가중치 감각 → 4-bit≈3.5GB 감각(설명용).  
LoRA $r=8,C=4096,N_{\mathrm{layer}}=32$ on $W_q,W_v$ → trainable ≈ $4.2\cdot10^6$.

NF4: 블록 스케일 $s$와 코드북 인덱스 $q_i$, $\hat w_i=s\cdot q_i$.

학습 양자화 ≠ 추론 양자화. Merge 후 BF16 서빙 가능.


## 워크드 예제 — 비트·trainable·함정

### 비트 산수

| 표현 | bytes/param |
|---|---:|
| FP32 | 4 |
| FP16 | 2 |
| 8-bit | 1 |
| 4-bit | 0.5 |

$N=7\times10^9$: FP16 가중치 $\approx14$GB 감각, 4-bit $\approx3.5$GB 감각.  
학습 VRAM ≠ 이 숫자(activation·LoRA·fragmentation).

### Trainable 세기

$C=4096$, $r=16$, layers $32$, modules $\{q,v,o\}$:

$$
\#=32\cdot3\cdot16\cdot(4096+4096)=12{,}582{,}912
$$

Base $7$B 대비 $\sim0.18\%$.

### 흔한 함정

1. 베이스 unfreeze → QLoRA가 아님  
2. loss에 프롬프트 포함 → 지시 복사  
3. “VRAM이 항상 N GB” 단정  
4. 학습 4-bit = 서빙 4-bit로 동일시

Fact: 저장 비트↓. Explanation: 같은 GPU에서 큰 베이스 SFT가 쉬워질 **수** 있음.

## LLM에서는 어디에 사용될까?

이번 74강에서 배운 개념은 이후 Transformer · GPT · 서빙 강의에서 반복해서 등장합니다. 각 수식·코드 블록을 “실제 모델의 어느 단계인가”와 연결해 다시 읽어 보세요.

## 핵심 요약
- QLoRA = 양자화된（동결）베이스 + LoRA 학습.
- 메모리 이득의 1차 출처는 **베이스 가중치 저장 비트 감소**다.
- NF4는 정규분포에 맞춘 4-bit 양자화 아이디어로 고수준 이해하면 충분하다.
- Activation·시퀀스 길이는 여전히 메모리를 크게 좌우한다.
- 학습 시 양자화와 추론 시 양자화/병합 전략을 분리해 설계한다.

## 용어 사전
| 용어 | 한 줄 의미 |
|---|---|
| Quantization | 고정도 가중치를 저비트 이산 표현으로 압축 |
| NF4 | 정규분포 가정 하의 4-bit 양자화 형식 |
| Dequantize | 연산 등을 위해 저비트 → 고정도로 복원 |
| QLoRA | 4-bit 베이스 + LoRA 미세조정 조합 |
| Compute dtype | 행렬곱 등에 쓰는 연산 정밀도（예: BF16） |
| Double quant | 양자화 상수까지 재압축하는 기법（티저） |

## 연습문제
### 문제 1（사실）

파라미터당 비트가 16→4로 줄 때, **가중치 저장량**의 이상적 비율은?

### 문제 2（구분）

다음 중 설명（해석）에 가까운 문장은?  
(a) NF4는 4비트 표현이다  
(b) 4-bit 베이스 덕분에 같은 카드에서 배치를 늘리기 쉬울 수 있다

### 문제 3（구조）

QLoRA에서 보통 그라디언트가 흐르는 파라미터와 흐르지 않는 파라미터를 나누어 쓰시오.

### 문제 4（추론）

학습은 QLoRA로 했다. 제품 추론에서 merge 후 BF16으로 서빙해도 되는가? 이유를 쓰시오.

### 문제 5（연결）

제75강으로 가기 전, QLoRA로 loss가 잘 내려가도 실패일 수 있는 이유를 한 줄로 예상하시오.

---

## 정답 및 해설
### 문제 1

약 $4/16 = 1/4$（메타데이터 무시）.

### 문제 2

(b). (a)는 형식에 대한 사실에 가깝다.

### 문제 3

흐름: LoRA $A,B$（및 설정상 일부）. 비흐름: 양자화된 베이스 $W_0$.

### 문제 4

된다. QLoRA는 학습 시 메모리 전략이며, 추론 정밀도·양자화는 별도 선택이다.

### 문제 5

지시 과적합, 평가 누수, 문체 붕괴 등 — loss ≠ 일반화（제75강）.

## 다음 강의와 연결
메모리를 줄여 SFT를 “돌릴 수 있게” 만드는 것과, SFT가 “잘된 것”은 다르다.  
**제75강. SFT 평가와 실패 사례**에서는 과적합·style collapse·데이터 누수·간단한 eval harness, 그리고 현장에서 반복되는 실패 패턴을 정리한다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [73강. LoRA](73강_LoRA.md)
- **다음 강:** [75강. SFT 평가와 실패 사례](75강_SFT_평가와_실패_사례.md)

<!-- /LECTURE_NAV -->
