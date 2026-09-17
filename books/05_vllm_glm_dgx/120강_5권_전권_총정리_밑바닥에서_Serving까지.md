# 120강. 5권·전권 총정리 — 밑바닥에서 Serving까지
## 이번 강에서 배우는 내용

- Python → … → Serving 전 경로에서 각 권이 담당한 소켓
- 5권이 답한 질문: 고정 $\theta$로 토큰을 빠르고·많이·안정적으로 생성하는 시스템
- 116~119 프로젝트가 증명한 범위와 증명한 것이 아닌 것
- 학습 성공과 서빙 성공이 다른 축이라는 최종 좌표

## 왜 중요한가?
120강은 길다. 각 권을 통과해도 머릿속에서는 조각으로 남는다. 총정리의 목적은 시험이 아니라 **좌표 재정렬**이다.

```text
조각 지식
  → 하나의 학습·정렬·추론 파이프라인
    → “지금 막히면 몇 권 몇 강으로 돌아가는가”
```

## 전권이 답하려 한 질문
한 문장:

> 텍스트를 숫자로 만들고, 미분 가능한 모델로 다음 토큰을 학습·정렬한 뒤, 그 가중치를 GPU 위에서 재현 가능하게 서빙할 수 있는가?

권별 한 줄:

| 권 | 한 줄 |
|---|---|
| 1 | 텐서 위에서 학습 루프가 돌게 한다 |
| 2 | 토큰과 Attention으로 Causal LM 뼈대를 만든다 |
| 3 | Pretrain·SFT로 지시 추종 초기 정책을 만든다 |
| 4 | 선호·보상·검증으로 정책을 다시 굽는다 |
| 5 | 고정 정책으로 Prefill/Decode 서빙 시스템을 돌린다 |

## 전권 경로 — 학습 지도
프로젝트 지침의 경로를 소켓에 붙인다.

```text
Python
  → 수학 · Tensor
    → Neural Network · Backpropagation
      → PyTorch
        → Tokenizer · Embedding
          → Attention · Transformer
            → GPT
              → Pretraining
                → SFT
                  → Preference Learning
                    → RLHF · PPO
                      → DPO · GRPO
                        → RLVR · Reasoning
                          → LLM Inference
                            → vLLM（및 동료 엔진）
                              → 모델 계열·MoE·스펙큘레이션
                                → GPU · TP · NCCL
                                  → DGX Spark 실환경
                                    → Serving 프로젝트 · GPU 최적화
                                      → 측정 리포트 · 운영 체크리스트
```

이 책의 끝점은 “더 큰 모델 이름”이 아니라 **이 경로를 스스로 다시 걸을 수 있는 상태**다.

## 권별 압축 지도
### 4.1 1권 — Python · Tensor · 수학 · PyTorch（1~26）

담당: 엔진실.

- 자료구조·NumPy·선형대수·미분
- Loss · Chain Rule · Backprop
- `Tensor` · Autograd · `nn.Module` · DataLoader · 학습 루프
- 미니 Neural Network 프로젝트（25）·총정리（26）

서빙에 남는 유산: **배치·손실·역전파를 구분하는 사고**, 실험 스크립트 습관, “숫자 배열 위의 모델”.

### 4.2 2권 — Tokenizer와 Transformer（27~54）

담당: 표현과 블록.

- Tokenization · Vocab · Embedding
- Softmax · Cross Entropy
- Q/K/V · Attention · RoPE · Norm · FFN
- Causal LM · Mini Transformer（49~50）·복잡도·변형

서빙에 남는 유산: **토큰·마스크·Attention 비용 $O(L^2)$**, KV가 무엇의 캐시인지（101로 연결）.

### 4.3 3권 — GPT Pretraining과 SFT（55~78）

담당: 대규모 학습과 지도 정렬.

- GPT · Causal LM · 생성·디코딩 손잡이
- Dataset packing · Training loop · AMP · Checkpoint · PPL
- Mini Pretrain（68）· Instruction · SFT · Chat template · LoRA/QLoRA
- 평가·역할 정리·Post-Training 입구（78）

서빙에 남는 유산: **chat template 일치**, 응답 길이·디코딩 파라미터가 지연에 미치는 영향.

### 4.4 4권 — RLHF · PPO · GRPO（79~98）

담당: 선호·보상·검증 신호.

- Policy gradient · Advantage · Preference · RM
- RLHF · PPO · KL · DPO · GRPO · RLVR · Reasoning
- Preference 프로젝트（95）·한계（96）·문헌（97）·Inference 다리（98）

서빙에 남는 유산: **정렬된 정책도 출력이 길어지면 Decode 비용이 오른다.** 보상 해킹과 서빙 메트릭은 다른 알람이다.

### 4.5 5권 — vLLM · GLM · DGX Spark（99~119）

담당: Inference · Serving · 최적화 · 운영.

이어 절에서 펼친다.

## 5권이 답하려 한 질문
> Training이 끝난 $\pi_\theta$를, Prefill/Decode·KV·배치·양자화·병렬·엔진·하드웨어 위에서 **측정 가능하고 운영 가능한 서비스**로 만들 수 있는가?

3권이 “지시 추종 모델”, 4권이 “정렬된 정책”을 만들었다면, 5권은 **그 파일을 트래픽 앞에 세우는 층**이다.

## 5권 개념 지도 — 한눈에
```text
Training ≠ Inference (99)
        │
        ▼
 Prefill / Decode (100) ──► KV Cache (101)
        │
        ▼
 Continuous Batching (102) · Quant (103)
        │
        ▼
 vLLM 구조 (104) · PagedAttention (105) · Scheduler (106)
        │
        ▼
 지표 TTFT/TPOT/Throughput (107)
        │
        ▼
 모델 계열 구조 (108) · MoE (109) · MTP/SpecDec (110)
        │
        ▼
 GPU·대역 (111) · Engine 비교 (112)
        │
        ▼
 TP (113) · NCCL/RoCE (114) · DGX Spark 환경 (115)
        │
        ▼
 Serving 프로젝트 (116)
        │
        ▼
 GPU 최적화 실험 (117) → 리포트 (118) → 운영·장애 (119)
        │
        ▼
 ★ 전권 총정리 (120)
```

## 5권 강의별 체크리스트
### 7.1 Inference 기초（99~103）

- [ ] Training과 Inference의 목표·메모리·성공 지표 차이를 말한다
- [ ] Prefill과 Decode 국면을 나눈다
- [ ] KV Cache가 왜 Decode에 필수인지 말한다
- [ ] Continuous Batching이 정적 배치와 다른 점을 말한다
- [ ] INT8/INT4/FP8이 무엇을 줄이고 무엇을 위협하는지 말한다

### 7.2 엔진·지표·모델（104~110）

- [ ] vLLM의 역할 위치를 한 줄로 말한다
- [ ] PagedAttention을 “KV 가상 메모리”로 설명한다
- [ ] 스케줄러가 요청·KV·배치를 고른다는 그림을 그린다
- [ ] TTFT·TPOT·Throughput을 정의하고 단위를 붙인다
- [ ] Dense / MoE / Hybrid 등 **계열별 구조**를 제품명 암기와 구분한다（108）
- [ ] MoE 라우팅이 서빙에 주는 부하를 말한다
- [ ] Speculative/MTP의 수락·검증 구조를 스케치한다

### 7.3 하드웨어·엔진 선택·병렬（111~115）

- [ ] SM·메모리 대역이 decode에 중요한 이유를 말한다
- [ ] vLLM · TensorRT-LLM · SGLang을 **선택 기준**으로 비교한다（112）
- [ ] Tensor Parallel 이득/통신 비용을 말한다
- [ ] NCCL/RoCE 실패 모드를 떠올린다
- [ ] 2× DGX Spark류 실환경 구성의 체크 항목을 안다（115）

### 7.4 프로젝트·운영（116~119）

- [ ] 실제 Serving 기동·호출 경로를 남겼다（116）
- [ ] Quant·Batch·Context·TP 통제 실험을 돌렸다（117）
- [ ] 11필드 리포트 템플릿으로 결과를 감사 가능하게 적었다（118）
- [ ] Pre-flight·장애 트리·롤백을 runbook으로 적었다（119）

하나라도 비면 해당 강으로 짧게 돌아간다. “전부 채울 때까지 실무 금지”는 아니다. **빈 칸을 아는 것**이 총정리의 성과다.

## 수식·정의 압축（5권）
Attention 한 줄（이미 2권）:

$$

\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V

$$

KV Cache: Decode 단계 $t$에서 $K_{1:t}, V_{1:t}$를 재사용.

대략 메모리:

$$

\mathrm{Mem}_{\mathrm{KV}} \propto B_{\mathrm{inflight}}\cdot L\cdot N\cdot d_{\mathrm{kv}}\cdot b

$$

성능 지표（정의만, 수치 없음）:

$$

\begin{aligned}
\mathrm{TTFT} &= t_{\mathrm{first\ token}} - t_{\mathrm{request}} \\
\mathrm{TPOT} &= \frac{t_{\mathrm{last}} - t_{\mathrm{first}}}{n_{\mathrm{out}}-1}\quad(n_{\mathrm{out}}>1) \\
\mathrm{Throughput} &= \frac{\#\mathrm{tokens\ or\ requests}}{\Delta t_{\mathrm{wall}}}
\end{aligned}

$$

## 116~119가 남긴 증거
| 증명한 것 | 증명한 것이 아닌 것 |
|---|---|
| 스택이 요청에 응답한다（116） | 특정 벤치 SOTA |
| 한 축씩 바꾼 비교 행렬（117） | 만능 최적 설정 |
| 11필드 감사 리포트 습관（118） | 창작된 가속 % |
| 배포·장애 runbook（119） | 무장애 보장 |

올바른 독자 자세:

> 서빙이 켜지고 측정·보고·운영 문장이 생겼으니, 이제 병목이 생기면 **몇 강 개념으로 돌아갈지** 안다.

## 학습 → 서빙 매핑（전권 통합）
| 앞에서 만든 것 | Serving에서 다시 열리는 문제 |
|---|---|
| Autograd 학습 그래프（1） | Inference 그래프·no_grad·CUDA 그래프 |
| Attention $L^2$（2） | Prefill 비용·긴 컨텍스트 TTFT |
| Chat template（3） | 학습≠서빙 문자열 사고 |
| 긴 CoT / reasoning（4） | Decode 토큰↑ · TPOT·비용 |
| 양자화（3 QLoRA / 5 INT4） | 커널·품질·VRAM 삼각관계 |
| 멀티 GPU 학습 직관 | TP/NCCL 서빙 병렬（113~114） |
| eval report（3~4） | perf report 11필드（118） |

## 추천 재학습 경로（막힐 때）
| 증상 | 돌아갈 곳 |
|---|---|
| 텐서·역전파가 흐리다 | 1권 11~20, 26 |
| Attention·마스크 | 2권 35~41, 48 |
| 학습 루프·SFT 템플릿 | 3권 62, 71~72, 78 |
| 정렬 신호·한계 | 4권 86~96, 98 |
| Prefill/KV/배치 | 5권 100~102 |
| 지표·리포트 | 107, 118 |
| 엔진·하드웨어 | 111~115 |
| 실험·운영 | 117~119 |

## 사실 / 해석 — 책 전체의 태도
이 시리즈가 반복한 규율:

1. **쉽게, 그러나 얕지 않게**
2. 직관 → 용어 → 수학 → 숫자 → 코드 → LLM 연결
3. **성능 수치 창작 금지**
4. 사실과 해석 분리
5. 챗봇 대화체로 기술서를 대체하지 않음

5권 말미에서도 동일하다. “우리 스택이 2.3×” 같은 문장은 118 템플릿의 두 행과 계산 없이 쓰지 않는다.

## 수식 보강 — 전권 한 줄

$$
\text{data}\to p_\theta(x_t\mid x_{<t})\to \text{align}\to \text{serve}(TTFT,TPOT,\mathrm{Mem})
$$

밑바닥 연산(내적·미분)이 서빙 지표까지 이어집니다.


<!-- enrich-block-120 -->
## 전권 수식 한 장

토큰화 → 임베딩 → Attention → CE → (SFT/RL) → 서빙:

$$
x=\mathrm{tok}(\cdot),\ 
e=E_x,\ 
A=\mathrm{softmax}(QK^\top/\sqrt{d}),\ 
L=-\log p(x_t\mid x_{<t})
$$

정렬:

$$
L_{\mathrm{align}}\in\{L_{\mathrm{SFT}},L_{\mathrm{PPO}},L_{\mathrm{DPO}}\}
$$

서빙 제약:

$$
\max \mathrm{TPS}\ \mathrm{s.t.}\ \mathrm{Mem}_{KV}+\mathrm{Mem}_W\le M,\ \mathrm{P99}\le\tau
$$

밑바닥 수식과 운영 지표가 같은 모델의 양면입니다.


<!-- enrich-extra-120 -->
## 전권 로드맵 한 장

| 권 | 핵심 식 |
|---|---|
| 1 | $y=xW+b$, $\nabla L$ |
| 2 | $A=\mathrm{softmax}(QK^\top/\sqrt{d})$ |
| 3 | $L_{\mathrm{CLM}}$, LoRA $BA$ |
| 4 | BT / PPO / DPO |
| 5 | KV mem, TPS, P99 |

```python
# 여정 체크: 로짓→샘플 한 줄
import torch
logits = torch.randn(5)
p = torch.softmax(logits, dim=-1)
idx = torch.multinomial(p, 1)
print(int(idx), float(p[idx]))
```

## LLM에서는 어디에 사용될까?

전권의 개념은 결국 **학습된 모델을 안정적으로 서빙**하는 일로 모입니다.

Embedding · Attention · KV Cache · Continuous Batching · 양자화 · TP는 각각 Prefill/Decode 지연과 처리량에 직접 영향을 줍니다.

## 핵심 요약 — 5권
- Inference는 Training과 목표·병목·메트릭이 다르다.
- Prefill/Decode·KV·Continuous Batching이 서빙 성능의 뼈대다.
- 엔진（vLLM 등）은 PagedAttention·스케줄로 KV와 배치를 현실화한다.
- 지표는 TTFT·TPOT·Throughput으로 분리해 보고한다.
- 모델 계열·MoE·스펙큘레이션·GPU·TP·패브릭이 선택 공간을 만든다.
- 프로젝트는 기동 → 통제 실험 → 리포트 → 운영으로 닫힌다.

## 전권 수식 포스터（추가로 손에 쥐기）
$$

\begin{aligned}
\mathrm{Attention}(Q,K,V)&=\mathrm{softmax}\!\Big(\frac{QK^\top}{\sqrt{d}}\Big)V \\
\rho&=\frac{\pi_\theta}{\pi_{\mathrm{old}}},\quad
L^{\mathrm{CLIP}}=\mathbb{E}[\min(\rho A,\mathrm{clip}(\rho)A)] \\
R&=r-\beta\,\mathrm{KL}(\pi\|\pi_{\mathrm{ref}}) \\
T_{\mathrm{total}}&\approx\mathrm{TTFT}+(n_{\mathrm{out}}-1)\mathrm{TPOT} \\
\mathrm{Throughput}&\approx\frac{N_{\mathrm{tokens}}}{\Delta t} \\
T_{\mathrm{AR}}&\gtrsim \frac{2(P-1)}{P}\frac{M}{B_{\mathrm{eff}}}
\end{aligned}

$$

위 여섯 줄이면 1~5권의 **미분 가능한 모델 → 정렬 → 서빙 → 통신**이 한 장에 붙는다.

### 학습 완료의 정의（이 책）

1. 기호를 코드·장애 트리에 대응시킨다  
2. 성능 숫자를 조건 없이 외우지 않는다  
3. 빈 칸（미측정）을 알고 있다  
4. 다음 병목의 강의 번호를 말할 수 있다  

## LLM에서는 어디에 사용될까?
전권의 개념은 결국 **학습된 모델을 안정적으로 서빙**하는 일로 모입니다. Embedding·Attention·KV·Batching·Quant·TP·NCCL은 Prefill/Decode 지연과 처리량·안정성에 직접 닿습니다. 정렬（4권）이 바꾼 길이·거부 정책은 서빙 SLO와 함께 관리합니다.

## 실습 F — 포스터 암기
위 포스터 여섯 줄을 가리고 다시 쓰시오. 막히면 해당 권으로 돌아가시오.


## 핵심 요약
- 1~2권: 미분 가능한 토큰 모델의 부품
- 3권: Pretrain·SFT로 초기 정책
- 4권: 선호·보상·검증으로 정렬
- 5권: 고정 정책을 측정·운영 가능한 Serving으로
- 끝: 데모가 아니라 **재진입 가능한 지도**

## 용어 사전
| 용어 | 한 줄 |
|---|---|
| Tensor | 다차원 배열 — 모든 계산의 용기 |
| Backpropagation | 연쇄법칙로 기울기 전달 |
| Tokenizer | 텍스트 ↔ 토큰 ID |
| Self-Attention | 서열 내 QKV 상호작용 |
| Transformer Block | Attn+FFN+잔차+정규화 |
| Causal LM | 다음 토큰 예측 언어모델 |
| Pretraining / SFT | 대량 NT / 지시 지도 미세조정 |
| Preference / RLHF / DPO | 선호 기반 정렬 경로 |
| GRPO / RLVR | 그룹 상대 / 검증 보상 |
| Prefill / Decode | 프롬프트 일괄 / 토큰 순차 |
| KV Cache | Decode용 K/V 재사용 |
| Continuous Batching | 동적 인플라이트 배치 |
| Quantization | 저비트 표현 |
| PagedAttention | 페이지 단위 KV 관리 |
| TTFT / TPOT / Throughput | 지연·토큰시간·처리량 |
| Tensor Parallel | 레이어 내 분할 |
| Perf report（11필드） | 감사 가능한 성능 행 |
| Runbook | 장애 절차서 |

## 연습문제
### 문제 1（경로）

빈칸을 채우시오.

```text
Python → 수학 → Tensor → NN → Backprop → PyTorch
 → Tokenizer → Embedding → Attention → Transformer → GPT
 → Pretraining → SFT → Preference → RLHF → PPO → DPO → GRPO
 → RLVR → Reasoning → LLM Inference → (    ) → (    ) 최적화·운영
```

### 문제 2（경계）

Training과 Inference의 성공 지표 차이를 한 줄로 쓰시오.

### 문제 3（5권）

TTFT와 Throughput을 한 점수로 합치면 안 되는 이유를 쓰시오.

### 문제 4（프로젝트）

제117강 실험에서 한 번에 축을 하나만 바꾸는 이유를 쓰시오.

### 문제 5（리포트）

성능 행 필수 11필드 중 `method`가 빠지면 생기는 문제를 쓰시오.

### 문제 6（전권）

4권에서 길어진 reasoning 출력이 5권 서빙에 주는 직접 부담을 쓰시오.

### 문제 7（태도）

이 책이 성능 숫자를 임의로 만들지 않는 이유를 “재현” 관점에서 한 문장 쓰시오.

### 문제 8（끝）

이 시리즈를 챗봇 데모로 끝내면 놓치는 산출물이 무엇인지 쓰시오.

---

## 정답 및 해설
### 문제 1

vLLM（또는 Inference Engine / Serving）, GPU（또는 DGX Spark / 하드웨어）. 동의 표현 가능.

### 문제 2

예: Training은 loss·보상·승률 등으로 $\theta$ 갱신을, Inference는 지연·처리량·안정성으로 생성 서비스를 본다.

### 문제 3

서로 다른 축（지연 vs 용량）이라 한 숫자로 합치면 회귀 판단이 오염된다.

### 문제 4

교락을 막아 원인 귀속을 가능하게 하려는 통제 실험 원칙.

### 문제 5

부하·시계·집계 절차가 불명이라 비교·재현이 불가능해진다.

### 문제 6

출력 토큰 증가 → Decode 비용·TPOT·처리량 부담 증가.

### 문제 7

환경·버전에 의존하는 수치를 창작하면 감사가 불가능하고 거짓 의사결정을 만든다.

### 문제 8

재진입 가능한 개념·실험·리포트·운영 지도（밑바닥→Serving 경로의 좌표）.

## 책을 덮은 뒤에도 남는 것
남겨야 하는 것:

- 막힐 때 돌아가는 **강 번호**
- 117 행렬과 118 템플릿 파일
- 119 runbook
- “사실 / 가설 / 측정” 문체

남겨도 되는 것:

- 특정 엔진의 단기 플래그 암기
- 특정 GPU 세대의 홍보 배수

엔진과 칩은 바뀐다. **Prefill/Decode·KV·배치·측정·통제 실험**은 더 오래 남는다.

## 다음 강의와 연결 — 다음이 아니라 닫음
4권 98강은 “다음 권”으로 문을 열었다. 5권 120강은 **시리즈의 문을 닫는다.**

추가 학습은 각자의 데이터·SLA·하드웨어에서 118 형식으로 이어가면 된다. 이 전자책의 역할은 그 작업을 **밑바닥 개념과 연결해 시작하게 하는 것**까지다.

```text
밑바닥（Python · 수학 · Tensor）
  …（모델 · 학습 · 정렬）…
    Serving（측정 · 최적화 · 운영）
```

여기까지가 《밑바닥부터 LLM》 120강의 좌표다.

<!-- enrich-120-depth -->
## 5권 핵심 식을 한 장에

Prefill / Decode:

$$
T_{\mathrm{total}}\approx\mathrm{TTFT}+(N_{\mathrm{out}}-1)\mathrm{TPOT}
$$

KV:

$$
M_{\mathrm{KV}}\approx 2\cdot L\cdot n_{\mathrm{kv}}\cdot d_h\cdot T\cdot b
$$

Continuous Batching 효율:

$$
\eta=\frac{\sum\ell_i}{B\cdot\ell_{\max}}
$$

스케줄러:

$$
\max \mathrm{tok/s}\ \mathrm{s.t.}\ \mathrm{Mem}\le M,\ \mathrm{SLO}
$$

통신:

$$
t_{\mathrm{tok}}\approx t_{\mathrm{compute}}+t_{\mathrm{comm}}
$$

### 전권 연결（1→5）

```text
텐서·미분（1）
 → 토크나이저·Transformer（2）
 → Pretrain·SFT（3）
 → RLHF/DPO（4）
 → Prefill/KV/Batching/Serving（5）
```

학습이 $\theta$를 바꾸고, 서빙은 고정 $\theta$로 **시간·메모리·통신**을 줄인다.

### 졸업 체크 10

1. TTFT/TPOT/Throughput 정의
2. KV 메모리 식의 각 기호
3. Continuous Batching이 $\eta$를 올리는 이유
4. PagedAttention이 스케줄러와 한 쌍인 이유
5. 스케줄러 제약 세 가지
6. 양자화가 대역·품질에 미치는 정성 효과
7. 스펙큘레이티브 $\alpha$와 speedup
8. NCCL allreduce 시간 모형
9. 리포트에 P99를 넣는 이유
10. 장애 런북 5단계

10개면 5권 지도가 손에서 그려진다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [119강. 실무 체크리스트와 장애 대응](119강_실무_체크리스트와_장애_대응.md)
- **다음 강:** 없음 (시리즈 끝)

<!-- /LECTURE_NAV -->
