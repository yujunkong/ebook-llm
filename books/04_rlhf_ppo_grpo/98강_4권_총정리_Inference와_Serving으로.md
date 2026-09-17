# 98강. 4권 총정리 — Inference와 Serving으로
## 이번 강에서 배우는 내용

- Post-Training이 Pretrain·SFT 위에 얹는 신호의 종류
- Preference / RM / PPO / DPO / GRPO / RLVR / Reasoning의 소켓 위치
- 제95강 프로젝트가 증명한 범위와 증명한 것이 아닌 것
- 정렬된 체크포인트가 서빙에서 다시 열리는 문제（지연·배치·KV·처리량）
- 왜 다음 권이 Training이 아니라 Inference·Serving인가

## 왜 중요한가?
4권까지 끝나면 “정렬된 모델을 얻었다”고 느끼기 쉽다. 그러나 사용자가 만나는 것은 학습 루프가 아니라 **매 요청의 prefill·decode**다.

```text
4권:  π_θ 를 선호·보상·검증 신호로 갱신한다
5권:  같은 π_θ 로 토큰을 빠르게·많이·안정적으로 생성한다
```

정렬 품질과 서빙 품질은 다른 축이다. 거부 정책이 훌륭해도 TTFT가 길면 제품은 실패하고, 처리량이 높아도 보상 해킹된 정책이면 제품은 실패한다. 총정리는 두 축이 **직렬로 붙는다**는 것을 명시한다.

## 4권이 답하려 한 질문
한 문장:

> SFT 이후, “무엇이 더 나은 응답인가 / 무엇이 맞는 응답인가”를 신호로 정책을 갱신하는 post-training을, 수식·구현·실패 모드·미니 실험까지 관통할 수 있는가?

3권이 “지시 추종 가능한 초기 정책”을 만들었다면, 4권은 **선호·보상·검증으로 그 정책을 다시 굽는 층**이다.

## 개념 지도 — 한눈에
```text
Pretrain → SFT (3권)
              │
              ▼
     Preference dataset (84)
              │
        ┌─────┴──────┐
        ▼            ▼
   Reward Model    Direct preference (DPO 90~91)
        │
        ▼
   RLHF + PPO/KL (86~89)
        │
        ▼
   GRPO 그룹 상대 (92) ──┐
                         ├─ policy updates
   RLVR 검증 보상 (93) ──┤
   Reasoning signals(94)─┘
              │
              ▼
     한계·부작용 (96) · 문헌 지도 (97)
              │
              ▼
     ★ Serving / Inference (5권)
```

수식 압축:

$$

\begin{aligned}
\text{SFT:}&\quad \max_\theta \mathbb{E}\log\pi_\theta(y\mid x) \\
\text{RM:}&\quad r_\phi(x,y_w) > r_\phi(x,y_l) \\
\text{RL:}&\quad \max_\theta \mathbb{E}[r] - \beta\,\mathrm{KL}(\pi_\theta\|\pi_{\mathrm{ref}}) \\
\text{DPO:}&\quad \min_\theta L_{\mathrm{DPO}}(\pi_\theta;\pi_{\mathrm{ref}},\mathcal{D}_{\mathrm{pref}}) \\
\text{RLVR:}&\quad r = V(x,a)\ \text{or process}
\end{aligned}

$$

## 강의별 체크리스트
### 4.1 기초·RLHF 코어（79~89）

- [ ] Post-Training이 Pretrain/SFT와 다른 **목표·데이터·성공 지표**를 쓴다
- [ ] State / Action / Reward / Policy를 LLM 토큰 생성에 대응시킨다
- [ ] Policy gradient와 Advantage의 역할을 한 줄로 말한다
- [ ] Preference pair와 BT식 RM 손실을 스케치한다
- [ ] RLHF 루프（생성→보상→업데이트）를 그린다
- [ ] PPO clip / 안정의 직관을 말한다
- [ ] KL to reference가 붕괴·과도 이동을 막는 이유를 말한다

### 4.2 직접 선호·그룹·검증（90~94）

- [ ] DPO가 RM+RL을 어떻게 우회하는지 말한다
- [ ] DPO 구현에서 ref freeze·logprob 정의를 안다
- [ ] GRPO를 “그룹 상대 baseline”으로 위치시킨다
- [ ] RLVR의 보상이 인간 선호와 다르다는 것을 안다
- [ ] Outcome vs Process reward를 구분한다

### 4.3 프로젝트·한계·지도（95~97）

- [ ] `ch95_preference_rl`에서 margin before/after를 남겼다
- [ ] （선택）RM accuracy·REINFORCE 로그를 남겼다
- [ ] Reward hacking / sycophancy / over-refusal / shift / eval limits를 구분한다
- [ ] InstructGPT류 → DPO류 → reasoning RL을 **사실/해석 구분** 하에 설명한다

하나라도 비면 해당 강으로 짧게 돌아간다. 전부 채울 때까지 5권을 금지하라는 뜻은 아니다.

## 제95강이 남긴 증거
| 증명한 것 | 증명한 것이 아닌 것 |
|---|---|
| 선호 신호→DPO 손실→margin 이동 | 실제 LLM 정렬 품질 |
| （선택）BT-RM + REINFORCE 스케치 | 안정적 PPO 프로덕션 |
| 라벨 뒤집기·β 극단의 실패 감각 | 벤치 SOTA |
| 평가 report 습관 | 자동 judge 일반론 완료 |

올바른 독자 자세:

> 미니 선호 루프가 돌아가니, 이제 그 체크포인트를 **서비스 트래픽 앞**에 세울 때 무엇이 병목인지 배울 준비가 되었다.

## 정렬 체크포인트 → 서빙 요구로 매핑
학습이 끝난 $\pi_\theta$는 파일이다. 서빙은 그 파일에 대한 **반복 생성 시스템**이다.

| 4권에서 만든 것 | 5권에서 다시 열리는 문제 |
|---|---|
| 긴 CoT / reasoning 궤적 | Decode 길이↑ → TPOT·비용↑ |
| 샘플 다수@train（GRPO류） | 온라인은 보통 1경로; 평가만 다샘플 |
| 거부·안전 정책 | 프롬프트 분류·가드와 파이프라인 결합 |
| Chat template·특수 토큰 | 토크나이저·템플릿 일치（학습≠서빙 불일치 사고） |
| KL로 묶인 문체 | temperature/top-p와 상호작용 |
| RM/judge（오프라인） | 온라인 모니터링은 다른 지표（지연·에러·환각 신고） |

특히 reasoning 정렬 후 **평균 출력 토큰 수**가 늘면, 동일 GPU에서 처리량이 떨어진다. 정렬 성공이 곧 서빙 부하 증가로 이어질 수 있다. 이 결합을 5권 성능 지표（TTFT, TPOT, Throughput）에서 수치로 다룬다 — 지금 숫자를 날조하지 않는다.

## Training vs Inference — 미리 보는 경계（제99강 예고）
| | Training（~4권） | Inference（5권~） |
|---|---|---|
| 목표 | $\theta$ 갱신 | $y\sim\pi_\theta$ 생성 |
| 배치 | 학습 배치·롤아웃 | 요청 배치·연속 배치 |
| 메모리 | 옵티마이저·그라드·활성화 | KV cache·가중치 |
| 성공 | loss·reward·선호 승률 | 지연·처리량·안정성 |
| 전형 병목 | 데이터·보상·분산 | 대역폭·스케줄·양자화 |

**사실:** 동일 가중치라도 training graph와 inference graph는 다르다.  
**해석:** 커리큘럼이 4권 끝에서 5권으로 꺾이는 이유는 “할 이야기가 없어서”가 아니라, **다음 병목이 하드웨어·스케줄러 쪽**으로 이동하기 때문이다.

제99강은 이 표를 본문으로 펼친다. Prefill/Decode（100）, KV Cache（101）, Continuous Batching（102）이 바로 이어진다.

## 한 페이지 구두 시험（자가）
제한 시간 10분. 메모 없이 답한다.

1. RLHF 3단계 상자  
2. $\rho$와 clip이 막는 것  
3. KL $\beta$를 키우면  
4. DPO가 접는 상자 / 못 접는 것  
5. Outcome vs Process  
6. Reward hacking 한 줄 정의  
7. TTFT와 TPOT 정의（기호）  
8. 왜 긴 CoT가 서빙 비용을 올리나  

틀리면 해당 강으로 돌아간다. 통과 기준은 “완벽 암기”가 아니라 **좌표를 잃지 않는 것**이다.

## 추천 재학습 경로（막힐 때）
| 증상 | 돌아갈 강 |
|---|---|
| Post-Training 좌표가 흐리다 | 79, （3권）77~78 |
| Policy gradient가 약하다 | 80~83 |
| RM·RLHF 루프가 안 그려진다 | 84~86 |
| PPO·KL이 헷갈린다 | 87~89 |
| DPO 수식·구현 | 90~91, 95 |
| 그룹 상대·검증 보상 | 92~94 |
| “잘 되는데 이상하다” | 96 |
| 논문 이야기 과신 | 97 |

## 수식 보강 — 학습에서 서빙으로

학습된 $\pi_\theta$의 추론은 Prefill+Decode입니다.

$$
T_{\mathrm{total}}\approx\mathrm{TTFT}+(T_{\mathrm{out}}-1)\mathrm{TPOT}
$$

5권은 이 지연·메모리를 줄이는 장치(KV, batching, quant)를 다룹니다.

## 수식 카드 — 4권을 한 장에
손에 남는 최소 세트:

$$

\begin{aligned}
\rho_t(\theta)
&=
\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
\\[0.5em]
L^{\mathrm{CLIP}}
&=
\mathbb{E}\big[
\min\big(\rho A,\;\mathrm{clip}(\rho,1-\epsilon,1+\epsilon)A\big)
\big]
\\[0.5em]
\mathrm{KL}(\pi\|\pi_{\mathrm{ref}})
&=
\mathbb{E}_{y\sim\pi}\Big[\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}\Big]
\\[0.5em]
R
&=
r-\beta\,\widehat{\mathrm{KL}}
\\[0.5em]
L_{\mathrm{DPO}}
&=
-\log\sigma\big(\beta(\Delta\log\pi_\theta-\Delta\log\pi_{\mathrm{ref}})\big)
\end{aligned}

$$

여기서 $\Delta\log\pi=\log\pi(y_w\mid x)-\log\pi(y_l\mid x)$ 형태（제90강）.  
**암기 팁:** clip은 “한 걸음”, KL/$\beta$는 “집에서 얼마나 멀어질지”, DPO는 “선호 쌍으로 그 거리를 직접 학습”.

## 제95강 미니 파이프라인 ↔ 이론 기호
| 코드/로그 | 기호 | 강의 |
|---|---|---|
| `margin` before/after | $\log\pi(y_w)-\log\pi(y_l)$ 감각 | 90~91, 95 |
| `beta` | $\beta$ | 89~91 |
| RM logit / score | $r_\phi$ | 85 |
| REINFORCE `(r-b)*logp` | $(r-b)\nabla\log\pi$ | 82, 95 |
| label flip 실험 | misspecification | 96 |

미니 실험이 “작은 절대 성능”을 준 것이 아니라, **기호↔코드 대응**을 몸에 심은 것이 성과다.

## 정렬 성공이 서빙 부하가 되는 경로（정성）
```text
reasoning RL → 평균 출력 토큰 L_out ↑
            → decode step ≈ L_out
            → 요청당 GPU-시간 ↑
            → 동일 GPU에서 Throughput ↓ 가능
            → TTFT는 prefill·큐, TPOT는 decode에 민감
```

정의만（제107강에서 본격）:

$$

\begin{aligned}
\mathrm{TTFT} &= t_{\mathrm{first}}-t_{\mathrm{req}} \\
\mathrm{TPOT} &\approx \frac{t_{\mathrm{last}}-t_{\mathrm{first}}}{n_{\mathrm{out}}-1} \\
\mathrm{Throughput} &\approx \frac{N_{\mathrm{tokens}}}{\Delta t_{\mathrm{wall}}}
\end{aligned}

$$

**사실:** 위는 정의·근사다.  
**비주장:** 특정 모델의 ms·tok/s 숫자.


<!-- enrich-block-98 -->
## 4권 → 5권 다리 수식

정렬된 정책 $\pi_\theta$를 서빙할 때 비용은 대략:

$$
\mathrm{Cost} \propto \underbrace{T_{\mathrm{prefill}}}_{\mathrm{prompt}} + \underbrace{T_{\mathrm{decode}}\cdot N_{\mathrm{gen}}}_{\mathrm{생성}}
$$

KV 캐시 메모리:

$$
\mathrm{Mem}_{\mathrm{KV}} \approx 2\cdot L\cdot H\cdot d_h\cdot T\cdot B\cdot b_{\mathrm{bytes}}
$$

다음 권은 throughput·latency·스케줄러로 이 식을 엔지니어링합니다.


<!-- enrich-extra-98 -->
## 서빙 입구 체크리스트

| 항목 | 식/질문 |
|---|---|
| 가중치 메모리 | $N\cdot b$ |
| KV | $2LHd_h T B b$ |
| 동시 요청 | continuous batching? |
| SLA | TTFT / TPOT P99 |

```python
# 대략 KV 바이트 추정
def kv_bytes(L, H, dh, T, B, nbytes=2):
    # nbytes=2 → fp16
    return 2 * L * H * dh * T * B * nbytes

print(kv_bytes(L=32, H=32, dh=128, T=2048, B=8) / 1e9, "GB")
```


<!-- enrich-batch2-98 -->
## Prefill / Decode 분해

$$
t_{e2e}\approx t_{\mathrm{prefill}}(T_{\mathrm{in}})+N_{\mathrm{out}}\cdot t_{\mathrm{decode}}
$$

```python
def e2e_ms(ttft, tpot, n_out):
    return ttft + max(n_out-1,0)*tpot
print(e2e_ms(200, 30, 64))
```


<!-- enrich-batch4-98 -->
## 4권에서 챙길 운영 숫자

정렬이 끝난 모델도 서빙에서 다시 측정합니다.

$$
\mathrm{quality}=f(\pi_\theta),\qquad
\mathrm{cost}=g(\mathrm{Mem},\mathrm{TPS},\mathrm{P99})
$$

```python
def serves_ok(mem_gb, budget_gb, p99_ms, sla_ms):
    # 메모리·SLA 동시 만족?
    return mem_gb <= budget_gb and p99_ms <= sla_ms
print(serves_ok(18, 24, 200, 300))
```

### Prefill vs Decode 자원

| 단계 | 특징 |
|---|---|
| Prefill | compute에 가깝게, 긴 프롬프트 민감 |
| Decode | memory-bound, KV 성장 |

$$
\mathrm{Mem}_{KV}(t)=\mathrm{Mem}_{KV}(0)+c\cdot t
$$

## 5권으로 넘기기

Continuous batching · PagedAttention · 스케줄러 · 멀티노드 통신을 이어갑니다.

## LLM에서는 어디에 사용될까?
4권 끝에서 팀이 실제로 들고 가는 산출물:

1. **정렬된 가중치** $\pi_\theta$ — 챗·코딩·거부 정책이 심긴 분포  
2. **평가 습관** — win-rate만이 아닌 다축·해킹 감사  
3. **실패 어휘** — hacking, sycophancy, over-refusal, shift  
4. **다음 질문** — 이 가중치를 어떤 엔진·배치·양자화로 서빙할까?

사용자에게 보이는 것은 (1)의 샘플링 결과이고, SRE에게 보이는 것은 (4)의 지연·오류다. 둘을 잇는 다리가 5권이다.

## 실습
### 실습 A — 빈 지도

제3절 다이어그램을 보지 말고 SFT 이후 상자를 다시 그리시오.

### 실습 B — 수식 카드

위 수식 카드에서 $\rho$, $L^{\mathrm{CLIP}}$, KL, $L_{\mathrm{DPO}}$ 각각을 한 문장 직관으로 옮기시오.

### 실습 C — 서빙 매핑

긴 CoT 체크포인트에 대해, TTFT/TPOT/Throughput 중 어떤 지표가 먼저 악화되기 쉬운지 가설을 쓰시오（숫자 없이）.

### 실습 D — 체크리스트 감사

제4절 체크리스트에서 비어 있는 항목 3개를 고르고, 해당 강으로 돌아가 한 단락 요약을 쓰시오.

### 실습 E — 구두 시험

위 한 페이지 구두 시험 8문항을 타이머로 풀고, 틀린 문항의 강 번호를 적으시오.

## 자주 하는 실수
1. 4권을 “PPO 튜닝 가이드”로만 축소  
2. DPO=정렬 완료로 선언  
3. 미니 프로젝트 숫자를 실LLM 성능처럼 인용  
4. 정렬 품질과 서빙 SLO를 한 지표로 섞음  
5. 5권을 건너뛰고 API 래퍼만 붙임  
6. KL·clip·β를 구분하지 못함  

## 핵심 요약
- 4권의 핵심은 SFT 이후 **선호·보상·검증 신호로 정책을 갱신**하는 것이다.
- DPO는 RM+RL을 접을 수 있지만 데이터·평가 한계는 남긴다.
- Reasoning/RLVR은 검증 가능 축을 열어 긴 추론 궤적을 강화한다.
- 정렬 실패 모드는 최적화 성공과 함께 온다（96）.
- 다음 문제는 그 정책을 **Inference·Serving**에서 살아남게 하는 것이다（99~）.

## 배포 직전 체크（4권→5권）
- [ ] 학습에 쓴 chat template과 서빙 템플릿 문자열이 동일한가
- [ ] 특수 토큰·거절 문구가 토크나이저에 존재하는가
- [ ] 평균/ p95 출력 길이를 측정할 계획이 있는가（숫자 창작 금지）
- [ ] temperature/top-p가 정렬 평가와 서빙 기본값에서 너무 다르지 않은가
- [ ] 롤백용 $\pi_{\mathrm{ref}}$/이전 체크포인트가 저장되어 있는가

하나라도 아니오이면 5권 프로젝트를 시작하기 전에 메모만 남겨도 된다. **빈 칸을 아는 것**이 총정리의 성과다.

## 짧은 스토리로 복습
```text
3권: 말을 배우게 가르쳤다
4권: 무엇이 나은지/맞는지 신호로 다시 굽는다
5권: 그 입을 빠르게·많이·안정적으로 연다
```

신호 없는 서빙은 빠른 무정렬이고, 서빙 없는 정렬은 연구 체크포인트일 뿐이다.


## 용어 사전
| 용어 | 한 줄 의미 |
|---|---|
| Post-Training | SFT 이후 정렬·적응 단계 |
| Preference | 응답 쌍의 상대 선호 |
| Reward Model | 선호를 점수화 |
| RLHF | 보상 기반 정책 최적화 틀 |
| PPO | 클리핑 등으로 안정화한 PG류 |
| KL penalty | 참조 정책으로부터의 과도 이탈 억제 |
| DPO | 선호 직접 정책 손실 |
| GRPO | 그룹 상대 비교 기반 업데이트 |
| RLVR | 검증 가능 보상 RL |
| Outcome/Process | 최종 답 vs 단계 보상 |
| Reward hacking | 대리 보상 허점 공략 |
| Inference / Serving | 학습 후 토큰 생성·배포 시스템 |

## 연습문제
### 문제 1（통합）

빈칸을 채우시오.

```text
Pretrain → SFT → (    ) / RL / (    ) → Inference·Serving
```

### 문제 2（신호）

DPO와 RLVR의 학습 신호 출처를 대조해 한 줄씩 쓰시오.

### 문제 3（매핑）

긴 chain-of-thought가 서빙에 주는 직접적 부담을 한 단어군으로 쓰시오.

### 문제 4（한계）

win-rate만 보고 정렬 완료를 선언할 때 빠지기 쉬운 함정 이름을 쓰시오.

### 문제 5（다리）

제99강의 제목을 쓰고, Training과 Inference의 목표 차이를 한 줄로 쓰시오.

---

## 정답 및 해설
### 문제 1

Preference（또는 RM/DPO）, Reasoning（또는 RLVR）. 동의어 표현 가능.

### 문제 2

DPO: 선호 쌍. RLVR: 검증기 등 verifiable reward.

### 문제 3

출력 토큰 증가에 따른 decode 비용·지연·처리량 부담.

### 문제 4

Goodhart / 평가 과적합 / proxy 숭배（택1 이상）.

### 문제 5

제목: Training과 Inference의 차이. 목표: 전자는 $\theta$ 갱신, 후자는 고정 $\theta$로 생성·서비스.

<!-- enrich-98-finale -->
## 4권 전체 스토리라인을 한 호흡으로

```text
79 지도
 → 80~83 RL 기초 (s,a,r / π,V / ∇logπ / A)
 → 84~85 선호 데이터와 RM (BT)
 → 86~89 RLHF·PPO·KL
 → 90~91 DPO
 → 92~94 GRPO·RLVR·Reasoning
 → 95 미니 실습
 → 96 한계·부작용
 → 97 논문→실무 읽기
 → 98 총정리 → 5권 Inference/Serving
```

각 블록에서 **지우면 안 되는 식**만 다시 적는다.

### 블록 A — RL 기초

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\Big[\sum_t\gamma^t r_t\Big]
$$

$$
A(s,a)=Q(s,a)-V(s)
$$

$$
\nabla_\theta J
\approx
\mathbb{E}\big[\nabla_\theta\log\pi_\theta(a\mid s)\,A\big]
$$

### 블록 B — 선호와 RM

$$
P(y_w\succ y_l\mid x)=\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)
$$

$$
\mathcal{L}_{\mathrm{RM}}=-\log\sigma(\Delta),\quad\Delta=r_w-r_l
$$

### 블록 C — RLHF / PPO / KL

$$
\max_\pi\mathbb{E}[r_\phi]-\beta\mathrm{KL}(\pi\|\pi_{\mathrm{ref}})
$$

$$
\rho_t=\frac{\pi_\theta}{\pi_{\mathrm{old}}},\quad
L^{\mathrm{CLIP}}=\mathbb{E}\big[\min(\rho A,\mathrm{clip}(\rho)A)\big]
$$

### 블록 D — DPO / 그룹 / 검증

$$
L_{\mathrm{DPO}}=-\log\sigma\big(\beta(\hat r_w-\hat r_l)\big)
$$

$$
\hat A_i=\frac{r_i-\mu_x}{\sigma_x+\varepsilon}
\quad\text{（GRPO 감각）}
$$

$$
r_{\mathrm{VR}}=V(x,y)\in\{0,1\}
\quad\text{（RLVR 감각）}
$$

## 미니 실습（95）에서 검증할 계약

코드로 옮길 때 깨지기 쉬운 네 가지:

1. chosen/rejected 마스킹이 반대로 붙지 않았는가
2. ref 로그확률이 **고정**인가（실수로 같이 학습）
3. $\beta$·clip $\epsilon$ 로그가 남았는가
4. 평가가 학습 쌍과 **같은 분포 복사**만은 아닌가

계약이 맞으면 절대 점수가 작아도 **기호↔텐서**가 학습된다.

## 한계（96）를 출시 게이트로

다축 예:

| 축 | 최소 질문 |
|---|---|
| 도움됨 | 인간/다운스트림이 올랐는가 |
| 안전 | over-refusal과 위험 통과를 함께 봤는가 |
| 정직 | sycophancy 프브가 있는가 |
| 비용 | 평균 $L_{\mathrm{out}}$이 SLO 안인가 |

한 축만 초록이면 배포하지 않는다는 팀 규칙을 문서화한다.

## 5권으로 넘기는 성능 언어

정렬이 끝난 가중치 $\theta^\star$를 고정하면, 사용자 체감은 대략

$$
T_{\mathrm{total}}
\approx
\mathrm{TTFT}+(N_{\mathrm{out}}-1)\cdot\mathrm{TPOT}
$$

메모리:

$$
M_{\mathrm{KV}}
\approx
2\cdot L\cdot H\cdot d_h\cdot T\cdot B\cdot b_{\mathrm{bytes}}
$$

처리량:

$$
\mathrm{Throughput}
\approx
\frac{\sum_i N_{\mathrm{out},i}}{\Delta t_{\mathrm{wall}}}
$$

이제 질문은 “보상을 어떻게 올리나”에서 “**같은 품질을 어떻게 더 싸게·빠르게 내보내나**”로 바뀐다. 그 공학이 5권이다.

## 자가 점검 12문항（답은 노트에）

1. BT에서 관측 가능한 것은 $r$인가 $\Delta r$인가
2. RM 손실의 기울기가 쉬운 쌍에서 작아지는 이유
3. $\pi_{\mathrm{ref}}$를 보통 SFT로 두는 이유
4. PPO에서 $\rho$와 $A$의 역할
5. clip_frac이 높을 때 의심할 것 두 가지
6. KL $\beta$를 키우면 생기는 전형적 증상
7. DPO가 제거하는 상자 / 못 제거하는 것
8. GRPO의 그룹 상대 $\hat A$가 필요한 상황
9. RLVR의 $V$가 preference와 다른 점
10. reward hacking의 최소 감사 트리거
11. 논문 6칸 템플릿을 빈칸 없이 채울 수 있는가
12. TTFT·TPOT·Throughput 정의를 식과 말로 쓸 수 있는가

12개 중 10개 이상이면 4권 졸업 감각이다. 부족한 칸만 해당 강으로 되돌아간다.

## 한 장 치트시트（인쇄용）

```text
선호: P(w≻l)=σ(rw−rl)
RM:   L=−logσ(Δ)
RLHF: max E[r]−β KL
PPO:  min(ρA, clip(ρ)A)
DPO:  −logσ(β( r̂w−r̂l ))
Serve: T≈TTFT+(N−1)TPOT
```

이 여섯 줄이 손에서 먼저 나오고, 그다음 구현 디테일로 내려가면 된다.


## 다음 강의와 연결
4권의 문을 닫는다.

다음 책은 **5권. vLLM · GLM · DGX Spark**다.  
첫 강의는 **제99강. Training과 Inference의 차이**다. 정렬된 가중치를 들고, 이제 **토큰이 하드웨어 위에서 어떻게 흐르는지**로 넘어간다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [97강. 논문·실무 흐름 정리](97강_논문_실무_흐름_정리.md)
- **다음 강:** [99강. Training과 Inference의 차이](../05_vllm_glm_dgx/99강_Training과_Inference의_차이.md)

<!-- /LECTURE_NAV -->
