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

## LLM에서는 어디에 사용될까?

이번 98강에서 배운 개념은 이후 Transformer · GPT · 서빙 강의에서 반복해서 등장합니다. 각 수식·코드 블록을 “실제 모델의 어느 단계인가”와 연결해 다시 읽어 보세요.

## 핵심 요약
- 4권의 핵심은 SFT 이후 **선호·보상·검증 신호로 정책을 갱신**하는 것이다.
- DPO는 RM+RL을 접을 수 있지만 데이터·평가 한계는 남긴다.
- Reasoning/RLVR은 검증 가능 축을 열어 긴 추론 궤적을 강화한다.
- 정렬 실패 모드는 최적화 성공과 함께 온다（96）.
- 다음 문제는 그 정책을 **Inference·Serving**에서 살아남게 하는 것이다（99~）.

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
