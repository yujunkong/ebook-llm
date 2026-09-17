# 106강. vLLM Scheduler
## 이번 강에서 배우는 내용

- waiting / running 큐가 각각 무엇을 기다리는가
- 한 스케줄 스텝에서 prefill과 decode가 어떻게 섞일 수 있는가
- 메모리(블록) 부족 시 preemption(선점) 이 왜 필요한가
- “FCFS + 예산 제약” 수준의 정책이 Continuous Batching과 어떻게 맞물리는가

## 왜 중요한가?
엔진을 “모델 forward 래퍼”로만 보면 장애 분석이 막힌다.

- TTFT가 길다 → 입장이 늦게 승인됐나, prefill이 밀렸나
- 처리량이 낮다 → running이 비었나, 블록이 고갈됐나
- 긴 요청 하나가 짧을 요청을 굶기나 → 스케줄·선점 정책

스케줄러는 **공정성·지연·처리량·OOM 회피**를 동시에 만지는 제어기다.

## 선수 개념
1. Prefill vs Decode 비용 특성 (제100강)
2. Continuous Batching (제102강)
3. KV 블록·free pool (제105강)
4. vLLM 구성 개요 (제104강)

## 두 큐 — Waiting과 Running
교육용으로 상태를 단순화한다.

```text
도착 요청
   │
   ▼
[ waiting ]  ──(블록·예산 충분)──►  [ running ]
   │                                    │
   │                                    ├── prefill 진행 / 완료
   │                                    ├── decode 토큰 생성
   │                                    └── 종료 → 블록 반납 → 사라짐
   │
   └──(메모리 부족·정책)── preemption 시 running→waiting(또는 swapped)
```

| 큐 | 의미 |
|---|---|
| **Waiting** | 아직 GPU에서 활발히 생성 중이 아님. 입장(admission) 대기, 또는 선점 후 재대기 |
| **Running** | 현재 스케줄 스텝의 배치에 들어갈 후보. KV 블록을 보유·사용 중 |

실제 엔진은 `swapped`, `preempted`, prefix-cache 대기 등 더 세분될 수 있다. **사실:** 이름·개수는 구현 의존. **해석:** “대기열 vs 실행열” 이분법이 이해의 최소 단위다.

## 한 스케줄 스텝의 직관
매 iteration(또는 엔진이 정한 스케줄 주기) roughly:

1. **종료 처리** — EOS·길이 상한에 도달한 시퀀스 제거, 블록 free
2. **Running 갱신** — 남은 시퀀스의 decode(또는 진행 중 prefill) 준비
3. **Admission** — waiting에서 새 요청을, **남은 블록·토큰 예산**이 허락하는 만큼 running으로
4. **실행** — 이번 배치에 대해 prefill/decode 커널 실행
5. **선점** — 예산이 부족하면 일부 running을 쫓아내 블록을 확보

```text
while serving:
  free_blocks_from_finished()
  batch = select(running)           # decode 위주 + 남은 prefill
  admit_from_waiting(free_budget)   # 새 prefill 입장
  maybe_preempt(if OOM risk)
  execute(batch)                    # GPU forward
```

Continuous Batching의 핵심은 **배치 구성원이 스텝마다 바뀐다**는 것이다. 고정 static batch가 아니다.

## Prefill과 Decode를 한 배치에?
Decode는 보통 시퀀스당 **새 토큰 1개**(스펙큘레이티브 제외)라서, 배칭하면 GPU 효율이 살아난다. Prefill은 프롬프트 길이만큼의 큰 연산이라 **한 번에 많이 넣으면 TTFT·버스트**가 커질 수 있다.

스케줄러가 흔히 보는 제약(개념):

- **토큰 예산:** 이번 스텝에서 처리할 토큰 수 상한
- **시퀀스 수 상한:** 동시 running 개수
- **KV 블록 수:** PagedAttention free list

**해석:** “항상 prefill 우선” 또는 “항상 decode 우선”이 만능은 아니다. 제품 SLO(TTFT vs 처리량)에 따라 정책이 갈린다. 특정 기본값이 영원한 정답이라고 쓰지 않는다.

## Preemption — 메모리가 부족할 때
생성 중 길이가 늘어나면 블록 수요가 **시간에 따라 증가**한다. Admission 시점에 넉넉해 보여도, 나중에 OOM에 가까워질 수 있다.

선점의 직관:

1. 정책에 따라 희생 시퀀스 선택 (예: 가장 최근 입장, 가장 긴 것 — **구현마다 다름**)
2. 해당 시퀀스의 KV 블록을 해제(또는 CPU로 스왑 — 기능이 있을 때)
3. 시퀀스를 waiting 쪽으로 되돌림
4. 확보한 블록으로 다른 running의 decode를 계속하거나 새 요청을 받음

```text
running:  [긴A, 짧은B, 짧은C]   free blocks → 0
                              ↓ preempt A
running:  [짧은B, 짧은C]       free blocks ↑
waiting:  [A, ...]
```

**사실:** 선점은 처리량·지연 모두에 영향을 준다. 스왑이 있으면 PCIe 비용이, 재계산(recompute) 전략이면 prefill 재비용이 붙을 수 있다.  
**해석:** “선점 = 버그”가 아니라 **가변 길이 + 유한 HBM** 세계의 정상 제어다. 다만 과도한 thrashing은 튜닝 신호다.

## Continuous Batching 스케줄러를 한 그림으로
```text
시간 →
요청1:  PPPDDDDDDD........
요청2:    ....PPPDDDDDDD..
요청3:         PPPDD......
GPU배치: [1] [1] [1,2] [1,2] [1,2,3] [2,3] [2] ...
         ↑        ↑              ↑
      1만 prefill  2 입장        3 입장 / 1 종료 후 재구성
```

P=prefill, D=decode. 스케줄러는 매 열에서 **누구를 배치에 넣을지·블록이 있는지**만 반복해 결정한다.

## 작은 의사코드
```python
from collections import deque
from dataclasses import dataclass, field

@dataclass
class Seq:
    sid: str
    prompt_left: int
    gen_left: int
    blocks: int = 0

@dataclass
class Scheduler:
    free_blocks: int
    waiting: deque = field(default_factory=deque)
    running: list = field(default_factory=list)
    max_seqs: int = 4

    def admit(self) -> None:
        while self.waiting and len(self.running) < self.max_seqs:
            s = self.waiting[0]
            need = 1  # 설명용: 입장에 최소 1블록
            if need > self.free_blocks:
                break
            self.waiting.popleft()
            self.free_blocks -= need
            s.blocks += need
            self.running.append(s)

    def preempt_one(self) -> None:
        if not self.running:
            return
        victim = self.running.pop()  # 설명용 LIFO
        self.free_blocks += victim.blocks
        victim.blocks = 0
        self.waiting.appendleft(victim)

    def step(self) -> None:
        # 1) 종료·블록 반납 생략 가능
        self.admit()
        # 2) decode 중 블록 추가가 필요하면
        for s in list(self.running):
            if s.prompt_left == 0 and s.gen_left > 0:
                if self.free_blocks == 0:
                    self.preempt_one()
                if self.free_blocks == 0:
                    break
                # 새 토큰이 새 블록을 요할 때만 감소한다고 가정
                self.free_blocks -= 0  # 자리표시
                s.gen_left -= 1
        # 3) 실제로는 여기서 execute(batch)
```

이 코드는 **동작하는 프로덕션 스케줄러가 아니다.** waiting/running/admit/preempt 단어만 몸에서 느껴 보려는 스케치다.

## 실제 LLM에서는 어떻게 보이는가
관측 포인트(도구·메트릭 이름은 엔진마다 다름):

| 현상 | 스케줄러 쪽 가설 |
|---|---|
| Waiting만 쌓임 | 블록 고갈, max sequences, rate limit |
| TTFT 꼬리 지연 | Admission 지연, 긴 prefill 큐 |
| 처리량 출렁 | 선점 thrashing, prefill 버스트 |
| 짧은 요청 기아 | 긴 요청 독점 — 정책·우선순위 검토 |

제107강의 TTFT/TPOT/Throughput은 스케줄러 건강의 **외부 증상**이다. 원인 분석은 큐 길이·블록 사용률과 함께 본다.

## 수식으로 보는 예산·입장
### 12.1 토큰·블록 예산（개념）

한 스케줄 스텝 $t$에서 처리할 토큰 상한을 $B_{\mathrm{tok}}$, 동시 시퀀스 상한을 $N_{\mathrm{seq}}$, free KV 블록을 $F$라 하자.

$$

\begin{aligned}
\mathrm{admit}
&\iff
(|\mathrm{running}| < N_{\mathrm{seq}})
\;\wedge\;
(F \ge F_{\mathrm{need}})
\;\wedge\;
(T_{\mathrm{step}} \le B_{\mathrm{tok}})
\end{aligned}

$$

$F_{\mathrm{need}}$는 새 요청의 프롬프트 길이·블록 크기에 비례한다（제105강）.  
**사실:** 실제 엔진의 술어는 더 많다（LoRA, priority, prefix cache…）.  
**해석:** 위 세 축만으로도 “왜 waiting만 쌓이는가”를 분류할 수 있다.

### 12.2 Prefill vs Decode 비용 스케치

시퀀스 $i$의 프롬프트 길이 $L_i$, 이미 생성한 길이 $\ell_i$일 때（아주 거친 모형）:

$$

\begin{aligned}
C_{\mathrm{prefill}}(i) &\propto L_i \\
C_{\mathrm{decode}}(i) &\propto 1 \quad\text{（토큰 1개; 스펙큘레이션 제외）}
\end{aligned}

$$

배치 비용 근사:

$$

C_{\mathrm{batch}}
\approx
\sum_{i\in\mathrm{prefill}} C_{\mathrm{prefill}}(i)
+
\sum_{j\in\mathrm{decode}} C_{\mathrm{decode}}(j)

$$

긴 prefill을 한 스텝에 많이 admit하면 $C_{\mathrm{batch}}$가 폭증 → TTFT 꼬리가 길어진다. Decode만 가득 채우면 처리량 효율은 좋아질 수 있으나 신규 요청 TTFT는 밀린다.

### 12.3 Preemption과 thrashing

선점 횟수를 $P$, 재개 시 재prefill 비용을 $C_{\mathrm{re}}$라 하면 thrashing 경고 신호（정성）:

$$

P \uparrow,\quad
\frac{C_{\mathrm{re}}}{C_{\mathrm{useful}}} \uparrow
\quad\Rightarrow\quad
\text{유효 처리량 하락 후보}
$$

절대 임계값은 환경마다 다르다. 로그에 preempt 카운트를 남겨 추세를 본다.

## 지표와 스케줄러 연결（제107강 예고）
$$

\begin{aligned}
\mathrm{TTFT} &= t_{\mathrm{first}}-t_{\mathrm{req}} \\
\mathrm{TPOT} &\approx \frac{t_{\mathrm{last}}-t_{\mathrm{first}}}{n_{\mathrm{out}}-1} \\
\mathrm{Throughput} &\approx \frac{N_{\mathrm{tokens}}}{\Delta t}
\end{aligned}

$$

| 증상 | 스케줄 쪽 레버 |
|---|---|
| TTFT↑ | admission 지연, prefill 버스트, waiting 길이 |
| TPOT↑ | running 과밀, 선점, 메모리 압박 |
| Throughput↓ | free blocks↓, thrashing, max_seqs 과소/과대 |

**비주장:** 특정 기본 설정이 항상 최적.


<!-- enrich-batch2-106 -->
## 큐잉 이론 직관

이용률 $\rho=\lambda/\mu$ 가 1에 가까우면 대기열 폭증.

$$
\mathbb{E}[W]\propto \frac{\rho}{1-\rho}
$$

(단순 M/M/1 스케치 — 실제 스케줄러는 KV 제약 포함)

```python
def mm1_wait(rho):
    return rho/(1-rho) if rho < 1 else float("inf")
print(mm1_wait(0.7), mm1_wait(0.95))
```

## LLM에서는 어디에 사용될까?
온콜에서 스케줄러 어휘가 바로 쓰인다.

1. “GPU는 노는데 응답이 안 나와요” → waiting vs running, 블록 고갈  
2. “짧은 질문이 굶어요” → 긴 요청 독점·우선순위  
3. “가끔 한 요청만 재시작처럼 느려요” → preemption/recompute  
4. “평균은 괜찮은데 p99 TTFT가…” → prefill 버스트·큐잉  

제116~119강 프로젝트·운영에서 이 가설 트리를 메트릭에 연결한다.

## 실습
### 실습 A — 예산 부등식

$N_{\mathrm{seq}}=4$, $F=10$, 새 요청이 블록 3을 필요로 하고 running이 이미 3개일 때 admit 여부를 판정하시오.

### 실습 B — 타임라인

제6절 그림에 요청4（짧은 프롬프트）가 t=3에 도착한다고 가정하고, 어느 열부터 배치에 들어갈지 가설을 쓰시오.

### 실습 C — 의사코드 확장

교육용 `Scheduler.step`에 “종료 시 블록 반납” 분기 2줄을 스케치하시오.

### 실습 D — SLO 충돌

TTFT SLO와 Throughput SLO가 충돌할 때, prefill admission을 조이는 정책의 득실을 문장으로 쓰시오.

## 자주 하는 실수
1. Static batching과 Continuous Batching을 같은 “batch size”로만 비교
2. Preemption을 장애로만 보고 로그를 무시
3. Prefill 대량 입장을 허용해 TTFT SLO를 깨고도 평균 처리량만 자랑
4. 엔진 버전 없이 “vLLM은 항상 X 정책”이라고 단정
5. PagedAttention 없이 스케줄만 논해 OOM 원인을 놓침 — 둘은 한 쌍이다
6. Throughput만 보고 p99 TTFT를 생략

## 스케줄 스텝 복잡도 감각（교육용）
시퀀스 $n$개, 평균 프롬프트 $L$, 블록 크기 $S$일 때 admission 검사 비용은 대략 $O(n)$이고,  
KV 블록 필요량은 대략

$$

F_{\mathrm{need}}(L) \approx \Big\lceil \frac{L}{S} \Big\rceil

$$

Decode 한 토큰이 새 블록을 여는 빈도는 $S$에 따라 달라진다. 스케줄러는 매 스텝 “지금 당장 부족한가”만 보지 말고 **앞으로 $\Delta$ 토큰 더 쓸 때**를 본다（lookahead는 구현·설정 의존）.

### 공정성 스케치

가중치 $w_i$（우선순위）가 있으면 선택 규칙 예:

$$

i^\star = \arg\max_{i\in\mathrm{waiting}} \frac{w_i}{a_i+\varepsilon}

$$

$a_i$는 이미 받은 서비스량. 구체 식은 엔진마다 다르다. **사실:** 공정성 정책은 제품 요구. **비주장:** 위 식이 vLLM 기본값.

## 장애 로그에 남길 최소 필드
```text
waiting_len, running_len, free_blocks, preempt_count,
prefill_tokens_this_step, decode_seqs_this_step,
ttft_p50/p99 (외부), throughput (외부)
```

숫자 해석은 제107·118강. 여기서는 **스케줄 내부 상태와 외부 SLO를 같은 티켓에 붙인다**는 규율만 고정한다.


## 핵심 요약
- vLLM류 스케줄러는 **waiting/running**을 중심으로 매 스텝 배치를 재구성한다.
- Admission은 **KV 블록·시퀀스·토큰 예산** 제약 하에 이뤄진다.
- 길이가 자라는 decode 때문에 **preemption**이 필요할 수 있다.
- Continuous Batching의 “연속”은 스케줄러가 매 스텝 **멤버십을 연속적으로 갱신**한다는 뜻에 가깝다.
- 정책 세부는 진화한다. **큐·예산·선점** 삼각만 먼저 몸에 넣자.

## 용어 사전
| 용어 | 설명 |
|---|---|
| Waiting queue | 입장·재개를 기다리는 시퀀스 집합 |
| Running queue | 현재 실행 배치 후보 시퀀스 집합 |
| Admission | waiting → running 전환 (자원 허가) |
| Preemption | 자원 확보를 위해 running에서 퇴장 |
| Schedule step | 배치 구성·실행을 결정하는 한 주기 |

## 연습문제
### 문제 1

Waiting과 Running의 차이를 한 문장으로 구분하시오.

### 문제 2

Continuous Batching에서 배치 구성원이 스텝마다 바뀌는 이유를 쓰시오.

### 문제 3

Decode 도중 free block이 0이 되면 스케줄러가 취할 수 있는 행동을 두 가지 이상 드시오.

### 문제 4

Prefill을 waiting에서 한꺼번에 너무 많이 admit하면 어떤 SLO가 깨지기 쉬운가?

### 문제 5（사실/해석）

“모든 vLLM 버전이 동일한 preemption 희생자 선택 규칙을 쓴다”는 문장을 평가하시오.

### 문제 6

제105강의 free block pool이 스케줄러의 어떤 결정에 직접 들어가나?

---

## 정답 및 해설
### 문제 1

Waiting은 아직(또는 더 이상) 활발 실행 배치에 없는 대기, Running은 KV를 쓰며 스케줄 배치 후보인 실행열.

### 문제 2

요청이 비동기적으로 도착·종료하고 길이가 서로 달라, 매 스텝 GPU에 올릴 집합을 다시 짜는 것이 이득이기 때문.

### 문제 3

일부 시퀀스 preemption, (지원 시) CPU 스왑, 신규 admission 중단, 종료 유도 등.

### 문제 4

TTFT(및 prefill 지연 꼬리). 긴 프롬프트 버스트로 첫 토큰까지 시간이 늘어난다.

### 문제 5

사실로 단정하기 어렵다. **해석·버전 의존.** 희생자 정책은 구현·설정에 따라 달라질 수 있다.

### 문제 6

Admission 가능 여부와 preemption 필요 여부(메모리 예산).

## 다음 강의와 연결
스케줄이 건강해도, **무엇으로 건강을 숫자로 말할지**가 없으면 튜닝이 감이다.  
다음 **제107강. LLM Serving 성능 지표 — TTFT, TPOT, Throughput**에서 지연·처리량 지표를 정의하고, 섞어 쓰면 안 되는 함정을 정리한다.

이전: 제105강 PagedAttention → 메모리 화폐. 이번: 그 화폐의 **배분 정책**.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [105강. PagedAttention](105강_PagedAttention.md)
- **다음 강:** [107강. LLM Serving 성능 지표 — TTFT, TPOT, Throughput](107강_LLM_Serving_성능_지표_TTFT_TPOT_Throughput.md)

<!-- /LECTURE_NAV -->
