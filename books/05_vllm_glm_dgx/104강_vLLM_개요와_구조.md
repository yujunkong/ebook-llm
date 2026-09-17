# 제104강. vLLM 개요와 구조

> **학습 목표**
> - vLLM이 무엇（역할）이고 무엇에 쓰이는지
> - LLM Serving 스택에서 API · Engine · Scheduler · Worker · KV Cache의 위치
> - PagedAttention이 개념적으로 어디에 꽂히는지（세부는 제105강）
> - “문서에 나온 역할”과 “소스의 특정 파일 추정”을 섞지 않는 태도
> - 다음 강의（PagedAttention·Scheduler·지표）로 가는 질문 목록

---
## 1. 왜 이것을 배우는가

한 줄 `model.generate()`로도 데모는 된다. 그러나 제품은 다음을 동시에 요구한다.

| 요구 | 부품 강의 |
|---|---|
| 낮은 첫 토큰 지연 | Prefill（100）·스케줄 |
| 안정적 토큰 간격 | Decode（100）·동시성 |
| 긴 문맥·다수 사용자 | KV（101）·Batching（102） |
| 한정 VRAM | Quantization（103）·페이지 메모리（105） |
| HTTP API · 멀티 모델 | Serving 계층 |

vLLM은 위 요구를 한 런타임에서 다루려는 **오픈소스 LLM 추론 엔진**으로 널리 쓰인다.  
**해석:** 엔진을 “마법의 속도 버튼”으로 보면 실패한다. **이미 배운 개념의 조립품**으로 봐야 디버깅이 된다.

## 2. 먼저 알아야 할 개념

1. 제99~103강의 용어 전부
2. Autoregressive generation — 제58강
3. （선택）OpenAI 호환 Chat Completions API의 존재 — 입출력 JSON 감각
4. 프로세스·GPU 워커·큐의 일반 시스템 직관

PagedAttention 수식·블록 테이블은 제105강.  
스케줄 정책 세부는 제106강.  
TTFT/TPOT 측정은 제107강.

## 3. 핵심 개념 설명

### 3.1 vLLM이란?

**vLLM**은 대규모 언어 모델을 효율적으로 서빙하기 위한 **오픈소스 Inference Engine（추론 엔진）** 이다.

공개적으로 자주 강조되는 축:

- **PagedAttention** — KV를 OS 가상메모리처럼 고정 크기 블록으로 관리하는 아이디어
- **Continuous Batching** — 요청을 동적으로 묶어 실행
- **고수준 API** — Python API 및（설정에 따라）OpenAI 호환 HTTP 서버

**사실:** 프로젝트명·목표가 오픈소스 추론 엔진이라는 점은 공개 문서·논문·저장소 소개에 따른다.  
**설명:** “항상 모든 워크로드에서 가장 빠르다”는 문장은 이 책이 주장하지 않는다. 비교는 제112강.

### 3.2 Inference Engine의 자리

```text
[Client] --HTTP/gRPC/SDK--> [API Server]
                               │
                               ▼
                          [LLM Engine]
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
         Scheduler        Cache/KV mgr       Workers
              │                │                │
              └──────── GPU model execution ────┘
```

| 층 | 책임（개념） |
|---|---|
| API Server | 요청 검증, 스트리밍, 인증·라우팅（배포에 따라） |
| Engine | 요청 수명주기, 생성 루프 orchestration |
| Scheduler | 누구를 이번 iteration에 넣을지 |
| Cache manager | KV 블록 할당·해제 |
| Worker | 실제 모델 forward（단일/다중 GPU） |

이 이름은 **역할 라벨**이다. 버전별 클래스명이 달라도 지도는 통한다.

### 3.3 요청의 수명（엔진 관점）

```text
arrive → waiting queue
      → admitted (KV 예산 OK)
      → prefill
      → decode loop (continuous batch의 일원)
      → finished → free KV → 응답 완료
```

제100~102강에서 본 수명이 엔진 안에서 **명시적 상태 기계**가 된다.

### 3.4 사용자가 만지는 표면

전형적 사용 형태（개념）:

1. **오프라인/스크립트:** Python에서 엔진을 띄우고 `generate`
2. **온라인 서버:** OpenAI 호환 endpoint로 chat/completions
3. **배치 추론:** 대량 프롬프트 처리

입출력에 공통으로 붙는 것:

- sampling params（temperature, top_p, max_tokens, stop …）
- （chat）messages → chat template → token ids

**사실:** 템플릿 불일치는 학습·서빙 불일치의 대표 사고다（제72·98강）. 엔진이 빨라도 템플릿이 틀리면 “다른 모델”이다.

### 3.5 PagedAttention이 끼는 위치（예고）

제101강: 정적 $T_{\max}$ 예약은 낭비·조각난다.  
**PagedAttention**은 KV를 **고정 길이 블록**의 연결로 두어, 필요할 때만 블록을 붙인다.

```text
논리 시퀀스:  [tok0 tok1 … tokN]
물리 블록:    [blk3]→[blk7]→[blk1]→…
```

엔진의 Cache manager가 블록 테이블을 들고, Attention 커널이 그 테이블을 따라 K/V를 읽는다.  
세부는 **제105강**. 오늘은 “왜 vLLM 이야기마다 페이지가 나오는지”만 연결한다.

### 3.6 단일 GPU와 다중 GPU

개념적으로 Worker는

- 단일 GPU에 모델 전체
- **Tensor Parallel** 등으로 모델을 쪼갬（제113강）

을 담당할 수 있다.  
**설명:** 병렬 전략은 엔진 설정·모델 크기에 따라 선택한다. 오늘은 존재만 표시.

## 4. 직관적으로 이해하기

항공 관제 비유（**설명**）:

| 구성 | 비유 |
|---|---|
| API | 항공권 발권 창구 |
| Scheduler | 관제탑 — 이륙·착륙 순서 |
| KV blocks | 주기장 게이트 — 필요 시 배정 |
| Worker/GPU | 활주로에서 실제로 움직이는 기체 |
| Continuous batching | 게이트가 비는 즉시 다음 편 배정 |

식당 비유:

- 메뉴판 양자화（103）= 재료 압축 보관
- Continuous batching（102）= 빈 화구에 즉시 새 주문
- PagedAttention（105）= 넓은 접시 대신 **같은 크기 쟁반**을 필요할 때 추가

vLLM은 주방을 **한 시스템으로 포장**한 것에 가깝다.

## 5. 수학적으로 이해하기（용량 감각만）

엔진이 매 iteration 대략 만족시키려는 제약（이상화）:

$$

\sum_{r\in\mathcal{R}} \mathrm{KVBlocks}(r) \le \mathrm{BlockPool}

$$

$$

|\mathcal{R}| \le \mathrm{MaxSeqs}

$$

$\mathcal{R}$은 running 요청 집합.  
입학·프리엠션·대기 결정은 이 부등식과 Prefill/Decode 비용을 본다（제106강）.

처리량 감각（정의는 제107강）:

$$

\mathrm{Throughput} \sim \frac{\text{완료 토큰 수}}{\text{벽시계 시간}}

$$

엔진 최적화의 목표는 이 비율과 지연 분포를 **워크로드에 맞게** 개선하는 것이다. 숫자를 여기서 단정하지 않는다.

## 6. 작은 시나리오（예시·가정）

가정 워크로드:

- GPU VRAM 여유: 가중치 적재 후 KV에 쓸 수 있는 예산이 “블록 1000개”라고 **가정**
- 블록 하나: 토큰 16개분 KV（가정）
- 요청 A: 길이 200 → 블록 $\lceil 200/16\rceil=13$
- 요청 B: 길이 800 → 50
- 동시 Decode 중 요청 12개 …

**설명:** 스케줄러는 “지금 새 Prefill을 넣을 블록이 있나?”를 본다.  
잔여 블록이 부족하면 대기열에 남긴다 — Continuous Batching이라도 **무한 입학은 없다**.

이 시나리오의 숫자는 교육용이다.

## 7. 코드로 구현하기 — 역할만 드러내는 미니 엔진

실제 vLLM을 재구현하지 않는다. **역할 분리**만 보여 준다.

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time

@dataclass
class SamplingParams:
    max_tokens: int = 32
    temperature: float = 0.0

@dataclass
class RequestState:
    request_id: str
    prompt_tokens: List[int]
    params: SamplingParams
    generated: List[int] = field(default_factory=list)
    kv_blocks: int = 0  # 장난감: 블록 개수만
    phase: str = "waiting"  # waiting|prefill|decode|done

class BlockPool:
    def __init__(self, n_blocks: int):
        self.free = n_blocks

    def alloc(self, n: int) -> bool:
        if self.free < n:
            return False
        self.free -= n
        return True

    def free_n(self, n: int):
        self.free += n

class MiniScheduler:
    def __init__(self, pool: BlockPool, max_running: int = 4):
        self.pool = pool
        self.max_running = max_running
        self.waiting: List[RequestState] = []
        self.running: List[RequestState] = []

    def add(self, req: RequestState):
        self.waiting.append(req)

    def schedule(self):
        # 종료분 free
        still = []
        for r in self.running:
            if r.phase == "done":
                self.pool.free_n(r.kv_blocks)
            else:
                still.append(r)
        self.running = still

        # 입학: prefill에 필요한 최소 블록(가정: 프롬프트/16)
        while self.waiting and len(self.running) < self.max_running:
            cand = self.waiting[0]
            need = max(1, (len(cand.prompt_tokens) + 15) // 16)
            if not self.pool.alloc(need):
                break
            cand.kv_blocks = need
            cand.phase = "prefill"
            self.running.append(self.waiting.pop(0))

        return list(self.running)

class MiniWorker:
    """GPU 대신: prefill/decode를 카운터로 흉내."""
    def run(self, batch: List[RequestState]):
        for r in batch:
            if r.phase == "prefill":
                r.phase = "decode"
                # 첫 토큰
                r.generated.append(1)
            elif r.phase == "decode":
                r.generated.append(1)
                # decode 중 길이 증가 → 가끔 블록 추가는 생략(교육용)
                if len(r.generated) >= r.params.max_tokens:
                    r.phase = "done"

class MiniEngine:
    def __init__(self):
        self.pool = BlockPool(100)
        self.sched = MiniScheduler(self.pool)
        self.worker = MiniWorker()
        self.outputs: Dict[str, List[int]] = {}

    def add_request(self, request_id, prompt_tokens, params):
        self.sched.add(RequestState(request_id, prompt_tokens, params))

    def step(self):
        batch = self.sched.schedule()
        self.worker.run(batch)
        for r in batch:
            if r.phase == "done":
                self.outputs[r.request_id] = r.generated

    def run_until_idle(self, max_steps=1000):
        for _ in range(max_steps):
            if not self.sched.waiting and not self.sched.running:
                break
            # running이 모두 done이면 schedule이 free
            if self.sched.running and all(r.phase == "done" for r in self.sched.running):
                self.sched.schedule()
                continue
            self.step()

# demo
eng = MiniEngine()
eng.add_request("A", [0] * 40, SamplingParams(max_tokens=5))
eng.add_request("B", [0] * 80, SamplingParams(max_tokens=3))
eng.run_until_idle()
print(eng.outputs)
print("free blocks", eng.pool.free)
```

관찰:

- `MiniScheduler`가 입학·해제를 담당
- `BlockPool`이 KV 예산
- `MiniWorker`가 Prefill/Decode 국면

실제 vLLM은 여기에 **진짜 모델·커널·페이지 테이블**이 붙는다.

## 8. “실제 사용” 스케치（버전 비의존）

아래는 **개념적 의사코드**다. 설치·인자명은 공식 문서의 현재 버전을 따른다.

```python
# 의사코드 — 실제 import/인자는 문서 확인
# from vllm import LLM, SamplingParams

# llm = LLM(model=".../model-id-or-path")
# params = SamplingParams(temperature=0.0, max_tokens=128)
# outputs = llm.generate(["안녕하세요"], params)
# print(outputs[0].outputs[0].text)
```

서버 형태（개념）:

```bash
# 의사코드 — 실제 CLI는 문서 확인
# vllm serve <model> --port 8000
# 이후 OpenAI 호환 클라이언트로 /v1/chat/completions 호출
```

**사실:** CLI·모듈 경로는 릴리스에 따라 변할 수 있다.  
**설명:** 책을 코드 스니펫의 절대 진리로 쓰지 말고, 학습한 **개념**으로 문서를 읽으라.

## 9. LLM Serving 연결 — 부품이 엔진에서 만나는 곳

| 부품 | 엔진 안에서의 만남 |
|---|---|
| Prefill/Decode | Worker step의 국면, 스케줄 단위 |
| KV Cache | Block pool / cache engine |
| Continuous Batching | Scheduler의 running set 갱신 |
| Quantization | 모델 로드 포맷·커널 경로 |
| Chat template | 토큰화 직전 전처리 |
| 지표 TTFT/TPOT | 큐·국면·동시성의 결과（107） |

장애를 볼 때 질문 순서 제안:

1. API에 들어온 입력이 학습과 같은가（템플릿·토크나이저）?
2. 대기열인가, Prefill인가, Decode인가?
3. KV 블록이 바닥났는가?
4. 양자화·병렬 설정이 의도한 경로인가?

## 10. 실습

### 실습 A — 지도 그리기

§4.2 다이어그램을 공백 없이 손으로 다시 그리고, 제100~103강 용어를 상자에 기입한다.

### 실습 B — 미니 엔진 실험

§8 코드에서 `BlockPool(20)`처럼 예산을 줄인 뒤, 긴 프롬프트 요청을 여러 개 넣어 `waiting`이 남는 상황을 재현한다.

### 실습 C — 상태 기계

`waiting → prefill → decode → done` 전이표를 만든다. 각 전이에서 블록 수가 어떻게 변해야 하는지（이상）를 쓴다.

### 실습 D — 문서 읽기 과제

공식 vLLM 문서（또는 README）에서 Continuous Batching / PagedAttention / OpenAI compatible server가 언급된 문장을 찾아, **인용 + 자신의 해석**을 분리해 노트한다.  
문서에 없는 내부 파일 구조를 추측해 “사실”로 쓰지 말 것.

### 실습 E — 비교 표 초안

제112강을 위해 vLLM / TensorRT-LLM / SGLang을 **빈 표**로만 만든다（오늘은 채우지 않아도 됨）. 열: 라이선스·강조 최적화·생태계.

### 실습 F — 문장 교정

원문: “vLLM을 쓰면 KV 관리 구현을 내가 몰라도 되고, 내부 스케줄러는 항상 최적이다.”  
교정: 역할은 숨겨 주지만 **지표·설정·워크로드**는 사용자가 책임진다는 형태로.

## 11. 자주 하는 실수

1. **엔진 = 모델**로 생각  
   엔진은 실행 시스템, 모델은 가중치·구조다.

2. **버전 무시하고 옛 튜토리얼 API 고정**  
   릴리스 노트를 본다.

3. **미공개 내부 함수명을 암기하며 “이해했다”고 착각**  
   역할·지표·실패 모드가 먼저다.

4. **템플릿·토크나이저를 엔진 바깥으로 미룸**  
   품질 버그의 절반은 전처리다（과장일 수 있으나 빈번）.

5. **OOM을 “모델이 커서”로만 단정**  
   KV 블록 고갈·동시성·긴 프롬프트를 점검.

6. **벤치 스크린샷을 재현 조건 없이 공유**  
   모델·GPU·동시성·입출력 길이 없이 무의미.

7. **제105강을 건너뛰고 페이지만 주문처럼 외움**  
   블록이 푸는 문제가 제101~102의 낭비다.

## 12. 핵심 정리

- vLLM은 LLM을 효율적으로 서빙하기 위한 오픈소스 추론 엔진이다.
- API · Engine · Scheduler · Cache · Worker의 **역할 지도**로 이해한다.
- Prefill/Decode·KV·Continuous Batching·Quantization이 그 지도 위에 놓인다.
- PagedAttention은 KV를 블록으로 관리하는 핵심 아이디어로 다음 강에서 전개한다.
- 내부 세부는 버전에 따라 변하므로, 이 책은 공개 개념과 교육용 모델을 사실과 구분한다.

## 13. 핵심 용어

| 용어 | 한 줄 의미 |
|---|---|
| Inference Engine | 고성능 추론 런타임 |
| vLLM | 오픈소스 LLM 서빙 엔진의 한 예 |
| API Server | 클라이언트 요청의 진입점 |
| Scheduler | iteration 멤버십·우선순위 결정 |
| Worker | 모델 실행 담당 |
| KV Block Pool | 페이지 단위 KV 자원 |
| PagedAttention | 블록 기반 Attention/KV 관리 아이디어 |
| OpenAI-compatible API | 널리 쓰이는 HTTP 스키마 호환 엔드포인트 |
| SamplingParams | 생성 제어 파라미터 묶음 |

## 14. 연습 문제
### 문제 1（역할）

vLLM을 “모델”이 아니라 무엇으로 정의해야 하는지 한 줄로 쓰시오.

### 문제 2（지도）

API Server / Scheduler / Worker / Cache manager의 책임을 한 줄씩.

### 문제 3（수명）

요청 상태 `waiting → … → done`을 채우시오.

### 문제 4（부품）

Continuous Batching과 PagedAttention이 각각 주로 해결하는 낭비를 한 줄씩.

### 문제 5（사실/해석）

“vLLM은 모든 엔진 중 최고다”를 이 책의 문장으로 고치시오.

### 문제 6（미니 엔진）

`BlockPool.alloc`이 실패하면 스케줄러가 해야 할 일은?

### 문제 7（다리）

제105강 제목을 쓰고, 논리 시퀀스와 물리 블록의 관계를 한 줄로.

---

## 정답 및 해설

### 문제 1

LLM을 효율적으로 실행·배치·메모리 관리하는 Inference Engine（서빙 런타임）.

### 문제 2

API: 입출력·프로토콜. Scheduler: 누구를 언제 실행. Worker: forward. Cache: KV 블록 할당·해제.

### 문제 3

waiting →（admit）prefill → decode → done（→ KV free）.

### 문제 4

Continuous: 정적 배치의 idle/호위·늦은 입학. PagedAttention: 거대 contiguous KV 예약 낭비·조각.

### 문제 5

예: “vLLM은 널리 쓰이는 오픈소스 엔진 중 하나이며, 워크로드별 최적은 측정으로 고른다（제112강）.”

### 문제 6

해당 요청을 입학시키지 않고 waiting에 남긴다（예산 확보까지 대기）.

### 문제 7

제목: PagedAttention. 관계: 논리 토큰 열을 고정 크기 물리 블록의 연결로 저장·조회한다.

## 15. 다음 강의와 연결

지도가 생겼다.  
다음 **제105강. PagedAttention**에서는 KV를 왜 페이지（블록）로 나누는지, 블록 테이블이 Attention과 어떻게 만나는지, 제101강의 낭비 식이 어떻게 줄어드는지（개념·수식·교육용 코드）를 본격적으로 푼다.

그 다음 **제106강. vLLM Scheduler**에서 입학·우선순위·Prefill/Decode 혼합 정책을 더 깊게 보고, **제107강**에서 TTFT·TPOT·Throughput으로 측정 언어를 고정한다.

> 엔진은 마법이 아니다. 이미 배운 Prefill·KV·Batching·Quantization의 운영체제에 가깝다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [제103강. Quantization — INT8, INT4, FP8](103강_Quantization_INT8_INT4_FP8.md)
- **다음 강:** [제105강. PagedAttention](105강_PagedAttention.md)

<!-- /LECTURE_NAV -->
