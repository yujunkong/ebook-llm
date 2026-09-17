# 제102강. Continuous Batching

> **학습 목표**
> - Static（고정）Batching과 Continuous Batching의 차이
> - 왜 온라인 트래픽에서 패딩·대기·버블이 생기는지
> - Iteration-level（스텝 단위）스케줄의 직관
> - Prefill과 Decode가 한 엔진 루프에 어떻게 공존하는지（개념）
> - 처리량·지연·공정성 사이의 트레이드오프（수치 단정 없이）
> - vLLM 등 엔진이 이 아이디어를 전제로 설계된다는 위치（제104강）

---
## 1. 왜 이것을 배우는가

학습 DataLoader의 배치는 대개

```text
길이를 맞추거나 비슷한 것끼리 모아 → 한 번에 forward/backward
```

이다. 온라인 서빙은 다르다.

```text
요청 A: 지금 도착, 프롬프트 2k, 앞으로 200 토큰 생성할지도
요청 B: 3초 전부터 decode 중, 앞으로 5 토큰 남음
요청 C: 방금 끝남 → KV 반환
요청 D: 큐에서 대기 중
```

고정 배치로 “4개를 모아 동시에 시작·동시에 끝”을 강제하면:

- 먼저 끝난 요청이 **빈 슬롯으로 남을** 수 있다
- 긴 요청을 기다리느라 **짧은 요청 TTFT**가 길어진다
- 패딩으로 GPU는 바빠 보여도 **유용 토큰**은 적다

**해석:** Continuous Batching은 “배치 크기를 키운다”가 아니라 **배치의 구성원을 시간에 따라 갈아끼운다**에 가깝다.

## 2. 먼저 알아야 할 개념

1. Prefill / Decode — 제100강
2. KV Cache · 요청당 상태 — 제101강
3. Training batch vs Serving batch — 제99강
4. GPU가 “큰 행렬곱”에 효율적이라는 감각 — 이후 제111강과 연결
5. 큐잉·대기 시간 직관（일반 시스템）

## 3. 핵심 개념 설명

### 3.1 Batching이란?

**Batching(배칭)**은 여러 입력을 묶어 한 번의 커널/스텝에서 처리해 하드웨어 효율을 올리려는 기법이다.

LLM 서빙에서 묶는 단위는 흔히

- 여러 **요청**의 Prefill 토큰
- 여러 **요청**의 Decode 토큰（각 길이 1）

이다.

### 3.2 Static Batching(정적 배치)

**Static Batching**은 배치를 구성한 뒤, 그 멤버가 **함께 시작해서 함께（또는 전원 종료까지）** 가는 방식에 가깝다.

```text
큐에서 B개 수집 → 패딩으로 길이 정렬 → 생성 루프
→ 전원 max_tokens 또는 전원 EOS까지（구현마다 다름）
→ 다음 배치
```

장점（설명）:

- 구현이 단순
- 텐서 shape가 비교적 고정

단점（설명）:

- 빠른 요청이 느린 요청을 기다림（convoy 효과）
- 패딩 낭비
- 중간에 생긴 GPU 여유를 새 요청에 바로 못 씀

### 3.3 Continuous Batching(연속 배치)

**Continuous Batching**（또는 dynamic / iteration-level batching으로 불리기도 함）은 **매 디코드 스텝（또는 매 엔진 iteration）마다** 배치 멤버십을 갱신하는 방식이다.

```text
iteration i:
  - 끝난 요청 제거 (KV free)
  - 대기 요청 중 예산 되는 것을 추가 (prefill 또는 decode 합류)
  - 현재 running set에 대해 GPU step 실행
iteration i+1:
  - 반복
```

**사실:** 명칭은 문헌·제품 문서에 따라 Continuous batching, in-flight batching, iteration-level scheduling 등으로 다양하다.  
**설명:** 공통 아이디어는 **요청의 수명이 배치 수명과 분리**된다는 것이다.

### 3.4 Iteration이란?

여기서 **Iteration(이터레이션)**은 엔진이 running 요청 집합에 대해 **한 번의 모델 진행**을 수행하는 주기이다.

- Decode만 있는 iteration: 각 요청에 토큰 약 1개
- Prefill이 섞인 iteration: 새 요청의 프롬프트를 처리（구현에 따라 chunk prefills）

제106강（vLLM Scheduler）에서 정책을 더 깊게 본다. 오늘은 **왜 연속이어야 하는지**다.

### 3.5 Prefill–Decode 간섭

같은 GPU에서

- 큰 Prefill（긴 $L$）과
- 많은 Decode（동시 사용자）

가 경쟁하면, Decode의 토큰 간격이 늘어나 체감이 나빠질 수 있다.

**설명:** 엔진은 예산·우선순위·chunked prefill 등으로 간섭을 완화하려 한다.  
**사실:** 구체 알고리즘은 엔진·버전마다 다르다. 이 책에서 한 구현을 “표준 사실”로 고정하지 않는다.

## 4. 직관적으로 이해하기

엘리베이터 비유（**설명**）:

| Static | Continuous |
|---|---|
| 4명이 다 탈 때까지 문 닫고, 모두 목적층까지 간 뒤 다시 탐 | 층마다 문 열어 내리고 태움 |
| 1층 가는 사람이 20층까지 동행 | 목적층에서 즉시 하차 |

식당 주방:

- Static: 한 테이블 전원 주문이 나와야 다음 테이블
- Continuous: 완성된 접시부터 내고, 빈 화구에 새 주문을 올림

**해석:** GPU의 “화구”는 행렬곱 유닛·대역폭이다. 빈 자리를 즉시 채우는 것이 Continuous Batching의 직관이다.

## 5. 수학적으로 이해하기（대기·낭비 감각）

### 5.1 패딩 낭비

배치 안 요청 길이가 $\ell_1,\ldots,\ell_B$이고 패딩 목표 길이 $\ell_{\max}=\max_i \ell_i$일 때, 유용 비율 감각은

$$

\frac{\sum_i \ell_i}{B\cdot \ell_{\max}}

$$

예（**가정**）: 길이 128, 512, 2048, 2048 → 합 4736, $B\cdot\max=8192$, 비율 $\approx 0.58$.  
약 42%는 패딩일 수 있다（이 숫자도 가정 시나리오）.

### 5.2 Convoy（호위）효과

배치 내 잔여 생성 토큰이 $r_1,\ldots,r_B$일 때, static이 “전원 종료”를 기다리면 배치 수명은 대략

$$

\max_i r_i

$$

에 묶인다. 이미 끝난 슬롯은 idle.

Continuous면 요청 $i$의 수명은 대략 자기 $r_i$（+ 스케줄 지연）에 가깝게 줄일 **여지가** 있다.

### 5.3 처리량 vs 지연

이상화 기호:

- $\lambda$: 요청 도착률
- $X$: 요청당 서비스 시간（랜덤）
- $S$: 시스템 용량（동시 running·KV 예산）

**설명:** Continuous Batching은 $S$를 더 꽉 채워 $\lambda$를 더 받게 돕는 쪽에 가깝다.  
그러나 running이 과도하면 Decode step 시간이 늘어 **개인 지연**이 악화될 수 있다.  
이 관계는 측정으로 확인한다（제107강）. 여기서 Little의 법칙 숫자를 날조하지 않는다.

## 6. 작은 시나리오로 직접 추적하기（예시·가정）

시간축을 스텝으로 본다. Decode만 단순화.

| 시각 t | 사건 | Static(배치=2) 가정 | Continuous 가정 |
|---|---|---|---|
| 0 | A,B 시작（잔여 4,2） | A,B running | A,B running |
| 1 | | A:3,B:1 | A:3,B:1 |
| 2 | B 종료, C 대기 | A:2,B:0（B 빈자리 유지?） | B free, C 합류 |
| 3 | | A:1,（빈） | A:2,C:… |
| 4 | A 종료 | 배치 종료 후 C 시작 | … |

Static의 빈 칸 처리는 구현마다 다르지만, **“다음 배치까지 C가 못 들어옴”** 이 전형적 낭비다.  
Continuous는 **t=2에 C 합류**가 가능하다는 스토리다.

다른 시나리오（Prefill）:

- t=0: Decode 8개 running
- t=1: 긴 Prefill D 도착
- D를 한 iteration에 전부 넣으면 Decode 8개의 간격이 한동안 벌어질 수 있다

**설명:** chunked prefill은 긴 Prefill을 조각내 Decode와 시간을 나누는 대응 중 하나다. 세부는 엔진 문서·제106강.

## 7. 코드로 구현하기 — 교육용 스케줄러

실제 GPU 커널 없이, **요청 집합의 멤버십**만 시뮬레이션한다.

```python
from dataclasses import dataclass, field
from typing import List, Optional
import itertools

@dataclass
class Request:
    rid: str
    remaining_decode: int
    waiting_prefill: bool = False
    prefill_left: int = 0  # chunk 단위 가정

@dataclass
class Engine:
    running: List[Request] = field(default_factory=list)
    waiting: List[Request] = field(default_factory=list)
    max_running: int = 4
    finished: List[str] = field(default_factory=list)

    def enqueue(self, req: Request):
        self.waiting.append(req)

    def admit(self):
        while self.waiting and len(self.running) < self.max_running:
            self.running.append(self.waiting.pop(0))

    def step(self):
        """Continuous: 매 스텝 종료자 제거 후 신규 입학."""
        self.admit()
        still = []
        for r in self.running:
            if r.waiting_prefill and r.prefill_left > 0:
                # 가정: 한 스텝에 prefill chunk 1
                r.prefill_left -= 1
                if r.prefill_left == 0:
                    r.waiting_prefill = False
                still.append(r)
            else:
                r.remaining_decode -= 1
                if r.remaining_decode > 0:
                    still.append(r)
                else:
                    self.finished.append(r.rid)
        self.running = still
        self.admit()

def demo():
    eng = Engine(max_running=2)
    eng.enqueue(Request("A", remaining_decode=4))
    eng.enqueue(Request("B", remaining_decode=2))
    eng.enqueue(Request("C", remaining_decode=3))
    for t in range(10):
        eng.step()
        print(t, "running", [r.rid for r in eng.running], "done", eng.finished)

demo()
```

관찰할 것:

- B가 끝나면 C가 **다음 배치를 기다리지 않고** 들어온다
- `max_running`이 KV/예산 상한의 장난감 모델이다

### Static 흉내

```python
def static_batches(reqs, batch_size):
    it = iter(reqs)
    while True:
        batch = list(itertools.islice(it, batch_size))
        if not batch:
            break
        # 배치 수명 = max remaining
        life = max(r.remaining_decode for r in batch)
        yield [r.rid for r in batch], life
```

같은 A,B,C에 대해 static `batch_size=2`면 첫 배치 수명이 `max(4,2)=4`인 동안 C는 대기한다（시나리오 가정）.

## 8. PyTorch와는 어떻게 연결되는가

Continuous Batching 자체는 **autograd API가 아니다**.  
다만 한 iteration에 올리는 텐서는 대략

```text
- prefill용 input_ids (가변 길이 → 패딩 또는 ragged/packed 표현)
- decode용 input_ids (대개 [R, 1])
- 각 요청의 KV 포인터/블록 테이블
```

형태가 된다.

교육용으로 Decode-only 배치를 패딩 없이 **스택**만 한다고 가정:

```python
import torch

def decode_batch_step(model, token_batch, kv_list):
    """
    token_batch: [R, 1]
    kv_list: 길이 R의 캐시 핸들（실제로는 블록 테이블）
    """
    # 실제 엔진은 커스텀 커널로 한번에 처리.
    # 아래는 “논리적으로 R번”을 드러내는 느린 참고 구현.
    outs = []
    new_kvs = []
    for i in range(token_batch.size(0)):
        logits, kv = model.forward_one(token_batch[i:i+1], kv_list[i])
        outs.append(logits)
        new_kvs.append(kv)
    return torch.cat(outs, dim=0), new_kvs
```

**설명:** 느린 루프는 개념용이다. 고성능 엔진은 배치를 **하나의 fused kernel**로 처리한다.  
**사실:** vLLM이 내부에서 어떤 커널을 쓰는지는 버전·백엔드에 따라 다르며, 여기서 파일 단위로 단정하지 않는다.

## 9. LLM Serving 연결

### 9.1 지표와의 관계（정의만）

| 목표 | Continuous가 돕는 방향（설명） |
|---|---|
| Throughput（처리량） | GPU idle·패딩 감소로 토큰/초↑ **여지** |
| TTFT | 입학이 빨라지면 개선 **여지** / Prefill 간섭 시 악화 **여지** |
| TPOT | running 과다 시 악화 **여지** |
| Fairness | 스케줄 정책에 의존 |

제107강에서 TTFT·TPOT·Throughput을 정식으로 다룬다. 오늘은 인과 화살표만.

### 9.2 KV 예산이 곧 동시성

`max_running`을 키우고 싶어도

$$

\sum_i \mathrm{KV}(t_i) \le \mathrm{Budget}

$$

을 넘으면 입학을 거절하거나 스왑해야 한다.  
Continuous Batching은 **빈 예산을 즉시 재사용**하게 해 예산을 더 잘 쓰게 한다. 예산 자체를 늘리지는 않는다（양자화·페이지 효율이 예산을 실질적으로 키울 수 있음）.

### 9.3 vLLM과의 위치

공개 개념상 vLLM은

- Continuous batching
- PagedAttention（KV 블록）
- 스케줄러

를 결합한 Inference Engine으로 널리 소개된다（제104~106강）.  
오늘은 “왜 continuous가 필요한가”까지만.

## 10. 실습

### 실습 A — 패딩 비율

길이 목록을 하나 정해

$$

\frac{\sum \ell_i}{B \max \ell_i}

$$

를 계산한다. 길이 편의를 키운 목록과 비교한다.

### 실습 B — 시뮬레이터 확장

§8 코드에

- Prefill 대기 필드
- `max_running`을 바꿔 finished까지의 스텝 수 비교

로그를 남긴다. **이 스텝 수는 가상의 이산 시간**이지 GPU ms가 아니다.

### 실습 C — Static vs Continuous 표

A(5), B(1), C(5), D(1), batch=2 가정으로 두 방식의 “C의 시작 시각”을 표로 만든다.

### 실습 D — 간섭 문장 쓰기

“긴 Prefill이 Decode TPOT를 해칠 수 있다”를 사실/설명으로 구분해 두 문장으로 쓴다.

### 실습 E — 제품 문장 교정

원문: “Continuous Batching을 켜면 항상 지연이 절반이 된다.”  
교정문을 작성（수치 제거·조건 명시）.

## 11. 자주 하는 실수

1. **배치 크기↑ = Continuous**라고 생각  
   핵심은 멤버십이 **시간에 따라 갱신**되는가다.

2. **학습 DataLoader 배치와 동일시**  
   온라인은 도착 과정·가변 길이·종료 시각이 다르다.

3. **처리량만 보고 running을 무한히 키움**  
   TPOT·TTFT·OOM이 동시에 움직일 수 있다.

4. **패딩 없는 ragged를 항상 실측 가능하다고 착각**  
   커널·프레임워크 지원에 의존한다.

5. **스케줄러 공정성 무시**  
   긴 요청이 항상 우선이면 짧은 요청이 굶는다（정책 문제）.

6. **벤치 그래프 숫자를 맥락 없이 이식**  
   워크로드（입출력 길이 분포）가 반이다.

## 12. 핵심 정리

- Static Batching은 배치 수명에 요청 수명을 묶기 쉽다.
- Continuous Batching은 iteration마다 종료·입학을 반영한다.
- 패딩·호위 효과·idle 슬롯이 동기 배치의 적이다.
- Prefill과 Decode의 간섭은 별도 설계 포인트다.
- KV 예산이 동시성의 천장이고, continuous는 그 천장을 **더 잘 사용**하게 돕는다.

## 13. 핵심 용어

| 용어 | 한 줄 의미 |
|---|---|
| Static Batching | 고정 멤버로 묶여 진행하는 배치 |
| Continuous Batching | 스텝마다 멤버십 갱신하는 배치 |
| Iteration-level scheduling | 엔진 iteration 단위로 스케줄 |
| Padding waste | 길이 정렬로 생기는 무효 계산 |
| Convoy effect | 느린 작업이 빠른 작업을 붙잡음 |
| Admission | KV/슬롯 예산 하에서 새 요청 입학 |
| Chunked prefill | 긴 Prefill을 조각내 실행하는 기법（개요） |
| Throughput / Latency | 처리량 / 지연（제107강） |

## 14. 연습 문제
### 문제 1（정의）

Continuous Batching을 “배치 크기”가 아니라 **무엇의 분리**로 정의하시오.

### 문제 2（낭비）

길이 100, 100, 800인 배치의 패딩 유용 비율을 가정 계산하시오.

### 문제 3（시나리오）

B가 먼저 끝났는데 static batch가 C를 못 넣는 이유를 한 줄로.

### 문제 4（트레이드오프）

running 요청 수를 과도하게 올리면 어떤 지표가 악화될 **수 있는지** 두 개 쓰시오.

### 문제 5（KV）

Continuous Batching이 KV 예산을 “늘리는”가, “잘 쓰는”가?

### 문제 6（용어）

Iteration-level scheduling을 한 줄로.

### 문제 7（다리）

제103강 제목을 쓰고, 양자화가 Continuous Batching의 동시성에 주는 간접 효과를 한 줄로.

---

## 정답 및 해설

### 문제 1

요청 수명과 배치（iteration 집합）수명의 분리. 멤버십의 동적 갱신.

### 문제 2

합 1000, $B\max=2400$, 비율 $1000/2400\approx 0.417$.

### 문제 3

배치 멤버가 고정되어 있거나 다음 배치 수집 시점까지 신규 입학이 막히기 때문（static 전형）.

### 문제 4

예: TPOT, TTFT（큐/간섭）, 또는 OOM 위험. 측정 의존.

### 문제 5

잘 쓰는 쪽（즉시 반환·재입학）. 예산 자체를 늘리지는 않음.

### 문제 6

매 엔진 스텝마다 running 집합을 스케줄·실행하는 방식.

### 문제 7

제목: Quantization — INT8, INT4, FP8.  
효과: 가중치（또는 KV）메모리가 줄면 동일 VRAM에서 KV/동시성 여유가 생길 **수 있다**.

## 15. 다음 강의와 연결

스케줄이 예산을 **잘 쓰게** 했다면, 다음은 예산 자체를 바꾸는 축이다.  
다음 **제103강. Quantization — INT8, INT4, FP8**에서는 가중치·활성값을 낮은 비트로 표현해 메모리·대역폭을 줄이는 아이디어를, 품질 트레이드오프와 함께 다룬다.

> 빈자리를 채우는 기술 다음에, 자리 자체를 좁히는 기술이 온다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [제101강. KV Cache](101강_KV_Cache.md)
- **다음 강:** [제103강. Quantization — INT8, INT4, FP8](103강_Quantization_INT8_INT4_FP8.md)

<!-- /LECTURE_NAV -->
